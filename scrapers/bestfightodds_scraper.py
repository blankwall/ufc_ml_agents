"""
BestFightOdds Scraper
Scrapes moneyline odds for UFC events from bestfightodds.com.

Usage (CLI):
    # Single event by search query
    python scrapers/bestfightodds_scraper.py --events "UFC 300" "UFC 301"

    # Range of numbered UFC events
    python scrapers/bestfightodds_scraper.py --ufc-range 200 310

    # Append to (or create) the master historical odds file
    python scrapers/bestfightodds_scraper.py --events "UFC 300" --output data/odds/historical_odds.csv

Output CSV columns:
    event_name, event_date, event_url,
    fighter1, fighter2, fighter1_odds, fighter2_odds,
    fighter1_prob, fighter2_prob
"""

import re
import sys
import time
import argparse
import unicodedata
from pathlib import Path
from typing import Optional
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger


# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL    = "https://www.bestfightodds.com"
SEARCH_URL  = "https://www.bestfightodds.com/search"
OUTPUT_DIR  = Path("data/odds")
MASTER_FILE = OUTPUT_DIR / "historical_odds.csv"
CACHE_DIR   = Path(".cache/bfo")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Referer": "https://www.bestfightodds.com/",
}

RATE_LIMIT_SECS = 2.0  # pause between requests to be polite


# ── Helpers ──────────────────────────────────────────────────────────────────

def _american_to_prob(odds: int) -> float:
    """Convert American moneyline odds to implied probability (0–1)."""
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace. Used for fight_key."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_name).strip().lower()


def _fight_key(f1: str, f2: str) -> str:
    """Stable fight key regardless of which fighter is listed first."""
    a, b = sorted([_normalize_name(f1), _normalize_name(f2)])
    return f"{a}_vs_{b}"


def _extract_odds_from_span(span) -> Optional[int]:
    """
    Pull an American odds integer from a BeautifulSoup span element.
    Strips arrow chars (▲▼) and whitespace.
    Returns None if text isn't a valid odds value.
    """
    if span is None:
        return None
    raw = span.get_text(strip=True).replace("▲", "").replace("▼", "").strip()
    try:
        val = int(raw)
        # sanity check: odds are in [-10000, 10000] range
        if -10000 <= val <= 10000 and val != 0:
            return val
    except (ValueError, TypeError):
        pass
    return None


# ── Main Scraper Class ────────────────────────────────────────────────────────

class BestFightOddsScraper:
    """
    Scrapes moneyline odds for UFC events from bestfightodds.com.

    The page has two <table class="odds-table"> elements:
      - Table 0: responsive sticky-header (only <th> rows, no odds data)
      - Table 1: full data table with <th> (fighter/prop label) + <td> (odds) per row

    Fighter matchup rows are identified by <tr id="mu-XXXXX">.
    Prop rows have class="pr" and are skipped.
    """

    def __init__(self, rate_limit: float = RATE_LIMIT_SECS):
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Network ──────────────────────────────────────────────────────────────

    def _get(self, url: str, use_cache: bool = True) -> str:
        """Fetch URL, use disk cache if available. Returns raw HTML string."""
        cache_key = re.sub(r"[^a-z0-9]", "_", url.lower()) + ".html"
        cache_path = CACHE_DIR / cache_key

        if use_cache and cache_path.exists():
            logger.debug(f"Cache hit: {cache_path.name}")
            return cache_path.read_text(encoding="utf-8")

        logger.info(f"GET {url}")
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        html = resp.text
        cache_path.write_text(html, encoding="utf-8")
        time.sleep(self.rate_limit)
        return html

    # ── Search ───────────────────────────────────────────────────────────────

    def _parse_search_results(self, html: str) -> list[dict]:
        """Parse event results from a BFO search results page."""
        soup = BeautifulSoup(html, "lxml")
        results = []
        for row in soup.select("table td"):
            a = row.find("a", href=re.compile(r"^/events/"))
            if a:
                href = a["href"].strip()
                name = a.get_text(strip=True)
                parent = row.parent
                tds = parent.find_all("td")
                date_str = tds[0].get_text(strip=True) if tds else ""
                results.append({
                    "name":     re.sub(r"\s+Odds$", "", name).strip(),
                    "url":      BASE_URL + href,
                    "date_str": date_str,
                })
        return results

    def _date_str_to_year(self, date_str: str) -> Optional[int]:
        """Parse year from BFO date strings like 'Apr 15th 2023'."""
        m = re.search(r"\b(\d{4})\b", date_str)
        return int(m.group(1)) if m else None

    def search_event(
        self,
        query: str,
        expected_year: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Search BFO for an event by name with smart fallback.

        Strategy:
        1. Search the full event name (exact match wins immediately)
        2. If no exact match and the name has a 'vs.' pattern, extract just
           the fighter names and search again — BFO finds Fight Night events
           much more reliably this way ("Holloway Imavov" beats
           "UFC Fight Night: Holloway vs. Imavov")
        3. If expected_year is set, prefer results from that year

        Returns dict: {name, url, date_str} or None if not found.
        """
        def _best_result(results: list[dict], name_clean: str) -> Optional[dict]:
            if not results:
                return None
            # Exact name match (case-insensitive)
            for r in results:
                if r["name"].lower() == name_clean:
                    return r
            # Year-filtered: pick the first result from the expected year
            if expected_year:
                year_matches = [r for r in results
                                if self._date_str_to_year(r["date_str"]) == expected_year]
                if year_matches:
                    return year_matches[0]
            # Last resort: first result only if it looks like a UFC event
            r = results[0]
            if "ufc" in r["name"].lower():
                return r
            return None

        query_clean = query.strip().lower()

        # ── Pass 1: exact name search ────────────────────────────────────────
        html = self._get(f"{SEARCH_URL}?query={requests.utils.quote(query)}")
        results = self._parse_search_results(html)

        # Immediate exact match
        for r in results:
            if r["name"].lower() == query_clean:
                logger.info(f"Exact match: {r['name']} → {r['url']}")
                return r

        # ── Pass 2: fighter-name search (for "Event: Fighter1 vs. Fighter2") ─
        # Extract "Fighter1 Fighter2" from event names containing a colon+vs.
        fighter_query = None
        if ":" in query:
            tail = query.split(":", 1)[1].strip()
            # "Fighter1 vs. Fighter2" → "Fighter1 Fighter2"
            fighters = re.split(r"\s+vs\.?\s+", tail, flags=re.IGNORECASE)
            if len(fighters) == 2:
                fighter_query = f"{fighters[0].strip()} {fighters[1].strip()}"

        if fighter_query:
            html2 = self._get(f"{SEARCH_URL}?query={requests.utils.quote(fighter_query)}")
            results2 = self._parse_search_results(html2)
            r = _best_result(results2, query_clean)
            if r:
                logger.info(
                    f"Fighter search match: {r['name']} ({r['date_str']}) → {r['url']}"
                )
                return r

        # ── Pass 3: fall back to pass-1 results with year filter ─────────────
        r = _best_result(results, query_clean)
        if r:
            logger.info(
                f"Year-filtered match: {r['name']} ({r['date_str']}) → {r['url']}"
            )
            return r

        logger.warning(f"No match found for: '{query}'")
        return None

    # ── Event Page Parsing ───────────────────────────────────────────────────

    def _parse_odds_from_row(self, tr) -> list[Optional[int]]:
        """
        Given a fighter <tr>, extract all non-empty moneyline odds across bookmakers.

        Priority:
        1. <span class="bestbet"> — the best available line highlighted by BFO
        2. Any <span> inside a <td> with a valid odds value

        Returns a list of integers (one per bookmaker column that has a value).
        """
        odds_values = []
        for td in tr.find_all("td"):
            # Skip non-odds cells (button cells, prop cells, etc.)
            if "button-cell" in td.get("class", []):
                continue
            if "prop-cell" in td.get("class", []):
                continue

            # Prefer the "bestbet" span (most competitive line)
            best = td.find("span", class_="bestbet")
            val = _extract_odds_from_span(best)
            if val is not None:
                odds_values.append(val)
                continue

            # Fall back to any non-arrow span in this cell
            for span in td.find_all("span"):
                cls = span.get("class", [])
                if "ard" in cls or "aru" in cls:
                    continue  # these are just the arrow indicators
                val = _extract_odds_from_span(span)
                if val is not None:
                    odds_values.append(val)
                    break

        return odds_values

    def _best_odds(self, values: list[int], is_favourite: bool = False) -> Optional[int]:
        """
        Choose a single representative odds value from the list of bookmaker lines.

        Strategy: return the BEST available line (highest value for underdog,
        least negative for favourite). This represents the true market consensus
        at the time because BFO showed this as the top line.

        If only one value, return it directly.
        If empty, return None.
        """
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        # Best line for favourite (negative odds) = closest to 0 (less juice)
        # Best line for underdog (positive odds) = highest number
        # Since we don't always know which is which, use the median to be
        # conservative and avoid outliers from boosted lines.
        return int(sorted(values)[len(values) // 2])

    def scrape_event(self, event_url: str, event_name: str = "", event_date: str = "") -> list[dict]:
        """
        Scrape all moneyline fight odds from a BFO event page.

        Returns list of dicts:
            event_name, event_date, event_url,
            fighter1, fighter2, fighter1_odds, fighter2_odds,
            fighter1_prob, fighter2_prob

        BFO renders TWO <table class="odds-table"> elements:
          - Table 0 (odds-table-responsive-header): sticky left column, only <th>,
            fighter rows carry id="mu-XXXXX" but have NO <td> cells.
          - Table 1 (odds-table): full rows with both <th> (name) and <td> (odds),
            but fighter rows have NO id attribute.

        Strategy: parse fighter names from Table 0 (has mu- IDs), extract odds
        from Table 1 (has <td> cells) using matching row positions.
        """
        html = self._get(event_url)
        soup = BeautifulSoup(html, "lxml")

        # ── Pull event metadata from page header if not provided ────────────
        if not event_name:
            h1 = soup.find("h1")
            event_name = h1.get_text(strip=True) if h1 else event_url.split("/")[-1]

        # BFO sometimes appends " Odds" or " for <Date>" to event names
        event_name = re.sub(r"\s+Odds$", "", event_name).strip()
        event_name = re.sub(r"\s+for\s+\w+\s+\d+$", "", event_name).strip()

        if not event_date:
            date_span = soup.find("span", class_="table-header-date")
            event_date = date_span.get_text(strip=True) if date_span else ""

        # ── Locate both tables ──────────────────────────────────────────────
        tables = soup.find_all("table", class_="odds-table")
        if len(tables) < 2:
            logger.warning(f"Expected 2 odds tables, found {len(tables)} on {event_url}")
            return []

        # Table 0 = sticky header (has mu- IDs but no td odds)
        # Table 1 = data table (has td odds but no mu- IDs)
        header_table = tables[0]
        data_table   = tables[1]

        header_rows = header_table.find("tbody").find_all("tr", recursive=False)
        data_rows   = data_table.find("tbody").find_all("tr", recursive=False)

        if len(header_rows) != len(data_rows):
            logger.warning(
                f"Row count mismatch: header={len(header_rows)} data={len(data_rows)} — "
                "will try positional matching anyway"
            )

        # ── Build ordered list of (row_index, fighter_name, is_f1) from header_table
        # Fighter rows: have a /fighters/ link. Non-prop rows without fighter links
        # are rare but possible (e.g. blank rows). We identify fights by mu-XXXXX rows.
        fighter_row_indices = []  # list of (index, name, is_matchup_start)
        for idx, row in enumerate(header_rows):
            row_classes = row.get("class", [])
            if "pr" in row_classes:
                continue  # prop row
            th = row.find("th", scope="row")
            if not th:
                continue
            link = th.find("a", href=re.compile(r"^/fighters/"))
            if link:
                is_start = row.get("id", "").startswith("mu-")
                fighter_row_indices.append((idx, link.get_text(strip=True), is_start))

        # ── Pair fighters and extract odds from data_table ──────────────────
        fights = []
        i = 0
        while i < len(fighter_row_indices):
            idx1, f1_name, is_start = fighter_row_indices[i]

            # A new fight always starts with a mu- row (is_start=True)
            if not is_start:
                i += 1
                continue

            # Next entry must be the second fighter (is_start=False)
            if i + 1 >= len(fighter_row_indices):
                break
            idx2, f2_name, is_f2_start = fighter_row_indices[i + 1]

            if is_f2_start:
                # Two consecutive mu- rows would be unusual; skip and try next
                logger.warning(f"Unexpected consecutive mu- rows at index {i}")
                i += 1
                continue

            # Extract odds from the data table using the same row indices
            if idx1 < len(data_rows) and idx2 < len(data_rows):
                f1_odds_raw = self._parse_odds_from_row(data_rows[idx1])
                f2_odds_raw = self._parse_odds_from_row(data_rows[idx2])
            else:
                logger.warning(f"Row index out of range for {f1_name} vs {f2_name}")
                i += 2
                continue

            f1_odds = self._best_odds(f1_odds_raw)
            f2_odds = self._best_odds(f2_odds_raw)

            if f1_odds is None or f2_odds is None:
                logger.warning(
                    f"Missing odds for {f1_name} vs {f2_name} "
                    f"(f1_raw={f1_odds_raw}, f2_raw={f2_odds_raw}) — skipping"
                )
                i += 2
                continue

            f1_prob = _american_to_prob(f1_odds)
            f2_prob = _american_to_prob(f2_odds)

            fights.append({
                "event_name":    event_name,
                "event_date":    event_date,
                "event_url":     event_url,
                "fighter1":      f1_name,
                "fighter2":      f2_name,
                "fighter1_odds": f1_odds,
                "fighter2_odds": f2_odds,
                "fighter1_prob": round(f1_prob, 4),
                "fighter2_prob": round(f2_prob, 4),
            })
            logger.info(
                f"  {f1_name} ({f1_odds:+d}) vs {f2_name} ({f2_odds:+d})"
            )
            i += 2

        logger.success(f"Scraped {len(fights)} fights from {event_name}")
        return fights

    # ── Batch ────────────────────────────────────────────────────────────────

    def scrape_events(
        self,
        queries: list[str],
        expected_years: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Search and scrape multiple events by query string.

        expected_years: optional list of ints (same length as queries) with the
        expected calendar year for each event. Used to disambiguate search results.

        Returns a combined DataFrame of all fights.
        """
        all_fights = []
        for i, query in enumerate(queries):
            year = expected_years[i] if expected_years and i < len(expected_years) else None
            logger.info(f"── Searching: {query}" + (f" ({year})" if year else ""))
            event = self.search_event(query, expected_year=year)
            if event is None:
                logger.warning(f"Skipping '{query}' — not found")
                continue
            fights = self.scrape_event(
                event_url=event["url"],
                event_name=event["name"],
                event_date=event["date_str"],
            )
            all_fights.extend(fights)

        if not all_fights:
            return pd.DataFrame()

        df = pd.DataFrame(all_fights)
        return df

    def scrape_ufc_range(self, start: int, end: int) -> pd.DataFrame:
        """
        Scrape a range of numbered UFC events: UFC {start} through UFC {end}.

        Example: scrape_ufc_range(200, 310) fetches UFC 200, 201, …, 310.
        """
        queries = [f"UFC {n}" for n in range(start, end + 1)]
        return self.scrape_events(queries)

    # ── Save / Merge ─────────────────────────────────────────────────────────

    def save(
        self,
        df: pd.DataFrame,
        output_path: str | Path = MASTER_FILE,
        dedupe: bool = True,
    ) -> Path:
        """
        Save scraped odds to CSV.

        If output_path already exists, APPENDS new rows (deduplicates by
        fighter1+fighter2+event_name to avoid double-counting).

        Returns the path written to.
        """
        if df.empty:
            logger.warning("Nothing to save — DataFrame is empty")
            return Path(output_path)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and dedupe:
            existing = pd.read_csv(output_path)
            combined = pd.concat([existing, df], ignore_index=True)

            # Deduplication key: event_name + fighter pair (order-insensitive)
            combined["_dedup_key"] = combined.apply(
                lambda r: f"{r['event_name']}|{_fight_key(r['fighter1'], r['fighter2'])}",
                axis=1,
            )
            before = len(combined)
            combined = combined.drop_duplicates(subset="_dedup_key", keep="last")
            combined = combined.drop(columns=["_dedup_key"])
            logger.info(f"Deduped: {before} → {len(combined)} rows")
        else:
            combined = df

        combined.to_csv(output_path, index=False)
        logger.success(f"Saved {len(combined)} rows → {output_path}")
        return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape UFC moneyline odds from BestFightOdds.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape specific events (enclose multi-word names in quotes):
  python scrapers/bestfightodds_scraper.py --events "UFC 300" "UFC 301" "UFC 302"

  # Scrape a range of numbered UFC events:
  python scrapers/bestfightodds_scraper.py --ufc-range 200 310

  # Custom output path (default: data/odds/historical_odds.csv):
  python scrapers/bestfightodds_scraper.py --events "UFC 307" --output data/odds/ufc_307.csv

  # Skip disk cache (force re-fetch):
  python scrapers/bestfightodds_scraper.py --events "UFC 300" --no-cache
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--events",
        nargs="+",
        metavar="QUERY",
        help="One or more event search queries, e.g. 'UFC 300' 'UFC Fight Night 240'",
    )
    group.add_argument(
        "--ufc-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Scrape UFC {START} through UFC {END}, inclusive",
    )

    parser.add_argument(
        "--output",
        default=str(MASTER_FILE),
        help=f"Output CSV path (default: {MASTER_FILE}). Appends if file exists.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass disk cache and re-fetch all pages",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=RATE_LIMIT_SECS,
        help=f"Seconds between requests (default: {RATE_LIMIT_SECS})",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Replace output file instead of appending",
    )

    args = parser.parse_args()

    scraper = BestFightOddsScraper(rate_limit=args.rate_limit)

    if args.events:
        df = scraper.scrape_events(args.events)
    else:
        start, end = args.ufc_range
        df = scraper.scrape_ufc_range(start, end)

    if df.empty:
        logger.warning("No data scraped. Check your event names or BFO connectivity.")
        sys.exit(1)

    print("\n── Preview ──────────────────────────────────────────────────────")
    print(df.to_string(index=False))
    print(f"\nTotal fights scraped: {len(df)}")

    scraper.save(df, output_path=args.output, dedupe=not args.no_merge)


if __name__ == "__main__":
    main()
