"""
Bovada Odds Scraper
Scrapes moneyline odds for UFC events from Bovada.lv API.

Usage (CLI):
    # Scrape all upcoming UFC events
    python scrapers/bovada_scraper.py

    # Scrape a specific event by URL slug
    python scrapers/bovada_scraper.py --event ufc-fight-night-evloev-murphy

    # Custom output path
    python scrapers/bovada_scraper.py --output data/odds/bovada_latest.csv

Output CSV columns:
    event_name, event_date, fighter1, fighter2,
    fighter1_odds, fighter2_odds,
    fighter1_prob, fighter2_prob, scrape_timestamp
"""

import argparse
import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL = "https://www.bovada.lv"
NAV_URL = f"{BASE_URL}/services/sports/event/v2/nav/A/description/ufc-mma/ufc"
COUPON_URL = f"{BASE_URL}/services/sports/event/coupon/events/A/description"
OUTPUT_DIR = Path("data/odds")
MASTER_FILE = OUTPUT_DIR / "bovada_odds.csv"
CACHE_DIR = Path(".cache/bovada")
CACHE_EXPIRY_SECS = 3600  # 1 hour

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{BASE_URL}/sports/ufc-mma",
    "Origin": BASE_URL,
}
RATE_LIMIT_SECS = 1.0


# ── Helpers ──────────────────────────────────────────────────────────────────


def _american_to_prob(odds: int) -> float:
    """Convert American moneyline odds to implied probability (0-1)."""
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _normalize_name(name: str) -> str:
    """Normalize fighter name for deduplication."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_name).strip().lower()


def _fight_key(f1: str, f2: str) -> str:
    """Create stable fight key regardless of fighter order."""
    a, b = sorted([_normalize_name(f1), _normalize_name(f2)])
    return f"{a}_vs_{b}"


# ── Main Scraper Class ────────────────────────────────────────────────────────


class BovadaScraper:
    """Scrapes moneyline odds for UFC events from Bovada.lv API."""

    def __init__(self, rate_limit: float = RATE_LIMIT_SECS):
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._request_count = 0
        self._error_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, url: str) -> Path:
        """Generate cache key from URL."""
        safe_key = re.sub(r"[^a-zA-Z0-9]", "_", url)
        return CACHE_DIR / f"{safe_key}.json"

    def _get_cached(self, url: str) -> Optional[dict]:
        """Get cached response if available and not expired."""
        cache_path = self._get_cache_key(url)
        if not cache_path.exists():
            self._cache_misses += 1
            return None
        try:
            content = cache_path.read_text()
            data = json.loads(content)
            cached_time = data.get("timestamp", 0)
            if time.time() - cached_time > CACHE_EXPIRY_SECS:
                self._cache_misses += 1
                return None
            self._cache_hits += 1
            return data.get("data")
        except (json.JSONDecodeError, KeyError):
            self._cache_misses += 1
            return None

    def _set_cached(self, url: str, data: dict) -> None:
        """Cache response data."""
        cache_path = self._get_cache_key(url)
        cache_data = {"data": data, "timestamp": time.time()}
        cache_path.write_text(json.dumps(cache_data))

    def _get_json(self, url: str, use_cache: bool = True) -> Optional[dict | list]:
        """Fetch URL and return JSON, use disk cache if available."""
        if use_cache:
            cached = self._get_cached(url)
            if cached:
                logger.debug(f"Cache hit: {url}")
                return cached

        logger.info(f"GET {url}")
        self._request_count += 1
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            self._set_cached(url, data)
            time.sleep(self.rate_limit)
            return data
        except requests.RequestException as e:
            self._error_count += 1
            logger.error(f"Request failed: {url} - {e}")
            return None

    def _parse_american_odds(self, price: dict) -> Optional[int]:
        """Extract American odds from price dict."""
        american = price.get("american")
        if american is None:
            return None
        try:
            if isinstance(american, str):
                return int(american)
            return int(american)
        except (ValueError, TypeError):
            logger.warning(f"Invalid American odds: {american}")
            return None

    def get_upcoming_events(self) -> list[dict]:
        """Get list of upcoming UFC events from navigation API."""
        data = self._get_json(f"{NAV_URL}?azSorting=true&lang=en")
        if not data:
            logger.error("Failed to fetch upcoming events")
            return []

        children = data.get("children", [])
        events = []
        for child in children:
            if child.get("numEvents", 0) > 0:
                events.append({
                    "name": child.get("description", ""),
                    "link": child.get("link", ""),
                    "num_events": child.get("numEvents", 0),
                })

        logger.info(f"Found {len(events)} upcoming UFC events")
        return events

    def scrape_event(self, event_slug: str) -> list[dict]:
        """
        Scrape all moneyline fight odds from a specific Bovada event.

        Args:
            event_slug: URL slug like "ufc-fight-night-evloev-murphy"

        Returns list of dicts with fighter names and odds.
        """
        url = f"{COUPON_URL}/ufc-mma/ufc/{event_slug}?marketFilterId=def&preMatchOnly=true"
        data = self._get_json(url)

        if not data:
            logger.error(f"Failed to fetch event: {event_slug}")
            return []

        # Handle response structure - can be list or dict
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        events = data.get("events", [])
        path_info = data.get("path", [])

        # Get event name from path
        event_name = event_slug
        if path_info:
            event_name = path_info[0].get("description", event_slug)

        scrape_time = datetime.now().isoformat()
        fights = []

        for event_data in events:
            fight = self._parse_fight(event_data, event_name, scrape_time)
            if fight:
                fights.append(fight)

        logger.success(f"Scraped {len(fights)} fights from {event_name}")
        return fights

    def _parse_fight(
        self, event: dict, event_name: str, scrape_time: str
    ) -> Optional[dict]:
        """Parse a single fight event into odds dict."""
        competitors = event.get("competitors", [])
        if len(competitors) < 2:
            return None

        fighter1 = competitors[0].get("name", "")
        fighter2 = competitors[1].get("name", "")

        if not fighter1 or not fighter2:
            return None

        # Parse start time
        start_time = event.get("startTime")
        if start_time:
            event_date = datetime.fromtimestamp(start_time / 1000).strftime("%Y-%m-%d")
        else:
            event_date = ""

        # Get moneyline (Fight Winner) odds from displayGroups
        f1_odds, f2_odds = self._extract_moneyline_odds(event)

        if f1_odds is None or f2_odds is None:
            logger.warning(f"Missing odds for {fighter1} vs {fighter2}")
            return None

        f1_prob = _american_to_prob(f1_odds)
        f2_prob = _american_to_prob(f2_odds)

        return {
            "event_name": event_name,
            "event_date": event_date,
            "fighter1": fighter1,
            "fighter2": fighter2,
            "fighter1_odds": f1_odds,
            "fighter2_odds": f2_odds,
            "fighter1_prob": round(f1_prob, 4),
            "fighter2_prob": round(f2_prob, 4),
            "scrape_timestamp": scrape_time,
        }

    def _extract_moneyline_odds(self, event: dict) -> tuple[Optional[int], Optional[int]]:
        """
        Extract moneyline (Fight Winner) odds from event data.

        Returns (fighter1_odds, fighter2_odds) or (None, None) if not found.
        """
        display_groups = event.get("displayGroups", [])

        for group in display_groups:
            markets = group.get("markets", [])

            for market in markets:
                # Look for "Fight Winner" market (moneyline)
                if market.get("descriptionKey") != "Fight Winner":
                    continue

                outcomes = market.get("outcomes", [])
                if len(outcomes) < 2:
                    continue

                f1_odds = None
                f2_odds = None

                for outcome in outcomes:
                    price = outcome.get("price", {})
                    odds = self._parse_american_odds(price)

                    # Type "H" = home (fighter1), Type "A" = away (fighter2)
                    if outcome.get("type") == "H":
                        f1_odds = odds
                    elif outcome.get("type") == "A":
                        f2_odds = odds

                if f1_odds is not None and f2_odds is not None:
                    return f1_odds, f2_odds

        return None, None

    def scrape_all_events(self) -> pd.DataFrame:
        """Scrape all upcoming UFC events from Bovada.lv."""
        upcoming_events = self.get_upcoming_events()

        if not upcoming_events:
            logger.warning("No upcoming events found")
            return pd.DataFrame()

        all_fights = []
        for event in upcoming_events:
            slug = event["link"].split("/")[-1]
            logger.info(f"Scraping: {event['name']}")
            fights = self.scrape_event(slug)
            all_fights.extend(fights)

        if not all_fights:
            return pd.DataFrame()

        return pd.DataFrame(all_fights)

    def save(
        self,
        df: pd.DataFrame,
        output_path: str | Path = MASTER_FILE,
        dedupe: bool = True,
    ) -> Path:
        """Save scraped odds to CSV."""
        if df.empty:
            logger.warning("Nothing to save - DataFrame is empty")
            return Path(output_path)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and dedupe:
            existing = pd.read_csv(output_path)
            combined = pd.concat([existing, df], ignore_index=True)

            # Dedupe by event + fighter pair (keep most recent)
            combined["_dedup_key"] = combined.apply(
                lambda r: f"{r['event_name']}|{_fight_key(r['fighter1'], r['fighter2'])}",
                axis=1,
            )
            before = len(combined)
            combined = combined.drop_duplicates(subset="_dedup_key", keep="last")
            combined = combined.drop(columns=["_dedup_key"])
            logger.info(f"Deduped: {before} -> {len(combined)} rows")
        else:
            combined = df

        combined.to_csv(output_path, index=False)
        logger.success(f"Saved {len(combined)} rows -> {output_path}")
        return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Scrape UFC moneyline odds from Bovada.lv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Scrape all upcoming UFC events:
    python scrapers/bovada_scraper.py

    # Scrape a specific event by URL slug:
    python scrapers/bovada_scraper.py --event ufc-fight-night-evloev-murphy

    # Custom output path:
    python scrapers/bovada_scraper.py --output data/odds/bovada_latest.csv
        """,
    )

    parser.add_argument(
        "--event",
        metavar="SLUG",
        help="Specific event slug to scrape (e.g., ufc-fight-night-evloev-murphy)",
    )
    parser.add_argument(
        "--output",
        default=str(MASTER_FILE),
        help=f"Output CSV path (default: {MASTER_FILE})",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=RATE_LIMIT_SECS,
        help=f"Seconds between requests (default: {RATE_LIMIT_SECS})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass disk cache and re-fetch all pages",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Replace output file instead of appending",
    )

    args = parser.parse_args()

    scraper = BovadaScraper(rate_limit=args.rate_limit)

    if args.event:
        fights = scraper.scrape_event(args.event)
        df = pd.DataFrame(fights) if fights else pd.DataFrame()
    else:
        df = scraper.scrape_all_events()

    if df.empty:
        logger.warning("No data scraped. Check your event slug or Bovada connectivity.")
        return

    print("\n── Preview ──────────────────────────────────────────────────────")
    print(df.to_string(index=False))
    print(f"\nTotal fights scraped: {len(df)}")

    scraper.save(df, output_path=args.output, dedupe=not args.no_merge)


if __name__ == "__main__":
    main()
