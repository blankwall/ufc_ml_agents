#!/usr/bin/env python3
"""
Scrape fight outcomes from UFCStats event pages.
Saves to data/future_fight_odds/outcomes.csv — does NOT touch the database.

Usage:
    python scrapers/scrape_outcomes.py \\
        http://ufcstats.com/event-details/0cfbbfa0ba6d9855 \\
        http://ufcstats.com/event-details/79ab17db3b40831a
"""

import sys
import re
import time
import unicodedata
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from bs4 import BeautifulSoup
import pandas as pd

OUTPUT_FILE = Path("data/future_fight_odds/outcomes.csv")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def norm(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_ = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_).strip().lower()


def fight_key(f1: str, f2: str) -> str:
    a, b = sorted([norm(f1), norm(f2)])
    return f"{a}_vs_{b}"


def scrape_event(url: str) -> list[dict]:
    """
    Parse a UFCStats event page and return fight results.

    Actual HTML structure (one <tr> per fight in the event table):
      col[0]: p[0] = "win"  (result, always the winner listed first)
      col[1]: p[0] = winner_name, p[1] = loser_name  (fighter links)
      col[7]: p[0] = method  (e.g. "KO/TKO", "Decision", "Submission")
      col[8]: p[0] = round
      col[9]: p[0] = time
    """
    print(f"\nScraping outcomes: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")

    # ── Event name ────────────────────────────────────────────────────────
    event_name = ""
    h2 = soup.find("h2", class_="b-content__title")
    if h2:
        event_name = h2.get_text(strip=True)
    print(f"  Event: {event_name}")

    # ── Fight rows ────────────────────────────────────────────────────────
    tbody = soup.find("tbody")
    if tbody is None:
        print("  No fight table found")
        return []

    rows = []
    for tr in tbody.find_all("tr", class_="b-fight-details__table-row"):
        # Only process rows with the clickable fight link class
        if "js-fight-details-click" not in " ".join(tr.get("class", [])):
            continue

        cols = tr.find_all("td")
        if len(cols) < 8:
            continue

        # col[0]: result indicator
        result_text = cols[0].get_text(strip=True).lower()
        if result_text not in ("win", "loss", "draw", "nc"):
            continue  # skip header or non-result rows

        # col[1]: fighter names — p[0] = winner (listed first on "win" row)
        name_ps = cols[1].find_all("p")
        if len(name_ps) < 2:
            continue
        f_winner = name_ps[0].get_text(strip=True)  # winner (top = result_text=="win")
        f_loser  = name_ps[1].get_text(strip=True)  # loser

        if result_text == "loss":
            # This row is from loser's perspective (shouldn't happen on event page but handle it)
            f_winner, f_loser = f_loser, f_winner
        elif result_text in ("draw", "nc"):
            f_winner = ""  # no winner for draws/no-contests

        # col[7]: method
        method_ps = cols[7].find_all("p") if len(cols) > 7 else []
        method = method_ps[0].get_text(strip=True) if method_ps else ""

        # col[8]: round, col[9]: time
        round_ps = cols[8].find_all("p") if len(cols) > 8 else []
        round_n  = round_ps[0].get_text(strip=True) if round_ps else ""

        fk = fight_key(f_winner or f_loser, f_loser)
        rows.append({
            "event_name": event_name,
            "event_url":  url,
            "fighter1":   f_winner,
            "fighter2":   f_loser,
            "winner":     f_winner,
            "method":     method,
            "round":      round_n,
            "fight_key":  fk,
        })
        status = f"→ {f_winner} ({method} R{round_n})" if f_winner else f"→ {result_text.upper()}"
        print(f"    {(f_winner or '?'):25s} vs {f_loser:25s}  {status}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Scrape fight outcomes from UFCStats (no DB write)")
    parser.add_argument("urls", nargs="+", help="UFCStats event-details URLs")
    parser.add_argument("--no-cache", action="store_true", help="(ignored, always fresh)")
    args = parser.parse_args()

    all_rows = []
    for url in args.urls:
        rows = scrape_event(url)
        all_rows.extend(rows)
        time.sleep(1.5)

    if not all_rows:
        print("\nNo outcomes scraped.")
        return

    new_df = pd.DataFrame(all_rows)

    # ── Merge with existing outcomes.csv ─────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="fight_key", keep="last")
    else:
        combined = new_df

    combined.to_csv(OUTPUT_FILE, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    total   = len(new_df)
    w_known = new_df["winner"].ne("").sum()
    print(f"\n{'='*60}")
    print(f"Scraped {total} fights from {len(args.urls)} event(s)")
    print(f"Winners identified: {w_known}/{total}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
