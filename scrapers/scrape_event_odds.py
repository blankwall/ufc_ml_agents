#!/usr/bin/env python3
"""
Scrape fight odds from BestFightOdds by direct event URL.

Saves each event to  data/future_fight_odds/<slug>.csv
Also appends to     data/future_fight_odds/all_events.csv  (deduped)

Usage:
    # By direct event URL(s)
    python scrapers/scrape_event_odds.py \\
        https://www.bestfightodds.com/events/ufc-3996 \\
        https://www.bestfightodds.com/events/ufc-mexico-4052

    # With a custom filename (useful when BFO event name is wrong)
    python scrapers/scrape_event_odds.py \\
        https://www.bestfightodds.com/events/ufc-4025 \\
        -n ufc_326

    # By event name (resolved via BestFightOdds search)
    python scrapers/scrape_event_odds.py "UFC 326"

    # Force re-scrape (bypass cache):
    python scrapers/scrape_event_odds.py --no-cache URL ...

Output columns:
    event_name, event_date, event_url,
    fighter1, fighter2, fighter1_odds, fighter2_odds,
    fighter1_prob, fighter2_prob
"""

import sys
import re
import argparse
from pathlib import Path

# Allow running from repo root or scrapers/ subdirectory
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from scrapers.bestfightodds_scraper import BestFightOddsScraper

OUTPUT_DIR   = Path("data/future_fight_odds")
MASTER_FILE  = OUTPUT_DIR / "all_events.csv"


def slug(url: str) -> str:
    """Turn a BFO URL into a clean filename slug."""
    part = url.rstrip("/").split("/events/")[-1]
    return re.sub(r"[^a-z0-9\-]", "-", part.lower())


def scrape_and_save(
    url: str,
    use_cache: bool = True,
    name_override: str | None = None,
) -> pd.DataFrame | None:
    """Scrape one BFO event URL and save results."""
    scraper = BestFightOddsScraper()

    if not use_cache:
        # Delete cached HTML so _get() fetches fresh
        cache_key = re.sub(r"[^a-z0-9]", "_", url.lower()) + ".html"
        cache_path = Path(".cache/bfo") / cache_key
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"  Cleared cache: {cache_key}")

    logger.info(f"Scraping: {url}")
    rows = scraper.scrape_event(url)

    if not rows:
        logger.warning(f"No fights scraped from {url}")
        return None

    df = pd.DataFrame(rows)
    logger.info(f"  → {len(df)} fights found: {df['event_name'].iloc[0]}")

    # Save per-event file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename_slug = name_override if name_override else slug(url)
    out_path = OUTPUT_DIR / f"{filename_slug}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  Saved: {out_path}")

    return df


def merge_to_master(new_df: pd.DataFrame):
    """Append new rows to all_events.csv, deduplicating on event_url + fight pair."""
    if MASTER_FILE.exists():
        existing = pd.read_csv(MASTER_FILE)
    else:
        existing = pd.DataFrame()

    combined = pd.concat([existing, new_df], ignore_index=True)

    # Dedup on event_url + sorted fighter pair
    def fkey(row):
        a, b = sorted([str(row.get("fighter1", "")), str(row.get("fighter2", ""))])
        return f"{row.get('event_url', '')}|{a}|{b}"

    combined["_fkey"] = combined.apply(fkey, axis=1)
    combined = combined.drop_duplicates(subset="_fkey", keep="last").drop(columns="_fkey")
    combined.to_csv(MASTER_FILE, index=False)
    logger.info(f"Master file updated: {MASTER_FILE}  ({len(combined)} total rows)")


def main():
    parser = argparse.ArgumentParser(description="Scrape BFO event odds by URL")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="",
        help="Optional custom base filename (e.g. 'ufc_326') for a single event",
    )
    parser.add_argument(
        "events",
        nargs="+",
        help="BestFightOdds event URLs or names (e.g. 'UFC 326')",
    )
    parser.add_argument("--no-cache", action="store_true", help="Bypass HTML cache")
    args = parser.parse_args()

    if args.name and len(args.events) != 1:
        parser.error("--name/-n can only be used when scraping exactly one event")

    scraper = BestFightOddsScraper()

    all_dfs = []
    for ident in args.events:
        # Allow either a full URL or a human-friendly name like "UFC 326"
        if ident.startswith("http://") or ident.startswith("https://"):
            url = ident
        else:
            query = ident.strip()
            result = scraper.search_event(query)
            if not result:
                logger.error(f'Could not find BestFightOdds event for query "{query}"')
                continue
            url = result["url"]
            logger.info(f'Resolved "{query}" → {result["name"]} ({url})')

        # Use name override only if provided (and only valid when a single event)
        name_override = args.name or None
        df = scrape_and_save(url, use_cache=not args.no_cache, name_override=name_override)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        merge_to_master(combined)
        print(f"\n{'='*60}")
        print(f"Scraped {len(all_dfs)} event(s), {len(combined)} fights total.")
        print(f"Saved to: {OUTPUT_DIR}/")
        print(f"{'='*60}")
        # Print summary
        for ev, grp in combined.groupby("event_name"):
            print(f"\n  {ev}  ({grp['event_date'].iloc[0]})  — {len(grp)} fights")
            for _, row in grp.iterrows():
                print(f"    {row['fighter1']:25s} ({row['fighter1_odds']:+d})  vs  "
                      f"{row['fighter2']:25s} ({row['fighter2_odds']:+d})")


if __name__ == "__main__":
    main()
