#!/usr/bin/env python3
"""
Backfill missing odds from BestFightOdds.

This script:
  1. Loads all events from the database
  2. Compares against existing historical_odds.csv
  3. Identifies events missing odds
  4. Scrapes missing events from BestFightOdds
  5. Appends to historical_odds.csv

Usage:
    # Dry run (shows what would be scraped)
    python scripts/backfill_odds.py --dry-run

    # Scrape all missing events
    python scripts/backfill_odds.py

    # Scrape only UFC events (skip DWCS, Road to UFC, etc.)
    python scripts/backfill_odds.py --ufc-only

    # Limit number of events to scrape (for testing)
    python scripts/backfill_odds.py --limit 10
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database.db_manager import DatabaseManager
from database.schema import Event
from scrapers.bestfightodds_scraper import BestFightOddsScraper

HISTORICAL_ODDS_PATH = ROOT / "data" / "odds" / "historical_odds.csv"


def parse_bfo_date(date_str: str) -> datetime | None:
    """Parse BFO date format: 'Mar 12th 2021'"""
    if not date_str:
        return None
    date_str = str(date_str)
    for suf in ("st", "nd", "rd", "th"):
        date_str = date_str.replace(suf, "")
    try:
        return datetime.strptime(date_str, "%b %d %Y")
    except ValueError:
        return None


def parse_db_date(date_str: str) -> datetime | None:
    """Parse DB date formats."""
    if not date_str:
        return None
    fmts = ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]
    for fmt in fmts:
        try:
            return datetime.strptime(str(date_str), fmt)
        except ValueError:
            continue
    return None


def get_events_with_odds() -> set[str]:
    """Load historical_odds.csv and return set of event names that have odds."""
    if not HISTORICAL_ODDS_PATH.exists():
        return set()

    df = pd.read_csv(HISTORICAL_ODDS_PATH)
    # Filter to UFC events only
    ufc_mask = df["event_name"].str.contains("UFC", case=False, na=False)
    return set(df[ufc_mask]["event_name"].unique())


def get_db_events() -> list[dict]:
    """Get all events from database."""
    db = DatabaseManager()
    session = db.get_session()
    try:
        events = session.query(Event).all()
        result = []
        for ev in events:
            dt = parse_db_date(ev.date or "")
            if dt:
                result.append({
                    "name": ev.name,
                    "date": dt,
                    "event_id": ev.event_id,
                    "year": dt.year,
                })
        return result
    finally:
        session.close()


def is_event_matched(db_event: dict, odds_event_names: set[str]) -> bool:
    """Check if a DB event already has odds (by name matching)."""
    db_name = db_event["name"].lower()

    # Direct match
    for odds_name in odds_event_names:
        if odds_name.lower() == db_name:
            return True

    # Fuzzy match: check if key parts match
    # e.g., "UFC Fight Night: Holloway vs. Imavov" matches "UFC Fight Night: Holloway vs Imavov"
    db_normalized = db_name.replace(".", "").replace("-", " ")
    for odds_name in odds_event_names:
        odds_normalized = odds_name.lower().replace(".", "").replace("-", " ")
        if db_normalized == odds_normalized:
            return True

    return False


def is_ufc_event(event_name: str) -> bool:
    """Check if event is a UFC event (not DWCS, Road to UFC, etc.)"""
    name = event_name.lower()
    if "ufc" not in name:
        return False
    # Exclude developmental/regional shows
    exclude_patterns = ["dwcs", "road to ufc", "ultimate fighter", "tuf"]
    return not any(p in name for p in exclude_patterns)


def main():
    parser = argparse.ArgumentParser(description="Backfill missing odds from BestFightOdds")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be scraped without scraping")
    parser.add_argument("--ufc-only", action="store_true", help="Only scrape UFC events (skip DWCS, etc.)")
    parser.add_argument("--limit", type=int, help="Limit number of events to scrape")
    parser.add_argument("--rate-limit", type=float, default=2.0, help="Seconds between requests")
    parser.add_argument("--start-year", type=int, help="Only scrape events from this year onwards")
    args = parser.parse_args()

    # Load existing odds
    events_with_odds = get_events_with_odds()
    logger.info(f"Events already in historical_odds.csv: {len(events_with_odds)}")

    # Get DB events
    db_events = get_db_events()
    logger.info(f"Events in database: {len(db_events)}")

    # Find missing events
    missing = []
    for ev in db_events:
        if is_event_matched(ev, events_with_odds):
            continue
        if args.ufc_only and not is_ufc_event(ev["name"]):
            continue
        if args.start_year and ev["year"] < args.start_year:
            continue
        missing.append(ev)

    # Sort by date (newest first - more likely to have odds available)
    missing.sort(key=lambda x: x["date"], reverse=True)

    logger.info(f"Events missing odds: {len(missing)}")

    if not missing:
        logger.success("All events already have odds!")
        return

    if args.limit:
        missing = missing[:args.limit]
        logger.info(f"Limited to {len(missing)} events")

    # Dry run: just print what would be scraped
    if args.dry_run:
        print("\nEvents to scrape:")
        print("-" * 80)
        for ev in missing[:50]:  # Show first 50
            print(f"  {ev['date'].strftime('%Y-%m-%d')}  {ev['name']}")
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")
        print(f"\nTotal: {len(missing)} events")
        return

    # Scrape
    scraper = BestFightOddsScraper(rate_limit=args.rate_limit)

    queries = [ev["name"] for ev in missing]
    years = [ev["year"] for ev in missing]

    logger.info(f"Starting scrape of {len(queries)} events...")
    df = scraper.scrape_events(queries, expected_years=years)

    if df.empty:
        logger.warning("No data scraped")
        return

    logger.info(f"Scraped {len(df)} fights from {df['event_name'].nunique()} events")

    # Save to historical_odds.csv
    scraper.save(df, output_path=HISTORICAL_ODDS_PATH, dedupe=True)
    logger.success(f"Done! Total rows in {HISTORICAL_ODDS_PATH}")


if __name__ == "__main__":
    main()
