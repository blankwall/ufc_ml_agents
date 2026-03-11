#!/usr/bin/env python3
"""
Scan database for fights missing betting odds by year.

Outputs a list of unique fighter names that need odds scraped.

Usage:
  python add_odds_db/scan_missing_odds.py --year 2024
  python add_odds_db/scan_missing_odds.py --year 2025 --format names
  python add_odds_db/scan_missing_odds.py --year 2023 --format detailed
  python add_odds_db/scan_missing_odds.py --all --format names > fighters_to_scrape.txt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database.db_manager import DatabaseManager
from database.schema import Event, Fight, Fighter, BettingOdds
from sqlalchemy.orm import joinedload


def parse_date(date_str: str) -> datetime | None:
    """Parse various date formats."""
    if not date_str:
        return None
    fmts = [
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def get_fights_missing_odds(session, year: int | None = None) -> list[dict]:
    """
    Get all fights missing betting odds, optionally filtered by year.

    Returns list of dicts with fight info.
    """
    query = (
        session.query(Fight)
        .options(
            joinedload(Fight.fighter_1),
            joinedload(Fight.fighter_2),
            joinedload(Fight.event),
            joinedload(Fight.betting_odds),
        )
        .join(Event, Fight.event_id == Event.id)
    )

    fights = query.all()
    missing = []

    for fight in fights:
        odds_records = fight.betting_odds if hasattr(fight, "betting_odds") else []

        # Check if any odds exist
        has_odds = False
        for o in odds_records:
            if o.fighter_1_odds is not None or o.fighter_2_odds is not None:
                has_odds = True
                break

        if has_odds:
            continue

        # Parse event date
        date_str = fight.event.date if fight.event else ""
        event_dt = parse_date(date_str)
        event_year = event_dt.year if event_dt else None

        # Filter by year if specified
        if year is not None and event_year != year:
            continue

        f1_name = fight.fighter_1.name if fight.fighter_1 else None
        f2_name = fight.fighter_2.name if fight.fighter_2 else None

        missing.append(
            {
                "fighter_1": f1_name,
                "fighter_2": f2_name,
                "event": fight.event.name if fight.event else "Unknown",
                "date": date_str,
                "year": event_year,
                "fight_id": fight.id,
            }
        )

    return missing


def print_summary(missing: list[dict]) -> None:
    """Print summary statistics."""
    by_year = defaultdict(int)
    for f in missing:
        year = f.get("year") or "Unknown"
        by_year[year] += 1

    print("\n=== SUMMARY ===")
    print(f"Total fights missing odds: {len(missing)}")
    print("\nBy year:")
    for year in sorted(by_year.keys(), key=lambda x: (x is None, x), reverse=True):
        print(f"  {year}: {by_year[year]}")


def print_names(missing: list[dict]) -> None:
    """Print unique fighter names that need scraping."""
    names = set()
    for f in missing:
        if f["fighter_1"]:
            names.add(f["fighter_1"])
        if f["fighter_2"]:
            names.add(f["fighter_2"])

    for name in sorted(names):
        print(name)


def print_detailed(missing: list[dict]) -> None:
    """Print detailed fight-by-fight list."""
    for f in missing:
        print(f"{f['fighter_1']} vs {f['fighter_2']}")
        print(f"  Event: {f['event']} ({f['date']})")
        print(f"  Fight ID: {f['fight_id']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan database for fights missing betting odds."
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Filter by year (e.g., 2024, 2025)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all years (no year filter)",
    )
    parser.add_argument(
        "--format",
        choices=["names", "detailed", "summary"],
        default="names",
        help="Output format: names (unique fighters), detailed (each fight), summary (counts only)",
    )
    args = parser.parse_args()

    if args.year is None and not args.all:
        parser.error("Specify --year YYYY or --all")

    db = DatabaseManager()
    session = db.get_session()

    try:
        year_filter = None if args.all else args.year
        missing = get_fights_missing_odds(session, year=year_filter)

        if args.format == "summary":
            print_summary(missing)
        elif args.format == "detailed":
            print_detailed(missing)
        else:
            print_names(missing)

    finally:
        session.close()


if __name__ == "__main__":
    main()
