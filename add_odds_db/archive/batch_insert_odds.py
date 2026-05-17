#!/usr/bin/env python3
"""
Batch insert odds for all fighters into the database.

This script:
  1. Gets the list of all fighters from the database
  2. For each fighter, scrapes odds from BestFightOdds
  3. Matches fights to DB events by date
  4. Finds corresponding Fight records
  5. Inserts opening and closing odds into the betting_odds table

Usage:
  python add_odds_db/batch_insert_odds.py
  python add_odds_db/batch_insert_odds.py --dry-run
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy.orm import Session

from database.db_manager import DatabaseManager
from database.schema import Event, Fight, Fighter, BettingOdds
from add_odds_db.fighter_odds_preview import find_fighter_url, parse_fighter_odds_page


from add_odds_db.insert_fighter_odds import insert_odds_for_fighter


def parse_bfo_event_date(label: str) -> Optional[datetime]:
    """Parse Bfo event date from label"""
    parts = label.strip().split()
    if len(parts) < 3:
        return None

    month_idx = None
    for i, tok in enumerate(parts):
        if tok[:3].isalpha() and tok[:3].title() in [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]:
            month_idx = i
            break
    if month_idx is None or month_idx + 2 >= len(parts):
        return None

    month = parts[month_idx]
    day_raw = parts[month_idx + 1]
    year = parts[month_idx + 2]

    for suf in ("st", "nd", "rd", "th"):
        if day_raw.lower().endswith(suf):
            day_raw = day_raw[: -len(suf)]
            break

    try:
        return datetime.strptime(f"{month} {day_raw} {year}", "%b %d %Y")
    except ValueError:
        return None


def parse_event_date_field(date_str: str) -> Optional[datetime]:
    """Parse our DB `Event.date` field"""
    if not date_str:
        return None

    fmts = [
        "%B %d, %Y", "%B %d, %Y",
        "%Y-%m-%d", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y", "%d/%m/%Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def parse_american_odds(american: str) -> Optional[int]:
    """Parse American odds string"""
    if not american:
        return None
    try:
        return int(str(american).replace("+", "").replace(" ", ""))
    except ValueError:
        return None


def american_to_implied_prob(american: int) -> float:
    """Convert American odds to implied probability"""
    if american > 0:
        return 100.0 / (american + 100.0)
    else:
        return abs(american) / (abs(american) + 100.0)


def normalize_name(name: str) -> str:
    """Normalize fighter name for matching"""
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def find_fight_by_event_and_fighters(
    session: Session, event_id: int, fighter_name: str, opponent_name: str,
) -> Optional[Fight]:
    """
    Find a Fight record by event and fighter names.
    """
    fights = session.query(Fight).filter_by(event_id=event_id).all()

    fighter_norm = normalize_name(fighter_name)
    opponent_norm = normalize_name(opponent_name)

    for fight in fights:
        f1_name = normalize_name(fight.fighter_1.name) if fight.fighter_1 else ""
        f2_name = normalize_name(fight.fighter_2.name) if fight.fighter_2 else ""

        # Check if fighter_1 is our target fighter
        if fighter_norm in f1_name or f1_name in fighter_norm:
            if opponent_norm in f2_name or f2_name in opponent_norm:
                return fight
        # Check if fighter_2 is our target fighter
        if fighter_norm in f2_name or f2_name in fighter_norm:
            if opponent_norm in f1_name or f1_name in opponent_norm:
                return fight

    return None


def get_fighter_position(fight: Fight, fighter_name: str) -> Optional[int]:
    """
    Determine if the fighter is fighter_1 or fighter_2.
    """
    fighter_norm = normalize_name(fighter_name)
    f1_name = normalize_name(fight.fighter_1.name) if fight.fighter_1 else ""
    f2_name = normalize_name(fight.fighter_2.name) if fight.fighter_2 else ""
    if fighter_norm in f1_name or f1_name in fighter_norm:
        return 1
    if fighter_norm in f2_name or f2_name in fighter_norm:
        return 2
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
                description="Batch insert odds for all fighters into the database"
            )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be inserted without actually inserting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing odds and re-insert",
    )
    args = parser.parse_args()

    db = DatabaseManager()
    session = db.get_session()

    try:
        # Get all fighter names from DB
        fighters = session.query(Fighter).all()
        fighter_names = [f.name for f in fighters]
        print(f"Found {len(fighter_names)} fighters in database")

        # Get all events for matching
        events = session.query(Event).all()

        print(f"Processing {len(fighter_names)} fighters...")

        total_stats = {
            "fights_found": 0,
            "events_matched": 0,
            "fights_matched": 0,
            "odds_inserted": 0,
            "odds_updated": 0,
            "skipped_existing": 0,
        }

        for fighter_name in fighter_names:
            stats = insert_odds_for_fighter(session, fighter_name, dry_run=args.dry_run, force=args.force)

            total_stats["fights_found"] += stats["fights_found"]
            total_stats["events_matched"] += stats["events_matched"]
            total_stats["fights_matched"] += stats["fights_matched"]
            total_stats["odds_inserted"] += stats["odds_inserted"]
            total_stats["odds_updated"] += stats["odds_updated"]
            total_stats["skipped_existing"] += stats["skipped_existing"]

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total fighters: {len(fighter_names)}")
        print(f"Total fights found on BFO: {total_stats['fights_found']}")
        print(f"Events matched to DB: {total_stats['events_matched']}")
        print(f"Fights matched to DB: {total_stats['fights_matched']}")
        print(f"Odds records inserted: {total_stats['odds_inserted']}")
        print(f"Odds records updated (force): {total_stats['odds_updated']}")
        print(f"Skipped (existing): {total_stats['skipped_existing']}")
        print("=" * 60)

    finally:
        session.close()


if __name__ == "__main__":
    main()
