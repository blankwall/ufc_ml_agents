#!/usr/bin/env python3
"""
Insert fighter odds into the database.

This script:
  1. Scrapes opening/closing odds for a fighter from BestFightOdds
  2. Matches those fights to DB events by date
  3. Finds the corresponding Fight records
  4. Inserts BettingOdds records (opening and closing lines)

Usage:
  python add_odds_db/insert_fighter_odds.py "Islam Makhachev"
  python add_odds_db/insert_fighter_odds.py "Islam Makhachev" --dry-run
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import logging
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(ROOT))

from database.db_manager import DatabaseManager  # type: ignore
from database.schema import Event, Fight, Fighter, BettingOdds  # type: ignore
from add_odds_db.fighter_odds_preview import (
    find_fighter_url,
    parse_fighter_odds_page,
)


def parse_bfo_event_date(label: str) -> Optional[datetime]:
    """
    Parse BFO-style label like "UFC Jan 19th 2025" into a datetime.
    """
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
    """Parse DB Event.date field into a datetime."""
    if not date_str:
        return None

    fmts = [
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def parse_american_odds(american: str) -> Optional[int]:
    """Parse American odds string like '+150' or '-200' to int."""
    if not american:
        return None
    try:
        return int(str(american).replace("+", "").replace(" ", ""))
    except ValueError:
        return None


def american_to_implied_prob(american: int) -> float:
    """Convert American odds to implied probability."""
    if american > 0:
        return 100.0 / (american + 100.0)
    else:
        return abs(american) / (abs(american) + 100.0)


def normalize_name(name: str) -> str:
    """Normalize fighter name for matching."""
    if not name:
        return ""
    # Lowercase, remove extra spaces, strip
    s = name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def find_fight_by_event_and_fighters(
    session: Session,
    event_id: int,
    fighter_name: str,
    opponent_name: str,
) -> Optional[Fight]:
    """
    Find a Fight record by event and fighter names.
    Returns the fight and indicates which position the fighter is in.
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
    Determine if the fighter is fighter_1 or fighter_2 in the fight.
    Returns 1 or 2, or None if not found.
    """
    fighter_norm = normalize_name(fighter_name)

    f1_name = normalize_name(fight.fighter_1.name) if fight.fighter_1 else ""
    f2_name = normalize_name(fight.fighter_2.name) if fight.fighter_2 else ""

    if fighter_norm in f1_name or f1_name in fighter_norm:
        return 1
    if fighter_norm in f2_name or f2_name in fighter_norm:
        return 2
    return None


def insert_odds_for_fighter(
    session: Session,
    fighter_name: str,
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Scrape odds for a fighter and insert into database.

    Args:
        session: SQLAlchemy session
        fighter_name: Name of the fighter to scrape odds for
        dry_run: If True, don't actually insert into DB
        force: If True, delete existing odds and re-insert

    Returns stats about what was processed.
    """
    stats = {
        "fights_found": 0,
        "events_matched": 0,
        "fights_matched": 0,
        "odds_inserted": 0,
        "odds_updated": 0,
        "skipped_existing": 0,
        "errors": [],
    }

    # Step 1: Get fighter URL and scrape odds
    logger.info(f"Searching BestFightOdds for fighter: {fighter_name}")
    try:
        fighter_url = find_fighter_url(fighter_name)
    except RuntimeError as e:
        logger.warning(f"Could not find fighter on BestFightOdds: {fighter_name} - {e}")
        stats["errors"].append(str(e))
        return stats

    logger.info(f"Using fighter URL: {fighter_url}")

    try:
        df = parse_fighter_odds_page(fighter_url, fighter_name)
    except RuntimeError as e:
        logger.warning(f"Could not parse odds page for {fighter_name}: {e}")
        stats["errors"].append(str(e))
        return stats

    if df.empty:
        logger.warning(f"No odds rows found for {fighter_name}")
        return stats

    stats["fights_found"] = len(df)
    logger.info(f"Found {len(df)} fights for {fighter_name}")

    # Step 2: Load all events from DB for matching
    events = session.query(Event).all()
    event_rows = []
    for ev in events:
        ev_dt = parse_event_date_field(ev.date or "")
        event_rows.append((ev, ev_dt))

    logger.info(f"Loaded {len(event_rows)} events from DB")

    # Step 3: Process each fight
    for _, row in df.iterrows():
        event_slug = row.get("event_slug", "")
        event_label = row.get("event_label", "")
        opponent = row.get("opponent", "")
        opening_american = row.get("opening_american", "")
        closing_american = row.get("closing_american", "")

        # Only process UFC events
        if not isinstance(event_slug, str) or not event_slug.startswith("ufc-"):
            continue

        # Parse BFO event date
        target_dt = parse_bfo_event_date(str(event_label))
        if not target_dt:
            logger.debug(f"Could not parse date from: {event_label}")
            continue

        # Find matching DB event within ±3 days
        window = timedelta(days=3)
        candidates = []
        for ev, ev_dt in event_rows:
            if ev_dt is None:
                continue
            delta = abs(ev_dt - target_dt)
            if delta <= window:
                candidates.append((delta, ev))

        if not candidates:
            logger.debug(f"No DB event match for {event_slug} ({event_label})")
            continue

        candidates.sort(key=lambda t: t[0])
        best_delta, best_event = candidates[0]
        stats["events_matched"] += 1

        # Find the Fight record
        fight = find_fight_by_event_and_fighters(
            session, best_event.id, fighter_name, opponent
        )
        if not fight:
            logger.debug(
                f"No fight match for {fighter_name} vs {opponent} at {best_event.name}"
            )
            continue

        stats["fights_matched"] += 1

        # Determine which position our fighter is in
        position = get_fighter_position(fight, fighter_name)
        if position is None:
            logger.warning(f"Could not determine fighter position for {fighter_name}")
            continue

        # Parse odds
        open_odds = parse_american_odds(str(opening_american))
        close_odds = parse_american_odds(str(closing_american))

        if open_odds is None and close_odds is None:
            logger.debug(f"No valid odds for {fighter_name} vs {opponent}")
            continue

        # Check for existing odds for this specific fighter position
        if position == 1:
            existing_opening = (
                session.query(BettingOdds)
                .filter_by(fight_id=fight.id, bookmaker="BestFightOdds", is_opening_line=True)
                .filter(BettingOdds.fighter_1_odds.isnot(None))
                .first()
            )
            existing_closing = (
                session.query(BettingOdds)
                .filter_by(fight_id=fight.id, bookmaker="BestFightOdds", is_closing_line=True)
                .filter(BettingOdds.fighter_1_odds.isnot(None))
                .first()
            )
        else:
            existing_opening = (
                session.query(BettingOdds)
                .filter_by(fight_id=fight.id, bookmaker="BestFightOdds", is_opening_line=True)
                .filter(BettingOdds.fighter_2_odds.isnot(None))
                .first()
            )
            existing_closing = (
                session.query(BettingOdds)
                .filter_by(fight_id=fight.id, bookmaker="BestFightOdds", is_closing_line=True)
                .filter(BettingOdds.fighter_2_odds.isnot(None))
                .first()
            )

        has_existing = existing_opening is not None or existing_closing is not None

        if has_existing and not force:
            logger.debug(
                f"Odds already exist for fight {fight.id} ({fighter_name} vs {opponent})"
            )
            stats["skipped_existing"] += 1
            continue
        elif has_existing and force:
            # Delete existing odds for this fighter's position before re-inserting
            if existing_opening:
                session.delete(existing_opening)
            if existing_closing:
                session.delete(existing_closing)
            logger.info(
                f"Deleted existing odds for fight {fight.id} ({fighter_name} vs {opponent})"
            )
            stats["odds_updated"] += 1

        if dry_run:
            logger.info(
                f"[DRY RUN] Would insert odds for {fighter_name} vs {opponent} "
                f"at {best_event.name}: open={opening_american}, close={closing_american}"
            )
            stats["odds_inserted"] += 2
            continue

        # Insert opening line
        if open_odds is not None:
            open_prob = american_to_implied_prob(open_odds)
            if position == 1:
                opening = BettingOdds(
                    fight_id=fight.id,
                    bookmaker="BestFightOdds",
                    fighter_1_odds=open_odds,
                    fighter_1_implied_prob=open_prob,
                    is_opening_line=True,
                    is_closing_line=False,
                )
            else:
                opening = BettingOdds(
                    fight_id=fight.id,
                    bookmaker="BestFightOdds",
                    fighter_2_odds=open_odds,
                    fighter_2_implied_prob=open_prob,
                    is_opening_line=True,
                    is_closing_line=False,
                )
            session.add(opening)
            stats["odds_inserted"] += 1

        # Insert closing line
        if close_odds is not None:
            close_prob = american_to_implied_prob(close_odds)
            if position == 1:
                closing = BettingOdds(
                    fight_id=fight.id,
                    bookmaker="BestFightOdds",
                    fighter_1_odds=close_odds,
                    fighter_1_implied_prob=close_prob,
                    is_opening_line=False,
                    is_closing_line=True,
                )
            else:
                closing = BettingOdds(
                    fight_id=fight.id,
                    bookmaker="BestFightOdds",
                    fighter_2_odds=close_odds,
                    fighter_2_implied_prob=close_prob,
                    is_opening_line=False,
                    is_closing_line=True,
                )
            session.add(closing)
            stats["odds_inserted"] += 1

        logger.info(
            f"Inserted odds for {fighter_name} vs {opponent} at {best_event.name}"
        )

    if not dry_run:
        session.commit()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insert fighter odds from BestFightOdds into the database."
    )
    parser.add_argument(
        "fighter_name", help="Fighter name (e.g. 'Islam Makhachev')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be inserted without actually inserting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing odds and re-insert",
    )
    parser.add_argument(
        "--fighter-url",
        help="Optional full BestFightOdds fighter URL",
    )
    args = parser.parse_args()

    db = DatabaseManager()
    session = db.get_session()

    try:
        stats = insert_odds_for_fighter(
            session, args.fighter_name, dry_run=args.dry_run, force=args.force
        )

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Fighter: {args.fighter_name}")
        print(f"Fights found on BFO: {stats['fights_found']}")
        print(f"Events matched to DB: {stats['events_matched']}")
        print(f"Fights matched to DB: {stats['fights_matched']}")
        print(f"Odds records {'would be ' if args.dry_run else ''}inserted: {stats['odds_inserted']}")
        print(f"Odds records updated (force): {stats['odds_updated']}")
        print(f"Skipped (existing): {stats['skipped_existing']}")
        if stats["errors"]:
            print(f"Errors: {len(stats['errors'])}")
        print("=" * 60)

    finally:
        session.close()


if __name__ == "__main__":
    main()
