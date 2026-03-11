#!/usr/bin/env python3
"""
Show a fighter's last 3 fights with odds.

Usage:
    python analysis/fighter_recent_fights.py "Islam Makhachev"
    python analysis/fighter_recent_fights.py "Jon Jones" --limit 5
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database.db_manager import DatabaseManager
from database.schema import Event, Fight, Fighter, BettingOdds


def parse_event_date(date_str: str) -> Optional[datetime]:
    """Parse event date string to datetime."""
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


def normalize_name(name: str) -> str:
    """Normalize fighter name for matching."""
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def find_fighter(session: Session, name: str) -> Optional[Fighter]:
    """Find a fighter by name (fuzzy match)."""
    target = normalize_name(name)

    # Try exact match first
    fighter = session.query(Fighter).filter(
        Fighter.name.ilike(name)
    ).first()
    if fighter:
        return fighter

    # Try partial match
    fighters = session.query(Fighter).all()
    for f in fighters:
        if target in normalize_name(f.name) or normalize_name(f.name) in target:
            return f

    return None


def get_fighter_fights(session: Session, fighter_id: int) -> list:
    """Get all fights for a fighter, sorted by date (most recent first)."""
    fights = session.query(Fight).filter(
        (Fight.fighter_1_id == fighter_id) | (Fight.fighter_2_id == fighter_id)
    ).all()

    # Attach event date for sorting
    fight_dates = []
    for fight in fights:
        event = session.query(Event).filter_by(id=fight.event_id).first()
        event_date = parse_event_date(event.date) if event else None
        fight_dates.append((fight, event, event_date))

    # Sort by date descending (most recent first)
    fight_dates.sort(key=lambda x: x[2] if x[2] else datetime.min, reverse=True)

    return fight_dates


def format_odds(odds: Optional[int]) -> str:
    """Format American odds for display."""
    if odds is None:
        return "N/A"
    if odds > 0:
        return f"+{odds}"
    return str(odds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show a fighter's recent fights with odds"
    )
    parser.add_argument("fighter_name", help="Fighter name (e.g., 'Islam Makhachev')")
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=3,
        help="Number of recent fights to show (default: 3)"
    )
    args = parser.parse_args()

    db = DatabaseManager()
    session = db.get_session()

    try:
        # Find fighter
        fighter = find_fighter(session, args.fighter_name)
        if not fighter:
            print(f"Fighter not found: {args.fighter_name}")
            sys.exit(1)

        print(f"\n{'='*70}")
        print(f"Fighter: {fighter.name}")
        print(f"Record: {fighter.wins}-{fighter.losses}-{fighter.draws}")
        print(f"{'='*70}\n")

        # Get fights
        fight_data = get_fighter_fights(session, fighter.id)
        recent_fights = fight_data[:args.limit]

        if not recent_fights:
            print("No fights found for this fighter.")
            return

        # Table header
        print(f"{'Result':<8} {'Opponent':<25} {'Event':<25} {'Open':<8} {'Close':<8}")
        print("-" * 74)

        for fight, event, event_date in recent_fights:
            # Determine opponent and result
            if fight.fighter_1_id == fighter.id:
                opponent = fight.fighter_2
                my_position = 1
            else:
                opponent = fight.fighter_1
                my_position = 2

            opponent_name = opponent.name if opponent else "Unknown"

            # Determine win/loss
            result = "N/A"
            if fight.result:
                if fight.result == f"fighter_{my_position}":
                    result = "W"
                elif fight.result in ("draw", "no_contest"):
                    result = "D" if fight.result == "draw" else "NC"
                else:
                    result = "L"

            # Get odds - find the record that has odds for this fighter's position
            if my_position == 1:
                opening_odds = session.query(BettingOdds).filter_by(
                    fight_id=fight.id,
                    is_opening_line=True
                ).filter(BettingOdds.fighter_1_odds.isnot(None)).first()

                closing_odds = session.query(BettingOdds).filter_by(
                    fight_id=fight.id,
                    is_closing_line=True
                ).filter(BettingOdds.fighter_1_odds.isnot(None)).first()
            else:
                opening_odds = session.query(BettingOdds).filter_by(
                    fight_id=fight.id,
                    is_opening_line=True
                ).filter(BettingOdds.fighter_2_odds.isnot(None)).first()

                closing_odds = session.query(BettingOdds).filter_by(
                    fight_id=fight.id,
                    is_closing_line=True
                ).filter(BettingOdds.fighter_2_odds.isnot(None)).first()

            # Extract odds for this fighter's position
            if my_position == 1:
                open_val = opening_odds.fighter_1_odds if opening_odds else None
                close_val = closing_odds.fighter_1_odds if closing_odds else None
            else:
                open_val = opening_odds.fighter_2_odds if opening_odds else None
                close_val = closing_odds.fighter_2_odds if closing_odds else None

            # Event name (shortened)
            event_name = event.name[:22] + "..." if event and len(event.name) > 25 else (event.name if event else "Unknown")

            # Print row
            print(f"{result:<8} {opponent_name:<25} {event_name:<25} {format_odds(open_val):<8} {format_odds(close_val):<8}")

        print()

    finally:
        session.close()


if __name__ == "__main__":
    main()
