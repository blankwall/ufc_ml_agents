#!/usr/bin/env python3
"""
Analyze Fighter - print last N fights (most recent first) with a few key stats.

Usage examples:
  python3 database/analyze_fighter.py --ufcstats-id d661ce4da776fc20 -N 3
  python3 database/analyze_fighter.py --db-id 123 -N 5
  python3 database/analyze_fighter.py --name "Petr Yan" -N 3

Notes:
- Event dates are stored as strings in the DB; we parse/sort in Python.
- FightStats totals are stored as strings (as scraped) and printed as-is.
"""

from __future__ import annotations

import sys
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

# Ensure project root is on sys.path when running as a script (e.g. `python3 database/analyze_fighter.py ...`)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db_manager import DatabaseManager
from database.schema import Fighter, Fight


def _parse_event_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    s = str(date_str).strip()
    if not s:
        return None
    # UFCStats typical: "December 06, 2025"
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _format_result_for_fighter(fight: Fight, fighter_db_id: int) -> str:
    if fight.result == "draw":
        return "DRAW"
    if fight.result == "no_contest":
        return "NC"
    if fight.winner_id == fighter_db_id:
        return "W"
    if fight.winner_id is None:
        return "?"
    return "L"


def _pick_totals_for_side(fight: Fight, is_f1: bool) -> Optional[dict]:
    if not fight.fight_stats:
        return None
    return fight.fight_stats.fighter_1_totals if is_f1 else fight.fight_stats.fighter_2_totals


@dataclass(frozen=True)
class FightRow:
    event_date: Optional[datetime]
    event_date_raw: str
    event_name: str
    opponent_name: str
    result: str
    weight_class: str
    method: str
    round_finished: str
    time_in_round: str
    totals: Optional[dict]


def _build_rows(fighter: Fighter, fights: List[Fight]) -> List[FightRow]:
    rows: List[FightRow] = []
    for fight in fights:
        is_f1 = fight.fighter_1_id == fighter.id
        opponent = fight.fighter_2 if is_f1 else fight.fighter_1

        event_name = fight.event.name if fight.event else "Unknown Event"
        event_date_raw = fight.event.date if fight.event and fight.event.date else ""
        event_date = _parse_event_date(event_date_raw)

        rows.append(
            FightRow(
                event_date=event_date,
                event_date_raw=str(event_date_raw or ""),
                event_name=str(event_name or ""),
                opponent_name=str(opponent.name if opponent else "Unknown Opponent"),
                result=_format_result_for_fighter(fight, fighter.id),
                weight_class=str(fight.weight_class or ""),
                method=str(fight.method or ""),
                round_finished=str(fight.round_finished or ""),
                time_in_round=str(fight.time or ""),
                totals=_pick_totals_for_side(fight, is_f1),
            )
        )

    # Most recent first
    rows.sort(key=lambda r: (r.event_date is not None, r.event_date or datetime.min), reverse=True)
    return rows


def _print_rows(fighter: Fighter, rows: List[FightRow], n: int) -> None:
    print("")
    print(f"Fighter: {fighter.name} (db_id={fighter.id}, ufcstats_id={fighter.fighter_id})")
    print(f"Showing last {min(n, len(rows))} fights (most recent first)")
    print("-" * 100)

    for i, row in enumerate(rows[:n], 1):
        date_display = row.event_date.strftime("%Y-%m-%d") if row.event_date else row.event_date_raw or "Unknown date"
        headline = f"{i:>2}. {date_display} | {row.result} vs {row.opponent_name} | {row.event_name}"
        details = f"    {row.weight_class} | {row.method} | R{row.round_finished} @ {row.time_in_round}"
        print(headline)
        print(details)

        if row.totals:
            # Print a few key totals (as scraped strings)
            keys = [
                ("knockdowns", "KD"),
                ("sig_strikes", "SIG STR"),
                ("total_strikes", "TOT STR"),
                ("takedowns", "TD"),
                ("control_time", "CTRL"),
            ]
            parts = []
            for k, label in keys:
                v = row.totals.get(k)
                if v is not None and str(v).strip() != "":
                    parts.append(f"{label}: {v}")
            if parts:
                print("    Stats:", " | ".join(parts))
            else:
                print("    Stats: (no totals available)")
        else:
            print("    Stats: (fight_stats not populated)")

        print("")


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Print last N fights and key stats for a fighter.")
    p.add_argument("-N", type=int, default=3, help="Number of most recent fights to show (default: 3)")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ufcstats-id", type=str, help="UFCStats fighter id (string)")
    g.add_argument("--db-id", type=int, help="Database fighter id (integer PK)")
    g.add_argument("--name", type=str, help="Fighter name (case-insensitive contains match)")

    p.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    return p


def main() -> int:
    args = _build_cli().parse_args()
    n = int(args.N)
    if n <= 0:
        raise SystemExit("N must be >= 1")

    db = DatabaseManager(config_path=args.config)
    session = db.get_session()
    try:
        fighter: Optional[Fighter] = None

        if args.ufcstats_id:
            fighter = session.query(Fighter).filter(Fighter.fighter_id == args.ufcstats_id.strip()).first()
        elif args.db_id:
            fighter = session.query(Fighter).filter(Fighter.id == int(args.db_id)).first()
        else:
            q = args.name.strip()
            matches = (
                session.query(Fighter)
                .filter(Fighter.name.ilike(f"%{q}%"))
                .order_by(Fighter.name)
                .all()
            )
            if not matches:
                print(f"No fighter found matching name: {q}")
                return 2
            if len(matches) > 1:
                print(f"Multiple fighters match '{q}'. Use --db-id or --ufcstats-id to disambiguate:")
                for m in matches[:25]:
                    print(f"  - {m.name} (db_id={m.id}, ufcstats_id={m.fighter_id})")
                if len(matches) > 25:
                    print(f"  ... and {len(matches) - 25} more")
                return 2
            fighter = matches[0]

        if not fighter:
            print("Fighter not found.")
            return 2

        fights = (
            session.query(Fight)
            .filter((Fight.fighter_1_id == fighter.id) | (Fight.fighter_2_id == fighter.id))
            .join(Fight.event)
            .all()
        )

        rows = _build_rows(fighter, fights)
        _print_rows(fighter, rows, n)
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())


