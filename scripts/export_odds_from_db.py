#!/usr/bin/env python3
"""
Export fights that have betting odds from the database to a CSV.

Output format matches what backtest_2025.py and evaluation/evaluate_model.py expect:
  date, fighter1, fighter2, fighter1_odds, fighter2_odds

Use this to run a full 2025 (or any year) backtest with odds from the DB:

  python scripts/export_odds_from_db.py --year 2025 -o backtest/odds/db_odds_2025.csv
  python backtest/backtest_2025.py --odds backtest/odds/db_odds_2025.csv --model mar_4_v2

  # Or with the evaluation pipeline (model vs market, Brier, ROI, etc.):
  python evaluation/evaluate_model.py --odds-path backtest/odds/db_odds_2025.csv --min-year 2025
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy.orm import aliased

from database.db_manager import DatabaseManager
from database.schema import Fight, Event, Fighter, BettingOdds


def export_odds(session, year: int | None = None, output_path: Path | None = None) -> pd.DataFrame:
    """
    Query all fights that have at least one betting_odds row.
    Prefer closing line, then opening, then any. One row per fight.
    """
    F1 = aliased(Fighter)
    F2 = aliased(Fighter)
    q = (
        session.query(
            Event.date,
            F1.name.label("fighter1"),
            F2.name.label("fighter2"),
            BettingOdds.fighter_1_odds,
            BettingOdds.fighter_2_odds,
            Fight.id.label("fight_id"),
            BettingOdds.is_closing_line,
            BettingOdds.is_opening_line,
        )
        .select_from(Fight)
        .join(Event, Fight.event_id == Event.id)
        .join(F1, Fight.fighter_1_id == F1.id)
        .join(F2, Fight.fighter_2_id == F2.id)
        .join(BettingOdds, Fight.id == BettingOdds.fight_id)
    )
    rows = q.all()
    if year is not None:
        def event_year(d):
            if d is None:
                return None
            s = str(d)
            if "-" in s:
                return int(s[:4]) if s[:4].isdigit() else None
            if "," in s:
                part = s.strip().split()[-1]
                return int(part) if part.isdigit() else None
            return int(s[:4]) if len(s) >= 4 and s[:4].isdigit() else None
        rows = [r for r in rows if event_year(r.date) == year]

    if not rows:
        return pd.DataFrame()

    def norm_date(d):
        if d is None:
            return ""
        if hasattr(d, "strftime"):
            return d.strftime("%Y-%m-%d")
        s = str(d).strip()
        if not s:
            return ""
        try:
            from datetime import datetime
            parsed = datetime.strptime(s, "%B %d, %Y")
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            pass
        if "-" in s and len(s) >= 10:
            return s[:10]
        return s

    df = pd.DataFrame(
        [
            {
                "date": norm_date(r.date),
                "fighter1": r.fighter1,
                "fighter2": r.fighter2,
                "fighter1_odds": r.fighter_1_odds,
                "fighter2_odds": r.fighter_2_odds,
                "_fight_id": r.fight_id,
                "_is_closing": r.is_closing_line or False,
                "_is_opening": r.is_opening_line or False,
            }
            for r in rows
        ]
    )
    # One row per fight: prefer closing, then opening, then first
    df["_priority"] = df["_is_closing"].astype(int) * 2 + df["_is_opening"].astype(int)
    df = df.sort_values("_priority", ascending=False).drop_duplicates(subset=["_fight_id"], keep="first")
    df = df.drop(columns=["_fight_id", "_is_closing", "_is_opening", "_priority"])
    df = df.sort_values("date").reset_index(drop=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} fights to {output_path}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Export fights with odds from DB to CSV")
    ap.add_argument("--year", type=int, default=None, help="Filter by event year (e.g. 2025)")
    ap.add_argument("-o", "--output", type=str, default=None, help="Output CSV path")
    ap.add_argument("--list", action="store_true", help="Only print count and sample; do not write CSV")
    args = ap.parse_args()

    try:
        db = DatabaseManager()
    except Exception as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)

    session = db.get_session()
    try:
        df = export_odds(session, year=args.year, output_path=Path(args.output) if args.output else None)
        if df.empty:
            print("No fights with odds found." + (" Try without --year." if args.year else ""))
            sys.exit(0)
        if args.list or not args.output:
            print(f"Fights with odds: {len(df)}" + (f" (year={args.year})" if args.year else ""))
            print(df.head(10).to_string(index=False))
    finally:
        session.close()


if __name__ == "__main__":
    main()
