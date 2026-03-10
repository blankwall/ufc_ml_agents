#!/usr/bin/env python3
"""
Backtest 2025: bet 1 unit on the market favorite in every fight.

- If the DB has betting odds for the year: uses odds and outcomes from the database.
- Otherwise (or if --odds-csv is set): uses odds from CSV and outcomes from the DB,
  matching fights by event date and fighter names.

  python scripts/backtest_favorites_2025.py
  python scripts/backtest_favorites_2025.py --year 2025
  python scripts/backtest_favorites_2025.py --odds-csv ufc_2025_odds.csv
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy.orm import aliased

from database.db_manager import DatabaseManager
from database.schema import Fight, Event, Fighter, BettingOdds


def american_profit_per_unit(odds: int) -> float:
    """Profit per 1 unit stake if the bet wins. American odds."""
    if odds is None:
        return 0.0
    o = int(odds)
    if o < 0:
        return 100.0 / abs(o)
    return o / 100.0


def _normalize_name(s: str) -> str:
    return " ".join(str(s).strip().split()) if s else ""


def _event_date_key(d) -> str | None:
    """Return YYYY-MM-DD for matching, or None."""
    if d is None:
        return None
    s = str(d).strip()
    # "January 12, 2025" or "2025-01-12" or "November 22, 2025 UFC Fight Night: ..."
    if "-" in s and len(s) >= 10 and s[:4].isdigit():
        return s[:10]
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})", s)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


def _csv_date_key(s: str) -> str | None:
    """Parse CSV date like 'January 12, 2025' or '"January 12, 2025"' -> YYYY-MM-DD."""
    s = str(s).strip().strip('"')
    try:
        dt = datetime.strptime(s, "%B %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def run_with_db_odds(session, year: int):
    """Use only DB: fights that have BettingOdds and winner_id."""
    F1 = aliased(Fighter)
    F2 = aliased(Fighter)
    q = (
        session.query(
            Event.date,
            F1.name.label("fighter1"),
            F2.name.label("fighter2"),
            Fight.fighter_1_id,
            Fight.fighter_2_id,
            Fight.winner_id,
            BettingOdds.fighter_1_odds,
            BettingOdds.fighter_2_odds,
            BettingOdds.is_closing_line,
            BettingOdds.is_opening_line,
            Fight.id.label("fight_id"),
        )
        .select_from(Fight)
        .join(Event, Fight.event_id == Event.id)
        .join(F1, Fight.fighter_1_id == F1.id)
        .join(F2, Fight.fighter_2_id == F2.id)
        .join(BettingOdds, Fight.id == BettingOdds.fight_id)
        .filter(Fight.winner_id.isnot(None))
    )
    rows = list(q.all())

    def event_year(d):
        if d is None:
            return None
        s = str(d)
        if "," in s:
            part = s.strip().split()[-1]
            return int(part) if part.isdigit() else None
        return int(s[:4]) if len(s) >= 4 and s[:4].isdigit() else None

    rows = [r for r in rows if event_year(r.date) == year]
    by_fight = {}
    for r in rows:
        fid = r.fight_id
        priority = (2 if (r.is_closing_line or False) else 0) + (1 if (r.is_opening_line or False) else 0)
        if fid not in by_fight or by_fight[fid][0] < priority:
            by_fight[fid] = (priority, r)
    rows = [r for _, r in by_fight.values()]

    n = wins = 0
    total_profit = 0.0
    for r in rows:
        o1, o2 = r.fighter_1_odds, r.fighter_2_odds
        if o1 is None or o2 is None:
            continue
        if o1 < o2:
            fav_id, fav_odds = r.fighter_1_id, o1
        else:
            fav_id, fav_odds = r.fighter_2_id, o2
        n += 1
        if r.winner_id == fav_id:
            wins += 1
            total_profit += american_profit_per_unit(fav_odds)
        else:
            total_profit -= 1.0
    return n, wins, total_profit, "DB"


def run_with_csv_odds(session, year: int, csv_path: Path):
    """Use odds from CSV; match to DB outcomes by date + fighter names."""
    df = pd.read_csv(csv_path)
    for c in ["date", "fighter1", "fighter2", "fighter1_odds", "fighter2_odds"]:
        if c not in df.columns:
            raise SystemExit(f"CSV must have columns: date, fighter1, fighter2, fighter1_odds, fighter2_odds. Got: {list(df.columns)}")
    # Build DB lookup: (date_key, frozenset({name1, name2})) -> (winner_id, fighter1_name, fighter2_name, fighter_1_id, fighter_2_id)
    F1 = aliased(Fighter)
    F2 = aliased(Fighter)
    q = (
        session.query(
            Event.date,
            F1.name.label("f1_name"),
            F2.name.label("f2_name"),
            Fight.fighter_1_id,
            Fight.fighter_2_id,
            Fight.winner_id,
        )
        .select_from(Fight)
        .join(Event, Fight.event_id == Event.id)
        .join(F1, Fight.fighter_1_id == F1.id)
        .join(F2, Fight.fighter_2_id == F2.id)
        .filter(Fight.winner_id.isnot(None))
    )
    rows = q.all()

    def event_year(d):
        if d is None:
            return None
        s = str(d)
        if "," in s:
            part = s.strip().split()[-1]
            return int(part) if part.isdigit() else None
        return int(s[:4]) if len(s) >= 4 and s[:4].isdigit() else None

    rows = [r for r in rows if event_year(r.date) == year]
    lookup = {}
    for r in rows:
        key = (_event_date_key(r.date), frozenset({_normalize_name(r.f1_name), _normalize_name(r.f2_name)}))
        if key[0] is None:
            continue
        # Keep one fight per key (duplicates possible if same two fighters twice on same card)
        if key not in lookup:
            lookup[key] = (r.winner_id, r.f1_name, r.f2_name, r.fighter_1_id, r.fighter_2_id)
    # If multiple CSV rows for same date+pair, we need to match order; CSV order is fighter1, fighter2
    n = wins = 0
    total_profit = 0.0
    for _, row in df.iterrows():
        date_key = _csv_date_key(row["date"])
        if date_key is None:
            continue
        f1 = _normalize_name(row["fighter1"])
        f2 = _normalize_name(row["fighter2"])
        o1 = row["fighter1_odds"]
        o2 = row["fighter2_odds"]
        if pd.isna(o1) or pd.isna(o2):
            continue
        o1, o2 = int(o1), int(o2)
        key = (date_key, frozenset({f1, f2}))
        if key not in lookup:
            continue
        winner_id, db_f1, db_f2, id1, id2 = lookup[key]
        if o1 < o2:
            fav_odds = o1
            fav_id = id1
        else:
            fav_odds = o2
            fav_id = id2
        n += 1
        if winner_id == fav_id:
            wins += 1
            total_profit += american_profit_per_unit(fav_odds)
        else:
            total_profit -= 1.0
    return n, wins, total_profit, "CSV"


def main():
    ap = argparse.ArgumentParser(description="Backtest ROI betting on all favorites")
    ap.add_argument("--year", type=int, default=2025, help="Event year (default 2025)")
    ap.add_argument("--odds-csv", type=Path, default=None, help="Use this CSV for odds and DB for outcomes (e.g. ufc_2025_odds.csv)")
    args = ap.parse_args()

    db = DatabaseManager()
    session = db.get_session()

    if args.odds_csv is not None:
        n, wins, total_profit, source = run_with_csv_odds(session, args.year, args.odds_csv)
        odds_src = f"odds from {args.odds_csv}, outcomes from DB"
    else:
        n, wins, total_profit, source = run_with_db_odds(session, args.year)
        odds_src = "odds and outcomes from DB"
        if n == 0:
            csv_default = Path(__file__).resolve().parent.parent / "ufc_2025_odds.csv"
            if csv_default.exists():
                print(f"No {args.year} fights with betting odds in the DB. Falling back to CSV odds + DB outcomes.\n")
                n, wins, total_profit, source = run_with_csv_odds(session, args.year, csv_default)
                odds_src = f"odds from {csv_default.name}, outcomes from DB"
    session.close()

    if n == 0:
        print(f"No {args.year} fights with odds and outcomes available.")
        sys.exit(0)

    roi_pct = (total_profit / n * 100)
    print(f"\n{'='*60}")
    print(f"  BACKTEST: Bet on ALL favorites — {args.year}")
    print(f"  {odds_src}")
    print(f"{'='*60}")
    print(f"  Fights:                         {n}")
    print(f"  Favorite won:                   {wins}  ({100*wins/n:.1f}%)")
    print(f"  Total profit (units):           {total_profit:+.2f}")
    print(f"  ROI (1 unit per fight):         {roi_pct:+.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
