#!/usr/bin/env python3
"""
Match BestFightOdds fighter odds previews to DB events.

Usage:
  python add_odds_db/fighter_event_match_preview.py \\
      --preview analysis/islam-makhachev_odds_preview.csv

This script expects a CSV produced by `fighter_odds_preview.py`, with at least:
  - event_slug
  - event_label
  - fighter
  - opponent
  - opening_american
  - closing_american

For each row it:
  - Parses the BFO event date from `event_label` (e.g. "UFC Jan 19th 2025")
  - Loads all `Event` rows from the DB
  - Parses `Event.date` using several known formats
  - Finds the closest DB event within ±3 days of the BFO date
  - Keeps only rows where:
      * event_slug starts with "ufc-"
      * a DB event match is found
  - Writes a merged CSV with:
      event_slug, db_event_name, db_event_id, opponent, opening_american, closing_american
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Dict, Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(ROOT))

from database.db_manager import DatabaseManager  # type: ignore
from database.schema import Event  # type: ignore


def parse_bfo_event_date(label: str) -> Optional[datetime]:
    """
    Parse BFO-style label like "UFC Jan 19th 2025" into a datetime.
    We look for a "<Mon> <day><suffix> <year>" substring.
    """
    parts = label.strip().split()
    if len(parts) < 3:
        return None

    month_idx = None
    for i, tok in enumerate(parts):
        if tok[:3].isalpha() and tok[:3].title() in [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
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
    """
    Parse our DB `Event.date` field into a datetime, trying several formats.
    """
    if not date_str:
        return None

    fmts = [
        "%B %d, %Y",          # "January 18, 2025"
        "%b %d, %Y",          # "Jan 18, 2025"
        "%Y-%m-%d",           # "2025-01-18"
        "%Y-%m-%d %H:%M:%S",  # "2025-01-18 00:00:00"
        "%Y-%m-%dT%H:%M:%S",  # "2025-01-18T00:00:00"
        "%m/%d/%Y",           # "01/18/2025"
        "%d/%m/%Y",           # "18/01/2025"
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Match fighter odds preview CSV to DB events.")
    parser.add_argument(
        "--preview",
        required=True,
        help="Path to *_odds_preview.csv produced by fighter_odds_preview.py",
    )
    parser.add_argument(
        "--output",
        help="Optional output CSV path (default: *_odds_with_events.csv next to preview).",
    )
    args = parser.parse_args()

    preview_path = Path(args.preview)
    if not preview_path.is_absolute():
        preview_path = ROOT / preview_path

    if not preview_path.exists():
        print(f"Preview CSV not found: {preview_path}")
        return

    df = pd.read_csv(preview_path)

    db = DatabaseManager()
    session = db.get_session()

    try:
        events: Iterable[Event] = session.query(Event).all()
        event_rows = []
        for ev in events:
            ev_dt = parse_event_date_field(ev.date or "")
            event_rows.append((ev, ev_dt))

        print(f"Loaded {len(df)} preview rows from {preview_path}.")
        print(f"Loaded {len(event_rows)} events from DB.\n")

        mappings: list[Dict[str, Any]] = []

        for _, row in df.iterrows():
            event_slug = row.get("event_slug", "")
            label = row.get("event_label", "")

            # Only consider UFC events; drop future/unconfirmed/other orgs.
            if not isinstance(event_slug, str) or not event_slug.startswith("ufc-"):
                continue

            target_dt = parse_bfo_event_date(str(label))
            if not target_dt:
                continue

            window = timedelta(days=3)
            candidates = []
            for ev, ev_dt in event_rows:
                if ev_dt is None:
                    continue
                delta = abs(ev_dt - target_dt)
                if delta <= window:
                    candidates.append((delta, ev))
            candidates.sort(key=lambda t: t[0])

            if not candidates:
                continue

            best_delta, best_ev = candidates[0]
            mappings.append(
                {
                    "event_slug": event_slug,
                    "db_event_name": best_ev.name,
                    "db_event_id": best_ev.event_id,
                    "db_event_date": best_ev.date,
                    "delta_days": best_delta.days,
                }
            )

        if not mappings:
            print("No DB event mappings found for this preview.")
            return

        map_df = pd.DataFrame(mappings)
        merged = df.merge(map_df, on="event_slug", how="inner")
        # Drop exact duplicate rows for the same event/opponent/odds.
        merged = merged.drop_duplicates(
            subset=["event_slug", "opponent", "opening_american", "closing_american"]
        ).copy()

        display_cols = [
            "event_slug",
            "db_event_name",
            "opponent",
            "opening_american",
            "closing_american",
        ]

        print("=" * 80)
        print("Fighter – opening/closing odds with matched DB event names:\n")
        print(merged[display_cols].to_string(index=False))

        if args.output:
            out_path = Path(args.output)
            if not out_path.is_absolute():
                out_path = ROOT / out_path
        else:
            out_path = preview_path.with_name(preview_path.stem.replace("_odds_preview", "_odds_with_events") + ".csv")

        merged.to_csv(out_path, index=False)
        print(f"\nWrote merged table → {out_path}")

    finally:
        session.close()


if __name__ == "__main__":
    main()

