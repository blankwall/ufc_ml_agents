#!/usr/bin/env python3
"""
Preview mapping of BestFightOdds event labels for Jalin Turner to our DB events.

Input:
  - analysis/jalin_turner_odds_preview.csv

For each row (one Jalin Turner fight on BestFightOdds), this script:
  - Parses the BFO event label (e.g. "UFC Dec 7th 2025") to a canonical date.
  - Looks up candidate `Event` rows in our database whose `date` field is
    within a small window of that date (±3 days), using several known date formats.
  - Prints the top candidate DB events (name + date) so we can see which mapping
    is correct, WITHOUT writing anything back to the DB.

This is a read-only, manual-validation helper before we wire any odds into
BettingOdds/Fight records.
"""

from __future__ import annotations

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
    Parse BFO-style label like "UFC Dec 7th 2025" into a datetime.
    We look for a "<Mon> <day><suffix> <year>" substr.
    """
    # Extract the last three tokens that look like "Dec 7th 2025"
    parts = label.strip().split()
    if len(parts) < 3:
        return None
    # Try to find month token (three-letter month)
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

    # Strip ordinal suffixes from day (7th -> 7)
    for suf in ("st", "nd", "rd", "th"):
        if day_raw.lower().endswith(suf):
            day_raw = day_raw[: -len(suf)]
            break

    try:
        dt = datetime.strptime(f"{month} {day_raw} {year}", "%b %d %Y")
        return dt
    except ValueError:
        return None


def parse_event_date_field(date_str: str) -> Optional[datetime]:
    """
    Parse our DB `Event.date` field into a datetime, trying several formats.
    """
    if not date_str:
        return None

    candidates = [
        "%B %d, %Y",  # "September 28, 2024"
        "%b %d, %Y",  # "Sep 28, 2024"
        "%Y-%m-%d",  # "2024-09-28"
        "%m/%d/%Y",  # "09/28/2024"
        "%d/%m/%Y",  # "28/09/2024"
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def main() -> None:
    csv_path = ROOT / "analysis" / "jalin_turner_odds_preview.csv"
    if not csv_path.exists():
        print(f"Preview CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    db = DatabaseManager()
    session = db.get_session()

    try:
        events: Iterable[Event] = session.query(Event).all()
        # Pre-parse event dates
        event_rows = []
        for ev in events:
            ev_dt = parse_event_date_field(ev.date or "")
            event_rows.append((ev, ev_dt))

        print(f"Loaded {len(df)} Jalin Turner fights from preview CSV.")
        print(f"Loaded {len(event_rows)} events from DB.\n")

        mappings: list[Dict[str, Any]] = []

        for _, row in df.iterrows():
            event_slug = row.get("event_slug", "")
            label = row.get("event_label", "")
            opponent = row.get("opponent", "")

            target_dt = parse_bfo_event_date(str(label))

            print("=" * 80)
            print(f"BFO event_slug: {event_slug}")
            print(f"BFO label:      {label}")
            print(f"Opponent:       {opponent}")
            if target_dt:
                print(f"Parsed BFO date: {target_dt.date()}")
            else:
                print("Parsed BFO date: (could not parse)")

            # Find candidate events within ±3 days of target date
            if target_dt is not None:
                window = timedelta(days=3)
                candidates = []
                for ev, ev_dt in event_rows:
                    if ev_dt is None:
                        continue
                    delta = abs(ev_dt - target_dt)
                    if delta <= window:
                        candidates.append((delta, ev))
                candidates.sort(key=lambda t: t[0])

                if candidates:
                    print("Closest DB events (within ±3 days):")
                    for delta, ev in candidates[:5]:
                        print(
                            f"  - {ev.name} | date={ev.date} | event_id={ev.event_id} | Δdays={delta.days}"
                        )
                    # Best match is the closest in time
                    best_delta, best_ev = candidates[0]
                    # Only keep UFC events (drop Bellator/other orgs). BFO slugs for UFC
                    # start with 'ufc-' (e.g. 'ufc-3895', 'ufc-2921', etc.).
                    if isinstance(event_slug, str) and event_slug.startswith("ufc-"):
                        mappings.append(
                            {
                                "event_slug": event_slug,
                                "db_event_name": best_ev.name,
                                "db_event_id": best_ev.event_id,
                                "db_event_date": best_ev.date,
                                "delta_days": best_delta.days,
                            }
                        )
                else:
                    print("No DB events found within ±3 days of this BFO date.")
            else:
                print("Skipping DB date match due to unparsed BFO date.")

            print()

        # If we built any mappings, join them back onto the preview CSV and show a clean table.
        if mappings:
            map_df = pd.DataFrame(mappings)
            merged = df.merge(map_df, on="event_slug", how="left")
            # Drop rows where we couldn't find a matching DB event
            merged = merged[merged["db_event_name"].notna()].copy()

            print("=" * 80)
            print("Jalin Turner – opening/closing odds with matched DB event names (DB-only):\n")
            display_cols = [
                "event_slug",
                "db_event_name",
                "opponent",
                "opening_american",
                "closing_american",
            ]

            print(merged[display_cols].to_string(index=False))

            out_path = ROOT / "analysis" / "jalin_turner_odds_with_events.csv"
            merged.to_csv(out_path, index=False)
            print(f"\nWrote merged table → {out_path}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

