#!/usr/bin/env python3
"""Filter a remote UFC events API card for MCP fight-review candidates."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any


DEFAULT_EVENTS_URL = "http://107.175.94.166:8002/api/events"


def _american_in_band(odds: Any, low: int, high: int) -> bool:
    if odds is None:
        return False
    try:
        value = int(float(odds))
    except (TypeError, ValueError):
        return False
    return low <= value <= high


def _fight_count_ok(count: Any, minimum: int) -> bool:
    if count is None:
        return False
    try:
        return int(count) > minimum
    except (TypeError, ValueError):
        return False


def _load_events(url: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError(f"Expected events API to return a list, got {type(payload).__name__}")
    return payload


def _find_event(events: list[dict[str, Any]], event_date: str) -> dict[str, Any]:
    matches = [
        event
        for event in events
        if str(event.get("event_date")) == event_date
        or event_date in str(event.get("event_name", ""))
    ]
    if not matches:
        raise ValueError(f"No event found for date {event_date}")
    if len(matches) > 1:
        names = ", ".join(str(event.get("event_name")) for event in matches)
        raise ValueError(f"Multiple events matched {event_date}: {names}")
    return matches[0]


def _eligible_fights(
    event: dict[str, Any],
    min_fights: int,
    odds_low: int,
    odds_high: int,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for fight in event.get("fights", []):
        if not isinstance(fight, dict):
            continue
        if not (
            _american_in_band(fight.get("f1_odds"), odds_low, odds_high)
            and _american_in_band(fight.get("f2_odds"), odds_low, odds_high)
            and _fight_count_ok(fight.get("f1_fight_count"), min_fights)
            and _fight_count_ok(fight.get("f2_fight_count"), min_fights)
        ):
            continue
        eligible.append(fight)
    return eligible


def _print_text(event: dict[str, Any], fights: list[dict[str, Any]]) -> None:
    print(f"{event.get('event_name')} ({event.get('event_date')})")
    print(f"eligible_fights={len(fights)}")
    for fight in fights:
        print(
            " - "
            f"{fight.get('fighter1')} ({fight.get('f1_odds')}, "
            f"count={fight.get('f1_fight_count')}) vs "
            f"{fight.get('fighter2')} ({fight.get('f2_odds')}, "
            f"count={fight.get('f2_fight_count')}) | "
            f"pick={fight.get('model_pick')} | "
            f"model_f1={fight.get('model_prob_f1')} | "
            f"edge={fight.get('edge')} | "
            f"skip={fight.get('skip_reason')} | "
            f"review={fight.get('review_label')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a remote /api/events payload and list fights eligible for quick "
            "UFC MCP card review."
        )
    )
    parser.add_argument("--url", default=DEFAULT_EVENTS_URL, help="Remote /api/events URL")
    parser.add_argument("--date", required=True, help="Event date, e.g. 2026-05-30")
    parser.add_argument("--min-fights", type=int, default=2, help="Require fight_count > this value")
    parser.add_argument("--odds-low", type=int, default=-300, help="Lowest allowed American odds")
    parser.add_argument("--odds-high", type=int, default=300, help="Highest allowed American odds")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    try:
        events = _load_events(args.url)
        event = _find_event(events, args.date)
        fights = _eligible_fights(event, args.min_fights, args.odds_low, args.odds_high)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "event": {key: event.get(key) for key in ("event_name", "event_date", "event_url")},
                    "filters": {
                        "min_fights_gt": args.min_fights,
                        "odds_low": args.odds_low,
                        "odds_high": args.odds_high,
                    },
                    "eligible_fights": fights,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_text(event, fights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
