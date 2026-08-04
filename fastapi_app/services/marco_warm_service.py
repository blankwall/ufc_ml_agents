"""Hourly warming of Marco results for all future cards."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .marco_service import run_marco_prediction
from .runtime_status import (
    configure_job,
    get_job_status,
    mark_check,
    mark_run_finished,
    mark_run_started,
)

JOB_NAME = "marco_cache_warm"
AUTO_WARM_ENV = "MARCO_AUTO_WARM"
CHECK_SECONDS_ENV = "MARCO_WARM_CHECK_SECONDS"
FALSEY = {"0", "false", "no", "off"}
FASTAPI_DIR = Path(__file__).resolve().parent.parent
if str(FASTAPI_DIR) not in sys.path:
    sys.path.insert(0, str(FASTAPI_DIR))


def scheduler_enabled() -> bool:
    return os.getenv(AUTO_WARM_ENV, "1").strip().lower() not in FALSEY


def _event_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    for fmt in ("%B %d", "%b %d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(year=datetime.now(timezone.utc).year).date()
        except ValueError:
            continue
    return None


def warm_future_cards(*, today: date | None = None) -> dict[str, Any]:
    from .predict_service import get_events_data
    from .ufc_schedule_service import load_allowlist

    configure_job(JOB_NAME, enabled=scheduler_enabled())
    mark_check(JOB_NAME)
    if not scheduler_enabled():
        return {
            "cards": 0,
            "fights": 0,
            "warmed": 0,
            "cached": 0,
            "unavailable": 0,
            "errors": 0,
        }

    mark_run_started(JOB_NAME, trigger="hourly")
    cutoff = today or datetime.now(timezone.utc).date()
    summary = {
        "cards": 0,
        "fights": 0,
        "warmed": 0,
        "cached": 0,
        "unavailable": 0,
        "errors": 0,
    }
    try:
        events = list(get_events_data())
        allowlist = load_allowlist() or {}
        events.extend(allowlist.get("events", []))
        seen_cards: set[tuple[date, str]] = set()
        seen_fights: set[tuple[date, tuple[str, str]]] = set()

        for event in events:
            fight_date = _event_date(event.get("event_date"))
            if fight_date is None:
                fight_date = _event_date(event.get("date"))
            if fight_date is None or fight_date <= cutoff:
                continue
            card_key = (fight_date, str(event.get("event_name") or event.get("name") or ""))
            if card_key not in seen_cards:
                seen_cards.add(card_key)
                summary["cards"] += 1
            for fight in event.get("fights", event.get("bouts", [])):
                if not isinstance(fight, dict):
                    continue
                fighter1 = fight.get("fighter1")
                fighter2 = fight.get("fighter2")
                if not fighter1 or not fighter2:
                    continue
                matchup = tuple(sorted((str(fighter1).strip().lower(), str(fighter2).strip().lower())))
                fight_key = (fight_date, matchup)
                if fight_key in seen_fights:
                    continue
                seen_fights.add(fight_key)
                summary["fights"] += 1
                result = run_marco_prediction(
                    str(fighter1),
                    str(fighter2),
                    fight_date=fight_date,
                )
                if result.get("status") != "complete":
                    if result.get("error_code") == "fighter_not_found":
                        summary["unavailable"] += 1
                    else:
                        summary["errors"] += 1
                elif result.get("cache_hit"):
                    summary["cached"] += 1
                else:
                    summary["warmed"] += 1
        all_failed = summary["fights"] > 0 and summary["errors"] == summary["fights"]
        mark_run_finished(
            JOB_NAME,
            success=not all_failed,
            summary=summary,
            error="Every future-card Marco prediction failed." if all_failed else None,
        )
        return summary
    except Exception as exc:
        mark_run_finished(JOB_NAME, success=False, summary=summary, error=str(exc))
        raise


async def run_sync_loop() -> None:
    check_seconds = int(os.getenv(CHECK_SECONDS_ENV, "3600"))
    while True:
        try:
            await asyncio.to_thread(warm_future_cards)
        except Exception:
            pass
        await asyncio.sleep(check_seconds)


def get_health_status() -> dict[str, Any]:
    return get_job_status(JOB_NAME)
