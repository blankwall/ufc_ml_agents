from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.schema import Event
from scrapers.event_populator import EventPopulator, PopulatorOptions
from scrapers.event_scraper import EventScraper
from fastapi_app.services.runtime_status import (
    configure_job,
    get_job_status,
    mark_check,
    mark_run_finished,
    mark_run_started,
)

CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
DB_PATH = ROOT_DIR / "data" / "ufc_database.db"
STATE_PATH = ROOT_DIR / "data" / "ufcstats_sync_state.json"

AUTO_SYNC_ENV = "UFCSTATS_AUTO_SYNC"
SYNC_INTERVAL_HOURS_ENV = "UFCSTATS_SYNC_INTERVAL_HOURS"
SYNC_CHECK_SECONDS_ENV = "UFCSTATS_SYNC_CHECK_SECONDS"
LOOKBACK_DAYS_ENV = "UFCSTATS_COMPLETED_LOOKBACK_DAYS"
MAX_PAGES_ENV = "UFCSTATS_COMPLETED_MAX_PAGES"
MIN_FIGHTS_ENV = "UFCSTATS_COMPLETED_MIN_FIGHTS"
MAX_EVENTS_PER_RUN_ENV = "UFCSTATS_COMPLETED_MAX_EVENTS_PER_RUN"

FALSEY = {"0", "false", "no", "off"}
DEFAULT_LOOKBACK_DAYS = 14
DEFAULT_MAX_PAGES = 1
DEFAULT_MIN_FIGHTS = 5
DEFAULT_MAX_EVENTS_PER_RUN = 1

logger = logging.getLogger(__name__)

JOB_NAME = "ufcstats_completed_sync"

_engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    temp_path.replace(path)


def _load_state() -> dict[str, Any]:
    state = _load_json(STATE_PATH, {})
    if not isinstance(state, dict):
        return {}
    state.setdefault("events", {})
    return state


def _save_state(state: dict[str, Any]) -> None:
    _write_json(STATE_PATH, state)


def get_health_status() -> dict[str, Any]:
    configure_job(JOB_NAME, enabled=scheduler_enabled())
    state = _load_state()
    return {
        **get_job_status(JOB_NAME),
        "state_file": str(STATE_PATH.relative_to(ROOT_DIR)),
        "persisted_last_run_started_at": state.get("last_run_started_at"),
        "persisted_last_run_finished_at": state.get("last_run_finished_at"),
        "persisted_last_success_at": state.get("last_success_at"),
        "persisted_last_result": {
            "recent_events_seen": state.get("recent_events_seen"),
            "candidates_considered": state.get("candidates_considered"),
            "synced_events": state.get("synced_events"),
            "dry_run_failed": state.get("dry_run_failed"),
            "validation_failed": state.get("validation_failed"),
        },
        "recent_failed_events": _recent_failed_events(state),
    }


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _parse_event_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        with suppress(ValueError):
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
    return None


def _known_event_ids_from_db() -> set[str]:
    session = _Session()
    try:
        rows = session.query(Event.event_id).all()
        return {row[0] for row in rows if row and row[0]}
    finally:
        session.close()


def _recent_completed_events(now: datetime | None = None) -> list[dict[str, str]]:
    now = now or _utc_now()
    scraper = EventScraper(config_path=str(CONFIG_PATH))
    raw_events = scraper.get_all_event_links(
        completed_only=True,
        max_pages=_int_env(MAX_PAGES_ENV, DEFAULT_MAX_PAGES),
    )
    lookback_days = _int_env(LOOKBACK_DAYS_ENV, DEFAULT_LOOKBACK_DAYS)
    cutoff = now - timedelta(days=lookback_days)

    recent: list[dict[str, str]] = []
    for event in raw_events:
        event_dt = _parse_event_date(event.get("date"))
        if event_dt is not None and event_dt < cutoff:
            continue
        recent.append(
            {
                "event_id": str(event.get("event_id", "")).strip(),
                "name": str(event.get("name", "")).strip(),
                "url": EventPopulator.normalize_ufcstats_url(str(event.get("url", "")).strip()),
                "date": str(event.get("date", "")).strip(),
                "location": str(event.get("location", "")).strip(),
            }
        )
    return [event for event in recent if event["event_id"] and event["url"]]


def _dry_run_summary_is_safe(summary: dict[str, Any]) -> tuple[bool, str | None]:
    fights_total = int(summary.get("fights_total") or 0)
    fights_upserted = int(summary.get("fights_upserted") or 0)
    fighters_failed = int(summary.get("fighters_failed") or 0)
    min_fights = _int_env(MIN_FIGHTS_ENV, DEFAULT_MIN_FIGHTS)

    if fights_total < min_fights:
        return False, f"fights_total below minimum ({fights_total} < {min_fights})"
    if fights_upserted != fights_total:
        return False, f"fights_upserted mismatch ({fights_upserted} != {fights_total})"
    if fighters_failed:
        return False, f"fighters_failed={fighters_failed}"
    return True, None


def _create_db_backup() -> str | None:
    if not DB_PATH.exists():
        return None
    backups_dir = DB_PATH.parent / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backups_dir / f"{DB_PATH.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{DB_PATH.suffix}"

    src = sqlite3.connect(str(DB_PATH))
    try:
        dst = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()
    return str(backup_path.relative_to(ROOT_DIR))


def _restore_db_backup(relative_path: str | None) -> bool:
    if not relative_path:
        return False
    backup_path = ROOT_DIR / relative_path
    if not backup_path.exists():
        return False
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(backup_path))
    try:
        dst = sqlite3.connect(str(DB_PATH))
        try:
            src.backup(dst)
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()
    return True


def _build_populator() -> EventPopulator:
    populator = EventPopulator(config_path=str(CONFIG_PATH))
    populator._backup_db_if_sqlite = lambda: None  # service creates one backup per run
    return populator


def _retry_command(event: dict[str, str]) -> str:
    return (
        ".venv/bin/python scrapers/event_populator.py "
        f"--event-id {event['event_id']} "
        "--include-fight-stats "
        "--force-refresh-fighters "
        "--validate "
        "--validate-details"
    )


def _recent_failed_events(state: dict[str, Any], *, limit: int = 3) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for event_id, payload in state.get("events", {}).items():
        if not isinstance(payload, dict):
            continue
        if payload.get("status") in {"synced", None}:
            continue
        failures.append(
            {
                "event_id": event_id,
                "event_name": payload.get("event_name"),
                "event_date": payload.get("event_date"),
                "status": payload.get("status"),
                "reason": payload.get("reason"),
                "last_attempt_at": payload.get("last_attempt_at"),
                "validation": payload.get("validation"),
                "retry_command": payload.get("retry_command"),
            }
        )
    failures.sort(key=lambda item: item.get("last_attempt_at") or "", reverse=True)
    return failures[:limit]


def _dry_run_event(event: dict[str, str]) -> dict[str, Any]:
    populator = _build_populator()
    dry_run_summary = populator.populate_event_from_url(
        event["url"],
        options=PopulatorOptions(
            include_fight_stats=True,
            force_refresh_fighters=True,
            bust_cache=True,
            commit=False,
        ),
    )
    is_safe, failure_reason = _dry_run_summary_is_safe(dry_run_summary)
    return {
        "event_id": event["event_id"],
        "event_name": event.get("name"),
        "event_date": event.get("date"),
        "status": "dry_run_ok" if is_safe else "dry_run_failed",
        "reason": failure_reason,
        "dry_run_summary": dry_run_summary,
        "retry_command": _retry_command(event),
    }


def _commit_validated_event(event: dict[str, str], *, dry_run_summary: dict[str, Any]) -> dict[str, Any]:
    populator = _build_populator()
    commit_summary = populator.populate_event_from_url(
        event["url"],
        options=PopulatorOptions(
            include_fight_stats=True,
            force_refresh_fighters=True,
            bust_cache=True,
            commit=True,
        ),
    )
    validation = populator.validate_event_against_db(event["url"], use_fight_details=True)
    status = "synced" if validation.get("missing_in_db", 0) == 0 and validation.get("mismatches", 0) == 0 else "validation_failed"
    result = {
        "event_id": event["event_id"],
        "event_name": commit_summary.get("event_name") or event.get("name"),
        "event_date": event.get("date"),
        "status": status,
        "dry_run_summary": dry_run_summary,
        "commit_summary": commit_summary,
        "validation": validation,
        "retry_command": _retry_command(event),
    }
    return result


def sync_completed_ufcstats_events(*, dry_run: bool = False) -> dict[str, Any]:
    configure_job(JOB_NAME, enabled=scheduler_enabled())
    mark_run_started(JOB_NAME, trigger="manual" if dry_run else "scheduled")
    started_at = _utc_now()
    logger.info("Starting UFCStats completed-event sync dry_run=%s", dry_run)
    state = _load_state()
    known_event_ids = _known_event_ids_from_db()
    recent_events = _recent_completed_events(now=started_at)

    synced_event_ids = {
        event_id
        for event_id, payload in state.get("events", {}).items()
        if isinstance(payload, dict) and payload.get("status") == "synced"
    }
    candidates = [event for event in recent_events if event["event_id"] not in known_event_ids and event["event_id"] not in synced_event_ids]
    candidates = candidates[: _int_env(MAX_EVENTS_PER_RUN_ENV, DEFAULT_MAX_EVENTS_PER_RUN)]

    backup_path = None
    processed: list[dict[str, Any]] = []
    synced_count = 0
    dry_run_failed = 0
    validation_failed = 0
    skipped_existing = max(0, len(recent_events) - len(candidates))

    safe_candidates: list[tuple[dict[str, str], dict[str, Any]]] = []
    for event in candidates:
        result = _dry_run_event(event)
        processed.append(result)
        if result["status"] == "dry_run_failed":
            dry_run_failed += 1
            state.setdefault("events", {})[event["event_id"]] = {
                "event_name": result.get("event_name"),
                "event_date": result.get("event_date"),
                "status": result["status"],
                "last_attempt_at": _iso_z(_utc_now()),
                "reason": result.get("reason"),
                "validation": result.get("validation"),
                "retry_command": result.get("retry_command"),
            }
        else:
            safe_candidates.append((event, result["dry_run_summary"]))

    if safe_candidates and not dry_run:
        backup_path = _create_db_backup()
        processed = [result for result in processed if result["status"] == "dry_run_failed"]
        for event, dry_run_summary in safe_candidates:
            result = _commit_validated_event(event, dry_run_summary=dry_run_summary)
            if (
                result["status"] == "validation_failed"
                and backup_path
                and len(safe_candidates) == 1
                and _restore_db_backup(backup_path)
            ):
                result["status"] = "validation_failed_restored"
                result["reason"] = "validation failed; restored DB backup"
            processed.append(result)
            if result["status"] == "synced":
                synced_count += 1
            elif result["status"] in {"validation_failed", "validation_failed_restored"}:
                validation_failed += 1
            state.setdefault("events", {})[event["event_id"]] = {
                "event_name": result.get("event_name"),
                "event_date": result.get("event_date"),
                "status": result["status"],
                "last_attempt_at": _iso_z(_utc_now()),
                "reason": result.get("reason"),
                "validation": result.get("validation"),
                "retry_command": result.get("retry_command"),
            }

    finished_at = _utc_now()
    payload = {
        "last_run_started_at": _iso_z(started_at),
        "last_run_finished_at": _iso_z(finished_at),
        "last_success_at": _iso_z(finished_at) if not dry_run else state.get("last_success_at"),
        "backup_path": backup_path,
        "recent_events_seen": len(recent_events),
        "candidates_considered": len(candidates),
        "skipped_existing": skipped_existing,
        "synced_events": synced_count,
        "dry_run_failed": dry_run_failed,
        "validation_failed": validation_failed,
        "processed": processed,
        "dry_run": dry_run,
    }

    if not dry_run:
        state.update(payload)
        _save_state(state)
    mark_run_finished(JOB_NAME, success=True, summary=payload)
    logger.info(
        "UFCStats completed-event sync finished recent_events_seen=%s candidates=%s synced=%s dry_run_failed=%s validation_failed=%s",
        payload["recent_events_seen"],
        payload["candidates_considered"],
        payload["synced_events"],
        payload["dry_run_failed"],
        payload["validation_failed"],
    )
    return payload


def scheduler_enabled() -> bool:
    return os.getenv(AUTO_SYNC_ENV, "0").strip().lower() not in FALSEY


def sync_due(now: datetime | None = None) -> bool:
    state = _load_state()
    last_success = _parse_iso(state.get("last_success_at"))
    if last_success is None:
        return True
    interval_hours = float(os.getenv(SYNC_INTERVAL_HOURS_ENV, "24"))
    now = now or _utc_now()
    return now - last_success >= timedelta(hours=interval_hours)


def sync_if_due() -> dict[str, Any] | None:
    configure_job(JOB_NAME, enabled=scheduler_enabled())
    mark_check(JOB_NAME)
    if not scheduler_enabled() or not sync_due():
        return None
    logger.info("Running scheduled UFCStats completed-event sync")
    result = sync_completed_ufcstats_events()
    logger.info(
        "UFCStats completed-event sync processed %s candidate events; synced=%s validation_failed=%s",
        result["candidates_considered"],
        result["synced_events"],
        result["validation_failed"],
    )
    return result


async def run_sync_loop() -> None:
    check_seconds = _int_env(SYNC_CHECK_SECONDS_ENV, 3600)
    while True:
        try:
            await asyncio.to_thread(sync_if_due)
        except Exception:
            mark_run_finished(JOB_NAME, success=False, error="Scheduled UFCStats completed-event sync failed")
            logger.exception("Scheduled UFCStats completed-event sync failed")
        await asyncio.sleep(check_seconds)
