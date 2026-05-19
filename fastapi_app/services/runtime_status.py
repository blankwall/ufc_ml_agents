from __future__ import annotations

import copy
from datetime import datetime, timezone
from threading import RLock
from typing import Any

_LOCK = RLock()
_STARTED_AT = datetime.now(timezone.utc)
_JOBS: dict[str, dict[str, Any]] = {}


def _iso_z(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _default_job(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "enabled": None,
        "task_active": False,
        "task_started_at": None,
        "task_stopped_at": None,
        "task_stop_reason": None,
        "checks_since_launch": 0,
        "runs_since_launch": 0,
        "successes_since_launch": 0,
        "failures_since_launch": 0,
        "last_check_at": None,
        "last_run_started_at": None,
        "last_run_finished_at": None,
        "last_success_at": None,
        "last_error_at": None,
        "last_error": None,
        "last_trigger": None,
        "last_summary": None,
    }


def _summarize(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if isinstance(child, list):
                result[f"{key}_count"] = len(child)
            else:
                result[key] = _summarize(child)
        return result
    if isinstance(value, list):
        return {"count": len(value)}
    return value


def configure_job(name: str, *, enabled: bool | None = None) -> None:
    with _LOCK:
        job = _JOBS.setdefault(name, _default_job(name))
        if enabled is not None:
            job["enabled"] = enabled


def mark_task_started(name: str) -> None:
    with _LOCK:
        job = _JOBS.setdefault(name, _default_job(name))
        now = _utc_now()
        job["task_active"] = True
        job["task_started_at"] = _iso_z(now)
        job["task_stopped_at"] = None
        job["task_stop_reason"] = None


def mark_task_stopped(name: str, *, reason: str | None = None) -> None:
    with _LOCK:
        job = _JOBS.setdefault(name, _default_job(name))
        job["task_active"] = False
        job["task_stopped_at"] = _iso_z(_utc_now())
        job["task_stop_reason"] = reason


def mark_check(name: str) -> None:
    with _LOCK:
        job = _JOBS.setdefault(name, _default_job(name))
        job["checks_since_launch"] += 1
        job["last_check_at"] = _iso_z(_utc_now())


def mark_run_started(name: str, *, trigger: str | None = None) -> None:
    with _LOCK:
        job = _JOBS.setdefault(name, _default_job(name))
        now = _utc_now()
        job["runs_since_launch"] += 1
        job["last_run_started_at"] = _iso_z(now)
        job["last_trigger"] = trigger


def mark_run_finished(
    name: str,
    *,
    success: bool,
    summary: Any = None,
    error: str | None = None,
) -> None:
    with _LOCK:
        job = _JOBS.setdefault(name, _default_job(name))
        now = _utc_now()
        job["last_run_finished_at"] = _iso_z(now)
        job["last_summary"] = _summarize(summary)
        if success:
            job["successes_since_launch"] += 1
            job["last_success_at"] = _iso_z(now)
            job["last_error"] = None
            job["last_error_at"] = None
        else:
            job["failures_since_launch"] += 1
            job["last_error_at"] = _iso_z(now)
            job["last_error"] = error


def get_job_status(name: str) -> dict[str, Any]:
    with _LOCK:
        return copy.deepcopy(_JOBS.get(name, _default_job(name)))


def get_runtime_health() -> dict[str, Any]:
    now = _utc_now()
    uptime_seconds = int((now - _STARTED_AT).total_seconds())
    with _LOCK:
        jobs = copy.deepcopy(_JOBS)
    return {
        "started_at": _iso_z(_STARTED_AT),
        "now": _iso_z(now),
        "uptime_seconds": uptime_seconds,
        "jobs": jobs,
    }
