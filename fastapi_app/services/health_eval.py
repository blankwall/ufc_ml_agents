"""Health evaluation helpers that turn raw job status into an actionable
verdict for the /health page.

The raw background-job status only answers "did the job run without throwing".
It happily reports success even when a scheduled run syncs 0 events for weeks
(e.g. the source scraper is silently blocked). These helpers add the signal a
human actually cares about:

* ``db_data_freshness`` — how old is the newest completed event in the DB. This
  is the primary "are we falling behind" alarm and is independent of whether a
  job "succeeded".
* ``evaluate_job`` — per-job health level (ok/degraded/error) derived from
  failures and last-success staleness.
* ``evaluate_health`` — rolls everything into an overall status + a plain-English
  list of issues to surface at the top of the page.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT_DIR / "data" / "ufc_database.db"

# No new completed event in this many days => data is considered stale.
DATA_STALE_DAYS = 14

# Per-job: how long since last success before we flag it as stale.
JOB_STALE_HOURS: dict[str, int] = {
    "ufcstats_completed_sync": 48,
    "the_odds_api_sync": 48,
    "sherdog_recovery": 168,
    "marco_cache_warm": 3,
}
DEFAULT_JOB_STALE_HOURS = 72

_STATUS_ORDER = {"ok": 0, "degraded": 1, "error": 2}

# --- Run evidence / raw-data artifacts -------------------------------------
# Explicit allowlist so the /health page can surface raw run data without
# exposing arbitrary files. Paths are relative to ROOT_DIR.
MAX_VIEW_BYTES = 5_000_000

_EVIDENCE_FILES: list[dict[str, str]] = [
    {"id": "ufcstats_sync_state", "label": "UFCStats completed-sync state", "path": "data/ufcstats_sync_state.json"},
    {"id": "odds_sync_state", "label": "The Odds API sync state", "path": "data/future_fight_odds/the_odds_api_sync.json"},
    {"id": "odds_events_store", "label": "The Odds API events store", "path": "data/future_fight_odds/the_odds_api_events.json"},
    {"id": "sherdog_recovery_state", "label": "Sherdog recovery state", "path": "data/future_fight_odds/sherdog_recovery.json"},
    {"id": "predictions_cache", "label": "Predictions cache", "path": "data/future_fight_odds/predictions_cache.json"},
]

_EVIDENCE_DIRS: list[dict[str, str]] = [
    {"id": "db_backups", "label": "Database backups (pre-sync snapshots)", "path": "data/backups", "glob": "*.db"},
]
_MAX_DIR_ENTRIES = 8


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _worse(a: str, b: str) -> str:
    return a if _STATUS_ORDER.get(a, 0) >= _STATUS_ORDER.get(b, 0) else b


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_event_date(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def humanize_age(seconds: int | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def db_data_freshness() -> dict[str, Any]:
    """Return freshness of the newest completed event in the DB."""
    result: dict[str, Any] = {
        "latest_event_date": None,
        "latest_event_name": None,
        "age_days": None,
        "age_human": "—",
        "stale": False,
        "threshold_days": DATA_STALE_DAYS,
        "error": None,
    }
    if not DB_PATH.exists():
        result["error"] = "database file not found"
        return result
    try:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT name, date FROM events WHERE date IS NOT NULL"
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error as exc:
        result["error"] = str(exc)
        return result

    now = _now()
    best: datetime | None = None
    best_name: str | None = None
    for name, raw_date in rows:
        parsed = _parse_event_date(raw_date)
        # Ignore future/scheduled events; we want the newest COMPLETED one.
        if parsed and parsed <= now and (best is None or parsed > best):
            best = parsed
            best_name = name
    if best is not None:
        age_seconds = int((now - best).total_seconds())
        age_days = age_seconds // 86400
        result.update(
            latest_event_date=best.date().isoformat(),
            latest_event_name=best_name,
            age_days=age_days,
            age_human=humanize_age(age_seconds),
            stale=age_days > DATA_STALE_DAYS,
        )
    return result


def evaluate_job(name: str, job: dict[str, Any]) -> dict[str, Any]:
    """Derive a health level + issues for a single background job."""
    issues: list[str] = []
    level = "ok"
    now = _now()

    failures = job.get("failures_since_launch") or 0
    last_error = job.get("last_error")
    enabled = job.get("enabled")

    last_success = _parse_iso(job.get("last_success_at")) or _parse_iso(
        job.get("persisted_last_success_at")
    )
    age_seconds = int((now - last_success).total_seconds()) if last_success else None
    stale_hours = JOB_STALE_HOURS.get(name, DEFAULT_JOB_STALE_HOURS)
    stale = age_seconds is not None and age_seconds > stale_hours * 3600

    if failures > 0:
        level = _worse(level, "error")
        issues.append(f"{failures} failed run(s) since launch")
    if last_error:
        level = _worse(level, "error")
    if name == "marco_cache_warm":
        warm_errors = (job.get("last_summary") or {}).get("errors", 0)
        if warm_errors:
            level = _worse(level, "degraded")
            issues.append(f"{warm_errors} future fight(s) failed Marco warming")

    # Staleness only alarms for jobs that are supposed to be running.
    if enabled:
        runs = job.get("runs_since_launch") or 0
        successes = job.get("successes_since_launch") or 0
        if last_success is not None and stale:
            level = _worse(level, "degraded")
            issues.append(
                f"last success {humanize_age(age_seconds)} ago (> {stale_hours}h)"
            )
        elif runs > 0 and successes == 0:
            # Has attempted runs this launch but none succeeded (not just idle).
            level = _worse(level, "degraded")
            issues.append(f"{runs} run(s) since launch, none succeeded")

    # Silent no-op: a run that "succeeded" but synced nothing.
    last_result = job.get("persisted_last_result") or {}
    synced = last_result.get("synced_events")
    seen = last_result.get("recent_events_seen")
    if enabled and synced == 0 and (seen or 0) == 0:
        issues.append("last run saw 0 events (source may be blocked)")
        level = _worse(level, "degraded")

    return {
        "level": level,
        "stale": stale,
        "enabled": bool(enabled),
        "last_success_age_seconds": age_seconds,
        "last_success_age_human": humanize_age(age_seconds),
        "issues": issues,
    }


def evaluate_health(background_jobs: dict[str, Any]) -> dict[str, Any]:
    """Roll job status + data freshness into an overall verdict."""
    status = "ok"
    issues: list[str] = []
    job_eval: dict[str, Any] = {}

    for name, job in background_jobs.items():
        result = evaluate_job(name, job)
        job_eval[name] = result
        status = _worse(status, result["level"])
        for message in result["issues"]:
            issues.append(f"{name.replace('_', ' ')}: {message}")

    freshness = db_data_freshness()
    if freshness.get("error"):
        status = _worse(status, "degraded")
        issues.append(f"data freshness check failed: {freshness['error']}")
    elif freshness.get("stale"):
        status = _worse(status, "degraded")
        issues.append(
            f"no new completed event in {freshness['age_days']} days "
            f"(latest: {freshness['latest_event_name']} on {freshness['latest_event_date']})"
        )

    return {
        "status": status,
        "issues": issues,
        "data_freshness": freshness,
        "job_eval": job_eval,
    }


# --- Evidence artifacts -----------------------------------------------------
def _human_size(num: int | None) -> str:
    if num is None:
        return "—"
    size = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _file_meta(path: Path, *, artifact_id: str, label: str, kind: str) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    modified = (
        datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc) if stat else None
    )
    age_seconds = int((_now() - modified).total_seconds()) if modified else None
    return {
        "id": artifact_id,
        "label": label,
        "path": str(path.relative_to(ROOT_DIR)),
        "kind": kind,
        "exists": exists,
        "size_bytes": stat.st_size if stat else None,
        "size_human": _human_size(stat.st_size) if stat else "—",
        "modified_at": modified.replace(microsecond=0).isoformat().replace("+00:00", "Z") if modified else None,
        "age_human": humanize_age(age_seconds),
        "viewable": kind == "json" and exists and (stat.st_size <= MAX_VIEW_BYTES if stat else False),
    }


def list_evidence() -> list[dict[str, Any]]:
    """Return metadata for all allowlisted run-evidence artifacts."""
    artifacts: list[dict[str, Any]] = []
    for entry in _EVIDENCE_FILES:
        path = ROOT_DIR / entry["path"]
        artifacts.append(_file_meta(path, artifact_id=entry["id"], label=entry["label"], kind="json"))
    for entry in _EVIDENCE_DIRS:
        directory = ROOT_DIR / entry["path"]
        matches = sorted(directory.glob(entry.get("glob", "*")), key=lambda p: p.stat().st_mtime, reverse=True) if directory.exists() else []
        recent = []
        for match in matches[:_MAX_DIR_ENTRIES]:
            recent.append(_file_meta(match, artifact_id=f"{entry['id']}:{match.name}", label=match.name, kind="file"))
        artifacts.append({
            "id": entry["id"],
            "label": entry["label"],
            "path": entry["path"],
            "kind": "dir",
            "exists": directory.exists(),
            "count": len(matches),
            "recent": recent,
        })
    return artifacts


def resolve_evidence_file(artifact_id: str) -> Path | None:
    """Return the absolute path of a viewable allowlisted JSON artifact, else None."""
    for entry in _EVIDENCE_FILES:
        if entry["id"] == artifact_id:
            path = (ROOT_DIR / entry["path"]).resolve()
            # Defense in depth: never escape the project root.
            if ROOT_DIR.resolve() in path.parents and path.exists() and path.is_file():
                return path
            return None
    return None
