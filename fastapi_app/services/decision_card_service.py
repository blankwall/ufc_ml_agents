"""Background, persistent card-level finish/decision analysis."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.finish_prediction_service import (
    SKILL_ROOT,
    _error,
    _shape_success,
    devig_finish_probability,
)
from services.ufc_schedule_service import find_upcoming_bout, refresh_allowlist

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = ROOT_DIR / "data" / "future_fight_odds" / "decision_card_cache.json"
BATCH_SCRIPT = ROOT_DIR / "scripts" / "run_finish_card_batch.py"
SKILL_PYTHON = SKILL_ROOT / ".venv" / "bin" / "python"
MANIFEST_PATH = SKILL_ROOT / "artifacts" / "manifest.json"
CARD_TIMEOUT_SECONDS = 900
CACHE_VERSION = "decision-card-v2"

_lock = threading.RLock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decision-card")
_active: set[str] = set()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {"version": CACHE_VERSION, "cards": {}}
    try:
        payload = json.loads(CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {"version": CACHE_VERSION, "cards": {}}
    if payload.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "cards": {}}
    payload.setdefault("cards", {})
    return payload


def _write_cache(payload: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = CACHE_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    temp.replace(CACHE_PATH)


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _reconcile_orphaned_jobs(cache: dict[str, Any]) -> bool:
    """Mark persisted jobs interrupted when no live worker can own them."""
    changed = False
    for card_key, card in cache.get("cards", {}).items():
        if card.get("status") not in {"queued", "running"}:
            continue
        owner_pid = card.get("owner_pid")
        owned_here = owner_pid == os.getpid() and card_key in _active
        owned_elsewhere = owner_pid != os.getpid() and _pid_alive(owner_pid)
        if owned_here or owned_elsewhere:
            continue
        card["status"] = "error"
        card["error_code"] = "interrupted"
        card["error_message"] = "Decision card analysis was interrupted by a server restart."
        card["completed_at"] = _now()
        changed = True
    return changed


def _manifest_identity() -> dict[str, Any]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {"model": None, "database_sha256": None}
    source = manifest.get("source", {})
    return {
        "model": manifest.get("model"),
        "database_sha256": source.get("database_sha256"),
        "protected_commit": source.get("protected_commit"),
    }


def _normalize_name(value: str) -> str:
    return " ".join(str(value).lower().replace("-", " ").replace("'", "").split())


def _card_key(event_date: str, fights: list[dict[str, Any]]) -> str:
    matchups = sorted(
        "::".join(sorted((_normalize_name(f["fighter1"]), _normalize_name(f["fighter2"]))))
        for f in fights
    )
    identity = _manifest_identity()
    raw = json.dumps({
        "version": CACHE_VERSION,
        "event_date": str(event_date),
        "matchups": matchups,
        "model": identity,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _save_entry(card_key: str, entry: dict[str, Any]) -> None:
    with _lock:
        cache = _load_cache()
        cache["cards"][card_key] = entry
        _write_cache(cache)


def _resolve_fights(event_date: str, fights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    missing = False
    for fight in fights:
        metadata = find_upcoming_bout(fight["fighter1"], fight["fighter2"], event_date)
        if metadata is None or not metadata.get("weight_class"):
            missing = True
        resolved.append({"request": fight, "metadata": metadata})

    if missing:
        try:
            refresh_allowlist()
        except Exception:
            pass
        resolved = [
            {
                "request": item["request"],
                "metadata": find_upcoming_bout(
                    item["request"]["fighter1"],
                    item["request"]["fighter2"],
                    event_date,
                ),
            }
            for item in resolved
        ]
    return resolved


def _run_card(card_key: str, request: dict[str, Any]) -> None:
    with _lock:
        _active.add(card_key)
    entry = {
        "card_key": card_key,
        "event_name": request.get("event_name"),
        "event_date": request["event_date"],
        "status": "running",
        "created_at": request["created_at"],
        "started_at": _now(),
        "completed_at": None,
        "total_fights": len(request["fights"]),
        "completed_fights": 0,
        "owner_pid": os.getpid(),
        "model": _manifest_identity(),
        "fights": [],
        "error_code": None,
        "error_message": None,
    }
    _save_entry(card_key, entry)

    try:
        if not SKILL_PYTHON.exists() or not BATCH_SCRIPT.exists():
            raise FileNotFoundError(
                f"Decision skill runtime missing: {SKILL_PYTHON} / {BATCH_SCRIPT}"
            )

        resolution = _resolve_fights(request["event_date"], request["fights"])
        batch_fights: list[dict[str, Any]] = []
        batch_indexes: list[int] = []
        output_fights: list[dict[str, Any]] = []

        for index, item in enumerate(resolution):
            requested = item["request"]
            metadata = item["metadata"]
            base = {
                "fighter1": requested["fighter1"],
                "fighter2": requested["fighter2"],
                "weight_class": metadata.get("weight_class") if metadata else None,
                "fight_number": metadata.get("fight_number") if metadata else None,
                "result": None,
            }
            output_fights.append(base)
            if metadata is None:
                base["result"] = _error(
                    "card_metadata_not_found",
                    "Matchup was not found on the UFCStats upcoming card.",
                )
                continue
            if not metadata.get("weight_class"):
                base["result"] = _error(
                    "missing_weight_class",
                    "UFCStats did not provide a weight class for this matchup.",
                )
                continue

            market_probability = None
            if requested.get("finish_odds") is not None and requested.get("decision_odds") is not None:
                market_probability = devig_finish_probability(
                    int(requested["finish_odds"]),
                    int(requested["decision_odds"]),
                )
            batch_indexes.append(index)
            batch_fights.append({
                "fighter1": metadata["fighter1"],
                "fighter2": metadata["fighter2"],
                "fight_date": metadata["event_date"],
                "weight_class": metadata["weight_class"],
                "fight_number": metadata.get("fight_number") or 5,
                "market_finish_probability": market_probability,
            })

        if batch_fights:
            proc = subprocess.run(
                [str(SKILL_PYTHON), str(BATCH_SCRIPT)],
                input=json.dumps({"fights": batch_fights}),
                capture_output=True,
                text=True,
                timeout=CARD_TIMEOUT_SECONDS,
                env={**os.environ, "UFC_DECISION_SKILL_ROOT": str(SKILL_ROOT)},
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip().splitlines()
                raise RuntimeError(detail[-1] if detail else f"batch exited {proc.returncode}")

            responses: dict[int, dict[str, Any]] = {}
            for line in proc.stdout.splitlines():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if message.get("type") == "fatal":
                    raise RuntimeError(message.get("error_message") or "Decision batch failed.")
                batch_index = message.get("index")
                if not isinstance(batch_index, int) or batch_index >= len(batch_indexes):
                    continue
                output_index = batch_indexes[batch_index]
                if message.get("type") == "result":
                    fight_number = output_fights[output_index]["fight_number"] or 5
                    responses[output_index] = _shape_success(message["result"], fight_number)
                elif message.get("type") == "error":
                    responses[output_index] = _error(
                        message.get("error_code") or "model_error",
                        message.get("error_message") or "Decision model failed.",
                    )

            for output_index in batch_indexes:
                output_fights[output_index]["result"] = responses.get(
                    output_index,
                    _error("invalid_output", "Decision batch omitted this matchup."),
                )

        entry["fights"] = output_fights
        entry["completed_fights"] = len(output_fights)
        entry["status"] = "complete"
        entry["completed_at"] = _now()
    except subprocess.TimeoutExpired:
        entry["status"] = "error"
        entry["error_code"] = "timeout"
        entry["error_message"] = (
            f"Decision card analysis exceeded {CARD_TIMEOUT_SECONDS} seconds."
        )
        entry["completed_at"] = _now()
    except Exception as exc:  # noqa: BLE001 - persist explicit card-level failure
        entry["status"] = "error"
        entry["error_code"] = "card_analysis_error"
        entry["error_message"] = str(exc)
        entry["completed_at"] = _now()
    finally:
        previous = request.get("previous_complete")
        if entry["status"] == "error" and previous:
            restored = dict(previous)
            restored["refresh_error_code"] = entry["error_code"]
            restored["refresh_error_message"] = entry["error_message"]
            restored["refresh_failed_at"] = entry["completed_at"]
            entry = restored
        _save_entry(card_key, entry)
        with _lock:
            _active.discard(card_key)


def start_card_analysis(
    *,
    event_name: str | None,
    event_date: str,
    fights: list[dict[str, Any]],
    force: bool = False,
) -> dict[str, Any]:
    card_key = _card_key(event_date, fights)
    with _lock:
        cache = _load_cache()
        if _reconcile_orphaned_jobs(cache):
            _write_cache(cache)
        existing = cache["cards"].get(card_key)
        if existing and existing.get("status") == "complete" and not force:
            return existing
        if card_key in _active:
            return existing or {"card_key": card_key, "status": "running"}

        queued = {
            "card_key": card_key,
            "event_name": event_name,
            "event_date": event_date,
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "completed_at": None,
            "total_fights": len(fights),
            "completed_fights": 0,
            "owner_pid": os.getpid(),
            "model": _manifest_identity(),
            "fights": existing.get("fights", []) if existing else [],
            "error_code": None,
            "error_message": None,
        }
        cache["cards"][card_key] = queued
        _write_cache(cache)
        _active.add(card_key)

    request = {
        "event_name": event_name,
        "event_date": event_date,
        "fights": fights,
        "created_at": queued["created_at"],
        "previous_complete": existing if existing and existing.get("status") == "complete" else None,
    }
    _executor.submit(_run_card, card_key, request)
    return queued


def get_card_analysis(card_key: str) -> dict[str, Any] | None:
    with _lock:
        cache = _load_cache()
        if _reconcile_orphaned_jobs(cache):
            _write_cache(cache)
        return cache["cards"].get(card_key)


def get_card_analysis_by_date(event_date: str) -> dict[str, Any] | None:
    with _lock:
        cache = _load_cache()
        if _reconcile_orphaned_jobs(cache):
            _write_cache(cache)
        cards = [
            card for card in cache["cards"].values()
            if str(card.get("event_date")) == str(event_date)
        ]
    if not cards:
        return None
    return max(cards, key=lambda card: str(card.get("created_at") or ""))
