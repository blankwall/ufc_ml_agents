"""Read-only Marco execution, caching, and bet-resurrection policy."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import threading
import unicodedata
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .fighter_identity import resolve_fighter
from .ufc_schedule_service import find_upcoming_bout

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = ROOT_DIR / "data" / "future_fight_odds" / "marco_cache.json"
LOCK_PATH = CACHE_PATH.with_suffix(".lock")
KEY_LOCK_DIR = CACHE_PATH.parent / ".marco_locks"
DB_PATH = ROOT_DIR / "data" / "ufc_database.db"
CACHE_VERSION = "marco-v1"
MODEL_TAG = "pre2025"
SIMULATIONS = 8000
DEFAULT_LBS = 170
DEFAULT_ROUNDS = 3
SUBPROCESS_TIMEOUT_SECONDS = 20

WEIGHT_LBS = {
    "Strawweight": 115,
    "Women's Strawweight": 115,
    "Flyweight": 125,
    "Women's Flyweight": 125,
    "Bantamweight": 135,
    "Women's Bantamweight": 135,
    "Featherweight": 145,
    "Women's Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 245,
    "Super Heavyweight": 265,
    "Catch Weight": 165,
    "Open Weight": 240,
}

_engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)
_Session = sessionmaker(bind=_engine)
_thread_lock = threading.RLock()


def _marco_root() -> Path:
    return Path(os.getenv("MARCO_ROOT", str(Path.home() / "marco_playground"))).expanduser()


def _marco_python() -> str:
    return os.getenv("MARCO_PYTHON", "python3")


def _predict_script() -> Path:
    return _marco_root() / "marco" / "predict.py"


def _normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _iso_date(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


def _git_identity(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    try:
        value = head.read_text().strip()
        if value.startswith("ref: "):
            ref = root / ".git" / value[5:]
            return ref.read_text().strip()
        return value
    except OSError:
        return None


def _runtime_identity() -> dict[str, Any]:
    try:
        stat = DB_PATH.stat()
        db_identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        db_identity = {"size": None, "mtime_ns": None}
    return {
        "cache_version": CACHE_VERSION,
        "model": MODEL_TAG,
        "simulations": SIMULATIONS,
        "marco_commit": _git_identity(_marco_root()),
        "database": db_identity,
    }


@contextmanager
def _cache_file_lock() -> Iterator[None]:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _key_file_lock(key: str) -> Iterator[None]:
    KEY_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = KEY_LOCK_DIR / f"{key}.lock"
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _empty_cache() -> dict[str, Any]:
    return {"version": CACHE_VERSION, "entries": {}}


def _load_cache() -> dict[str, Any]:
    try:
        payload = json.loads(CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return _empty_cache()
    if payload.get("version") != CACHE_VERSION:
        return _empty_cache()
    payload.setdefault("entries", {})
    return payload


def _write_cache(payload: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = CACHE_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    temp.replace(CACHE_PATH)


def _cache_key(
    fighter_a: str,
    fighter_b: str,
    fight_date: str,
    lbs: int,
    rounds: int,
) -> str:
    matchup = [_normalize_name(fighter_a), _normalize_name(fighter_b)]
    raw = json.dumps(
        {
            "matchup": matchup,
            "fight_date": fight_date,
            "lbs": lbs,
            "rounds": rounds,
            "runtime": _runtime_identity(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _error(code: str, message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error_code": code,
        "error_message": message,
        "cache_hit": False,
        "pick": None,
        "pick_probability": None,
        "probabilities": None,
        "history": None,
        "metadata": None,
    }


def resolve_canonical_names(fighter1: str, fighter2: str) -> tuple[str, str] | None:
    session = _Session()
    try:
        f1 = resolve_fighter(session, fighter1)
        f2 = resolve_fighter(session, fighter2)
        if f1 is None or f2 is None:
            return None
        return f1.name, f2.name
    finally:
        session.close()


def resolve_marco_metadata(
    fighter1: str,
    fighter2: str,
    fight_date: date | datetime | str,
) -> dict[str, Any]:
    date_str = _iso_date(fight_date)
    upcoming = find_upcoming_bout(fighter1, fighter2, date_str)
    weight_class = upcoming.get("weight_class") if upcoming else None
    fight_number = upcoming.get("fight_number") if upcoming else None
    is_title_fight = bool(upcoming.get("is_title_fight")) if upcoming else False
    lbs = WEIGHT_LBS.get(str(weight_class).strip(), DEFAULT_LBS) if weight_class else DEFAULT_LBS
    rounds = 5 if is_title_fight or fight_number == 1 else DEFAULT_ROUNDS
    return {
        "fight_date": upcoming.get("event_date", date_str) if upcoming else date_str,
        "weight_class": weight_class,
        "lbs": lbs,
        "rounds": rounds,
        "fight_number": fight_number,
        "is_title_fight": is_title_fight,
        "metadata_source": "ufcstats" if upcoming else "defaults",
    }


def run_marco_prediction(
    fighter1: str,
    fighter2: str,
    *,
    fight_date: date | datetime | str | None,
    force: bool = False,
    timeout: int = SUBPROCESS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if fight_date is None:
        return _error("missing_date", "fight_date is required for Marco.")

    canonical = resolve_canonical_names(fighter1, fighter2)
    if canonical is None:
        return _error("fighter_not_found", "A fighter could not be resolved in the model database.")
    fighter_a, fighter_b = canonical
    metadata = resolve_marco_metadata(fighter_a, fighter_b, fight_date)
    date_str = _iso_date(metadata["fight_date"])
    key = _cache_key(fighter_a, fighter_b, date_str, metadata["lbs"], metadata["rounds"])

    if not force:
        with _thread_lock, _cache_file_lock():
            cached = _load_cache()["entries"].get(key)
        if cached is not None:
            return {**cached, "cache_hit": True}

    with _key_file_lock(key):
        if not force:
            with _thread_lock, _cache_file_lock():
                cached = _load_cache()["entries"].get(key)
            if cached is not None:
                return {**cached, "cache_hit": True}
        return _run_uncached(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            date_str=date_str,
            metadata=metadata,
            key=key,
            timeout=timeout,
        )


def _run_uncached(
    *,
    fighter_a: str,
    fighter_b: str,
    date_str: str,
    metadata: dict[str, Any],
    key: str,
    timeout: int,
) -> dict[str, Any]:
    script = _predict_script()
    if not script.exists():
        return _error("marco_not_installed", f"Marco predictor not found at {script}.")

    cmd = [
        _marco_python(),
        str(script),
        fighter_a,
        fighter_b,
        "--date",
        date_str,
        "--lbs",
        str(metadata["lbs"]),
        "--rounds",
        str(metadata["rounds"]),
        "--n",
        str(SIMULATIONS),
        "--model",
        MODEL_TAG,
        "--json",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_marco_root()),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, OSError) as exc:
        return _error("marco_not_installed", str(exc))
    except subprocess.TimeoutExpired:
        return _error("timeout", f"Marco did not respond within {timeout}s.")

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        message = detail[-1] if detail else f"Marco exited with code {proc.returncode}."
        code = "fighter_not_found" if "fighter not found" in message.lower() else "subprocess_error"
        return _error(code, message)

    try:
        raw = json.loads(proc.stdout)
        p_a = float(raw["p_A"])
        p_b = float(raw["p_B"])
        a_prior = int(raw["a_fights_prior"])
        b_prior = int(raw["b_fights_prior"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return _error("invalid_output", f"Marco returned an unexpected response: {exc}")

    pick = fighter_a if p_a >= p_b else fighter_b
    result = {
        "status": "complete",
        "error_code": None,
        "error_message": None,
        "cache_hit": False,
        "cache_key": key,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fighter1": fighter_a,
        "fighter2": fighter_b,
        "pick": pick,
        "pick_probability": max(p_a, p_b),
        "probabilities": {"fighter1": p_a, "fighter2": p_b},
        "history": {"fighter1_prior": a_prior, "fighter2_prior": b_prior},
        "metadata": metadata,
        "runtime": _runtime_identity(),
    }
    with _thread_lock, _cache_file_lock():
        cache = _load_cache()
        cache["entries"][key] = result
        if len(cache["entries"]) > 500:
            ordered = sorted(
                cache["entries"].items(),
                key=lambda item: str(item[1].get("generated_at") or ""),
                reverse=True,
            )
            cache["entries"] = dict(ordered[:500])
        _write_cache(cache)
    return result


def evaluate_resurrection(
    *,
    marco: dict[str, Any],
    model_pick: str,
    original_bet: bool,
    skip_code: str | None,
    skip_reason: str | None,
    pick_model_prob: float,
    pick_market_prob: float,
    pick_odds: int | None,
) -> dict[str, Any]:
    edge_pct = round((pick_model_prob - pick_market_prob) * 100, 10)
    base = {
        "resurrected": False,
        "final_bet": bool(original_bet),
        "stake_multiplier": None,
        "original_skip_code": skip_code,
        "original_skip_reason": skip_reason,
        "agreement": None,
        "history_eligible": None,
        "reason_code": None,
        "reason": None,
    }
    if marco.get("status") != "complete":
        return {
            **base,
            "reason_code": "marco_unavailable",
            "reason": marco.get("error_message") or "Marco is unavailable.",
        }

    agreement = _normalize_name(marco["pick"]) == _normalize_name(model_pick)
    history = marco.get("history") or {}
    history_eligible = min(
        int(history.get("fighter1_prior", 0)),
        int(history.get("fighter2_prior", 0)),
    ) >= 2
    base.update(agreement=agreement, history_eligible=history_eligible)

    if original_bet:
        return {
            **base,
            "reason_code": "existing_bet",
            "reason": "The original betting config already approved this bet.",
        }
    if skip_code == "D1" or not history_eligible:
        return {
            **base,
            "reason_code": "insufficient_history",
            "reason": "Marco cannot resurrect min-fights or low-history skips.",
        }
    if pick_odds is None:
        return {
            **base,
            "reason_code": "missing_odds",
            "reason": "Market odds are required for resurrection.",
        }
    if not agreement:
        return {
            **base,
            "reason_code": "marco_disagrees",
            "reason": "Marco does not agree with the model-selected side.",
        }

    if pick_odds < 0:
        eligible = 0 <= edge_pct < 10 and pick_model_prob >= 0.55
        reason = (
            "Marco resurrected a skipped favorite with non-negative edge under 10% "
            "and model confidence of at least 55%."
        )
        failure = "Favorite resurrection requires edge from 0% to under 10% and confidence of at least 55%."
    elif pick_odds > 0:
        eligible = edge_pct >= 5 and pick_odds <= 400
        reason = "Marco resurrected a skipped underdog with at least 5% edge and odds no higher than +400."
        failure = "Underdog resurrection requires at least 5% edge and odds no higher than +400."
    else:
        eligible = False
        reason = ""
        failure = "Invalid market odds."

    if not eligible:
        return {
            **base,
            "reason_code": "threshold_not_met",
            "reason": failure,
        }
    return {
        **base,
        "resurrected": True,
        "final_bet": True,
        "stake_multiplier": 1.0,
        "reason_code": "marco_reclaim",
        "reason": reason,
    }
