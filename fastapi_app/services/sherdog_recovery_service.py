from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from database.db_manager import DatabaseManager
from fastapi_app.services.runtime_status import configure_job, get_job_status, mark_run_finished, mark_run_started
from scrapers.sherdog_scraper import SherdogScraper

CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
STATE_PATH = ROOT_DIR / "data" / "future_fight_odds" / "sherdog_recovery.json"

RECOVERY_ENABLED_ENV = "SHERDOG_RECOVERY_ENABLED"
RECOVERY_MAX_FIGHTERS_ENV = "SHERDOG_RECOVERY_MAX_FIGHTERS_PER_RUN"
FALSEY = {"0", "false", "no", "off"}

logger = logging.getLogger(__name__)

JOB_NAME = "sherdog_recovery"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_name(name: str) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"['.`]", "", value)
    value = value.replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def recovery_enabled() -> bool:
    return os.getenv(RECOVERY_ENABLED_ENV, "1").strip().lower() not in FALSEY


def _max_fighters_per_run() -> int:
    raw = os.getenv(RECOVERY_MAX_FIGHTERS_ENV, "10").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 10


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
        state = {}
    state.setdefault("fighters", {})
    return state


def _save_state(state: dict[str, Any]) -> None:
    _write_json(STATE_PATH, state)


def get_health_status() -> dict[str, Any]:
    configure_job(JOB_NAME, enabled=recovery_enabled())
    state = _load_state()
    return {
        **get_job_status(JOB_NAME),
        "state_file": str(STATE_PATH.relative_to(ROOT_DIR)),
        "persisted_last_run_started_at": state.get("last_run_started_at"),
        "persisted_last_run_finished_at": state.get("last_run_finished_at"),
        "persisted_last_result": state.get("last_result"),
        "last_trigger": state.get("last_trigger"),
        "tracked_fighters": len(state.get("fighters", {})),
    }


def _collect_missing_fighters(odds_df, session) -> list[dict[str, Any]]:
    from fastapi_app.services.predict_service import FIGHTER_ALIASES, _resolve_fighter

    candidates: dict[str, dict[str, Any]] = {}
    for row in odds_df.to_dict(orient="records"):
        event_name = str(row.get("event_name", "")).strip()
        event_date = str(row.get("event_date", "")).strip()
        source_type = str(row.get("source_type", "")).strip()
        for fighter_field, opponent_field in (("fighter1", "fighter2"), ("fighter2", "fighter1")):
            requested_name = str(row.get(fighter_field, "")).strip()
            opponent_name = str(row.get(opponent_field, "")).strip()
            if not requested_name:
                continue
            canonical_name = FIGHTER_ALIASES.get(requested_name, requested_name)
            if _resolve_fighter(session, canonical_name):
                continue
            key = _normalize_name(canonical_name)
            entry = candidates.setdefault(
                key,
                {
                    "requested_name": requested_name,
                    "canonical_name": canonical_name,
                    "source_rows": [],
                },
            )
            source_row = {
                "event_name": event_name,
                "event_date": event_date,
                "source_type": source_type,
                "opponent": opponent_name,
            }
            if source_row not in entry["source_rows"]:
                entry["source_rows"].append(source_row)
    return sorted(candidates.values(), key=lambda item: item["canonical_name"].lower())


def _split_name_nickname(raw: str | None) -> tuple[str, str | None]:
    """
    Split a Sherdog display name like 'Richard "The Hammer" Harris' into a clean
    name ('Richard Harris') and nickname ('The Hammer'). Handles straight and
    curly quotes. Returns (name, nickname) with nickname None when absent.
    """
    if not raw:
        return "", None
    text = str(raw).strip()
    match = re.search(r'["\u201c\u201d\u2018\u2019\'](.+?)["\u201c\u201d\u2018\u2019\']', text)
    nickname = None
    if match:
        nickname = match.group(1).strip() or None
        text = (text[: match.start()] + " " + text[match.end():])
    clean = re.sub(r"\s+", " ", text).strip()
    return clean, nickname


def _map_recovered_fighter(scraped: dict[str, Any], *, requested_name: str) -> dict[str, Any]:
    breakdown = scraped.get("method_breakdown", {})
    wins = int((breakdown.get("wins") or {}).get("total") or 0)
    losses = int((breakdown.get("losses") or {}).get("total") or 0)
    clean_name, nickname = _split_name_nickname(scraped.get("name") or requested_name)
    return {
        "fighter_id": f"sherdog:{scraped['fighter_id']}",
        "name": clean_name or requested_name,
        "nickname": nickname,
        "height": scraped.get("height"),
        "weight": scraped.get("weight"),
        "date_of_birth": scraped.get("date_of_birth"),
        "age": scraped.get("age"),
        "wins": wins,
        "losses": losses,
        "draws": 0,
        "no_contests": 0,
        "url": scraped.get("url"),
        "scraped_at": scraped.get("scraped_at"),
    }


def _validate_sherdog_fighter_url(fighter_url: str) -> str:
    value = str(fighter_url).strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Sherdog URL must start with http:// or https://")
    if "sherdog.com" not in parsed.netloc.lower():
        raise ValueError("Sherdog URL must point to sherdog.com")
    if "/fighter/" not in parsed.path:
        raise ValueError("Sherdog URL must be a fighter profile link")
    return value


def recover_fighter_from_url(
    *,
    fighter_url: str,
    requested_name: str | None = None,
    dry_run: bool = False,
    bust_cache: bool = False,
    trigger: str = "manual_url",
) -> dict[str, Any]:
    configure_job(JOB_NAME, enabled=recovery_enabled())
    mark_run_started(JOB_NAME, trigger=trigger)

    fighter_url = _validate_sherdog_fighter_url(fighter_url)
    state = _load_state()
    state["last_run_started_at"] = _iso_z(_utc_now())
    state["last_trigger"] = trigger

    scraper = SherdogScraper(config_path=str(CONFIG_PATH))
    db = DatabaseManager(config_path=str(CONFIG_PATH))
    fighter_id = scraper._extract_fighter_id(fighter_url)
    state_key = _normalize_name(requested_name or fighter_id)
    fighter_state = state["fighters"].setdefault(state_key, {})
    fighter_state.update(
        {
            "requested_name": requested_name,
            "canonical_name": requested_name or fighter_id,
            "fighter_url": fighter_url,
            "last_seen_at": _iso_z(_utc_now()),
            "status": "pending",
        }
    )

    try:
        scraped = scraper.scrape_fighter(fighter_url, fighter_id=fighter_id, bust_cache=bust_cache)
        if not scraped:
            raise ValueError("Unable to scrape Sherdog fighter page")

        recovered_name, _nick = _split_name_nickname(
            scraped.get("name") or requested_name or fighter_id
        )
        recovered_name = recovered_name or requested_name or fighter_id
        mapped = _map_recovered_fighter(scraped, requested_name=recovered_name)
        fighter_state["requested_name"] = requested_name or recovered_name
        fighter_state["canonical_name"] = recovered_name
        fighter_state["scraped_name"] = recovered_name
        fighter_state["fighter_url"] = scraped.get("url") or fighter_url
        fighter_state["sherdog_fighter_id"] = scraped.get("fighter_id")

        result = {
            "status": "recovered",
            "requested_name": requested_name or recovered_name,
            "scraped_name": recovered_name,
            "nickname": mapped.get("nickname"),
            "fighter_url": scraped.get("url") or fighter_url,
            "fighter_id": mapped["fighter_id"],
            "dry_run": dry_run,
        }

        if not dry_run:
            session = db.get_session()
            try:
                fighter_obj = db.add_fighter(session, mapped)
                session.commit()
                fighter_state["db_fighter_pk"] = fighter_obj.id
                fighter_state["db_fighter_id"] = fighter_obj.fighter_id
                fighter_state["db_name"] = fighter_obj.name
                result["db_fighter_pk"] = fighter_obj.id
                result["db_fighter_id"] = fighter_obj.fighter_id
                result["db_name"] = fighter_obj.name
            finally:
                session.close()

        fighter_state["status"] = "recovered"
        fighter_state["recovered_at"] = _iso_z(_utc_now())
        state["last_run_finished_at"] = _iso_z(_utc_now())
        state["last_result"] = result
        _save_state(state)
        mark_run_finished(JOB_NAME, success=True, summary=result)
        return result
    except Exception as exc:
        fighter_state["status"] = "error"
        fighter_state["error"] = str(exc)
        fighter_state["last_attempt_at"] = _iso_z(_utc_now())
        state["last_run_finished_at"] = _iso_z(_utc_now())
        state["last_result"] = {
            "status": "error",
            "fighter_url": fighter_url,
            "requested_name": requested_name,
            "error": str(exc),
        }
        _save_state(state)
        mark_run_finished(JOB_NAME, success=False, summary=state["last_result"], error=str(exc))
        raise


def recover_missing_fighters_from_odds(
    *,
    odds_df=None,
    dry_run: bool = False,
    trigger: str = "manual",
    max_fighters: int | None = None,
) -> dict[str, Any]:
    from fastapi_app.services.predict_service import _load_all_odds

    configure_job(JOB_NAME, enabled=recovery_enabled())
    mark_run_started(JOB_NAME, trigger=trigger)
    odds_df = odds_df if odds_df is not None else _load_all_odds()
    if odds_df is None or odds_df.empty:
        result = {
            "trigger": trigger,
            "queued": 0,
            "attempted": 0,
            "recovered": 0,
            "search_misses": 0,
            "errors": 0,
            "dry_run": dry_run,
            "processed": [],
        }
        mark_run_finished(JOB_NAME, success=True, summary=result)
        logger.info("Sherdog recovery skipped: no odds rows available")
        return result

    state = _load_state()
    state["last_run_started_at"] = _iso_z(_utc_now())
    state["last_trigger"] = trigger

    db = DatabaseManager(config_path=str(CONFIG_PATH))
    scraper = SherdogScraper(config_path=str(CONFIG_PATH))

    session = db.get_session()
    try:
        missing = _collect_missing_fighters(odds_df, session)
    finally:
        session.close()

    result = {
        "trigger": trigger,
        "queued": len(missing),
        "attempted": 0,
        "recovered": 0,
        "search_misses": 0,
        "errors": 0,
        "dry_run": dry_run,
        "processed": [],
    }
    logger.info(
        "Starting Sherdog recovery trigger=%s dry_run=%s queued=%s max_fighters=%s",
        trigger,
        dry_run,
        result["queued"],
        max_fighters if max_fighters is not None else _max_fighters_per_run(),
    )

    limit = max_fighters if max_fighters is not None else _max_fighters_per_run()
    for entry in missing[:limit]:
        key = _normalize_name(entry["canonical_name"])
        fighter_state = state["fighters"].setdefault(key, {})
        fighter_state.update(
            {
                "requested_name": entry["requested_name"],
                "canonical_name": entry["canonical_name"],
                "source_rows": entry["source_rows"],
                "last_seen_at": _iso_z(_utc_now()),
                "status": "pending",
            }
        )

        result["attempted"] += 1
        processed = {
            "canonical_name": entry["canonical_name"],
            "requested_name": entry["requested_name"],
            "status": None,
        }

        try:
            search_result = scraper.search_fighter(entry["canonical_name"])
            if not search_result:
                fighter_state["status"] = "search_not_found"
                fighter_state["last_attempt_at"] = _iso_z(_utc_now())
                result["search_misses"] += 1
                processed["status"] = "search_not_found"
                result["processed"].append(processed)
                logger.warning("Sherdog recovery search miss for %s", entry["canonical_name"])
                continue

            fighter_state["search_result"] = search_result
            fighter_state["last_attempt_at"] = _iso_z(_utc_now())

            scraped = scraper.scrape_fighter(search_result["url"], fighter_id=search_result["fighter_id"])
            if not scraped:
                fighter_state["status"] = "scrape_failed"
                result["errors"] += 1
                processed["status"] = "scrape_failed"
                result["processed"].append(processed)
                continue

            mapped = _map_recovered_fighter(scraped, requested_name=entry["requested_name"])
            fighter_state["scraped_name"] = scraped.get("name")
            fighter_state["fighter_url"] = scraped.get("url")
            fighter_state["sherdog_fighter_id"] = scraped.get("fighter_id")

            if not dry_run:
                session = db.get_session()
                try:
                    fighter_obj = db.add_fighter(session, mapped)
                    session.commit()
                    fighter_state["db_fighter_pk"] = fighter_obj.id
                    fighter_state["db_fighter_id"] = fighter_obj.fighter_id
                    fighter_state["db_name"] = fighter_obj.name
                finally:
                    session.close()

            fighter_state["status"] = "recovered"
            fighter_state["recovered_at"] = _iso_z(_utc_now())
            result["recovered"] += 1
            processed["status"] = "recovered"
            processed["fighter_url"] = scraped.get("url")
            result["processed"].append(processed)
            logger.info(
                "Sherdog recovery recovered fighter requested=%s db_name=%s fighter_id=%s",
                entry["requested_name"],
                fighter_state.get("db_name") or scraped.get("name"),
                fighter_state.get("db_fighter_id") or f"sherdog:{scraped.get('fighter_id')}",
            )
        except Exception as exc:
            fighter_state["status"] = "error"
            fighter_state["error"] = str(exc)
            fighter_state["last_attempt_at"] = _iso_z(_utc_now())
            result["errors"] += 1
            processed["status"] = "error"
            processed["error"] = str(exc)
            result["processed"].append(processed)
            logger.exception("Sherdog recovery failed for %s", entry["canonical_name"])

    state["last_run_finished_at"] = _iso_z(_utc_now())
    state["last_result"] = result
    if not dry_run:
        _save_state(state)
    mark_run_finished(JOB_NAME, success=result["errors"] == 0, summary=result, error=None if result["errors"] == 0 else "errors during recovery")
    logger.info(
        "Sherdog recovery finished trigger=%s attempted=%s recovered=%s search_misses=%s errors=%s",
        trigger,
        result["attempted"],
        result["recovered"],
        result["search_misses"],
        result["errors"],
    )
    return result
