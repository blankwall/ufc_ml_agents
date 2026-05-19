from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi_app.services.runtime_status import (
    configure_job,
    get_job_status,
    mark_check,
    mark_run_finished,
    mark_run_started,
)
from services.sherdog_recovery_service import recover_missing_fighters_from_odds, recovery_enabled

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ODDS_DIR = ROOT_DIR / "data" / "future_fight_odds"
USER_EVENTS_DIR = ROOT_DIR / "data" / "user_events"
RAW_DIR = ROOT_DIR / "data" / "raw" / "the_odds_api"
STORE_PATH = ODDS_DIR / "the_odds_api_events.json"
OUTPUT_CSV = ODDS_DIR / "the_odds_api_new_events.csv"
STATE_PATH = ODDS_DIR / "the_odds_api_sync.json"

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
API_KEY_ENV = "THE_ODDS_API_KEY"
AUTO_SYNC_ENV = "THE_ODDS_API_AUTO_SYNC"
SYNC_INTERVAL_HOURS_ENV = "THE_ODDS_API_SYNC_INTERVAL_HOURS"
SYNC_CHECK_SECONDS_ENV = "THE_ODDS_API_SYNC_CHECK_SECONDS"
WINDOW_DAYS_ENV = "THE_ODDS_API_WINDOW_DAYS"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_DATE_FORMAT = "iso"
DEFAULT_WINDOW_DAYS = 31
FALSEY = {"0", "false", "no", "off"}
PREFERRED_BOOKMAKERS = (
    "fanduel",
    "draftkings",
    "betmgm",
    "caesars",
    "espnbet",
    "espn_bet",
    "bovada",
)
CSV_FIELDS = [
    "event_name",
    "event_date",
    "event_url",
    "fighter1",
    "fighter2",
    "fighter1_odds",
    "fighter2_odds",
    "fighter1_prob",
    "fighter2_prob",
    "source_event_id",
    "bookmaker",
    "last_update",
    "commence_time",
]

logger = logging.getLogger(__name__)

JOB_NAME = "the_odds_api_sync"


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


def _window_days() -> int:
    raw = os.getenv(WINDOW_DAYS_ENV, str(DEFAULT_WINDOW_DAYS)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_WINDOW_DAYS


def _window_bounds(now: datetime | None = None) -> tuple[datetime, datetime]:
    now = now or _utc_now()
    start = now - timedelta(hours=6)
    end = now + timedelta(days=_window_days())
    return start, end


def _within_window(commence_dt: datetime, now: datetime | None = None) -> bool:
    start, end = _window_bounds(now)
    return start <= commence_dt <= end


def _normalize_name(name: str) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"['.`]", "", value)
    value = value.replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def fight_key(fighter1: str, fighter2: str) -> str:
    return "_vs_".join(sorted([_normalize_name(fighter1), _normalize_name(fighter2)]))


def american_to_prob(odds: int | float) -> float:
    price = float(odds)
    if price > 0:
        return round(100 / (price + 100), 4)
    return round(abs(price) / (abs(price) + 100), 4)


def _event_date_key(commence_dt: datetime) -> str:
    return commence_dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _event_key(date_key: str) -> str:
    return f"the_odds_api|{date_key}"


def _event_name(date_key: str) -> str:
    return f"MMA Card · {date_key}"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    temp_path.replace(path)


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


def _load_sync_state() -> dict[str, Any]:
    return _load_json(STATE_PATH, {})


def _save_sync_state(state: dict[str, Any]) -> None:
    _write_json(STATE_PATH, state)


def get_health_status() -> dict[str, Any]:
    configure_job(JOB_NAME, enabled=scheduler_enabled())
    state = _load_sync_state()
    return {
        **get_job_status(JOB_NAME),
        "state_file": str(STATE_PATH.relative_to(ROOT_DIR)),
        "persisted_last_run_started_at": state.get("last_run_started_at"),
        "persisted_last_run_finished_at": state.get("last_run_finished_at"),
        "persisted_last_success_at": state.get("last_success_at"),
        "persisted_last_result": {
            "payload_events": state.get("payload_events"),
            "new_rows_added": state.get("new_rows_added"),
            "updated_rows": state.get("updated_rows"),
            "deactivated_rows": state.get("deactivated_rows"),
            "skipped_existing": state.get("skipped_existing"),
            "skipped_invalid": state.get("skipped_invalid"),
            "sherdog_recovery": state.get("sherdog_recovery"),
        },
    }


def _bookmaker_rank(bookmaker: dict[str, Any]) -> tuple[int, str]:
    key = str(bookmaker.get("key", "")).strip().lower()
    title = str(bookmaker.get("title", "")).strip().lower()
    for idx, preferred in enumerate(PREFERRED_BOOKMAKERS):
        if key == preferred or title == preferred:
            return (idx, title or key)
    return (len(PREFERRED_BOOKMAKERS), title or key)


def _extract_complete_market(bookmaker: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    for market in bookmaker.get("markets", []):
        if market.get("key") != "h2h":
            continue
        outcomes = []
        for outcome in market.get("outcomes", []):
            name = str(outcome.get("name", "")).strip()
            price = outcome.get("price")
            if not name or price is None:
                continue
            try:
                price_int = int(price)
            except (TypeError, ValueError):
                continue
            outcomes.append({"name": name, "price": price_int})
        if len(outcomes) >= 2:
            return market, outcomes[:2]
    return None


def _select_bookmaker(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]] | None:
    candidates: list[tuple[tuple[int, str], dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    for bookmaker in event.get("bookmakers", []):
        extracted = _extract_complete_market(bookmaker)
        if extracted is None:
            continue
        market, outcomes = extracted
        candidates.append((_bookmaker_rank(bookmaker), bookmaker, market, outcomes))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, bookmaker, market, outcomes = candidates[0]
    return bookmaker, market, outcomes


def _normalize_api_event(event: dict[str, Any]) -> dict[str, Any] | None:
    commence_dt = _parse_iso(event.get("commence_time"))
    if commence_dt is None:
        return None

    selection = _select_bookmaker(event)
    if selection is None:
        return None

    bookmaker, market, outcomes = selection
    outcome_map = {item["name"]: item["price"] for item in outcomes}
    home = str(event.get("home_team", "")).strip()
    away = str(event.get("away_team", "")).strip()
    if home in outcome_map and away in outcome_map:
        fighter1, fighter2 = home, away
    else:
        fighter1, fighter2 = outcomes[0]["name"], outcomes[1]["name"]

    odds1 = outcome_map.get(fighter1)
    odds2 = outcome_map.get(fighter2)
    if odds1 is None or odds2 is None:
        return None

    date_key = _event_date_key(commence_dt)
    return {
        "event_key": _event_key(date_key),
        "event_name": _event_name(date_key),
        "event_date": date_key,
        "event_url": "",
        "fight_key": fight_key(fighter1, fighter2),
        "fighter1": fighter1,
        "fighter2": fighter2,
        "fighter1_odds": odds1,
        "fighter2_odds": odds2,
        "fighter1_prob": american_to_prob(odds1),
        "fighter2_prob": american_to_prob(odds2),
        "source_event_id": str(event.get("id", "")).strip(),
        "bookmaker": str(bookmaker.get("title", "")).strip(),
        "last_update": str(market.get("last_update") or bookmaker.get("last_update") or "").strip(),
        "commence_time": _iso_z(commence_dt),
    }


def _int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _history_entry(row: dict[str, Any], captured_at: str) -> dict[str, Any]:
    return {
        "captured_at": captured_at,
        "fighter1_odds": _int_or_none(row.get("fighter1_odds")),
        "fighter2_odds": _int_or_none(row.get("fighter2_odds")),
        "fighter1_prob": _float_or_none(row.get("fighter1_prob")),
        "fighter2_prob": _float_or_none(row.get("fighter2_prob")),
        "bookmaker": str(row.get("bookmaker", "")).strip(),
        "last_update": str(row.get("last_update", "")).strip(),
        "commence_time": str(row.get("commence_time", "")).strip(),
    }


def _same_snapshot(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("fighter1_odds") == right.get("fighter1_odds")
        and left.get("fighter2_odds") == right.get("fighter2_odds")
        and left.get("bookmaker", "") == right.get("bookmaker", "")
        and left.get("last_update", "") == right.get("last_update", "")
        and left.get("commence_time", "") == right.get("commence_time", "")
    )


def _empty_store() -> dict[str, Any]:
    return {"events": []}


def _bootstrap_store_from_existing_csv() -> dict[str, Any]:
    store = _empty_store()
    rows = _read_csv_rows(OUTPUT_CSV)
    if not rows:
        return store

    state = _load_sync_state()
    captured_at = state.get("last_success_at") or _iso_z(_utc_now())
    events_by_key: dict[str, dict[str, Any]] = {}

    for row in rows:
        event_dt = _parse_iso(row.get("commence_time")) or _parse_iso(row.get("event_date"))
        if event_dt is not None:
            date_key = _event_date_key(event_dt)
        else:
            date_key = str(row.get("event_date", "")).strip() or "unknown"

        event_key = _event_key(date_key)
        event = events_by_key.setdefault(
            event_key,
            {
                "event_key": event_key,
                "event_name": _event_name(date_key),
                "event_date": date_key,
                "event_url": "",
                "source_type": "the_odds_api",
                "first_seen_at": captured_at,
                "last_synced_at": captured_at,
                "fights": [],
            },
        )

        current = {
            "event_key": event_key,
            "event_name": event["event_name"],
            "event_date": event["event_date"],
            "event_url": "",
            "fight_key": fight_key(row.get("fighter1", ""), row.get("fighter2", "")),
            "fighter1": row.get("fighter1", ""),
            "fighter2": row.get("fighter2", ""),
            "fighter1_odds": _int_or_none(row.get("fighter1_odds")),
            "fighter2_odds": _int_or_none(row.get("fighter2_odds")),
            "fighter1_prob": _float_or_none(row.get("fighter1_prob")),
            "fighter2_prob": _float_or_none(row.get("fighter2_prob")),
            "source_event_id": str(row.get("source_event_id", "")).strip(),
            "bookmaker": str(row.get("bookmaker", "")).strip(),
            "last_update": str(row.get("last_update", "")).strip(),
            "commence_time": str(row.get("commence_time", "")).strip(),
        }
        history = _history_entry(current, captured_at)
        event["fights"].append(
            {
                **current,
                "active": True,
                "first_seen_at": captured_at,
                "last_seen_at": captured_at,
                "removed_at": None,
                "source_event_ids": [current["source_event_id"]] if current["source_event_id"] else [],
                "odds_history": [history],
            }
        )

    store["events"] = sorted(events_by_key.values(), key=lambda event: event["event_date"])
    return store


def _load_store() -> dict[str, Any]:
    store = _load_json(STORE_PATH, None)
    if store is None:
        return _bootstrap_store_from_existing_csv()
    if not isinstance(store, dict) or "events" not in store:
        return _bootstrap_store_from_existing_csv()
    return store


def _save_store(store: dict[str, Any]) -> None:
    ordered_events = []
    for event in sorted(store.get("events", []), key=lambda item: item.get("event_date", "")):
        fights = sorted(
            event.get("fights", []),
            key=lambda fight: (fight.get("commence_time", ""), fight.get("fighter1", ""), fight.get("fighter2", "")),
        )
        ordered_events.append({**event, "fights": fights})
    _write_json(STORE_PATH, {"events": ordered_events})


def _collect_legacy_fight_keys() -> set[str]:
    keys: set[str] = set()

    if ODDS_DIR.exists():
        for csv_path in sorted(ODDS_DIR.glob("ufc*.csv")):
            try:
                for row in _read_csv_rows(csv_path):
                    keys.add(fight_key(row.get("fighter1", ""), row.get("fighter2", "")))
            except Exception:
                continue

    if USER_EVENTS_DIR.exists():
        for json_path in sorted(USER_EVENTS_DIR.glob("*.json")):
            try:
                payload = json.loads(json_path.read_text())
            except json.JSONDecodeError:
                continue
            for fight in payload.get("fights", []):
                keys.add(fight_key(fight.get("fighter1", ""), fight.get("fighter2", "")))

    return {key for key in keys if key != "_vs_"}


def _event_index(store: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {event["event_key"]: event for event in store.get("events", [])}


def _fight_index(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {fight["fight_key"]: fight for fight in event.get("fights", [])}


def _export_store_rows(store: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in sorted(store.get("events", []), key=lambda item: item.get("event_date", "")):
        for fight in sorted(
            event.get("fights", []),
            key=lambda item: (item.get("commence_time", ""), item.get("fighter1", ""), item.get("fighter2", "")),
        ):
            if not fight.get("active", True):
                continue
            rows.append(
                {
                    "event_name": event.get("event_name", ""),
                    "event_date": event.get("event_date", ""),
                    "event_url": "",
                    "fighter1": fight.get("fighter1", ""),
                    "fighter2": fight.get("fighter2", ""),
                    "fighter1_odds": fight.get("fighter1_odds", ""),
                    "fighter2_odds": fight.get("fighter2_odds", ""),
                    "fighter1_prob": fight.get("fighter1_prob", ""),
                    "fighter2_prob": fight.get("fighter2_prob", ""),
                    "source_event_id": ",".join(fight.get("source_event_ids", [])),
                    "bookmaker": fight.get("bookmaker", ""),
                    "last_update": fight.get("last_update", ""),
                    "commence_time": fight.get("commence_time", ""),
                }
            )
    return rows


def _find_fight(store: dict[str, Any], *, event_date: str, fighter1: str, fighter2: str) -> tuple[dict[str, Any], dict[str, Any]] | tuple[None, None]:
    target_event_key = _event_key(str(event_date).strip())
    target_fight_key = fight_key(fighter1, fighter2)
    for event in store.get("events", []):
        if event.get("event_key") != target_event_key:
            continue
        for fight in event.get("fights", []):
            if fight.get("fight_key") == target_fight_key:
                return event, fight
    return None, None


def _estimated_snapshot(entry: dict[str, Any], *, delta: int, captured_at: str, label: str) -> dict[str, Any]:
    def _shift(odds: Any) -> int | None:
        value = _int_or_none(odds)
        if value is None:
            return None
        return value + delta

    fighter1_odds = _shift(entry.get("fighter1_odds"))
    fighter2_odds = _shift(entry.get("fighter2_odds"))
    return {
        "label": label,
        "captured_at": captured_at,
        "fighter1_odds": fighter1_odds,
        "fighter2_odds": fighter2_odds,
        "fighter1_prob": american_to_prob(fighter1_odds) if fighter1_odds is not None else None,
        "fighter2_prob": american_to_prob(fighter2_odds) if fighter2_odds is not None else None,
        "bookmaker": "Estimated",
        "last_update": captured_at,
        "estimated": True,
    }


def _build_sampled_history(history: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    if len(history) == 1:
        latest = history[0]
        anchor_dt = _parse_iso(latest.get("captured_at")) or _parse_iso(latest.get("last_update")) or _utc_now()
        first = _estimated_snapshot(
            latest,
            delta=20,
            captured_at=_iso_z(anchor_dt - timedelta(days=7)),
            label="Estimated first",
        )
        middle = _estimated_snapshot(
            latest,
            delta=10,
            captured_at=_iso_z(anchor_dt - timedelta(days=3)),
            label="Estimated middle",
        )
        latest_sample = {
            "label": "Most recent",
            "captured_at": latest.get("captured_at"),
            "fighter1_odds": latest.get("fighter1_odds"),
            "fighter2_odds": latest.get("fighter2_odds"),
            "fighter1_prob": latest.get("fighter1_prob"),
            "fighter2_prob": latest.get("fighter2_prob"),
            "bookmaker": latest.get("bookmaker"),
            "last_update": latest.get("last_update"),
            "estimated": False,
        }
        return [first, middle, latest_sample], True

    sample_indexes = [0]
    if len(history) > 2:
        sample_indexes.append(len(history) // 2)
    if len(history) > 1:
        sample_indexes.append(len(history) - 1)
    sample_indexes = sorted(set(sample_indexes))

    labels = {
        0: "First",
        len(history) // 2: "Middle",
        len(history) - 1: "Most recent",
    }
    samples = []
    for idx in sample_indexes:
        entry = history[idx]
        samples.append(
            {
                "label": labels.get(idx, f"Sample {idx + 1}"),
                "captured_at": entry.get("captured_at"),
                "fighter1_odds": entry.get("fighter1_odds"),
                "fighter2_odds": entry.get("fighter2_odds"),
                "fighter1_prob": entry.get("fighter1_prob"),
                "fighter2_prob": entry.get("fighter2_prob"),
                "bookmaker": entry.get("bookmaker"),
                "last_update": entry.get("last_update"),
                "estimated": False,
            }
        )
    return samples, False


def get_bet_placed_map() -> dict[tuple[str, str], dict[str, Any]]:
    store = _load_store()
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for event in store.get("events", []):
        event_date = str(event.get("event_date", "")).strip()
        for fight in event.get("fights", []):
            bet_placed = fight.get("bet_placed")
            fight_id = str(fight.get("fight_key", "")).strip()
            if event_date and fight_id and bet_placed:
                out[(event_date, fight_id)] = bet_placed
    return out


def get_sampled_odds_history(event_date: str, fighter1: str, fighter2: str) -> dict[str, Any] | None:
    store = _load_store()
    event, fight = _find_fight(store, event_date=event_date, fighter1=fighter1, fighter2=fighter2)
    if event is None or fight is None:
        return None

    history = fight.get("odds_history", [])
    if not history:
        return None

    samples, uses_estimated_samples = _build_sampled_history(history)
    return {
        "event_name": event.get("event_name"),
        "event_date": event.get("event_date"),
        "fighter1": fight.get("fighter1"),
        "fighter2": fight.get("fighter2"),
        "history_count": len(samples),
        "real_history_count": len(history),
        "uses_estimated_samples": uses_estimated_samples,
        "current_fighter1_odds": fight.get("fighter1_odds"),
        "current_fighter2_odds": fight.get("fighter2_odds"),
        "bet_placed": fight.get("bet_placed"),
        "samples": samples,
    }


def toggle_bet_placed(
    event_date: str,
    fighter1: str,
    fighter2: str,
    bet_fighter: str,
    *,
    stake: float | None = None,
    custom_odds: int | None = None,
) -> dict[str, Any] | None:
    store = _load_store()
    event, fight = _find_fight(store, event_date=event_date, fighter1=fighter1, fighter2=fighter2)
    if event is None or fight is None:
        return None

    selected = str(bet_fighter).strip()
    if selected == fight.get("fighter1"):
        opponent = fight.get("fighter2")
        current_odds = fight.get("fighter1_odds")
        opponent_odds = fight.get("fighter2_odds")
    elif selected == fight.get("fighter2"):
        opponent = fight.get("fighter1")
        current_odds = fight.get("fighter2_odds")
        opponent_odds = fight.get("fighter1_odds")
    else:
        return None

    current_bet = fight.get("bet_placed")
    if current_bet and current_bet.get("fighter") == selected:
        fight["bet_placed"] = None
        active = False
    else:
        stake_value = None if stake is None else round(float(stake), 2)
        if stake_value is None or stake_value <= 0:
            return None
        bet_odds = int(custom_odds) if custom_odds is not None else current_odds
        if bet_odds in (None, 0):
            return None
        fight["bet_placed"] = {
            "fighter": selected,
            "opponent": opponent,
            "listed_odds": current_odds,
            "opponent_listed_odds": opponent_odds,
            "bet_odds": bet_odds,
            "stake": stake_value,
            "placed_at": _iso_z(_utc_now()),
        }
        active = True

    _save_store(store)
    return {
        "event_name": event.get("event_name"),
        "event_date": event.get("event_date"),
        "fighter1": fight.get("fighter1"),
        "fighter2": fight.get("fighter2"),
        "bet_placed": fight.get("bet_placed"),
        "active": active,
    }


def fetch_odds_payload(api_key: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    start, end = _window_bounds()
    response = requests.get(
        ODDS_API_URL,
        params={
            "apiKey": api_key,
            "regions": DEFAULT_REGIONS,
            "markets": DEFAULT_MARKETS,
            "oddsFormat": DEFAULT_ODDS_FORMAT,
            "dateFormat": DEFAULT_DATE_FORMAT,
            "commenceTimeFrom": _iso_z(start),
            "commenceTimeTo": _iso_z(end),
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected The Odds API response shape")
    headers = {
        "x-requests-remaining": response.headers.get("x-requests-remaining", ""),
        "x-requests-used": response.headers.get("x-requests-used", ""),
        "x-requests-last": response.headers.get("x-requests-last", ""),
    }
    return payload, headers


def _save_raw_snapshot(payload: list[dict[str, Any]], headers: dict[str, str]) -> Path:
    fetched_at = _utc_now()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{fetched_at.strftime('%Y%m%d_%H%M%S')}_mma_odds.json"
    out_path.write_text(
        json.dumps(
            {
                "fetched_at": _iso_z(fetched_at),
                "headers": headers,
                "payload": payload,
            },
            indent=2,
        )
    )
    return out_path


def sync_new_the_odds_api_events(api_key: str | None = None, *, dry_run: bool = False) -> dict[str, Any]:
    api_key = api_key or os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is required")

    configure_job(JOB_NAME, enabled=scheduler_enabled())
    mark_run_started(JOB_NAME, trigger="manual" if dry_run else "scheduled")
    started_at = _utc_now()
    logger.info("Starting The Odds API sync dry_run=%s", dry_run)
    payload, headers = fetch_odds_payload(api_key)
    raw_snapshot = _save_raw_snapshot(payload, headers)

    store = _load_store()
    events_by_key = _event_index(store)
    legacy_keys = _collect_legacy_fight_keys()
    export_rows_before = _export_store_rows(store)

    added_fights = 0
    updated_fights = 0
    unchanged_fights = 0
    skipped_existing = 0
    skipped_invalid = 0
    deactivated_fights = 0
    seen_pairs: set[tuple[str, str]] = set()
    now_utc = _utc_now()
    captured_at = _iso_z(now_utc)

    for event in payload:
        row = _normalize_api_event(event)
        if row is None:
            skipped_invalid += 1
            continue

        commence_dt = _parse_iso(row["commence_time"])
        if commence_dt is None or not _within_window(commence_dt, now_utc):
            skipped_invalid += 1
            continue

        event_key = row["event_key"]
        fight_id = row["fight_key"]
        pair_key = (event_key, fight_id)
        seen_pairs.add(pair_key)

        event_entry = events_by_key.get(event_key)
        existing_fight = _fight_index(event_entry).get(fight_id) if event_entry is not None else None
        if existing_fight is None and fight_id in legacy_keys:
            skipped_existing += 1
            continue

        if event_entry is None:
            event_entry = {
                "event_key": event_key,
                "event_name": row["event_name"],
                "event_date": row["event_date"],
                "event_url": "",
                "source_type": "the_odds_api",
                "first_seen_at": captured_at,
                "last_synced_at": captured_at,
                "fights": [],
            }
            store["events"].append(event_entry)
            events_by_key[event_key] = event_entry
        else:
            event_entry["event_name"] = row["event_name"]
            event_entry["event_date"] = row["event_date"]
            event_entry["last_synced_at"] = captured_at

        fights_by_key = _fight_index(event_entry)
        fight_entry = fights_by_key.get(fight_id)

        history_entry = _history_entry(row, captured_at)

        if fight_entry is None:
            event_entry["fights"].append(
                {
                    **row,
                    "active": True,
                    "first_seen_at": captured_at,
                    "last_seen_at": captured_at,
                    "removed_at": None,
                    "source_event_ids": [row["source_event_id"]] if row["source_event_id"] else [],
                    "odds_history": [history_entry],
                }
            )
            added_fights += 1
            continue

        if row["source_event_id"] and row["source_event_id"] not in fight_entry.get("source_event_ids", []):
            fight_entry.setdefault("source_event_ids", []).append(row["source_event_id"])

        previous_snapshot = fight_entry.get("odds_history", [])[-1] if fight_entry.get("odds_history") else None
        if previous_snapshot is None or not _same_snapshot(previous_snapshot, history_entry):
            fight_entry.setdefault("odds_history", []).append(history_entry)
            updated_fights += 1
        else:
            unchanged_fights += 1

        fight_entry.update(
            {
                **row,
                "active": True,
                "last_seen_at": captured_at,
                "removed_at": None,
            }
        )

    for event_entry in store.get("events", []):
        event_entry["last_synced_at"] = captured_at
        for fight_entry in event_entry.get("fights", []):
            pair_key = (event_entry["event_key"], fight_entry["fight_key"])
            if pair_key in seen_pairs:
                continue
            if fight_entry.get("active", True):
                fight_entry["active"] = False
                fight_entry["removed_at"] = captured_at
                deactivated_fights += 1

    export_rows = _export_store_rows(store)
    export_rows.sort(key=lambda row: (row.get("event_date", ""), row.get("commence_time", ""), row.get("fighter1", ""), row.get("fighter2", "")))

    finished_at = _utc_now()
    state = {
        "last_run_started_at": _iso_z(started_at),
        "last_run_finished_at": _iso_z(finished_at),
        "last_success_at": _iso_z(finished_at),
        "raw_snapshot": str(raw_snapshot.relative_to(ROOT_DIR)),
        "payload_events": len(payload),
        "existing_rows": len(export_rows_before),
        "new_rows_added": added_fights,
        "updated_rows": updated_fights,
        "unchanged_rows": unchanged_fights,
        "deactivated_rows": deactivated_fights,
        "exported_rows": len(export_rows),
        "skipped_existing": skipped_existing,
        "skipped_invalid": skipped_invalid,
        "requests_remaining": headers.get("x-requests-remaining", ""),
        "requests_used": headers.get("x-requests-used", ""),
        "requests_last": headers.get("x-requests-last", ""),
        "window_days": _window_days(),
        "output_csv": str(OUTPUT_CSV.relative_to(ROOT_DIR)),
        "store_json": str(STORE_PATH.relative_to(ROOT_DIR)),
        "dry_run": dry_run,
    }

    sherdog_recovery: dict[str, Any] | None = None
    if not dry_run:
        _save_store(store)
        _write_csv_rows(OUTPUT_CSV, export_rows)
        if recovery_enabled():
            try:
                sherdog_recovery = recover_missing_fighters_from_odds(trigger="the_odds_api_sync")
            except Exception as exc:
                logger.exception("Sherdog recovery run failed after The Odds API sync")
                sherdog_recovery = {"status": "error", "error": str(exc)}
        state["sherdog_recovery"] = sherdog_recovery
        _save_sync_state(state)

    mark_run_finished(JOB_NAME, success=True, summary=state)
    logger.info(
        "The Odds API sync finished payload_events=%s new_rows_added=%s updated_rows=%s deactivated_rows=%s sherdog_recovered=%s",
        state["payload_events"],
        state["new_rows_added"],
        state.get("updated_rows", 0),
        state.get("deactivated_rows", 0),
        (state.get("sherdog_recovery") or {}).get("recovered", 0) if isinstance(state.get("sherdog_recovery"), dict) else 0,
    )
    return state


def scheduler_enabled() -> bool:
    if not os.getenv(API_KEY_ENV):
        return False
    return os.getenv(AUTO_SYNC_ENV, "1").strip().lower() not in FALSEY


def sync_due(now: datetime | None = None) -> bool:
    state = _load_sync_state()
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
    logger.info("Running scheduled The Odds API sync")
    result = sync_new_the_odds_api_events()
    logger.info(
        "The Odds API sync added %s new rows, updated %s rows, deactivated %s rows",
        result["new_rows_added"],
        result.get("updated_rows", 0),
        result.get("deactivated_rows", 0),
    )
    return result


async def run_sync_loop() -> None:
    check_seconds = int(os.getenv(SYNC_CHECK_SECONDS_ENV, "3600"))
    while True:
        try:
            await asyncio.to_thread(sync_if_due)
        except Exception:
            mark_run_finished(JOB_NAME, success=False, error="Scheduled The Odds API sync failed")
            logger.exception("Scheduled The Odds API sync failed")
        await asyncio.sleep(check_seconds)
