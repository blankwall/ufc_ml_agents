from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "fastapi_app") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "fastapi_app"))

from backtest.confidence_profile import describe_confidence
from backtest.deterministic_signal_filter import (
    evaluate_deterministic_signal_filter,
    evaluate_elo_market_signal,
)
from fastapi_app.services.fighter_snapshot import build_fighter_snapshot
from fastapi_app.services.predict_service import (
    FIGHTER_ALIASES,
    MatchupFeatureExtractor,
    _fight_key,
    _fight_count_as_of,
    _is_wmma,
    _load_all_odds,
    _parse_event_date_any,
    _prediction_cutoff_datetime,
    _resolve_fighter,
    _score_row,
)

_DB_PATH = ROOT_DIR / "data" / "ufc_database.db"
_engine = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine)


def _issue(code: str, message: str, *, field: str | None = None) -> dict[str, str]:
    payload = {"code": code, "message": message}
    if field is not None:
        payload["field"] = field
    return payload


def _american_to_prob(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _round_prob(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _round_pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100, 1)


def _coerce_fight_date(fight_date: date | datetime | str | None) -> tuple[date | None, datetime | None]:
    if fight_date is None:
        return None, None
    if isinstance(fight_date, datetime):
        parsed = fight_date.replace(tzinfo=None)
    elif isinstance(fight_date, date):
        parsed = datetime.combine(fight_date, datetime.min.time())
    else:
        parsed = _parse_event_date_any(str(fight_date).strip())
    if parsed is None:
        raise ValueError("fight_date must be a parseable date string.")
    return parsed.date(), _prediction_cutoff_datetime(parsed)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _lookup_market_odds(
    fighter1: str,
    fighter2: str,
    fight_date: date | None,
) -> dict[str, Any] | None:
    odds = _load_all_odds()
    if odds.empty:
        return None

    alias_f1 = FIGHTER_ALIASES.get(fighter1, fighter1)
    alias_f2 = FIGHTER_ALIASES.get(fighter2, fighter2)
    target_key = _fight_key(alias_f1, alias_f2)

    matches: list[dict[str, Any]] = []
    for _, row in odds.iterrows():
        row_f1 = str(row.get("fighter1", "") or "").strip()
        row_f2 = str(row.get("fighter2", "") or "").strip()
        if _fight_key(row_f1, row_f2) != target_key:
            continue
        row_event_date = row.get("event_date")
        parsed_row_date = _parse_event_date_any(str(row_event_date).strip()) if row_event_date else None
        row_date = parsed_row_date.date() if parsed_row_date is not None else None
        matches.append(
            {
                "fighter1_odds": _coerce_optional_int(row.get("fighter1_odds")),
                "fighter2_odds": _coerce_optional_int(row.get("fighter2_odds")),
                "source_type": row.get("source_type"),
                "source_file": row.get("source_file"),
                "event_name": row.get("event_name"),
                "event_date": row_date.isoformat() if row_date else None,
                "event_url": row.get("event_url"),
                "last_update": row.get("last_update"),
            }
        )

    if not matches:
        return None
    if fight_date is not None:
        dated_matches = [row for row in matches if row["event_date"] == fight_date.isoformat()]
        if dated_matches:
            return dated_matches[0]
    return matches[0]


def _market_inputs_with_lookup(
    fighter1: str,
    fighter2: str,
    fight_date: date | None,
    fighter1_odds: int | None,
    fighter2_odds: int | None,
) -> tuple[int | None, int | None, dict[str, Any] | None]:
    lookup = None
    resolved_f1_odds = fighter1_odds
    resolved_f2_odds = fighter2_odds

    if fighter1_odds is None or fighter2_odds is None:
        lookup = _lookup_market_odds(fighter1, fighter2, fight_date)
        if lookup is not None:
            if resolved_f1_odds is None:
                resolved_f1_odds = lookup["fighter1_odds"]
            if resolved_f2_odds is None:
                resolved_f2_odds = lookup["fighter2_odds"]

    return resolved_f1_odds, resolved_f2_odds, lookup


def _compact_fighter_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    if not snapshot.get("resolved"):
        return snapshot

    elo = snapshot.get("elo") or {}
    compact_elo = {
        key: elo.get(key)
        for key in (
            "available",
            "fighter_id",
            "resolved_name",
            "resolution_source",
            "elo_current",
            "elo_peak",
            "elo_decline_from_peak",
            "elo_current_source",
            "fights_in_sidecar",
            "fights_total_all_time",
        )
    }

    return {
        "query_name": snapshot.get("query_name"),
        "as_of_date": snapshot.get("as_of_date"),
        "resolved": snapshot.get("resolved"),
        "identity": snapshot.get("identity"),
        "record": snapshot.get("record"),
        "profile": snapshot.get("profile"),
        "stats": snapshot.get("stats"),
        "elo": compact_elo,
        "recent_results": snapshot.get("recent_results", []),
        "qualitative": snapshot.get("qualitative"),
    }


def get_elo_market_signal(
    *,
    fighter1: str,
    fighter2: str,
    fight_date: date | datetime | str | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
) -> dict[str, Any]:
    analysis = init_fight_analysis(
        fighter1=fighter1,
        fighter2=fighter2,
        fight_date=fight_date,
        fighter1_odds=fighter1_odds,
        fighter2_odds=fighter2_odds,
    )
    return evaluate_elo_market_signal(analysis)


def get_deterministic_signal_filter(
    *,
    fighter1: str,
    fighter2: str,
    fight_date: date | datetime | str | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
) -> dict[str, Any]:
    analysis = init_fight_analysis(
        fighter1=fighter1,
        fighter2=fighter2,
        fight_date=fight_date,
        fighter1_odds=fighter1_odds,
        fighter2_odds=fighter2_odds,
    )
    return evaluate_deterministic_signal_filter(analysis)


def normalize_market_odds(
    fighter1_odds: int | None,
    fighter2_odds: int | None,
) -> dict[str, Any]:
    raw_f1 = raw_f2 = None
    normalized_f1 = normalized_f2 = 0.5
    method = "even_money_default"
    warnings: list[str] = []

    if fighter1_odds is not None and fighter2_odds is not None:
        raw_f1 = _american_to_prob(fighter1_odds)
        raw_f2 = _american_to_prob(fighter2_odds)
        vig = raw_f1 + raw_f2
        normalized_f1 = raw_f1 / vig
        normalized_f2 = raw_f2 / vig
        method = "vig_normalized"
    elif fighter1_odds is not None:
        raw_f1 = _american_to_prob(fighter1_odds)
        normalized_f1 = raw_f1
        normalized_f2 = 1.0 - raw_f1
        method = "single_sided_fighter1"
        warnings.append("Only fighter1 odds provided; fighter2 market probability inferred as the complement.")
        vig = None
    elif fighter2_odds is not None:
        raw_f2 = _american_to_prob(fighter2_odds)
        normalized_f2 = raw_f2
        normalized_f1 = 1.0 - raw_f2
        method = "single_sided_fighter2"
        warnings.append("Only fighter2 odds provided; fighter1 market probability inferred as the complement.")
        vig = None
    else:
        warnings.append("No odds provided; defaulted market probabilities to 50/50.")
        vig = None

    return {
        "odds": {
            "fighter1": fighter1_odds,
            "fighter2": fighter2_odds,
        },
        "has_market_odds": fighter1_odds is not None or fighter2_odds is not None,
        "normalization_method": method,
        "raw_implied_probabilities": {
            "fighter1": _round_prob(raw_f1),
            "fighter2": _round_prob(raw_f2),
        },
        "normalized_probabilities": {
            "fighter1": _round_prob(normalized_f1),
            "fighter2": _round_prob(normalized_f2),
        },
        "normalized_probabilities_pct": {
            "fighter1": _round_pct(normalized_f1),
            "fighter2": _round_pct(normalized_f2),
        },
        "overround": _round_prob(vig),
        "overround_pct": _round_pct(vig - 1.0) if vig is not None else None,
        "warnings": warnings,
        "provenance": {
            "source": "user_input",
            "normalizer": "mcp_server.fight_init.normalize_market_odds",
        },
    }


def _matchup_wmma_flag(session, fighter1_id: int, fighter2_id: int) -> bool | None:
    w1 = _is_wmma(session, fighter1_id)
    w2 = _is_wmma(session, fighter2_id)
    if w1 is True or w2 is True:
        return True
    if w1 is None and w2 is None:
        return None
    return False


def _fighter_payload(requested_name: str, resolved, lookup_name: str) -> dict[str, Any]:
    if resolved is None:
        return {
            "requested_name": requested_name,
            "lookup_name": lookup_name,
            "alias_applied": lookup_name != requested_name,
            "resolved": False,
            "fighter_id": None,
            "resolved_name": None,
            "record": None,
        }
    return {
        "requested_name": requested_name,
        "lookup_name": lookup_name,
        "alias_applied": lookup_name != requested_name,
        "resolved": True,
        "fighter_id": resolved.id,
        "resolved_name": resolved.name,
        "record": f"{resolved.wins}-{resolved.losses}-{resolved.draws}",
    }


def init_fight_analysis(
    *,
    fighter1: str,
    fighter2: str,
    fight_date: date | datetime | str | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    resolved_f1_odds = fighter1_odds
    resolved_f2_odds = fighter2_odds
    market_lookup: dict[str, Any] | None = None
    market = {
        "odds": {"fighter1": fighter1_odds, "fighter2": fighter2_odds},
        "has_market_odds": fighter1_odds is not None or fighter2_odds is not None,
        "normalization_method": None,
        "raw_implied_probabilities": {"fighter1": None, "fighter2": None},
        "normalized_probabilities": {"fighter1": None, "fighter2": None},
        "normalized_probabilities_pct": {"fighter1": None, "fighter2": None},
        "overround": None,
        "overround_pct": None,
        "warnings": [],
        "provenance": {
            "source": "user_input",
            "normalizer": "mcp_server.fight_init.normalize_market_odds",
        },
    }
    resolution = {
        "fighter1": _fighter_payload(fighter1, None, FIGHTER_ALIASES.get(fighter1, fighter1)),
        "fighter2": _fighter_payload(fighter2, None, FIGHTER_ALIASES.get(fighter2, fighter2)),
        "fight_date": {
            "input": fight_date.isoformat() if isinstance(fight_date, (date, datetime)) else fight_date,
            "parsed": None,
            "as_of_datetime": None,
            "used_for_feature_cutoff": False,
        },
    }
    fighters: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None

    if not fighter1.strip():
        errors.append(_issue("missing_fighter", "fighter1 must not be empty.", field="fighter1"))
    if not fighter2.strip():
        errors.append(_issue("missing_fighter", "fighter2 must not be empty.", field="fighter2"))
    if fighter1.strip() and fighter2.strip() and fighter1.strip().lower() == fighter2.strip().lower():
        errors.append(_issue("duplicate_fighters", "fighter1 and fighter2 must be different fighters."))

    try:
        parsed_date, as_of = _coerce_fight_date(fight_date)
        if parsed_date is None:
            warnings.append(
                _issue("missing_fight_date", "No fight_date provided; using the latest available fighter history.")
            )
        else:
            resolution["fight_date"] = {
                "input": resolution["fight_date"]["input"],
                "parsed": parsed_date.isoformat(),
                "as_of_datetime": as_of.isoformat() if as_of else None,
                "used_for_feature_cutoff": True,
            }
    except ValueError as exc:
        parsed_date = None
        as_of = None
        errors.append(_issue("invalid_fight_date", str(exc), field="fight_date"))

    try:
        resolved_f1_odds, resolved_f2_odds, market_lookup = _market_inputs_with_lookup(
            fighter1,
            fighter2,
            parsed_date,
            fighter1_odds,
            fighter2_odds,
        )
        market = normalize_market_odds(resolved_f1_odds, resolved_f2_odds)
        if market_lookup is not None:
            if fighter1_odds is None and fighter2_odds is None:
                market["provenance"]["source"] = "app_odds_lookup"
            else:
                market["provenance"]["source"] = "mixed_input_plus_app_lookup"
            market["provenance"]["lookup"] = market_lookup
        market["odds"] = {"fighter1": resolved_f1_odds, "fighter2": resolved_f2_odds}
        for warning in market["warnings"]:
            warnings.append(_issue("market_input", warning, field="odds"))
    except ValueError as exc:
        errors.append(_issue("invalid_odds", str(exc), field="odds"))

    if errors:
        return {
            "status": "invalid",
            "request": {
                "fighter1": fighter1,
                "fighter2": fighter2,
                "fight_date": resolution["fight_date"]["input"],
                "fighter1_odds": resolved_f1_odds,
                "fighter2_odds": resolved_f2_odds,
            },
            "validation": {
                "ok": False,
                "errors": errors,
                "warnings": warnings,
            },
            "resolution": resolution,
            "fighters": fighters,
            "market": market,
            "prediction": prediction,
            "provenance": {
                "source": "mcp_server.fight_init.init_fight_analysis",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "steps": {
                    "fighter_resolution": "not_run",
                    "market_normalization": "completed" if market["normalization_method"] else "not_run",
                    "model_prediction": "not_run",
                },
            },
        }

    session = _Session()
    try:
        f1_lookup = FIGHTER_ALIASES.get(fighter1, fighter1)
        f2_lookup = FIGHTER_ALIASES.get(fighter2, fighter2)
        f1 = _resolve_fighter(session, f1_lookup)
        f2 = _resolve_fighter(session, f2_lookup)

        resolution["fighter1"] = _fighter_payload(fighter1, f1, f1_lookup)
        resolution["fighter2"] = _fighter_payload(fighter2, f2, f2_lookup)

        if f1 is None:
            errors.append(_issue("fighter_not_found", f"Could not resolve fighter1: {fighter1}", field="fighter1"))
        if f2 is None:
            errors.append(_issue("fighter_not_found", f"Could not resolve fighter2: {fighter2}", field="fighter2"))
        if f1 is not None and f2 is not None and f1.id == f2.id:
            errors.append(
                _issue("duplicate_resolution", "fighter1 and fighter2 resolved to the same database fighter.")
            )

        if errors:
            return {
                "status": "invalid",
                "request": {
                    "fighter1": fighter1,
                    "fighter2": fighter2,
                    "fight_date": resolution["fight_date"]["input"],
                    "fighter1_odds": fighter1_odds,
                    "fighter2_odds": fighter2_odds,
                },
                "validation": {
                    "ok": False,
                    "errors": errors,
                    "warnings": warnings,
                },
                "resolution": resolution,
                "fighters": fighters,
                "market": market,
                "prediction": prediction,
                "provenance": {
                    "source": "mcp_server.fight_init.init_fight_analysis",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "steps": {
                        "fighter_resolution": "completed",
                        "market_normalization": "completed",
                        "model_prediction": "not_run",
                    },
                },
            }

        extractor = MatchupFeatureExtractor(session)
        pred = _score_row(
            session,
            extractor,
            f1.id,
            f2.id,
            market["normalized_probabilities"]["fighter1"],
            as_of_date=as_of,
        )

        model_prob_f1 = pred["model_prob_f1"]
        model_prob_f2 = 1.0 - model_prob_f1
        if model_prob_f1 >= 0.5:
            pick_slot = "fighter1"
            pick_name = f1.name
            pick_prob = model_prob_f1
            pick_market_prob = market["normalized_probabilities"]["fighter1"]
        else:
            pick_slot = "fighter2"
            pick_name = f2.name
            pick_prob = model_prob_f2
            pick_market_prob = market["normalized_probabilities"]["fighter2"]

        confidence = describe_confidence(pick_prob)
        f1_count = _fight_count_as_of(session, f1.id, as_of)
        f2_count = _fight_count_as_of(session, f2.id, as_of)
        is_wmma = _matchup_wmma_flag(session, f1.id, f2.id)
        thin_data = f1_count < 3 or f2_count < 3
        if thin_data:
            warnings.append(
                _issue("thin_data", "At least one fighter has fewer than 3 prior fights before the analysis cutoff.")
            )

        fighters = {
            "fighter1": _compact_fighter_snapshot(build_fighter_snapshot(
                fighter1,
                as_of=as_of.isoformat() if as_of is not None else None,
                recent_elo_fights=2,
                session=session,
            )),
            "fighter2": _compact_fighter_snapshot(build_fighter_snapshot(
                fighter2,
                as_of=as_of.isoformat() if as_of is not None else None,
                recent_elo_fights=2,
                session=session,
            )),
        }

        prediction = {
            "model_source": pred["model_source"],
            "probabilities": {
                "fighter1": _round_prob(model_prob_f1),
                "fighter2": _round_prob(model_prob_f2),
            },
            "probabilities_pct": {
                "fighter1": _round_pct(model_prob_f1),
                "fighter2": _round_pct(model_prob_f2),
            },
            "pick": {
                "slot": pick_slot,
                "fighter_name": pick_name,
                "probability": _round_prob(pick_prob),
                "probability_pct": _round_pct(pick_prob),
                "market_probability": _round_prob(pick_market_prob),
                "market_probability_pct": _round_pct(pick_market_prob),
                "edge": _round_prob(pick_prob - pick_market_prob),
                "edge_pct": _round_pct(pick_prob - pick_market_prob),
            },
            "confidence": {
                "score": confidence["confidence_score"],
                "historical_win_rate_pct": confidence["confidence_historical_win_rate"],
                "band": {
                    "min_prob_pct": confidence["confidence_prob_min"],
                    "max_prob_pct": confidence["confidence_prob_max"],
                    "avg_prob_pct": confidence["confidence_avg_prob"],
                    "sample_size": confidence["confidence_sample_size"],
                },
                "method": confidence["confidence_method"],
            },
            "fighter_metadata": {
                "fighter1": {
                    "fight_count_as_of": f1_count,
                    "record": resolution["fighter1"]["record"],
                },
                "fighter2": {
                    "fight_count_as_of": f2_count,
                    "record": resolution["fighter2"]["record"],
                },
                "is_wmma": is_wmma,
                "thin_data_warning": thin_data,
            },
        }

        return {
            "status": "ok",
            "request": {
                "fighter1": fighter1,
                "fighter2": fighter2,
                "fight_date": resolution["fight_date"]["input"],
                "fighter1_odds": resolved_f1_odds,
                "fighter2_odds": resolved_f2_odds,
            },
            "validation": {
                "ok": True,
                "errors": [],
                "warnings": warnings,
            },
            "resolution": resolution,
            "fighters": fighters,
            "market": market,
            "prediction": prediction,
            "provenance": {
                "source": "mcp_server.fight_init.init_fight_analysis",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "helpers": {
                    "fighter_aliases": "fastapi_app.services.predict_service.FIGHTER_ALIASES",
                    "fighter_resolution": "fastapi_app.services.predict_service._resolve_fighter",
                    "model_scoring": "fastapi_app.services.predict_service._score_row",
                    "fight_counts": "fastapi_app.services.predict_service._fight_count_as_of",
                    "wmma_detection": "fastapi_app.services.predict_service._is_wmma",
                    "confidence": "backtest.confidence_profile.describe_confidence",
                    "fighter_snapshot": "fastapi_app.services.fighter_snapshot.build_fighter_snapshot",
                },
                "steps": {
                    "fighter_resolution": "completed",
                    "market_normalization": "completed",
                    "model_prediction": "completed",
                },
            },
        }
    finally:
        session.close()
