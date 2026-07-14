from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi_app.services.fighter_snapshot import build_fighter_snapshot

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "betting_config.json"

SKIP_REASONS = {
    "F1": "Favorite low confidence",
    "F2": "Favorite odds cap exceeded",
    "F3": "Favorite low edge",
    "U1": "Underdog low confidence",
    "U2": "Underdog low edge",
    "U3": "Underdog odds cap exceeded",
    "W1": "WMMA edge below WMMA minimum",
    "D1": "Insufficient fight data",
    "ERR": "Prediction failed",
}

_SIGNAL_PRIORITY = (
    "skip_50_65_elo_100_plus",
    "skip_50_65_elo_50_plus_not_expensive",
    "skip_50_65_elo_50_plus",
    "validated_cardio_advantage",
    "elo_100_plus",
    "elo_50_plus",
)

_SIGNAL_LABELS = {
    "skip_50_65_elo_100_plus": "ELO 100+ confidence bucket",
    "skip_50_65_elo_50_plus_not_expensive": "Golden ELO not-expensive bucket",
    "skip_50_65_elo_50_plus": "ELO-supported mid-confidence bucket",
    "validated_cardio_advantage": "Cardio support present",
    "elo_100_plus": "ELO 100+ support",
    "elo_50_plus": "ELO 50+ support",
}


def load_betting_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _snapshot_cache_key(fighter_name: str, as_of: datetime | date | None) -> tuple[str, str | None]:
    if isinstance(as_of, datetime):
        as_of_key = as_of.isoformat()
    elif isinstance(as_of, date):
        as_of_key = as_of.isoformat()
    else:
        as_of_key = None
    return fighter_name, as_of_key


def _fighter_snapshot(
    fighter_name: str,
    *,
    as_of: datetime | date | None,
    session,
    snapshot_cache: dict[tuple[str, str | None], dict[str, Any]] | None,
) -> dict[str, Any]:
    if not hasattr(session, "query"):
        return {"resolved": False}
    cache_key = _snapshot_cache_key(fighter_name, as_of)
    if snapshot_cache is not None and cache_key in snapshot_cache:
        return snapshot_cache[cache_key]
    snapshot = build_fighter_snapshot(fighter_name, as_of=as_of, session=session)
    if snapshot_cache is not None:
        snapshot_cache[cache_key] = snapshot
    return snapshot


def _signal_details(
    *,
    fighter1_name: str,
    fighter2_name: str,
    pick_slot: str,
    pick_model_prob: float,
    pick_odds: int | None,
    as_of: datetime | date | None,
    session,
    review_cfg: dict[str, Any],
    snapshot_cache: dict[tuple[str, str | None], dict[str, Any]] | None,
) -> dict[str, Any]:
    fighter1 = _fighter_snapshot(fighter1_name, as_of=as_of, session=session, snapshot_cache=snapshot_cache)
    fighter2 = _fighter_snapshot(fighter2_name, as_of=as_of, session=session, snapshot_cache=snapshot_cache)
    if not fighter1.get("resolved") or not fighter2.get("resolved"):
        return {
            "signal_bucket": None,
            "signal_label": None,
            "signal_reason": None,
            "signal_tags": [],
            "pick_elo_diff": None,
            "cardio_score_diff": None,
            "review_candidate": False,
            "review_bucket": None,
            "review_label": None,
            "review_reason": None,
        }

    pick_snapshot = fighter1 if pick_slot == "fighter1" else fighter2
    opp_snapshot = fighter2 if pick_slot == "fighter1" else fighter1
    pick_name = pick_snapshot["identity"]["canonical_name"]

    pick_elo = (pick_snapshot.get("elo") or {}).get("elo_current")
    opp_elo = (opp_snapshot.get("elo") or {}).get("elo_current")
    pick_elo_diff = round(float(pick_elo) - float(opp_elo), 1) if pick_elo is not None and opp_elo is not None else None

    pick_cardio = (pick_snapshot.get("qualitative") or {}).get("cardio_score")
    opp_cardio = (opp_snapshot.get("qualitative") or {}).get("cardio_score")
    cardio_diff = round(float(pick_cardio) - float(opp_cardio), 1) if pick_cardio is not None and opp_cardio is not None else None

    conf_min = float(review_cfg.get("confidence_min", 0.50))
    conf_max = float(review_cfg.get("confidence_max", 0.65))
    elo_diff_min = float(review_cfg.get("min_elo_diff", 50))
    strong_elo_diff = float(review_cfg.get("strong_elo_diff", 100))
    min_pick_odds = int(review_cfg.get("min_pick_odds", -300))
    cardio_signal_diff = float(review_cfg.get("cardio_signal_diff", 10))

    mid_confidence = conf_min <= pick_model_prob < conf_max
    not_expensive = pick_odds is not None and pick_odds > min_pick_odds

    signal_tags: list[str] = []
    if pick_elo_diff is not None and pick_elo_diff >= elo_diff_min:
        signal_tags.append("elo_50_plus")
    if pick_elo_diff is not None and pick_elo_diff >= strong_elo_diff:
        signal_tags.append("elo_100_plus")
    if mid_confidence and pick_elo_diff is not None and pick_elo_diff >= elo_diff_min:
        signal_tags.append("skip_50_65_elo_50_plus")
    if mid_confidence and pick_elo_diff is not None and pick_elo_diff >= strong_elo_diff:
        signal_tags.append("skip_50_65_elo_100_plus")
    if mid_confidence and not_expensive and pick_elo_diff is not None and pick_elo_diff >= elo_diff_min:
        signal_tags.append("skip_50_65_elo_50_plus_not_expensive")
    if cardio_diff is not None and cardio_diff >= cardio_signal_diff:
        signal_tags.append("validated_cardio_advantage")

    signal_bucket = next((tag for tag in _SIGNAL_PRIORITY if tag in signal_tags), None)
    signal_label = _SIGNAL_LABELS.get(signal_bucket)
    signal_reason = None
    if signal_bucket == "skip_50_65_elo_100_plus":
        signal_reason = (
            f"{pick_name} lands in the 50-65% model-confidence bucket with a +{pick_elo_diff:.0f} ELO edge."
        )
    elif signal_bucket == "skip_50_65_elo_50_plus_not_expensive":
        signal_reason = (
            f"{pick_name} lands in the 50-65% model-confidence bucket with a +{pick_elo_diff:.0f} ELO edge "
            f"and a price better than {min_pick_odds}."
        )
    elif signal_bucket == "skip_50_65_elo_50_plus":
        signal_reason = f"{pick_name} lands in the tracked 50-65% confidence bucket with a +{pick_elo_diff:.0f} ELO edge."
    elif signal_bucket == "validated_cardio_advantage":
        signal_reason = f"{pick_name} also carries a +{cardio_diff:.0f} cardio-score edge."
    elif signal_bucket == "elo_100_plus":
        signal_reason = f"{pick_name} carries a +{pick_elo_diff:.0f} current-ELO edge."
    elif signal_bucket == "elo_50_plus":
        signal_reason = f"{pick_name} carries a +{pick_elo_diff:.0f} current-ELO edge."

    review_candidate = bool(
        review_cfg.get("enabled", False)
        and mid_confidence
        and not_expensive
        and pick_elo_diff is not None
        and pick_elo_diff >= elo_diff_min
    )
    review_label = "Human review: Golden ELO not-expensive"
    review_reason = None
    if review_candidate:
        review_reason = (
            f"{pick_name} meets the configured review gate: {conf_min * 100:.0f}-{conf_max * 100:.0f}% confidence, "
            f"+{elo_diff_min:.0f} ELO minimum, and odds better than {min_pick_odds}."
        )

    return {
        "signal_bucket": signal_bucket,
        "signal_label": signal_label,
        "signal_reason": signal_reason,
        "signal_tags": signal_tags,
        "pick_elo_diff": pick_elo_diff,
        "cardio_score_diff": cardio_diff,
        "review_candidate": review_candidate,
        "review_bucket": str(review_cfg.get("label", "golden_elo_not_expensive")) if review_candidate else None,
        "review_label": review_label if review_candidate else None,
        "review_reason": review_reason,
    }


def evaluate_bet_decision(
    *,
    fighter1_name: str,
    fighter2_name: str,
    pick_slot: str,
    pick_model_prob: float,
    pick_mkt_prob: float,
    pick_odds: int | None,
    is_favorite: bool,
    is_wmma: bool | None,
    f1_count: int,
    f2_count: int,
    as_of: datetime | date | None,
    session,
    snapshot_cache: dict[tuple[str, str | None], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg = load_betting_config()
    filters = cfg.get("filters", {})
    wmma_rules = cfg.get("wmma_rules", {})
    review_cfg = (cfg.get("dynamic_overrides", {}) or {}).get("golden_elo_not_expensive", {})

    min_fights = filters.get("min_fights", 2)
    fav_conf = filters.get("favorite_confidence_min", 0.65)
    ud_conf = filters.get("underdog_confidence_min", 0.53)
    fav_cap = filters.get("favorite_odds_cap", -300)
    ud_cap = filters.get("underdog_odds_cap", 300)
    edge_min = filters.get("edge_min", 0.04)
    ud_edge_min = filters.get("underdog_edge_min", edge_min)
    wmma_min_edge = wmma_rules.get("min_edge", 0.10)

    edge_pct = pick_model_prob - pick_mkt_prob

    if f1_count < min_fights or f2_count < min_fights:
        result = {"bet": False, "skip_code": "D1", "skip_reason": SKIP_REASONS["D1"]}
    elif is_wmma and edge_pct < wmma_min_edge:
        result = {"bet": False, "skip_code": "W1", "skip_reason": SKIP_REASONS["W1"]}
    elif is_favorite:
        if pick_model_prob < fav_conf:
            result = {"bet": False, "skip_code": "F1", "skip_reason": SKIP_REASONS["F1"]}
        elif pick_odds is not None and pick_odds < fav_cap:
            result = {"bet": False, "skip_code": "F2", "skip_reason": SKIP_REASONS["F2"]}
        elif edge_pct < edge_min:
            result = {"bet": False, "skip_code": "F3", "skip_reason": SKIP_REASONS["F3"]}
        else:
            result = {"bet": True, "skip_code": None, "skip_reason": None}
    else:
        if pick_model_prob < ud_conf:
            result = {"bet": False, "skip_code": "U1", "skip_reason": SKIP_REASONS["U1"]}
        elif pick_odds is not None and pick_odds > ud_cap:
            result = {"bet": False, "skip_code": "U3", "skip_reason": SKIP_REASONS["U3"]}
        elif edge_pct < ud_edge_min:
            result = {"bet": False, "skip_code": "U2", "skip_reason": SKIP_REASONS["U2"]}
        else:
            result = {"bet": True, "skip_code": None, "skip_reason": None}

    signal = _signal_details(
        fighter1_name=fighter1_name,
        fighter2_name=fighter2_name,
        pick_slot=pick_slot,
        pick_model_prob=pick_model_prob,
        pick_odds=pick_odds,
        as_of=as_of,
        session=session,
        review_cfg=review_cfg,
        snapshot_cache=snapshot_cache,
    )
    if review_cfg.get("surface_on_skip_only", True) and result["bet"]:
        signal["review_candidate"] = False
        signal["review_bucket"] = None
        signal["review_label"] = None
        signal["review_reason"] = None

    if not result["bet"] and signal["review_candidate"]:
        review_reason = signal["review_reason"]
        if result["skip_reason"]:
            review_reason = f"{result['skip_reason']}. {review_reason}"
        signal["review_reason"] = review_reason

    return {
        **result,
        "decision_source": "static_config" if result["bet"] else ("elo_review_gate" if signal["review_candidate"] else "static_skip"),
        **signal,
    }
