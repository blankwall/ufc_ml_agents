from __future__ import annotations

from datetime import date, datetime
from typing import Any

from services.golden_elo_service import evaluate_golden_elo_reopen

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
    filters: dict[str, Any],
    wmma_rules: dict[str, Any],
    as_of_date: date | datetime | str | None = None,
) -> dict[str, Any]:
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

    golden = evaluate_golden_elo_reopen(
        fighter1_name=fighter1_name,
        fighter2_name=fighter2_name,
        pick_slot=pick_slot,
        pick_model_prob=pick_model_prob,
        pick_odds=pick_odds,
        as_of_date=as_of_date,
    )
    payload = {
        "decision_source": "static_config" if result["bet"] else "static_skip",
        "review_bucket": None,
        "review_tier": None,
        "review_label": None,
        "review_stats": None,
        "pick_elo_diff": golden.get("pick_elo_diff"),
    }
    if not result["bet"] and golden.get("reopen"):
        payload.update(
            {
                "decision_source": "golden_elo_reopen",
                "review_bucket": golden.get("review_bucket"),
                "review_tier": golden.get("review_tier"),
                "review_label": golden.get("review_label"),
                "review_stats": golden.get("review_stats"),
            }
        )
        result = {"bet": True, "skip_code": None, "skip_reason": None}

    return {**result, **payload}
