from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONTEXT_POOL = ROOT_DIR / "data" / "enrichment" / "context_pool.sqlite"

CARDIO_VALIDATION = {
    "label": "pace_retention",
    "assessment_corr": 0.454,
    "status": "first_pass_aligned",
}


def _round_prob(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _pct(value: float | None) -> float | None:
    return None if value is None else round(float(value) * 100, 1)


def elo_implied_probability(elo_diff: float | None) -> float | None:
    if elo_diff is None:
        return None
    return round(1 / (1 + 10 ** (-float(elo_diff) / 400)), 4)


def _load_pattern_stats(
    pattern_names: list[str],
    *,
    context_pool_path: Path = DEFAULT_CONTEXT_POOL,
) -> dict[str, Any]:
    ordered_names = list(dict.fromkeys(pattern_names))
    source_path = str(context_pool_path.relative_to(ROOT_DIR) if context_pool_path.is_relative_to(ROOT_DIR) else context_pool_path)
    if not ordered_names:
        return {
            "available": False,
            "matched_patterns": [],
            "missing_patterns": [],
            "source_table": "pattern_stats",
            "source_path": source_path,
            "reason": "No deterministic ELO bucket patterns applied.",
        }
    if not context_pool_path.exists():
        return {
            "available": False,
            "matched_patterns": [],
            "missing_patterns": ordered_names,
            "source_table": "pattern_stats",
            "source_path": source_path,
            "reason": "context_pool.sqlite was not found.",
        }

    conn = sqlite3.connect(f"file:{context_pool_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ", ".join("?" for _ in ordered_names)
        rows = conn.execute(
            f"""
            SELECT *
            FROM pattern_stats
            WHERE pattern_name IN ({placeholders})
            """,
            ordered_names,
        ).fetchall()
    finally:
        conn.close()

    by_name = {row["pattern_name"]: dict(row) for row in rows}
    matched = []
    for name in ordered_names:
        row = by_name.get(name)
        if row is None:
            continue
        matched.append(
            {
                "pattern_name": row["pattern_name"],
                "description": row["description"],
                "sample_size": row["sample_size"],
                "graded_sample_size": row.get("graded_sample_size", row["sample_size"]),
                "ungraded_sample_size": row.get("ungraded_sample_size", 0),
                "wins": row["wins"],
                "losses": row["losses"],
                "win_rate": row["win_rate"],
                "roi": row["roi"],
                "avg_confidence": row["avg_confidence"],
                "avg_edge": row["avg_edge"],
                "avg_elo_diff": row["avg_elo_diff"],
                "last_graded_date": row["last_graded_date"],
                "provenance": {
                    "source_table": "pattern_stats",
                    "source_key": row["pattern_name"],
                },
            }
        )

    return {
        "available": bool(matched),
        "matched_patterns": matched,
        "missing_patterns": [name for name in ordered_names if name not in by_name],
        "source_table": "pattern_stats",
        "source_path": source_path,
        "note": (
            "Pattern stats are aggregate historical bucket evidence from context_pool; "
            "they are deterministic decision-support filters, not standalone recommendations."
        ),
    }


def _analysis_slots(analysis: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    prediction = analysis["prediction"]
    pick = prediction["pick"]
    pick_slot = pick["slot"]
    opponent_slot = "fighter2" if pick_slot == "fighter1" else "fighter1"
    return pick, pick_slot, opponent_slot


def evaluate_elo_market_signal(
    analysis: dict[str, Any],
    *,
    context_pool_path: Path = DEFAULT_CONTEXT_POOL,
) -> dict[str, Any]:
    if analysis.get("status") != "ok":
        return {
            "status": "invalid",
            "analysis": analysis,
            "signal": None,
        }

    fighters = analysis["fighters"]
    market = analysis["market"]
    pick, pick_slot, opp_slot = _analysis_slots(analysis)
    pick_snapshot = fighters[pick_slot]
    opp_snapshot = fighters[opp_slot]
    pick_elo = (pick_snapshot.get("elo") or {}).get("elo_current")
    opp_elo = (opp_snapshot.get("elo") or {}).get("elo_current")
    if pick_elo is None or opp_elo is None:
        return {
            "status": "unavailable",
            "analysis": analysis,
            "signal": {
                "available": False,
                "reason": "Current ELO was not available for both fighters.",
            },
        }

    pick_elo_diff = pick_elo - opp_elo
    elo_implied_prob = elo_implied_probability(pick_elo_diff)
    pick_market_prob = pick.get("market_probability")
    pick_model_prob = pick.get("probability")
    market_minus_elo_prob = (
        round(pick_market_prob - elo_implied_prob, 4)
        if pick_market_prob is not None and elo_implied_prob is not None
        else None
    )
    model_minus_elo_prob = (
        round(pick_model_prob - elo_implied_prob, 4)
        if pick_model_prob is not None and elo_implied_prob is not None
        else None
    )
    pick_odds = market["odds"][pick_slot]

    triggers: list[dict[str, Any]] = []
    cautions: list[dict[str, Any]] = []
    pattern_names = ["all_oriented_elo"]
    score = 0
    pick_prob = pick.get("probability") or 0.0
    mid_confidence = 0.50 <= pick_prob < 0.65
    not_expensive = pick_odds is not None and pick_odds > -300
    plus_money = pick_odds is not None and pick_odds > 0

    if pick_elo_diff >= 50:
        score += 1
        triggers.append(
            {
                "tag": "elo_50_plus",
                "boost": 1,
                "reason": "Pick carries at least +50 current-ELO support versus the opponent.",
            }
        )
    if pick_elo_diff >= 100:
        score += 1
        triggers.append(
            {
                "tag": "elo_100_plus",
                "boost": 1,
                "reason": "Pick carries at least +100 current-ELO support.",
            }
        )
    if mid_confidence and pick_elo_diff >= 50:
        score += 1
        pattern_names.append("skip_50_65_elo_50_plus")
        triggers.append(
            {
                "tag": "mid_confidence_elo_support",
                "boost": 1,
                "reason": "Model confidence sits in the 50-65% range where +50 ELO support is tracked as an upgrade bucket.",
            }
        )
    if mid_confidence and pick_elo_diff >= 50 and not_expensive:
        score += 1
        pattern_names.append("skip_50_65_elo_50_plus_not_expensive")
        triggers.append(
            {
                "tag": "not_expensive_elo_support",
                "boost": 1,
                "reason": "Price is better than -300 while the pick still has +50 ELO support.",
            }
        )
    if mid_confidence and pick_elo_diff >= 50 and market_minus_elo_prob is not None and market_minus_elo_prob <= -0.10:
        score += 1
        pattern_names.append("skip_50_65_elo_50_plus_market_under_elo_10")
        triggers.append(
            {
                "tag": "market_under_elo_10",
                "boost": 1,
                "reason": "Market implied probability is at least 10 points below the ELO-implied probability.",
            }
        )
    if plus_money and pick_elo_diff > 0:
        score += 1
        pattern_names.append("underdog_elo_support")
        triggers.append(
            {
                "tag": "underdog_elo_support",
                "boost": 1,
                "reason": "The model pick is plus money while still holding the ELO edge.",
            }
        )

    if pick_elo_diff <= -50:
        score -= 2
        pattern_names.append("model_pick_lower_elo")
        cautions.append(
            {
                "tag": "elo_against_pick",
                "penalty": -2,
                "reason": "Current ELO disagrees materially with the pick by at least 50 points.",
            }
        )
    if plus_money and pick_elo_diff < 0:
        score -= 1
        pattern_names.append("underdog_elo_against")
        cautions.append(
            {
                "tag": "underdog_without_elo_support",
                "penalty": -1,
                "reason": "The pick is plus money but does not carry the ELO edge.",
            }
        )

    if pick_elo_diff > 0:
        pattern_names.append("model_pick_higher_elo")
    elif pick_elo_diff < 0 and "model_pick_lower_elo" not in pattern_names:
        pattern_names.append("model_pick_lower_elo")
    if mid_confidence and pick_elo_diff >= 100:
        pattern_names.append("skip_50_65_elo_100_plus")

    if score >= 5:
        tier = "very_strong_boost"
        boost_points = 3
    elif score >= 3:
        tier = "strong_boost"
        boost_points = 2
    elif score >= 1:
        tier = "mild_boost"
        boost_points = 1
    else:
        tier = "no_boost"
        boost_points = 0

    summary = (
        f"{pick['fighter_name']} shows a current-ELO edge of {pick_elo_diff:+} at {pick_odds} odds. "
        f"ELO implies {round((elo_implied_prob or 0) * 100, 1) if elo_implied_prob is not None else 'n/a'}% "
        f"while the market implies {round((pick_market_prob or 0) * 100, 1) if pick_market_prob is not None else 'n/a'}%. "
        f"This maps to a {tier.replace('_', ' ')} signal with {boost_points} boost point(s)."
    )

    return {
        "status": "ok",
        "request": analysis["request"],
        "pick": {
            "slot": pick_slot,
            "fighter_name": pick["fighter_name"],
            "probability": pick_model_prob,
            "probability_pct": pick["probability_pct"],
            "odds": pick_odds,
            "market_probability": pick_market_prob,
            "market_probability_pct": pick["market_probability_pct"],
        },
        "elo": {
            "pick_current_elo": pick_elo,
            "opponent_current_elo": opp_elo,
            "pick_elo_diff": pick_elo_diff,
            "elo_implied_probability": elo_implied_prob,
            "market_minus_elo_probability": market_minus_elo_prob,
            "model_minus_elo_probability": model_minus_elo_prob,
        },
        "historical_signal": {
            "score": score,
            "tier": tier,
            "boost_points": boost_points,
            "triggers": triggers,
            "cautions": cautions,
            "summary": summary,
            "historical_evidence": _load_pattern_stats(pattern_names, context_pool_path=context_pool_path),
        },
        "provenance": {
            "source": "backtest.deterministic_signal_filter.evaluate_elo_market_signal",
            "based_on": "dynamic fight analysis payload",
            "heuristics": [
                "elo_50_plus",
                "elo_100_plus",
                "mid_confidence_50_65",
                "not_expensive_better_than_minus_300",
                "market_under_elo_10",
                "underdog_elo_support",
            ],
        },
    }


def evaluate_cardio_signal(analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis.get("status") != "ok":
        return {
            "status": "invalid",
            "analysis": analysis,
            "signal": None,
        }

    fighters = analysis["fighters"]
    pick, pick_slot, opp_slot = _analysis_slots(analysis)
    pick_quality = (fighters[pick_slot] or {}).get("qualitative") or {}
    opp_quality = (fighters[opp_slot] or {}).get("qualitative") or {}
    pick_cardio = pick_quality.get("cardio_score")
    opp_cardio = opp_quality.get("cardio_score")
    if pick_cardio is None or opp_cardio is None:
        return {
            "status": "unavailable",
            "signal": {
                "available": False,
                "reason": "Cardio score was not available for both fighters.",
            },
        }

    cardio_diff = float(pick_cardio) - float(opp_cardio)
    triggers: list[dict[str, Any]] = []
    cautions: list[dict[str, Any]] = []
    score = 0
    if cardio_diff >= 10:
        score += 1
        triggers.append(
            {
                "tag": "validated_cardio_advantage",
                "boost": 1,
                "reason": "Pick has a material cardio-score edge; cardio is a first-pass aligned pace-retention trait.",
            }
        )
    elif cardio_diff <= -10:
        score -= 1
        cautions.append(
            {
                "tag": "cardio_disadvantage",
                "penalty": -1,
                "reason": "Pick has a material cardio-score disadvantage.",
            }
        )

    if score > 0:
        tier = "cardio_support"
    elif score < 0:
        tier = "cardio_risk"
    else:
        tier = "neutral"

    return {
        "status": "ok",
        "pick": {
            "slot": pick_slot,
            "fighter_name": pick["fighter_name"],
        },
        "cardio": {
            "pick_cardio_score": pick_cardio,
            "opponent_cardio_score": opp_cardio,
            "cardio_score_diff": _round_prob(cardio_diff),
            "cardio_score_diff_pct_points": _pct(cardio_diff / 100.0),
            "pick_trait_confidence": pick_quality.get("trait_confidence"),
            "opponent_trait_confidence": opp_quality.get("trait_confidence"),
        },
        "signal": {
            "score": score,
            "tier": tier,
            "triggers": triggers,
            "cautions": cautions,
            "validation": CARDIO_VALIDATION,
            "summary": (
                f"{pick['fighter_name']} cardio diff is {cardio_diff:+.1f}. "
                f"Cardio is tracked as {CARDIO_VALIDATION['label']} with {CARDIO_VALIDATION['status']} validation."
            ),
        },
        "provenance": {
            "source": "backtest.deterministic_signal_filter.evaluate_cardio_signal",
            "validation_source": "backtest.build_context_pool.TRAIT_VALIDATION_NOTES.cardio_score_diff",
        },
    }


def evaluate_deterministic_signal_filter(
    analysis: dict[str, Any],
    *,
    context_pool_path: Path = DEFAULT_CONTEXT_POOL,
) -> dict[str, Any]:
    elo = evaluate_elo_market_signal(analysis, context_pool_path=context_pool_path)
    cardio = evaluate_cardio_signal(analysis)
    elo_score = ((elo.get("historical_signal") or {}).get("score") or 0) if elo.get("status") == "ok" else 0
    cardio_score = ((cardio.get("signal") or {}).get("score") or 0) if cardio.get("status") == "ok" else 0
    combined_score = elo_score + cardio_score

    support_flags = []
    risk_flags = []
    if elo.get("status") == "ok":
        support_flags.extend(item["tag"] for item in (elo.get("historical_signal") or {}).get("triggers", []))
        risk_flags.extend(item["tag"] for item in (elo.get("historical_signal") or {}).get("cautions", []))
    if cardio.get("status") == "ok":
        support_flags.extend(item["tag"] for item in (cardio.get("signal") or {}).get("triggers", []))
        risk_flags.extend(item["tag"] for item in (cardio.get("signal") or {}).get("cautions", []))

    if combined_score >= 3:
        action = "positive_filter_review"
    elif combined_score <= -2:
        action = "risk_filter_review"
    elif support_flags or risk_flags:
        action = "mixed_filter_review"
    else:
        action = "neutral_no_filter_edge"

    return {
        "status": "ok" if analysis.get("status") == "ok" else "invalid",
        "request": analysis.get("request"),
        "filter_version": "deterministic_elo_cardio_v1",
        "summary": {
            "combined_score": combined_score,
            "action": action,
            "support_flags": sorted(set(support_flags)),
            "risk_flags": sorted(set(risk_flags)),
            "not_a_recommendation": True,
            "use_case": "Fast deterministic screening before any MCP deep dive.",
        },
        "model_market": None if analysis.get("status") != "ok" else {
            "pick": analysis["prediction"]["pick"],
            "market": analysis["market"],
        },
        "signals": {
            "elo": elo,
            "cardio": cardio,
        },
        "provenance": {
            "source": "backtest.deterministic_signal_filter.evaluate_deterministic_signal_filter",
            "inputs": "dynamic fight analysis payload",
        },
    }
