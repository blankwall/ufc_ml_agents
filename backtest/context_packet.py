#!/usr/bin/env python3
"""
Generate a deterministic fight context packet from context_pool.sqlite.

This is the first on-demand report layer for the Sergey/context sidecar work.
It does not use an LLM. It pulls a target fight from the generated context pool,
attaches relevant aggregate pattern stats, and retrieves nearest historical
examples by model confidence + ELO delta.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.elo_analysis import (  # noqa: E402
    DEFAULT_ALIAS_SOURCES,
    canonical_name,
    load_aliases,
)
from backtest.build_context_pool import PATTERNS  # noqa: E402


DEFAULT_POOL = ROOT_DIR / "data" / "enrichment" / "context_pool.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deterministic fight context packet.")
    parser.add_argument("--fighter1", required=True)
    parser.add_argument("--fighter2", required=True)
    parser.add_argument("--date", default=None, help="Optional YYYY-MM-DD target date to disambiguate rematches.")
    parser.add_argument("--season", type=int, default=None, help="Optional season filter.")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--similar-limit", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write packet JSON.")
    parser.add_argument("--json-only", action="store_true", help="Print only packet JSON.")
    parser.add_argument(
        "--expand-source-pattern",
        action="store_true",
        help="After the summary, print the rows behind the selected pattern_score_v0 source pattern.",
    )
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="With --expand-source-pattern, include pending/ungraded rows. Default is graded rows only.",
    )
    return parser.parse_args()


def bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("pick_correct", "bet", "female", "model_agrees_with_elo", "elo_pick_correct"):
        data[key] = bool_or_none(data.get(key))
    return data


def parse_json_list(value: str | None) -> list[dict[str, Any]]:
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def pair_key(fighter1: str, fighter2: str, aliases: dict[str, str]) -> tuple[str, str]:
    return tuple(sorted((canonical_name(fighter1, aliases), canonical_name(fighter2, aliases))))


def find_target(
    conn: sqlite3.Connection,
    *,
    fighter1: str,
    fighter2: str,
    date: str | None,
    season: int | None,
    aliases: dict[str, str],
) -> tuple[dict[str, Any], int]:
    wanted = pair_key(fighter1, fighter2, aliases)
    rows = []
    for row in conn.execute("SELECT * FROM backtest_fight_pool ORDER BY date DESC, id DESC"):
        data = dict(row)
        if season is not None and data["season"] != season:
            continue
        if date is not None and data["date"] != date:
            continue
        if pair_key(data["fighter1"], data["fighter2"], aliases) == wanted:
            rows.append(row_to_dict(row))

    if not rows:
        raise SystemExit(
            f"No context-pool row found for {fighter1} vs {fighter2}"
            + (f" on {date}" if date else "")
            + ". Rebuild the pool or check fighter names."
        )

    # Latest row wins by default; candidate_count tells the caller if date should
    # be supplied for a rematch.
    return rows[0], len(rows)


def fetch_pattern_stats(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute("SELECT * FROM pattern_stats").fetchall()
    return {row["pattern_name"]: dict(row) for row in rows}


def applicable_pattern_names(target: dict[str, Any]) -> list[str]:
    names = ["all_oriented_elo"]
    pick_elo_diff = target.get("pick_elo_diff")
    pick_prob = target.get("pick_prob") or 0.0
    pick_odds = target.get("pick_odds")
    bet = target.get("bet") is True

    if target.get("model_agrees_with_elo") is True:
        names.append("model_pick_higher_elo")
    elif target.get("model_agrees_with_elo") is False:
        names.append("model_pick_lower_elo")

    if pick_odds is not None and pick_odds > 0:
        names.append("model_pick_underdog")
        if pick_elo_diff is not None and pick_elo_diff > 0:
            names.append("underdog_elo_support")
        elif pick_elo_diff is not None and pick_elo_diff < 0:
            names.append("underdog_elo_against")

    if bet and pick_elo_diff is not None:
        names.append("bet_elo_support" if pick_elo_diff > 0 else "bet_elo_against")

    if not bet and 0.50 <= pick_prob < 0.65 and pick_elo_diff is not None:
        if pick_elo_diff >= 50:
            names.append("skip_50_65_elo_50_plus")
            if pick_odds is not None and pick_odds > -300:
                names.append("skip_50_65_elo_50_plus_not_expensive")
            market_minus_elo = target.get("market_minus_elo_prob")
            if market_minus_elo is not None and market_minus_elo <= -0.10:
                names.append("skip_50_65_elo_50_plus_market_under_elo_10")
            opp_quality_diff = target.get("pick_opponent_quality_diff")
            if opp_quality_diff is not None:
                if opp_quality_diff >= 0:
                    names.append("skip_50_65_elo_50_plus_opp_quality_support")
                else:
                    names.append("skip_50_65_elo_50_plus_opp_quality_against")
        if pick_elo_diff >= 100:
            names.append("skip_50_65_elo_100_plus")

    return names


def pattern_payload(conn: sqlite3.Connection, target: dict[str, Any]) -> list[dict[str, Any]]:
    stats_by_name = fetch_pattern_stats(conn)
    patterns = []
    for name in applicable_pattern_names(target):
        stats = stats_by_name.get(name)
        if not stats:
            continue
        patterns.append(
            {
                "pattern_name": stats["pattern_name"],
                "description": stats["description"],
                "sample_size": stats["sample_size"],
                "graded_sample_size": stats.get("graded_sample_size", stats["sample_size"]),
                "ungraded_sample_size": stats.get("ungraded_sample_size", 0),
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": stats["win_rate"],
                "profit": stats["profit"],
                "roi": stats["roi"],
                "avg_confidence": stats["avg_confidence"],
                "avg_edge": stats["avg_edge"],
                "avg_elo_diff": stats["avg_elo_diff"],
                "last_graded_date": stats.get("last_graded_date"),
                "filters": json.loads(stats["filters_json"]),
                "evidence_role": "decision_support",
            }
        )
    return patterns


def fetch_similar_rows(conn: sqlite3.Connection, target: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    target_edge = target.get("edge") or 0.0
    rows = conn.execute(
        """
        SELECT
            id,
            season,
            date,
            fighter1,
            fighter2,
            pick,
            winner,
            pick_prob,
            pick_odds,
            edge,
            pick_elo_diff,
            model_agrees_with_elo,
            bet,
            skip_reason,
            pick_correct,
            actual_pnl,
            (
                ABS(pick_prob - ?)
                + ABS(COALESCE(pick_elo_diff, 0) - ?) / 300.0
                + ABS(COALESCE(edge, 0) - ?) / 2.0
            ) AS distance
        FROM backtest_fight_pool
        WHERE id != ?
          AND join_status = 'matched'
          AND pick_elo_diff IS NOT NULL
          AND pick_correct IS NOT NULL
        ORDER BY distance ASC
        LIMIT ?
        """,
        (
            target["pick_prob"],
            target.get("pick_elo_diff") or 0,
            target_edge,
            target["id"],
            limit,
        ),
    ).fetchall()
    return [row_to_dict(row) for row in rows]


def fetch_trait_delta_evidence(conn: sqlite3.Connection, target: dict[str, Any]) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT summary, data_json, source_table, source_key
        FROM evidence_items
        WHERE fight_pool_id = ?
          AND evidence_type = 'trait_delta'
        ORDER BY evidence_id
        LIMIT 1
        """,
        (target["id"],),
    ).fetchone()
    if row is None:
        return None
    payload = json.loads(row["data_json"])
    return {
        "evidence_role": "context_metric",
        "evidence_type": "trait_delta",
        "summary": row["summary"],
        "source_table": row["source_table"],
        "source_key": row["source_key"],
        **payload,
    }


def filter_pattern_rows(
    rows: list[dict[str, Any]],
    *,
    pattern_name: str,
    include_pending: bool,
) -> list[dict[str, Any]]:
    pattern = next((pattern for pattern in PATTERNS if pattern[0] == pattern_name), None)
    if pattern is None:
        raise ValueError(f"Unknown source pattern: {pattern_name}")

    _, _, _, predicate = pattern
    return [
        row for row in rows
        if predicate(row) and (include_pending or row.get("pick_correct") is not None)
    ]


def fetch_pattern_expansion_rows(
    conn: sqlite3.Connection,
    *,
    pattern_name: str,
    include_pending: bool,
) -> list[dict[str, Any]]:
    rows = [
        row_to_dict(row)
        for row in conn.execute(
            """
            SELECT *
            FROM backtest_fight_pool
            ORDER BY date, fighter1, fighter2, id
            """
        )
    ]
    return filter_pattern_rows(rows, pattern_name=pattern_name, include_pending=include_pending)


def support_level(pick_elo_diff: int | None) -> str:
    if pick_elo_diff is None:
        return "missing"
    if pick_elo_diff >= 100:
        return "strong"
    if pick_elo_diff >= 50:
        return "moderate"
    if pick_elo_diff > 0:
        return "thin"
    if pick_elo_diff <= -100:
        return "strong_against"
    if pick_elo_diff <= -50:
        return "moderate_against"
    if pick_elo_diff < 0:
        return "thin_against"
    return "neutral"


def build_flags(
    target: dict[str, Any],
    patterns: list[dict[str, Any]],
    trait_delta: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    support: list[str] = []
    risk: list[str] = []
    pick_elo_diff = target.get("pick_elo_diff")
    opponent_quality_diff = target.get("pick_opponent_quality_diff")
    edge = target.get("edge")
    pick_prob = target.get("pick_prob") or 0.0
    pick_odds = target.get("pick_odds")

    if pick_elo_diff is None:
        risk.append("missing_oriented_elo")
    elif pick_elo_diff >= 100:
        support.append("strong_elo_advantage")
    elif pick_elo_diff >= 50:
        support.append("moderate_elo_advantage")
    elif pick_elo_diff > 0:
        support.append("thin_elo_advantage")
    elif pick_elo_diff <= -50:
        risk.append("elo_disagrees_with_pick")

    if target.get("model_agrees_with_elo") is True:
        support.append("model_elo_agreement")
    elif target.get("model_agrees_with_elo") is False:
        risk.append("model_elo_disagreement")

    if opponent_quality_diff is not None and opponent_quality_diff >= 50:
        support.append("opponent_quality_advantage")
    elif opponent_quality_diff is not None and opponent_quality_diff <= -50:
        risk.append("opponent_quality_disadvantage")

    if edge is not None and edge > 0:
        support.append("positive_market_edge")
    elif edge is not None and edge < 0:
        risk.append("negative_market_edge")

    if target.get("bet") is False:
        risk.append("current_rules_skip")
    if pick_odds is not None and pick_odds < 0 and pick_prob < 0.65:
        risk.append("below_favorite_confidence_threshold")
    if pick_odds is not None and pick_odds <= -300:
        risk.append("expensive_favorite")

    for pattern in patterns:
        if pattern["sample_size"] >= 20 and (pattern["roi"] or 0.0) > 0 and (pattern["win_rate"] or 0.0) >= 0.60:
            support.append(f"positive_historical_pattern:{pattern['pattern_name']}")
        elif pattern["sample_size"] < 10:
            risk.append(f"small_pattern_sample:{pattern['pattern_name']}")

    if trait_delta:
        deltas = trait_delta.get("deltas", {})
        validation = trait_delta.get("validation_notes", {})
        cardio = deltas.get("cardio_score_diff")
        striking_eff = deltas.get("striking_efficiency_score_diff")
        defensive_exposure = deltas.get("defensive_exposure_score_diff")
        anti_control = deltas.get("anti_control_score_diff")

        if cardio is not None and cardio >= 10 and validation.get("cardio_score_diff", {}).get("status") == "first_pass_aligned":
            support.append("validated_trait_context:cardio_advantage")
        elif cardio is not None and cardio <= -10:
            risk.append("trait_context:cardio_disadvantage")

        if striking_eff is not None and striking_eff >= 10:
            support.append("trait_context:striking_efficiency_advantage")
        elif striking_eff is not None and striking_eff <= -10:
            risk.append("trait_context:striking_efficiency_disadvantage")

        if defensive_exposure is not None and defensive_exposure >= 10:
            risk.append("experimental_trait_context:higher_defensive_exposure_risk")
        elif defensive_exposure is not None and defensive_exposure <= -10:
            support.append("experimental_trait_context:lower_defensive_exposure_risk")

        if anti_control is not None and abs(anti_control) >= 10:
            risk.append("experimental_trait_context:anti_control_formula_review_needed")

    return {"support": sorted(set(support)), "risk": sorted(set(risk))}


def pattern_grade(pattern: dict[str, Any]) -> tuple[int, str]:
    sample_size = pattern["sample_size"] or 0
    win_rate = pattern["win_rate"] or 0.0
    roi = pattern["roi"] or 0.0

    if sample_size >= 50 and win_rate >= 0.70 and roi >= 0.15:
        return 8, "strong"
    if sample_size >= 30 and win_rate >= 0.65 and roi >= 0.05:
        return 7, "moderate"
    if sample_size >= 20 and win_rate >= 0.60 and roi > 0:
        return 6, "mild"
    if sample_size >= 20 and roi <= 0:
        return 4, "negative_or_unprofitable"
    return 5, "insufficient_or_neutral"


def pattern_specificity(pattern_name: str) -> int:
    """Prefer narrow, decision-relevant patterns over broad base-rate patterns."""
    if "opp_quality" in pattern_name:
        return 50
    if "market_under_elo" in pattern_name:
        return 48
    if "not_expensive" in pattern_name:
        return 46
    if pattern_name.startswith("skip_50_65"):
        return 40
    if pattern_name.startswith("bet_"):
        return 30
    if pattern_name.startswith("underdog_"):
        return 25
    if pattern_name.startswith("model_pick_"):
        return 10
    return 0


def pattern_allowed_for_target_score(target: dict[str, Any], pattern_name: str) -> bool:
    if target.get("bet") is False:
        return pattern_name.startswith("skip_")
    return True


def build_pattern_score(target: dict[str, Any], patterns: list[dict[str, Any]]) -> dict[str, Any]:
    decision_patterns = [
        pattern for pattern in patterns
        if pattern["pattern_name"] != "all_oriented_elo"
        and pattern_allowed_for_target_score(target, pattern["pattern_name"])
    ]
    candidates = []
    for pattern in decision_patterns:
        score, support = pattern_grade(pattern)
        specificity = pattern_specificity(pattern["pattern_name"])
        candidates.append((score, specificity, pattern["sample_size"], pattern, support))

    if candidates:
        score, _, _, source_pattern, support = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    else:
        source_pattern = None
        score = 5
        support = "insufficient_or_neutral"

    warnings = [
        "Pattern score is still aggregate ELO-pattern evidence only; trait deltas are reported separately as context.",
        "Nearest historical examples are illustrative only and do not affect this score.",
    ]
    if target.get("edge") is not None and target["edge"] < 0:
        warnings.append("Negative market edge remains a risk even when historical ELO pattern is positive.")
    if source_pattern and source_pattern["sample_size"] < 30:
        warnings.append("Source pattern sample size is small; treat as exploratory.")
    if target.get("bet") is False and source_pattern is None:
        warnings.append("Skipped fights require a specific validated skip pattern; broad ELO agreement is context only.")

    action = "neutral_empirical_context"
    if score >= 8:
        action = "strong_empirical_support"
    elif score >= 7:
        action = "moderate_empirical_support"
    elif score <= 4:
        action = "negative_empirical_signal"

    return {
        "score": score,
        "support_level": support,
        "action": action,
        "decision_scope": "empirical_evidence_only",
        "not_a_recommendation": True,
        "source_pattern": source_pattern["pattern_name"] if source_pattern else None,
        "basis": None if source_pattern is None else {
            "sample_size": source_pattern["sample_size"],
            "graded_sample_size": source_pattern.get("graded_sample_size", source_pattern["sample_size"]),
            "ungraded_sample_size": source_pattern.get("ungraded_sample_size", 0),
            "wins": source_pattern["wins"],
            "losses": source_pattern["losses"],
            "win_rate": source_pattern["win_rate"],
            "roi": source_pattern["roi"],
            "avg_confidence": source_pattern["avg_confidence"],
            "avg_edge": source_pattern["avg_edge"],
            "avg_elo_diff": source_pattern["avg_elo_diff"],
            "last_graded_date": source_pattern.get("last_graded_date"),
        },
        "thresholds": {
            "strong": "N>=50, win_rate>=70%, ROI>=15%",
            "moderate": "N>=30, win_rate>=65%, ROI>=5%",
            "mild": "N>=20, win_rate>=60%, ROI>0%",
            "negative_or_unprofitable": "N>=20, ROI<=0%",
        },
        "warnings": warnings,
    }


def build_packet(
    conn: sqlite3.Connection,
    *,
    target: dict[str, Any],
    candidate_count: int,
    similar_limit: int,
    pool_path: Path,
) -> dict[str, Any]:
    patterns = pattern_payload(conn, target)
    similar_rows = fetch_similar_rows(conn, target, limit=similar_limit)
    trait_delta = fetch_trait_delta_evidence(conn, target)
    flags = build_flags(target, patterns, trait_delta)
    pattern_score = build_pattern_score(target, patterns)

    return {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {
            "pool": str(pool_path.relative_to(ROOT_DIR) if pool_path.is_relative_to(ROOT_DIR) else pool_path),
            "target_candidate_count": candidate_count,
        },
        "target": {
            "pool_id": target["id"],
            "season": target["season"],
            "date": target["date"],
            "fight": f"{target['fighter1']} vs {target['fighter2']}",
            "fighter1": target["fighter1"],
            "fighter2": target["fighter2"],
            "pick": target["pick"],
            "winner": target["winner"],
            "pick_correct": target["pick_correct"],
            "actual_pnl": target["actual_pnl"],
        },
        "model_market": {
            "pick_prob": target["pick_prob"],
            "pick_odds": target["pick_odds"],
            "market_implied_prob": target.get("market_implied_prob"),
            "edge": target["edge"],
            "elo_implied_prob": target.get("elo_implied_prob"),
            "model_minus_elo_prob": target.get("model_minus_elo_prob"),
            "market_minus_elo_prob": target.get("market_minus_elo_prob"),
            "model_market_elo_triangle": target.get("model_market_elo_triangle"),
            "current_decision": "bet" if target["bet"] else "skip",
            "skip_reason": target["skip_reason"],
        },
        "elo": {
            "fighter1_elo": target["fighter1_elo"],
            "fighter2_elo": target["fighter2_elo"],
            "pick_elo": target["pick_elo"],
            "opponent_elo": target["opponent_elo"],
            "pick_elo_diff": target["pick_elo_diff"],
            "abs_elo_diff": target["abs_elo_diff"],
            "model_agrees_with_elo": target["model_agrees_with_elo"],
            "support_level": support_level(target["pick_elo_diff"]),
            "join_status": target["join_status"],
            "join_method": target["join_method"],
        },
        "opponent_quality": {
            "pick_prior_fight_count": target.get("pick_prior_fight_count"),
            "opponent_prior_fight_count": target.get("opponent_prior_fight_count"),
            "pick_avg_prior_opponent_elo": target.get("pick_avg_prior_opponent_elo"),
            "opponent_avg_prior_opponent_elo": target.get("opponent_avg_prior_opponent_elo"),
            "pick_recent3_prior_opponent_elo": target.get("pick_recent3_prior_opponent_elo"),
            "opponent_recent3_prior_opponent_elo": target.get("opponent_recent3_prior_opponent_elo"),
            "pick_best_win_opponent_elo": target.get("pick_best_win_opponent_elo"),
            "opponent_best_win_opponent_elo": target.get("opponent_best_win_opponent_elo"),
            "pick_opponent_quality_diff": target.get("pick_opponent_quality_diff"),
            "pick_recent_opponent_quality_diff": target.get("pick_recent_opponent_quality_diff"),
            "pick_best_win_quality_diff": target.get("pick_best_win_quality_diff"),
            "pick_current_vs_peak_decline": target.get("pick_current_vs_peak_decline"),
            "opponent_current_vs_peak_decline": target.get("opponent_current_vs_peak_decline"),
            "pick_decline_diff": target.get("pick_decline_diff"),
            "pick_recent_fights": parse_json_list(target.get("pick_recent_fights_json")),
            "opponent_recent_fights": parse_json_list(target.get("opponent_recent_fights_json")),
        },
        "trait_deltas_v0": trait_delta,
        "flags": flags,
        "pattern_score_v0": pattern_score,
        "matching_patterns": {
            "evidence_role": "decision_support",
            "items": patterns,
        },
        "nearest_historical_examples": {
            "evidence_role": "qualitative_sanity_check",
            "warning": "Nearest examples are illustrative only; decision support comes from aggregate pattern stats.",
            "items": similar_rows,
        },
    }


def fmt_pct(value: float | None) -> str:
    return "--" if value is None else f"{value:.1%}"


def fmt_signed_pct(value: float | None) -> str:
    return "--" if value is None else f"{value:+.1%}"


def fmt_num(value: int | float | None) -> str:
    if value is None:
        return "--"
    return str(int(value)) if float(value).is_integer() else f"{value:.0f}"


def fmt_pnl(value: float | None) -> str:
    return "--" if value is None else f"{value:+.2f}"


def fmt_elo(value: float | int | None) -> str:
    return "--" if value is None else f"{value:.0f}"


def fmt_signed_elo(value: float | int | None) -> str:
    return "--" if value is None else f"{value:+.0f}"


def fmt_signed_score(value: float | int | None) -> str:
    return "--" if value is None else f"{value:+.1f}"


def print_recent_fights(label: str, rows: list[dict[str, Any]]) -> None:
    print(f"  {label}:")
    if not rows:
        print("    --")
        return
    for row in rows:
        result = row.get("result") or "--"
        method = row.get("method") or "method unknown"
        print(
            f"    {row.get('date', '--')} vs {row.get('opponent', '--')} "
            f"oppELO={fmt_elo(row.get('opponent_elo'))} {result} ({method})"
        )


def print_summary(packet: dict[str, Any]) -> None:
    target = packet["target"]
    model = packet["model_market"]
    elo = packet["elo"]
    opponent_quality = packet["opponent_quality"]
    print("=" * 90)
    print(f"CONTEXT PACKET: {target['fight']} ({target['date']})")
    print("=" * 90)
    print(
        f"Pick: {target['pick']} | Prob: {fmt_pct(model['pick_prob'])} | "
        f"Odds: {model['pick_odds']} | Edge: {fmt_pct(model['edge'])}"
    )
    print(f"Current decision: {model['current_decision'].upper()}" + (f" ({model['skip_reason']})" if model["skip_reason"] else ""))
    print(
        f"ELO: pick {elo['pick_elo']} vs opp {elo['opponent_elo']} "
        f"diff={elo['pick_elo_diff']} support={elo['support_level']}"
    )
    if model.get("elo_implied_prob") is not None:
        print("Model / market / ELO triangle")
        print(f"  Model probability:        {fmt_pct(model['pick_prob'])}")
        print(f"  Market implied:          {fmt_pct(model['market_implied_prob'])}")
        print(f"  ELO implied:             {fmt_pct(model['elo_implied_prob'])}")
        print(f"  Model - ELO:             {fmt_signed_pct(model['model_minus_elo_prob'])}")
        print(f"  Market - ELO:            {fmt_signed_pct(model['market_minus_elo_prob'])}")
        print(f"  Triangle:                {model['model_market_elo_triangle']}")
    if opponent_quality["pick_opponent_quality_diff"] is not None:
        print("Opponent quality")
        print(
            f"  Avg prior opponent ELO:    diff={opponent_quality['pick_opponent_quality_diff']:+.0f} "
            f"(pick {fmt_elo(opponent_quality['pick_avg_prior_opponent_elo'])} vs "
            f"opp {fmt_elo(opponent_quality['opponent_avg_prior_opponent_elo'])})"
        )
        print(
            f"  Recent 3 opponent ELO:     diff={fmt_signed_elo(opponent_quality['pick_recent_opponent_quality_diff'])} "
            f"(pick {fmt_elo(opponent_quality['pick_recent3_prior_opponent_elo'])} vs "
            f"opp {fmt_elo(opponent_quality['opponent_recent3_prior_opponent_elo'])})"
        )
        print(
            f"  Best win opponent ELO:     diff={fmt_signed_elo(opponent_quality['pick_best_win_quality_diff'])} "
            f"(pick {fmt_elo(opponent_quality['pick_best_win_opponent_elo'])} vs "
            f"opp {fmt_elo(opponent_quality['opponent_best_win_opponent_elo'])})"
        )
        print(
            f"  Prior fight count:         pick {opponent_quality['pick_prior_fight_count']} "
            f"vs opp {opponent_quality['opponent_prior_fight_count']}"
        )
        print(
            f"  Current-vs-peak decline:   diff={fmt_signed_elo(opponent_quality['pick_decline_diff'])} "
            f"(pick {fmt_elo(opponent_quality['pick_current_vs_peak_decline'])} vs "
            f"opp {fmt_elo(opponent_quality['opponent_current_vs_peak_decline'])})"
        )
        print_recent_fights("Pick recent fights", opponent_quality["pick_recent_fights"])
        print_recent_fights("Opponent recent fights", opponent_quality["opponent_recent_fights"])
    trait_delta = packet.get("trait_deltas_v0")
    if trait_delta:
        deltas = trait_delta.get("deltas", {})
        validation = trait_delta.get("validation_notes", {})
        print("Trait deltas v0 (context only)")
        print(f"  {trait_delta['summary']}")
        print(
            f"  Samples: pick fights={trait_delta.get('fight_count')} "
            f"opp fights={trait_delta.get('opponent_fight_count')} "
            f"confidence={fmt_pct(trait_delta.get('trait_confidence'))}"
        )
        for field, label in [
            ("cardio_score_diff", "Cardio / late-fight proxy"),
            ("striking_efficiency_score_diff", "Striking efficiency"),
            ("offensive_control_score_diff", "Offensive control"),
            ("anti_control_score_diff", "Anti-control"),
            ("defensive_exposure_score_diff", "Defensive exposure risk"),
            ("durability_risk_score_diff", "Durability risk"),
            ("grappling_threat_score_diff", "Grappling threat"),
            ("finishing_threat_score_diff", "Finishing threat"),
            ("variance_score_diff", "Variance"),
        ]:
            note = validation.get(field, {})
            status = note.get("status", "exploratory_unvalidated")
            print(f"  {label:<29} {fmt_signed_score(deltas.get(field)):>7}  [{status}]")
        print("  Note: positive ability deltas favor the pick; positive risk deltas mean more risk.")
    print(f"Flags +: {', '.join(packet['flags']['support']) or '-'}")
    print(f"Flags -: {', '.join(packet['flags']['risk']) or '-'}")

    score = packet["pattern_score_v0"]
    print(
        f"Pattern evidence v0: {score['score']}/10 "
        f"({score['support_level']}; {score['action']})"
        + (f" via {score['source_pattern']}" if score["source_pattern"] else "")
    )
    for warning in score["warnings"]:
        print(f"  Score warning: {warning}")

    print("\nMatching patterns (decision support)")
    for pattern in packet["matching_patterns"]["items"]:
        roi = "--" if pattern["roi"] is None else f"{pattern['roi'] * 100:+.1f}%"
        wr = "--" if pattern["win_rate"] is None else f"{pattern['win_rate']:.1%}"
        pending = pattern.get("ungraded_sample_size", 0)
        pending_text = f" + {pending} pending" if pending else ""
        print(
            f"  {pattern['pattern_name']:<30} "
            f"N={pattern['sample_size']:>3} graded{pending_text} "
            f"W-L={pattern['wins']}-{pattern['losses']} WR={wr:>6} ROI={roi:>7}"
            + (f" last={pattern['last_graded_date']}" if pattern.get("last_graded_date") else "")
        )

    print("\nNearest historical examples (qualitative sanity check only)")
    print("  Warning: nearest examples are illustrative only; use aggregate pattern stats for decision support.")
    for row in packet["nearest_historical_examples"]["items"]:
        result = "W" if row["pick_correct"] else "L"
        print(
            f"  {row['date']} {row['fighter1']} vs {row['fighter2']} | "
            f"pick={row['pick']} @{row['pick_odds']} prob={fmt_pct(row['pick_prob'])} "
            f"ELO={row['pick_elo_diff']:+} {result} pnl={row['actual_pnl']:+.2f}"
        )
    print()


def print_source_pattern_expansion(packet: dict[str, Any], rows: list[dict[str, Any]], *, include_pending: bool) -> None:
    score = packet["pattern_score_v0"]
    basis = score["basis"] or {}
    source_pattern = score["source_pattern"]
    graded_count = basis.get("graded_sample_size", basis.get("sample_size", 0))
    pending_count = basis.get("ungraded_sample_size", 0)
    print("=" * 90)
    print(f"SOURCE PATTERN EXPANSION: {source_pattern}")
    print("=" * 90)
    print(
        f"N={graded_count} graded + {pending_count} pending "
        f"(showing {'graded + pending' if include_pending else 'graded only'})"
    )
    print("| # | Date | Fight | Pick | Odds | Prob | Edge | ELO | Result | PnL | Skip reason |")
    print("|---:|---|---|---|---:|---:|---:|---:|---|---:|---|")
    for idx, row in enumerate(rows, 1):
        if row["pick_correct"] is True:
            result = "W"
        elif row["pick_correct"] is False:
            result = "L"
        else:
            result = "pending"
        print(
            f"| {idx} | {row['date']} | {row['fighter1']} vs {row['fighter2']} | "
            f"{row['pick']} | {fmt_num(row['pick_odds'])} | {fmt_pct(row['pick_prob'])} | "
            f"{fmt_signed_pct(row['edge'])} | {fmt_num(row['pick_elo_diff'])} | "
            f"{result} | {fmt_pnl(row['actual_pnl'])} | {row['skip_reason'] or '-'} |"
        )
    print()


def main() -> None:
    args = parse_args()
    if not args.pool.exists():
        raise SystemExit(f"Context pool not found: {args.pool}. Run backtest/build_context_pool.py first.")

    aliases = load_aliases(DEFAULT_ALIAS_SOURCES)
    conn = sqlite3.connect(args.pool)
    conn.row_factory = sqlite3.Row
    try:
        target, candidate_count = find_target(
            conn,
            fighter1=args.fighter1,
            fighter2=args.fighter2,
            date=args.date,
            season=args.season,
            aliases=aliases,
        )
        packet = build_packet(
            conn,
            target=target,
            candidate_count=candidate_count,
            similar_limit=args.similar_limit,
            pool_path=args.pool,
        )
        expansion_rows = []
        if args.expand_source_pattern and not args.json_only:
            source_pattern = packet["pattern_score_v0"]["source_pattern"]
            if source_pattern is None:
                raise SystemExit("No source pattern available to expand.")
            expansion_rows = fetch_pattern_expansion_rows(
                conn,
                pattern_name=source_pattern,
                include_pending=args.include_pending,
            )
    finally:
        conn.close()

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(packet, indent=2, sort_keys=True))
        if not args.json_only:
            print(f"Packet JSON written: {args.json_out}")

    if args.json_only:
        print(json.dumps(packet, indent=2, sort_keys=True))
    else:
        print_summary(packet)
        if args.expand_source_pattern:
            print_source_pattern_expansion(packet, expansion_rows, include_pending=args.include_pending)


if __name__ == "__main__":
    main()
