from __future__ import annotations

import json
import sqlite3
from typing import Any

from backtest.context_packet import (
    build_pattern_score,
    fetch_trait_delta_evidence,
    pattern_payload,
    row_to_dict,
)

RISK_TRAITS = {"durability_risk_score_diff", "defensive_exposure_score_diff", "variance_score_diff"}
TRAIT_SCORE_FIELDS = [
    "experience_score",
    "recent_form_score",
    "cardio_score",
    "durability_risk_score",
    "defensive_exposure_score",
    "offensive_control_score",
    "anti_control_score",
    "scramble_score",
    "striking_pressure_score",
    "striking_efficiency_score",
    "grappling_threat_score",
    "finishing_threat_score",
    "variance_score",
]
QUANT_PROFILE_FIELDS = [
    "fight_count",
    "recent3_win_rate",
    "recent5_win_rate",
    "finish_win_rate",
    "finish_loss_rate",
    "ko_loss_rate",
    "avg_sig_landed_per_min",
    "avg_sig_absorbed_per_min",
    "avg_sig_diff_per_min",
    "avg_control_diff_minutes_per_15",
]
PROFILE_FIELD_SCALES = {
    "fight_count": 8.0,
    "recent3_win_rate": 0.30,
    "recent5_win_rate": 0.30,
    "finish_win_rate": 0.25,
    "finish_loss_rate": 0.20,
    "ko_loss_rate": 0.15,
    "avg_sig_landed_per_min": 1.5,
    "avg_sig_absorbed_per_min": 1.5,
    "avg_sig_diff_per_min": 1.5,
    "avg_control_diff_minutes_per_15": 3.0,
    **{field: 12.0 for field in TRAIT_SCORE_FIELDS},
}
TRAIT_LABELS = {
    "cardio_score_diff": "cardio",
    "durability_risk_score_diff": "durability risk",
    "defensive_exposure_score_diff": "defensive exposure",
    "offensive_control_score_diff": "offensive control",
    "grappling_threat_score_diff": "grappling threat",
    "striking_efficiency_score_diff": "striking efficiency",
    "striking_pressure_score_diff": "striking pressure",
    "anti_control_score_diff": "anti-control",
    "scramble_score_diff": "scramble",
    "finishing_threat_score_diff": "finishing threat",
    "recent_form_score_diff": "recent form",
    "experience_score_diff": "experience",
    "variance_score_diff": "variance risk",
}
TRAIT_ARCHETYPES: dict[str, dict[str, Any]] = {
    "weak_chin_vs_wrestler": {
        "description": (
            "Pick-side control/grappling pressure with the opponent carrying more "
            "durability or defensive-exposure risk."
        ),
        "positive_traits": ["offensive_control_score_diff", "grappling_threat_score_diff"],
        "opponent_risk_traits": ["durability_risk_score_diff", "defensive_exposure_score_diff"],
    },
    "wrestler_vs_striker": {
        "description": (
            "Pick-side wrestling/control threat against an opponent whose best comparative "
            "case is striking efficiency or pressure."
        ),
        "positive_traits": ["offensive_control_score_diff", "grappling_threat_score_diff"],
        "opponent_risk_traits": ["striking_efficiency_score_diff", "striking_pressure_score_diff"],
    },
    "grappling_control_vs_striking_efficiency": {
        "description": (
            "Pick-side grappling and control advantages contrasted with opponent-side "
            "striking efficiency."
        ),
        "positive_traits": ["grappling_threat_score_diff", "offensive_control_score_diff"],
        "opponent_risk_traits": ["striking_efficiency_score_diff"],
    },
    "cardio_pressure": {
        "description": "Pick-side cardio plus pressure/control advantages.",
        "positive_traits": ["cardio_score_diff", "offensive_control_score_diff", "striking_pressure_score_diff"],
        "opponent_risk_traits": [],
    },
    "clean_striker_vs_hittable_opponent": {
        "description": "Pick-side striking efficiency with the opponent showing more exposure risk.",
        "positive_traits": ["striking_efficiency_score_diff", "striking_pressure_score_diff"],
        "opponent_risk_traits": ["defensive_exposure_score_diff"],
    },
}


def _locator(row: dict[str, Any]) -> dict[str, Any]:
    locator = {
        "fight_pool_id": row["id"],
        "date": row["date"],
        "season": row["season"],
        "fighter1": row["fighter1"],
        "fighter2": row["fighter2"],
        "fight": f"{row['fighter1']} vs {row['fighter2']}",
        "pick": row["pick"],
    }
    if row.get("source_table") == "dynamic_synthetic_target":
        locator["target_type"] = "dynamic_synthetic"
    return locator


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _pct_points(value: float | None) -> float | None:
    return None if value is None else round(float(value) * 100, 1)


def _historical_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [row for row in rows if row.get("pick_correct") is not None]
    wins = sum(1 for row in graded if row.get("pick_correct") is True)
    losses = sum(1 for row in graded if row.get("pick_correct") is False)
    profit = round(sum(float(row.get("actual_pnl") or 0.0) for row in graded), 4)
    return {
        "example_count": len(rows),
        "graded_example_count": len(graded),
        "wins": wins,
        "losses": losses,
        "win_rate": _round(wins / len(graded)) if graded else None,
        "profit": profit if graded else None,
        "roi": _round(profit / len(graded)) if graded else None,
        "date_range": None if not rows else {
            "from": min(str(row["date"]) for row in rows),
            "to": max(str(row["date"]) for row in rows),
        },
    }


def _row_provenance(
    row: dict[str, Any],
    *,
    source_table: str = "backtest_fight_pool",
    source_key: str | None = None,
    trait_source_table: str | None = None,
    trait_source_key: str | None = None,
) -> dict[str, Any]:
    odds = {
        "odds_source_file": row.get("odds_source_file"),
        "odds_source_line": row.get("odds_source_line"),
        "odds_source_type": row.get("odds_source_type"),
        "source_event_id": row.get("source_event_id"),
        "bookmaker": row.get("bookmaker"),
        "odds_timestamp": row.get("odds_timestamp"),
    }
    return {
        "source_table": row.get("source_table") or source_table,
        "source_key": source_key or str(row["id"]),
        "source_row_key": row.get("source_row_key"),
        "source_results": row.get("source_results"),
        "odds": {key: value for key, value in odds.items() if value is not None},
        "trait_evidence": None if trait_source_table is None else {
            "source_table": trait_source_table,
            "source_key": trait_source_key,
        },
    }


def _load_rows(
    conn: sqlite3.Connection,
    *,
    include_pending: bool,
    require_elo: bool = False,
    require_market: bool = False,
) -> list[dict[str, Any]]:
    clauses = []
    if not include_pending:
        clauses.append("pick_correct IS NOT NULL")
    if require_elo:
        clauses.extend(["join_status = 'matched'", "pick_elo_diff IS NOT NULL"])
    if require_market:
        clauses.append("pick_odds IS NOT NULL")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(f"SELECT * FROM backtest_fight_pool {where} ORDER BY date DESC, id DESC").fetchall()
    return [row_to_dict(row) for row in rows]


def _elo_match_reason(row: dict[str, Any], *, elo_gap: float, pick_prob: float | None, edge: float | None) -> str:
    reasons = [f"ELO gap {row['pick_elo_diff']:+.0f} vs target {elo_gap:+.0f} ({abs(row['pick_elo_diff'] - elo_gap):.0f}-point diff)"]
    if pick_prob is not None:
        reasons.append(f"confidence within {abs(row['pick_prob'] - pick_prob) * 100:.1f} pts")
    if edge is not None and row.get("edge") is not None:
        reasons.append(f"edge within {abs(row['edge'] - edge) * 100:.1f} pts")
    return "; ".join(reasons)


def _market_match_reason(
    row: dict[str, Any],
    *,
    pick_odds: float,
    market_implied_prob: float | None,
    edge: float | None,
    pick_prob: float | None,
) -> str:
    reasons = [f"price {row['pick_odds']} vs target {pick_odds:.0f} ({abs(row['pick_odds'] - pick_odds):.0f}-point diff)"]
    if market_implied_prob is not None and row.get("market_implied_prob") is not None:
        reasons.append(
            f"market implied probability within {abs(row['market_implied_prob'] - market_implied_prob) * 100:.1f} pts"
        )
    if pick_prob is not None:
        reasons.append(f"model confidence within {abs(row['pick_prob'] - pick_prob) * 100:.1f} pts")
    if edge is not None and row.get("edge") is not None:
        reasons.append(f"edge within {abs(row['edge'] - edge) * 100:.1f} pts")
    return "; ".join(reasons)


def _trait_value_text(field: str, value: float) -> str:
    label = TRAIT_LABELS.get(field, field)
    if field in RISK_TRAITS and value <= 0:
        return f"opponent {label} higher by {abs(value):.1f}"
    return f"{label} {value:+.1f}"


def _signature_from_payload(
    payload: dict[str, Any],
    *,
    min_trait_gap: float,
) -> dict[str, Any]:
    deltas = payload.get("deltas", {})
    positive_traits = sorted(
        [
            field
            for field, value in deltas.items()
            if field not in RISK_TRAITS and value is not None and float(value) >= min_trait_gap
        ],
        key=lambda field: float(deltas[field]),
        reverse=True,
    )[:3]
    opponent_risk_traits = sorted(
        [
            field
            for field, value in deltas.items()
            if field in RISK_TRAITS and value is not None and float(value) <= -min_trait_gap
        ],
        key=lambda field: float(deltas[field]),
    )[:2]
    if not positive_traits and not opponent_risk_traits:
        raise ValueError("Target fight does not expose a strong enough trait signature for historical matching.")
    description_parts = [_trait_value_text(field, float(deltas[field])) for field in positive_traits + opponent_risk_traits]
    return {
        "type": "target_signature",
        "description": ", ".join(description_parts),
        "positive_traits": positive_traits,
        "opponent_risk_traits": opponent_risk_traits,
    }


def _trait_signature(
    conn: sqlite3.Connection,
    *,
    target: dict[str, Any] | None,
    target_payload: dict[str, Any] | None,
    archetype: str | None,
    min_trait_gap: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if archetype is not None:
        config = TRAIT_ARCHETYPES.get(archetype)
        if config is None:
            supported = ", ".join(sorted(TRAIT_ARCHETYPES))
            raise ValueError(f"Unknown archetype '{archetype}'. Supported archetypes: {supported}")
        return {
            "type": "archetype",
            "archetype": archetype,
            **config,
        }, None
    if target_payload is not None:
        return _signature_from_payload(target_payload, min_trait_gap=min_trait_gap), target_payload
    if target is None:
        raise ValueError("Pass a target fight or an archetype.")
    payload = fetch_trait_delta_evidence(conn, target)
    if payload is None:
        raise ValueError("No trait delta evidence found for the requested target fight.")
    return _signature_from_payload(payload, min_trait_gap=min_trait_gap), payload


def _trait_examples(conn: sqlite3.Connection, *, include_pending: bool) -> list[tuple[dict[str, Any], dict[str, Any], sqlite3.Row]]:
    pending_clause = "" if include_pending else "AND p.pick_correct IS NOT NULL"
    rows = conn.execute(
        f"""
        SELECT p.*, e.data_json, e.source_table AS trait_source_table, e.source_key AS trait_source_key
        FROM evidence_items e
        JOIN backtest_fight_pool p ON p.id = e.fight_pool_id
        WHERE e.evidence_type = 'trait_delta'
        {pending_clause}
        ORDER BY p.date DESC, p.id DESC
        """
    ).fetchall()
    examples = []
    for row in rows:
        pool_row = row_to_dict(row)
        trait_payload = json.loads(row["data_json"])
        examples.append((pool_row, trait_payload, row))
    return examples


def find_similar_elo_gap_fights(
    conn: sqlite3.Connection,
    *,
    target: dict[str, Any] | None = None,
    elo_gap: float | None = None,
    pick_prob: float | None = None,
    edge: float | None = None,
    limit: int = 5,
    include_pending: bool = False,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if target is not None:
        elo_gap = float(target.get("pick_elo_diff")) if elo_gap is None and target.get("pick_elo_diff") is not None else elo_gap
        pick_prob = target.get("pick_prob") if pick_prob is None else pick_prob
        edge = target.get("edge") if edge is None else edge
    if elo_gap is None:
        raise ValueError("elo_gap is required when no target fight is provided.")

    candidates = []
    for row in _load_rows(conn, include_pending=include_pending, require_elo=True):
        if target is not None and row["id"] == target["id"]:
            continue
        distance = abs(float(row["pick_elo_diff"]) - float(elo_gap)) / 100.0
        if pick_prob is not None:
            distance += abs(float(row["pick_prob"]) - float(pick_prob)) / 0.05
        if edge is not None and row.get("edge") is not None:
            distance += abs(float(row["edge"]) - float(edge)) / 0.05
        candidates.append((distance, row))

    ranked = [row for _, row in sorted(candidates, key=lambda item: (item[0], item[1]["date"], item[1]["id"]), reverse=False)[:limit]]
    examples = [
        {
            **_locator(row),
            "winner": row.get("winner"),
            "pick_correct": row.get("pick_correct"),
            "actual_pnl": row.get("actual_pnl"),
            "match_reason": _elo_match_reason(row, elo_gap=float(elo_gap), pick_prob=pick_prob, edge=edge),
            "metrics": {
                "pick_elo_diff": row.get("pick_elo_diff"),
                "pick_prob": row.get("pick_prob"),
                "edge": row.get("edge"),
            },
            "provenance": _row_provenance(row),
        }
        for row in ranked
    ]
    return {
        "query": {
            "elo_gap": _round(float(elo_gap), 2),
            "pick_prob": _round(pick_prob),
            "edge": _round(edge),
            "from_target": None if target is None else _locator(target),
        },
        "summary": {
            **_historical_summary(ranked),
            "average_pick_elo_diff": _round(_mean([float(row["pick_elo_diff"]) for row in ranked if row.get("pick_elo_diff") is not None])),
            "average_pick_prob": _round(_mean([float(row["pick_prob"]) for row in ranked if row.get("pick_prob") is not None])),
            "average_edge": _round(_mean([float(row["edge"]) for row in ranked if row.get("edge") is not None])),
        },
        "examples": examples,
    }


def find_similar_market_fights(
    conn: sqlite3.Connection,
    *,
    target: dict[str, Any] | None = None,
    pick_odds: float | None = None,
    market_implied_prob: float | None = None,
    edge: float | None = None,
    pick_prob: float | None = None,
    limit: int = 5,
    include_pending: bool = False,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if target is not None:
        pick_odds = float(target.get("pick_odds")) if pick_odds is None and target.get("pick_odds") is not None else pick_odds
        market_implied_prob = target.get("market_implied_prob") if market_implied_prob is None else market_implied_prob
        edge = target.get("edge") if edge is None else edge
        pick_prob = target.get("pick_prob") if pick_prob is None else pick_prob
    if pick_odds is None:
        raise ValueError("pick_odds is required when no target fight is provided.")

    candidates = []
    for row in _load_rows(conn, include_pending=include_pending, require_market=True):
        if target is not None and row["id"] == target["id"]:
            continue
        distance = abs(float(row["pick_odds"]) - float(pick_odds)) / 50.0
        if market_implied_prob is not None and row.get("market_implied_prob") is not None:
            distance += abs(float(row["market_implied_prob"]) - float(market_implied_prob)) / 0.05
        if pick_prob is not None:
            distance += abs(float(row["pick_prob"]) - float(pick_prob)) / 0.05
        if edge is not None and row.get("edge") is not None:
            distance += abs(float(row["edge"]) - float(edge)) / 0.05
        candidates.append((distance, row))

    ranked = [row for _, row in sorted(candidates, key=lambda item: (item[0], item[1]["date"], item[1]["id"]), reverse=False)[:limit]]
    examples = [
        {
            **_locator(row),
            "winner": row.get("winner"),
            "pick_correct": row.get("pick_correct"),
            "actual_pnl": row.get("actual_pnl"),
            "match_reason": _market_match_reason(
                row,
                pick_odds=float(pick_odds),
                market_implied_prob=market_implied_prob,
                edge=edge,
                pick_prob=pick_prob,
            ),
            "metrics": {
                "pick_odds": row.get("pick_odds"),
                "market_implied_prob": row.get("market_implied_prob"),
                "pick_prob": row.get("pick_prob"),
                "edge": row.get("edge"),
            },
            "provenance": _row_provenance(row),
        }
        for row in ranked
    ]
    return {
        "query": {
            "pick_odds": _round(float(pick_odds), 1),
            "market_implied_prob": _round(market_implied_prob),
            "pick_prob": _round(pick_prob),
            "edge": _round(edge),
            "from_target": None if target is None else _locator(target),
        },
        "summary": {
            **_historical_summary(ranked),
            "average_pick_odds": _round(_mean([float(row["pick_odds"]) for row in ranked if row.get("pick_odds") is not None]), 1),
            "average_market_implied_prob": _round(
                _mean([float(row["market_implied_prob"]) for row in ranked if row.get("market_implied_prob") is not None])
            ),
            "average_edge": _round(_mean([float(row["edge"]) for row in ranked if row.get("edge") is not None])),
        },
        "examples": examples,
    }


def find_trait_matchup_examples(
    conn: sqlite3.Connection,
    *,
    target: dict[str, Any] | None = None,
    target_payload: dict[str, Any] | None = None,
    archetype: str | None = None,
    limit: int = 5,
    include_pending: bool = False,
    min_trait_gap: float = 8.0,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    signature, target_payload = _trait_signature(
        conn,
        target=target,
        target_payload=target_payload,
        archetype=archetype,
        min_trait_gap=min_trait_gap,
    )
    positive_traits = signature.get("positive_traits", [])
    opponent_risk_traits = signature.get("opponent_risk_traits", [])

    scored = []
    for row, payload, source in _trait_examples(conn, include_pending=include_pending):
        if target is not None and row["id"] == target["id"]:
            continue
        deltas = payload.get("deltas", {})
        positive_hits = [field for field in positive_traits if deltas.get(field) is not None and float(deltas[field]) >= min_trait_gap]
        risk_hits = [field for field in opponent_risk_traits if deltas.get(field) is not None and float(deltas[field]) <= -min_trait_gap]
        total_hits = len(positive_hits) + len(risk_hits)
        if total_hits == 0:
            continue
        distance = 0.0
        if target_payload is not None:
            target_deltas = target_payload.get("deltas", {})
            for field in positive_hits + risk_hits:
                distance += abs(float(deltas[field]) - float(target_deltas.get(field) or 0.0))
        scored.append((total_hits, distance, row, payload, source, positive_hits, risk_hits))

    ranked = sorted(scored, key=lambda item: (-item[0], item[1], item[2]["date"], item[2]["id"]))[:limit]
    rows = [item[2] for item in ranked]
    examples = []
    for _, _, row, payload, source, positive_hits, risk_hits in ranked:
        deltas = payload.get("deltas", {})
        matched = [_trait_value_text(field, float(deltas[field])) for field in positive_hits + risk_hits]
        examples.append(
            {
                **_locator(row),
                "winner": row.get("winner"),
                "pick_correct": row.get("pick_correct"),
                "actual_pnl": row.get("actual_pnl"),
                "match_reason": "; ".join(matched),
                "trait_summary": {field: deltas.get(field) for field in positive_hits + risk_hits},
                "provenance": _row_provenance(
                    row,
                    trait_source_table=source["trait_source_table"],
                    trait_source_key=source["trait_source_key"],
                ),
            }
        )

    aggregate_fields = positive_traits + opponent_risk_traits
    return {
        "query": {
            "signature": signature,
            "min_trait_gap": min_trait_gap,
            "from_target": None if target is None else _locator(target),
        },
        "summary": {
            **_historical_summary(rows),
            "average_trait_deltas": {
                field: _round(
                    _mean(
                        [
                            float(item[3].get("deltas", {}).get(field))
                            for item in ranked
                            if item[3].get("deltas", {}).get(field) is not None
                        ]
                    ),
                    2,
                )
                for field in aggregate_fields
            },
        },
        "examples": examples,
    }


def _snapshot_profile_values(target_snapshot: dict[str, Any]) -> dict[str, float]:
    qualitative = target_snapshot.get("qualitative") or {}
    stats = target_snapshot.get("stats") or {}
    record = target_snapshot.get("record") or {}
    values: dict[str, float] = {}
    for field in TRAIT_SCORE_FIELDS:
        value = qualitative.get(field)
        if value is not None:
            values[field] = float(value)
    for field in QUANT_PROFILE_FIELDS:
        value = qualitative.get(field)
        if value is None and field == "fight_count":
            value = record.get("fight_count_as_of")
        if value is None and field == "avg_sig_landed_per_min":
            value = stats.get("sig_strikes_landed_per_min")
        if value is None and field == "avg_sig_absorbed_per_min":
            value = stats.get("sig_strikes_absorbed_per_min")
        if value is not None:
            values[field] = float(value)
    return values


def _latest_trait_snapshot_rows(
    conn: sqlite3.Connection,
    *,
    as_of_date: str | None,
    min_fight_count: int,
) -> list[dict[str, Any]]:
    clauses = ["fight_count >= ?"]
    params: list[Any] = [min_fight_count]
    if as_of_date is not None:
        clauses.append("as_of_date <= ?")
        params.append(as_of_date)
    rows = conn.execute(
        f"""
        SELECT *
        FROM fighter_trait_snapshots
        WHERE {' AND '.join(clauses)}
        ORDER BY fighter_id, as_of_date DESC, snapshot_id DESC
        """,
        params,
    ).fetchall()
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        data = dict(row)
        fighter_id = int(data["fighter_id"])
        if fighter_id not in latest:
            latest[fighter_id] = data
    return list(latest.values())


def _profile_distance(
    target_values: dict[str, float],
    row: dict[str, Any],
) -> tuple[float, list[str], dict[str, Any]]:
    distance = 0.0
    matched_fields: list[str] = []
    deltas: dict[str, Any] = {}
    for field, target_value in target_values.items():
        row_value = row.get(field)
        if row_value is None:
            continue
        diff = float(row_value) - target_value
        scale = PROFILE_FIELD_SCALES.get(field, 10.0)
        distance += abs(diff) / scale
        matched_fields.append(field)
        deltas[field] = {
            "target": _round(target_value, 3),
            "historical": _round(float(row_value), 3),
            "delta": _round(diff, 3),
        }
    return distance, matched_fields, deltas


def find_similar_fighter_profiles(
    conn: sqlite3.Connection,
    *,
    target_snapshot: dict[str, Any],
    as_of_date: str | None = None,
    limit: int = 5,
    min_fight_count: int = 3,
) -> dict[str, Any]:
    """Return historical fighter-state neighbors from trait/quant snapshots."""
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if min_fight_count < 0:
        raise ValueError("min_fight_count must be non-negative.")
    if not target_snapshot.get("resolved"):
        raise ValueError("target_snapshot must be a resolved fighter snapshot.")

    target_values = _snapshot_profile_values(target_snapshot)
    if not target_values:
        raise ValueError("No qualitative or quantitative profile fields are available for the target fighter.")

    identity = target_snapshot.get("identity") or {}
    target_main_id = identity.get("main_fighter_id")
    scored = []
    for row in _latest_trait_snapshot_rows(conn, as_of_date=as_of_date, min_fight_count=min_fight_count):
        if target_main_id is not None and row.get("fighter_id") == target_main_id:
            continue
        distance, matched_fields, deltas = _profile_distance(target_values, row)
        if not matched_fields:
            continue
        scored.append((distance, row, matched_fields, deltas))

    ranked = sorted(scored, key=lambda item: (item[0], item[1]["as_of_date"], item[1]["fighter_name"]))[:limit]
    examples = []
    for distance, row, matched_fields, deltas in ranked:
        trait_fields = [field for field in matched_fields if field in TRAIT_SCORE_FIELDS]
        quantitative_fields = [field for field in matched_fields if field in QUANT_PROFILE_FIELDS]
        examples.append(
            {
                "fighter_name": row["fighter_name"],
                "as_of_date": row["as_of_date"],
                "opponent_context": {
                    "main_fight_id": row.get("main_fight_id"),
                    "opponent_name": row.get("opponent_name"),
                },
                "similarity_score": _round(1 / (1 + distance), 4),
                "distance": _round(distance, 4),
                "matched_fields": {
                    "qualitative": trait_fields,
                    "quantitative": quantitative_fields,
                },
                "metric_deltas": deltas,
                "profile": {
                    "fight_count": row.get("fight_count"),
                    "recent5_win_rate": row.get("recent5_win_rate"),
                    "ko_loss_rate": row.get("ko_loss_rate"),
                    "avg_sig_diff_per_min": row.get("avg_sig_diff_per_min"),
                    "avg_control_diff_minutes_per_15": row.get("avg_control_diff_minutes_per_15"),
                    "trait_confidence": row.get("trait_confidence"),
                    "trait_version": row.get("trait_version"),
                },
                "provenance": {
                    "source_table": "fighter_trait_snapshots",
                    "source_key": str(row["snapshot_id"]),
                },
            }
        )

    return {
        "query": {
            "fighter_name": (identity.get("resolved_name") or target_snapshot.get("query_name")),
            "as_of_date": as_of_date,
            "min_fight_count": min_fight_count,
            "matched_field_count": len(target_values),
            "source": "synthetic_fighter_profile_from_snapshot",
        },
        "summary": {
            "example_count": len(examples),
            "average_similarity_score": _round(
                _mean([example["similarity_score"] for example in examples if example.get("similarity_score") is not None])
            ),
            "evidence_lanes": ["qualitative_trait_scores", "quantitative_performance_stats"],
        },
        "examples": examples,
    }


def _filters_reason(filters: dict[str, Any]) -> str:
    parts = []
    for key, value in filters.items():
        if isinstance(value, list) and len(value) == 2:
            parts.append(f"{key} between {value[0]:.2f} and {value[1]:.2f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def get_historical_pattern_summary(conn: sqlite3.Connection, *, target: dict[str, Any]) -> dict[str, Any]:
    patterns = pattern_payload(conn, target)
    score = build_pattern_score(target, patterns)
    source_pattern = next((pattern for pattern in patterns if pattern["pattern_name"] == score["source_pattern"]), None)
    return {
        "target": _locator(target),
        "summary": {
            "action": score["action"],
            "support_level": score["support_level"],
            "source_pattern": score["source_pattern"],
            "sample_size": None if score["basis"] is None else score["basis"]["sample_size"],
            "win_rate": None if score["basis"] is None else score["basis"]["win_rate"],
            "roi": None if score["basis"] is None else score["basis"]["roi"],
            "match_reason": None if source_pattern is None else source_pattern["description"],
        },
        "patterns": [
            {
                "pattern_name": pattern["pattern_name"],
                "description": pattern["description"],
                "match_reason": _filters_reason(pattern["filters"]),
                "stats": {
                    "sample_size": pattern["sample_size"],
                    "graded_sample_size": pattern["graded_sample_size"],
                    "ungraded_sample_size": pattern["ungraded_sample_size"],
                    "wins": pattern["wins"],
                    "losses": pattern["losses"],
                    "win_rate": pattern["win_rate"],
                    "profit": pattern["profit"],
                    "roi": pattern["roi"],
                    "avg_confidence": pattern["avg_confidence"],
                    "avg_edge": pattern["avg_edge"],
                    "avg_elo_diff": pattern["avg_elo_diff"],
                    "last_graded_date": pattern["last_graded_date"],
                },
                "provenance": {
                    "source_table": "pattern_stats",
                    "source_key": pattern["pattern_name"],
                },
            }
            for pattern in patterns
        ],
        "warnings": score["warnings"],
        "provenance": _row_provenance(target),
    }
