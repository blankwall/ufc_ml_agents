#!/usr/bin/env python3
"""
UFC context MCP server.

Exposes read-only analysis tools over the generated context sidecars, selected
SQLite databases, and a small set of whitelisted backtest/document files.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mcp.server.fastmcp import FastMCP

from backtest.context_agent_review import build_llm_review, build_review, load_evidence
from backtest.context_packet import (
    DEFAULT_POOL,
    build_flags,
    build_packet,
    build_pattern_score,
    fetch_similar_rows,
    fetch_trait_delta_evidence,
    find_target,
    pattern_payload,
    support_level,
)
from backtest.historical_evidence import (
    TRAIT_LABELS,
    find_similar_elo_gap_fights as build_similar_elo_gap_fights,
    find_similar_fighter_profiles as build_similar_fighter_profiles,
    find_similar_market_fights as build_similar_market_fights,
    find_trait_matchup_examples as build_trait_matchup_examples,
    get_historical_pattern_summary as build_historical_pattern_summary,
)
from backtest.elo_analysis import DEFAULT_ALIAS_SOURCES, load_aliases, normalize_name
from fastapi_app.services.fighter_snapshot import build_fighter_snapshot

from backtest.validate_combined_evidence import (
    RULES,
    build_rule_rows,
    enrich_with_main_db,
    fetch_rows as fetch_combined_rows,
    matching_rows_for_rule,
)
from mcp_server.fight_init import (
    get_deterministic_signal_filter as build_deterministic_signal_filter,
    get_elo_market_signal as build_elo_market_signal,
    init_fight_analysis as build_init_fight_analysis,
)

DEFAULT_CONTEXT_POOL = DEFAULT_POOL
DEFAULT_TRAITS_DB = ROOT_DIR / "data" / "enrichment" / "trait_snapshots.sqlite"
DEFAULT_SERGEY_DB = ROOT_DIR / "data" / "enrichment" / "sergey_sidecar.sqlite"
FRAGILITY_CASES_FILE = ROOT_DIR / "analysis" / "fragility_cases.jsonl"
SQLDatabase = Literal["context_pool", "trait_snapshots", "sergey_sidecar", "main"]

mcp = FastMCP("ufc-context-analysis", json_response=True)


def main_db_path() -> Path:
    for candidate in (ROOT_DIR / "data" / "ufc_database.db", ROOT_DIR / "ufc_database.db"):
        if candidate.exists():
            return candidate
    return ROOT_DIR / "data" / "ufc_database.db"


DATABASES: dict[SQLDatabase, Path] = {
    "context_pool": DEFAULT_CONTEXT_POOL,
    "trait_snapshots": DEFAULT_TRAITS_DB,
    "sergey_sidecar": DEFAULT_SERGEY_DB,
    "main": main_db_path(),
}

WHITELISTED_FILE_ROOTS = [
    ROOT_DIR / "backtest",
    ROOT_DIR / "docs",
    ROOT_DIR / ".claude" / "skills",
]
WHITELISTED_FILES = {
    ROOT_DIR / "README.md",
    ROOT_DIR / "embeding_sergey.md",
}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def resolve_database_path(database: SQLDatabase) -> Path:
    path = DATABASES[database]
    if path.exists():
        return path
    raise FileNotFoundError(f"Database not found for {database}: {path}")


def readonly_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_read_only_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise ValueError("Query must not be empty.")
    if ";" in normalized.rstrip(";"):
        raise ValueError("Only a single read-only SQL statement is allowed.")

    first_token = normalized.lstrip().split(None, 1)[0].lower()
    if first_token not in {"select", "with", "explain"}:
        raise ValueError("Only read-only SELECT/WITH/EXPLAIN queries are allowed.")
    return normalized


def serialize_rows(rows: list[sqlite3.Row], *, limit: int) -> dict[str, Any]:
    serialized = [dict(row) for row in rows[:limit]]
    columns = list(rows[0].keys()) if rows else []
    return {
        "columns": columns,
        "rows": serialized,
        "returned_rows": len(serialized),
        "truncated": len(rows) > limit,
    }


def _load_fragility_cases() -> list[dict[str, Any]]:
    if not FRAGILITY_CASES_FILE.exists():
        return []

    cases: list[dict[str, Any]] = []
    with FRAGILITY_CASES_FILE.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {display_path(FRAGILITY_CASES_FILE)} line {line_number}: {exc}") from exc
            if isinstance(payload, dict):
                payload.setdefault("_line_number", line_number)
                cases.append(payload)
    return cases


def _case_text_blob(case: dict[str, Any]) -> str:
    return json.dumps(case, sort_keys=True, default=str).lower()


def _case_fighters(case: dict[str, Any]) -> set[str]:
    fighters = case.get("fighters") or {}
    return {
        normalize_name(str(value))
        for value in fighters.values()
        if value is not None and str(value).strip()
    }


def _case_matches_date_range(
    case: dict[str, Any],
    *,
    min_date: str | None,
    max_date: str | None,
) -> bool:
    event_date = _normalize_lookup_date(case.get("event_date"))
    if event_date is None:
        return min_date is None and max_date is None
    if min_date is not None and event_date < min_date:
        return False
    if max_date is not None and event_date > max_date:
        return False
    return True


@lru_cache(maxsize=1)
def aliases() -> dict[str, str]:
    return load_aliases(DEFAULT_ALIAS_SOURCES)


def resolve_fight_pool_id(
    conn: sqlite3.Connection,
    *,
    fight_pool_id: int | None,
    fighter1: str | None,
    fighter2: str | None,
    date: str | None,
    season: int | None,
) -> int:
    if fight_pool_id is not None:
        return fight_pool_id
    if not fighter1 or not fighter2:
        raise ValueError("Pass fight_pool_id or both fighter1 and fighter2.")
    try:
        target, _ = find_target(
            conn,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
            season=season,
            aliases=aliases(),
        )
    except SystemExit as exc:
        raise ValueError(str(exc)) from None
    return int(target["id"])


def resolve_target_row(
    conn: sqlite3.Connection,
    *,
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
) -> dict[str, Any]:
    resolved_id = resolve_fight_pool_id(
        conn,
        fight_pool_id=fight_pool_id,
        fighter1=fighter1,
        fighter2=fighter2,
        date=date,
        season=season,
    )
    row = conn.execute(
        "SELECT * FROM backtest_fight_pool WHERE id = ?",
        (resolved_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"fight_pool_id not found: {resolved_id}")
    return dict(row)


def fight_locator_payload(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "fight_pool_id": target["id"],
        "date": target["date"],
        "season": target["season"],
        "fighter1": target["fighter1"],
        "fighter2": target["fighter2"],
        "fight": f"{target['fighter1']} vs {target['fighter2']}",
        "pick": target["pick"],
        "source_row_key": target.get("source_row_key"),
    }


def _elo_implied_probability(elo_diff: float | None) -> float | None:
    if elo_diff is None:
        return None
    return round(1 / (1 + 10 ** (-float(elo_diff) / 400)), 4)


def _metric_delta_payload(analysis: dict[str, Any]) -> dict[str, Any] | None:
    prediction = analysis.get("prediction") or {}
    fighters = analysis.get("fighters") or {}
    pick = prediction.get("pick") or {}
    pick_slot = pick.get("slot")
    if pick_slot not in {"fighter1", "fighter2"}:
        return None
    opponent_slot = "fighter2" if pick_slot == "fighter1" else "fighter1"
    pick_snapshot = fighters.get(pick_slot) or {}
    opponent_snapshot = fighters.get(opponent_slot) or {}
    pick_quality = pick_snapshot.get("qualitative") or {}
    opponent_quality = opponent_snapshot.get("qualitative") or {}
    if not pick_quality.get("available") or not opponent_quality.get("available"):
        return None

    deltas: dict[str, Any] = {}
    for diff_field in TRAIT_LABELS:
        base_field = diff_field.removesuffix("_diff")
        pick_value = pick_quality.get(base_field)
        opponent_value = opponent_quality.get(base_field)
        deltas[diff_field] = None if pick_value is None or opponent_value is None else float(pick_value) - float(opponent_value)

    return {
        "trait_version": pick_quality.get("trait_version") or opponent_quality.get("trait_version"),
        "fighter_name": pick.get("fighter_name"),
        "opponent_name": (opponent_snapshot.get("identity") or {}).get("resolved_name"),
        "fight_count": pick_quality.get("fight_count"),
        "opponent_fight_count": opponent_quality.get("fight_count"),
        "trait_confidence": pick_quality.get("trait_confidence"),
        "opponent_trait_confidence": opponent_quality.get("trait_confidence"),
        "deltas": deltas,
        "validation_notes": {
            field: {"status": "dynamic_snapshot_delta"}
            for field in deltas
        },
        "interpretation_note": (
            "Synthetic trait deltas were derived from the latest point-in-time fighter snapshots. "
            "Positive ability-score deltas favor the model pick; positive risk-score deltas mean the pick carries more of that risk."
        ),
    }


def _dynamic_synthetic_target(
    *,
    fighter1: str,
    fighter2: str,
    date: str | None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    analysis = build_init_fight_analysis(
        fighter1=fighter1,
        fighter2=fighter2,
        fight_date=date,
        fighter1_odds=fighter1_odds,
        fighter2_odds=fighter2_odds,
    )
    if analysis.get("status") != "ok":
        raise ValueError(f"Dynamic fight init failed: {analysis.get('validation')}")

    prediction = analysis["prediction"]
    market = analysis["market"]
    fighters = analysis["fighters"]
    resolution = analysis["resolution"]
    pick = prediction["pick"]
    pick_slot = pick["slot"]
    opponent_slot = "fighter2" if pick_slot == "fighter1" else "fighter1"
    pick_snapshot = fighters[pick_slot] or {}
    opponent_snapshot = fighters[opponent_slot] or {}
    pick_elo = (pick_snapshot.get("elo") or {}).get("elo_current")
    opponent_elo = (opponent_snapshot.get("elo") or {}).get("elo_current")
    pick_elo_diff = None if pick_elo is None or opponent_elo is None else pick_elo - opponent_elo
    elo_prob = _elo_implied_probability(pick_elo_diff)
    model_minus_elo = None if elo_prob is None else round(pick["probability"] - elo_prob, 4)
    market_minus_elo = None if elo_prob is None or pick.get("market_probability") is None else round(pick["market_probability"] - elo_prob, 4)
    parsed_date = (resolution.get("fight_date") or {}).get("parsed") or date
    season = int(parsed_date[:4]) if parsed_date and len(parsed_date) >= 4 and parsed_date[:4].isdigit() else None
    f1_name = (resolution["fighter1"].get("resolved_name") or fighter1)
    f2_name = (resolution["fighter2"].get("resolved_name") or fighter2)

    target = {
        "id": f"dynamic:{f1_name}:{f2_name}:{parsed_date or 'latest'}",
        "source_table": "dynamic_synthetic_target",
        "season": season,
        "source_results": "dynamic_init",
        "source_row_key": None,
        "date": parsed_date or "dynamic-latest",
        "fighter1": f1_name,
        "fighter2": f2_name,
        "pick": pick["fighter_name"],
        "winner": None,
        "pick_prob": pick["probability"],
        "pick_odds": market["odds"].get(pick_slot),
        "pick_correct": None,
        "actual_pnl": None,
        "bet": None,
        "skip_reason": "dynamic_synthetic_target_no_config_decision",
        "female": prediction.get("fighter_metadata", {}).get("is_wmma"),
        "edge": pick.get("edge"),
        "join_status": "matched" if pick_elo_diff is not None else "dynamic_missing_elo",
        "join_method": "dynamic_fighter_snapshot",
        "fighter1_elo": (fighters["fighter1"].get("elo") or {}).get("elo_current") if fighters.get("fighter1") else None,
        "fighter2_elo": (fighters["fighter2"].get("elo") or {}).get("elo_current") if fighters.get("fighter2") else None,
        "pick_elo": pick_elo,
        "opponent_elo": opponent_elo,
        "pick_elo_diff": pick_elo_diff,
        "abs_elo_diff": None if pick_elo_diff is None else abs(pick_elo_diff),
        "model_agrees_with_elo": None if pick_elo_diff is None else pick_elo_diff > 0,
        "pick_prior_fight_count": (pick_snapshot.get("record") or {}).get("fight_count_as_of"),
        "opponent_prior_fight_count": (opponent_snapshot.get("record") or {}).get("fight_count_as_of"),
        "market_implied_prob": pick.get("market_probability"),
        "elo_implied_prob": elo_prob,
        "model_minus_elo_prob": model_minus_elo,
        "market_minus_elo_prob": market_minus_elo,
        "model_market_elo_triangle": None,
    }
    if model_minus_elo is not None and market_minus_elo is not None:
        if model_minus_elo < 0 and market_minus_elo < 0:
            target["model_market_elo_triangle"] = "model_and_market_under_elo"
        elif model_minus_elo > 0 and market_minus_elo > 0:
            target["model_market_elo_triangle"] = "model_and_market_over_elo"
        elif model_minus_elo >= 0 and market_minus_elo < 0:
            target["model_market_elo_triangle"] = "model_over_market_under_elo"
        else:
            target["model_market_elo_triangle"] = "model_under_market_over_elo"

    return target, _metric_delta_payload(analysis), analysis


def _target_should_use_dynamic_packet(target: dict[str, Any]) -> bool:
    if target.get("pick_correct") is None:
        return True

    target_date = target.get("date")
    if not isinstance(target_date, str):
        return False
    try:
        parsed = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        return False
    return parsed > datetime.now().date()


def _evidence_chain_from_flags(flags: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "support": [
            {"code": code, "source": "context_packet.flags", "evidence_role": "context_signal"}
            for code in flags.get("support", [])
        ],
        "concerns": [
            {"code": code, "source": "context_packet.flags", "evidence_role": "risk_signal"}
            for code in flags.get("risk", [])
        ],
    }


def _build_dynamic_context_packet(
    conn: sqlite3.Connection,
    *,
    target: dict[str, Any],
    dynamic_trait_delta: dict[str, Any] | None,
    analysis: dict[str, Any],
    candidate_count: int,
    similar_limit: int,
    pool_path: Path,
    dynamic_reason: str,
    historical_lookup_error: str | None = None,
) -> dict[str, Any]:
    packet = build_packet(
        conn,
        target=target,
        candidate_count=candidate_count,
        similar_limit=similar_limit,
        pool_path=pool_path,
    )

    patterns = packet["matching_patterns"]["items"]
    packet["packet_type"] = "dynamic_future_fight"
    packet["source"].update(
        {
            "target_source": "init_fight_analysis",
            "historical_pool_role": "evidence_library",
            "exact_context_pool_row": False,
            "dynamic_reason": dynamic_reason,
        }
    )
    if historical_lookup_error is not None:
        packet["source"]["historical_lookup_error"] = historical_lookup_error

    if dynamic_trait_delta is not None:
        packet["trait_deltas_v0"] = dynamic_trait_delta
        packet["flags"] = build_flags(target, patterns, dynamic_trait_delta)
        packet["pattern_score_v0"] = build_pattern_score(target, patterns)

    market = analysis.get("market") or {}
    pricing_context = market.get("pricing_context") or {}
    packet["model_market"].update(
        {
            "edge_type": pricing_context.get("edge_type"),
            "pricing_context_degraded": pricing_context.get("pricing_context_degraded"),
            "market_missing": pricing_context.get("market_missing"),
            "market_provenance": market.get("provenance"),
        }
    )
    packet["request"] = analysis.get("request")
    packet["resolution"] = analysis.get("resolution")
    packet["market"] = market
    packet["model"] = analysis.get("prediction")
    packet["fighters"] = analysis.get("fighters")
    packet["pricing_context"] = pricing_context
    packet["validation"] = analysis.get("validation")
    packet["dynamic_provenance"] = analysis.get("provenance")
    packet["historical_examples"] = packet["nearest_historical_examples"]
    packet["evidence_chain"] = _evidence_chain_from_flags(packet["flags"])
    return packet


def _historical_elo_fighter_neighbors(
    *,
    target_snapshot: dict[str, Any],
    as_of_date: str | None,
    limit: int,
) -> dict[str, Any]:
    target_elo = (target_snapshot.get("elo") or {}).get("elo_current")
    if target_elo is None:
        return {
            "available": False,
            "reason": "Target fighter does not have current ELO in the Sergey sidecar.",
            "examples": [],
        }

    sidecar_path = resolve_database_path("sergey_sidecar")
    conn = readonly_connection(sidecar_path)
    try:
        date_clause = "AND event_date <= ?" if as_of_date else ""
        params: list[Any] = [target_elo]
        if as_of_date:
            params.append(as_of_date)
        params.append(target_elo)
        params.append(limit)
        rows = conn.execute(
            f"""
            WITH fighter_states AS (
                SELECT
                    fight_id,
                    event_date,
                    event_name,
                    fighter_red_id AS fighter_id,
                    fighter_red_name AS fighter_name,
                    fighter_red_elo AS fighter_pre_elo,
                    fighter_blue_name AS opponent_name,
                    fighter_blue_elo AS opponent_pre_elo,
                    winner_id,
                    winner_name,
                    short_method,
                    division
                FROM fights
                WHERE promotion LIKE '%Ultimate Fighting%'
                  AND fighter_red_elo IS NOT NULL
                UNION ALL
                SELECT
                    fight_id,
                    event_date,
                    event_name,
                    fighter_blue_id AS fighter_id,
                    fighter_blue_name AS fighter_name,
                    fighter_blue_elo AS fighter_pre_elo,
                    fighter_red_name AS opponent_name,
                    fighter_red_elo AS opponent_pre_elo,
                    winner_id,
                    winner_name,
                    short_method,
                    division
                FROM fights
                WHERE promotion LIKE '%Ultimate Fighting%'
                  AND fighter_blue_elo IS NOT NULL
            )
            SELECT *,
                   ABS(fighter_pre_elo - ?) AS elo_distance,
                   fighter_pre_elo - opponent_pre_elo AS fight_elo_diff
            FROM fighter_states
            WHERE fighter_pre_elo IS NOT NULL
              {date_clause}
            ORDER BY ABS(fighter_pre_elo - ?), event_date DESC, fight_id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    examples = []
    for row in rows:
        if row["winner_id"] == row["fighter_id"] or (
            row["winner_name"] and normalize_name(row["winner_name"]) == normalize_name(row["fighter_name"])
        ):
            result = "win"
        elif row["winner_name"]:
            result = "loss"
        else:
            result = "unknown"
        examples.append(
            {
                "fighter_name": row["fighter_name"],
                "fight_date": row["event_date"],
                "event_name": row["event_name"],
                "opponent_name": row["opponent_name"],
                "result": result,
                "method": row["short_method"],
                "division": row["division"],
                "fighter_pre_elo": row["fighter_pre_elo"],
                "opponent_pre_elo": row["opponent_pre_elo"],
                "fight_elo_diff": row["fight_elo_diff"],
                "elo_distance": row["elo_distance"],
                "provenance": {
                    "source_table": "sergey_sidecar.fights",
                    "source_key": str(row["fight_id"]),
                },
            }
        )

    return {
        "available": True,
        "target_elo": target_elo,
        "as_of_date": as_of_date,
        "examples": examples,
    }


def allowed_file(path: Path) -> bool:
    if path in WHITELISTED_FILES:
        return True
    return any(root == path or root in path.parents for root in WHITELISTED_FILE_ROOTS)


def resolve_whitelisted_file(relative_path: str) -> Path:
    path = (ROOT_DIR / relative_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {relative_path}")
    if not allowed_file(path):
        raise ValueError(f"File is outside the MCP whitelist: {relative_path}")
    return path


@mcp.tool()
def list_data_sources() -> dict[str, Any]:
    """List MCP-accessible databases, core files, and validation rules."""
    databases = {}
    for name, path in DATABASES.items():
        databases[name] = {
            "path": display_path(path),
            "exists": path.exists(),
        }
    return {
        "databases": databases,
        "core_files": sorted(display_path(path) for path in WHITELISTED_FILES),
        "analysis_files": {
            "fragility_cases": {
                "path": display_path(FRAGILITY_CASES_FILE),
                "exists": FRAGILITY_CASES_FILE.exists(),
            }
        },
        "whitelisted_roots": sorted(display_path(path) for path in WHITELISTED_FILE_ROOTS),
        "combined_evidence_rules": [rule[0] for rule in RULES],
    }


@mcp.tool()
def query_fragility_cases(
    fighter: str | None = None,
    fragility_flag: str | None = None,
    failure_mode: str | None = None,
    text_query: str | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
    limit: int = 10,
    include_full_case: bool = False,
) -> dict[str, Any]:
    """Query curated model fragility cases from analysis/fragility_cases.jsonl."""
    if limit <= 0 or limit > 100:
        raise ValueError("limit must be between 1 and 100.")

    normalized_fighter = normalize_name(fighter) if fighter else None
    normalized_flag = fragility_flag.strip().lower() if fragility_flag else None
    normalized_failure_mode = failure_mode.strip().lower() if failure_mode else None
    normalized_text_query = text_query.strip().lower() if text_query else None
    normalized_min_date = _normalize_lookup_date(min_date)
    normalized_max_date = _normalize_lookup_date(max_date)

    if normalized_min_date and normalized_max_date and normalized_min_date > normalized_max_date:
        raise ValueError("min_date must be <= max_date.")

    all_cases = _load_fragility_cases()
    matches: list[dict[str, Any]] = []
    for case in all_cases:
        match_reasons: list[str] = []
        if normalized_fighter:
            fighter_names = _case_fighters(case)
            if not any(normalized_fighter in name or name in normalized_fighter for name in fighter_names):
                continue
            match_reasons.append("fighter")

        flags = [str(flag).lower() for flag in case.get("fragility_flags", [])]
        if normalized_flag:
            if not any(normalized_flag == flag or normalized_flag in flag for flag in flags):
                continue
            match_reasons.append("fragility_flag")

        failure_tags = [
            str(tag).lower()
            for tag in ((case.get("post_fight") or {}).get("failure_mode_tags") or [])
        ]
        if normalized_failure_mode:
            if not any(normalized_failure_mode == tag or normalized_failure_mode in tag for tag in failure_tags):
                continue
            match_reasons.append("failure_mode")

        if normalized_text_query:
            if normalized_text_query not in _case_text_blob(case):
                continue
            match_reasons.append("text_query")

        if not _case_matches_date_range(case, min_date=normalized_min_date, max_date=normalized_max_date):
            continue
        if normalized_min_date or normalized_max_date:
            match_reasons.append("date_range")

        if include_full_case:
            payload = dict(case)
        else:
            pre_fight = case.get("pre_fight") or {}
            post_fight = case.get("post_fight") or {}
            payload = {
                "fight_id": case.get("fight_id"),
                "event_date": case.get("event_date"),
                "event_name": case.get("event_name"),
                "weight_class": case.get("weight_class"),
                "fighters": case.get("fighters"),
                "result": case.get("result"),
                "pre_fight": {
                    key: pre_fight.get(key)
                    for key in (
                        "model_pick_prob",
                        "market_pick_prob",
                        "edge",
                        "confidence_score",
                        "pick_odds",
                        "pick_elo",
                        "opponent_elo",
                        "elo_diff",
                        "market_resistance_level",
                        "core_win_condition",
                        "known_concerns",
                    )
                },
                "fragility_flags": case.get("fragility_flags", []),
                "post_fight": {
                    "why_pick_lost": post_fight.get("why_pick_lost"),
                    "failure_mode_tags": failure_tags,
                    "key_stats": post_fight.get("key_stats"),
                },
                "lesson": case.get("lesson"),
                "review_notes": case.get("review_notes", []),
            }
        payload["match_reasons"] = match_reasons or ["all_cases"]
        matches.append(payload)

    return {
        "source_file": display_path(FRAGILITY_CASES_FILE),
        "filters": {
            "fighter": fighter,
            "fragility_flag": fragility_flag,
            "failure_mode": failure_mode,
            "text_query": text_query,
            "min_date": min_date,
            "max_date": max_date,
            "limit": limit,
            "include_full_case": include_full_case,
        },
        "total_cases": len(all_cases),
        "matched_cases": len(matches),
        "returned_cases": min(len(matches), limit),
        "truncated": len(matches) > limit,
        "cases": matches[:limit],
    }


@mcp.tool()
def describe_database(database: SQLDatabase) -> dict[str, Any]:
    """Describe tables, views, and columns for one read-only SQLite database."""
    path = resolve_database_path(database)
    conn = readonly_connection(path)
    try:
        objects = conn.execute(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        described = []
        for obj in objects:
            columns = [
                {
                    "cid": row["cid"],
                    "name": row["name"],
                    "type": row["type"],
                    "notnull": bool(row["notnull"]),
                    "default": row["dflt_value"],
                    "pk": bool(row["pk"]),
                }
                for row in conn.execute(f"PRAGMA table_info('{obj['name']}')").fetchall()
            ]
            described.append(
                {
                    "name": obj["name"],
                    "type": obj["type"],
                    "columns": columns,
                }
            )
        return {
            "database": database,
            "path": display_path(path),
            "objects": described,
        }
    finally:
        conn.close()


@mcp.tool()
def run_readonly_sql(database: SQLDatabase, query: str, limit: int = 200) -> dict[str, Any]:
    """Run one read-only SQL query against a selected SQLite database."""
    if limit <= 0 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000.")
    normalized = ensure_read_only_query(query)
    path = resolve_database_path(database)
    conn = readonly_connection(path)
    try:
        rows = conn.execute(normalized).fetchall()
        result = serialize_rows(rows, limit=limit)
        result["database"] = database
        result["path"] = display_path(path)
        result["query"] = normalized
        return result
    finally:
        conn.close()


@mcp.tool()
def search_context_targets(
    fighter_query: str | None = None,
    season: int | None = None,
    date: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Search fight targets in context_pool for packet/review lookup."""
    if limit <= 0 or limit > 100:
        raise ValueError("limit must be between 1 and 100.")

    clauses = []
    params: list[Any] = []
    if fighter_query:
        clauses.append("(LOWER(fighter1) LIKE ? OR LOWER(fighter2) LIKE ? OR LOWER(pick) LIKE ?)")
        needle = f"%{fighter_query.lower()}%"
        params.extend([needle, needle, needle])
    if season is not None:
        clauses.append("season = ?")
        params.append(season)
    if date is not None:
        clauses.append("date = ?")
        params.append(date)

    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        rows = conn.execute(
            f"""
            SELECT
                id AS fight_pool_id,
                date,
                season,
                fighter1,
                fighter2,
                pick,
                pick_prob,
                pick_odds,
                skip_reason,
                source_row_key
            FROM backtest_fight_pool
            {where_clause}
            ORDER BY date DESC, id DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
        return {
            "query": fighter_query,
            "results": [dict(row) for row in rows],
        }
    finally:
        conn.close()


@mcp.tool()
def find_similar_elo_gap_fights(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
    elo_gap: float | None = None,
    pick_prob: float | None = None,
    edge: float | None = None,
    limit: int = 8,
    include_pending: bool = False,
) -> dict[str, Any]:
    """Return structured historical comps for a target or requested ELO gap."""
    if limit <= 0 or limit > 50:
        raise ValueError("limit must be between 1 and 50.")
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        target = None
        if fight_pool_id is not None:
            target = resolve_target_row(
                conn,
                fight_pool_id=fight_pool_id,
                season=season,
            )
        elif fighter1 and fighter2:
            target, _, _ = _dynamic_synthetic_target(
                fighter1=fighter1,
                fighter2=fighter2,
                date=date,
                fighter1_odds=fighter1_odds,
                fighter2_odds=fighter2_odds,
            )
        return build_similar_elo_gap_fights(
            conn,
            target=target,
            elo_gap=elo_gap,
            pick_prob=pick_prob,
            edge=edge,
            limit=limit,
            include_pending=include_pending,
        )
    finally:
        conn.close()


@mcp.tool()
def find_similar_market_fights(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
    pick_odds: float | None = None,
    market_implied_prob: float | None = None,
    edge: float | None = None,
    pick_prob: float | None = None,
    limit: int = 8,
    include_pending: bool = False,
) -> dict[str, Any]:
    """Return structured historical comps for a target or requested market profile."""
    if limit <= 0 or limit > 50:
        raise ValueError("limit must be between 1 and 50.")
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        target = None
        if fight_pool_id is not None:
            target = resolve_target_row(
                conn,
                fight_pool_id=fight_pool_id,
                season=season,
            )
        elif fighter1 and fighter2:
            target, _, _ = _dynamic_synthetic_target(
                fighter1=fighter1,
                fighter2=fighter2,
                date=date,
                fighter1_odds=fighter1_odds,
                fighter2_odds=fighter2_odds,
            )
        return build_similar_market_fights(
            conn,
            target=target,
            pick_odds=pick_odds,
            market_implied_prob=market_implied_prob,
            edge=edge,
            pick_prob=pick_prob,
            limit=limit,
            include_pending=include_pending,
        )
    finally:
        conn.close()


@mcp.tool()
def find_trait_matchup_examples(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
    archetype: str | None = None,
    limit: int = 8,
    include_pending: bool = False,
    min_trait_gap: float = 8.0,
) -> dict[str, Any]:
    """Return trait/archetype historical examples with structured provenance."""
    if limit <= 0 or limit > 50:
        raise ValueError("limit must be between 1 and 50.")
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        target = None
        target_payload = None
        if fight_pool_id is not None:
            target = resolve_target_row(
                conn,
                fight_pool_id=fight_pool_id,
                season=season,
            )
        elif fighter1 and fighter2 and archetype is None:
            target, target_payload, _ = _dynamic_synthetic_target(
                fighter1=fighter1,
                fighter2=fighter2,
                date=date,
                fighter1_odds=fighter1_odds,
                fighter2_odds=fighter2_odds,
            )
        return build_trait_matchup_examples(
            conn,
            target=target,
            target_payload=target_payload,
            archetype=archetype,
            limit=limit,
            include_pending=include_pending,
            min_trait_gap=min_trait_gap,
        )
    finally:
        conn.close()


@mcp.tool()
def get_historical_pattern_summary(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
) -> dict[str, Any]:
    """Return a structured historical pattern summary for one fight."""
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        if fight_pool_id is not None:
            target = resolve_target_row(conn, fight_pool_id=fight_pool_id, season=season)
            analysis = None
        elif fighter1 and fighter2:
            target, _, analysis = _dynamic_synthetic_target(
                fighter1=fighter1,
                fighter2=fighter2,
                date=date,
                fighter1_odds=fighter1_odds,
                fighter2_odds=fighter2_odds,
            )
        else:
            raise ValueError("Pass fight_pool_id or both fighter1 and fighter2.")
        result = build_historical_pattern_summary(conn, target=target)
        if analysis is not None:
            result["dynamic_source"] = {
                "source": "init_fight_analysis",
                "request": analysis["request"],
                "market_provenance": analysis["market"].get("provenance"),
            }
        return result
    finally:
        conn.close()


@mcp.tool()
def get_context_packet(
    fighter1: str,
    fighter2: str,
    date: str | None = None,
    season: int | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
    similar_limit: int = 10,
) -> dict[str, Any]:
    """Build the deterministic context packet for one fight."""
    if similar_limit <= 0 or similar_limit > 50:
        raise ValueError("similar_limit must be between 1 and 50.")
    pool_path = resolve_database_path("context_pool")
    conn = readonly_connection(pool_path)
    try:
        historical_lookup_error = None
        try:
            target, candidate_count = find_target(
                conn,
                fighter1=fighter1,
                fighter2=fighter2,
                date=date,
                season=season,
                aliases=aliases(),
            )
        except SystemExit as exc:
            target = None
            candidate_count = 0
            historical_lookup_error = str(exc)

        if target is not None and not _target_should_use_dynamic_packet(target):
            packet = build_packet(
                conn,
                target=target,
                candidate_count=candidate_count,
                similar_limit=similar_limit,
                pool_path=pool_path,
            )
            packet["packet_type"] = "historical_context_pool"
            packet["source"].update(
                {
                    "target_source": "context_pool",
                    "historical_pool_role": "target_and_evidence_library",
                    "exact_context_pool_row": True,
                }
            )
            return packet

        dynamic_target, dynamic_trait_delta, analysis = _dynamic_synthetic_target(
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
            fighter1_odds=fighter1_odds,
            fighter2_odds=fighter2_odds,
        )
        return _build_dynamic_context_packet(
            conn,
            target=dynamic_target,
            dynamic_trait_delta=dynamic_trait_delta,
            analysis=analysis,
            candidate_count=candidate_count,
            similar_limit=similar_limit,
            pool_path=pool_path,
            dynamic_reason="missing_context_pool_row" if target is None else "pending_or_future_context_row",
            historical_lookup_error=historical_lookup_error,
        )
    finally:
        conn.close()


@mcp.tool()
def get_fight_basics(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
) -> dict[str, Any]:
    """Return the core identifying/model fields for one fight row."""
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        target = resolve_target_row(
            conn,
            fight_pool_id=fight_pool_id,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
            season=season,
        )
        return {
            **fight_locator_payload(target),
            "winner": target.get("winner"),
            "pick_correct": target.get("pick_correct"),
            "actual_pnl": target.get("actual_pnl"),
        }
    finally:
        conn.close()


@mcp.tool()
def get_fight_model_market(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
) -> dict[str, Any]:
    """Return model probability, odds, edge, and pricing fields for one fight."""
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        target = resolve_target_row(
            conn,
            fight_pool_id=fight_pool_id,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
            season=season,
        )
        return {
            **fight_locator_payload(target),
            "pick_prob": target.get("pick_prob"),
            "pick_odds": target.get("pick_odds"),
            "market_implied_prob": target.get("market_implied_prob"),
            "edge": target.get("edge"),
            "elo_implied_prob": target.get("elo_implied_prob"),
            "model_minus_elo_prob": target.get("model_minus_elo_prob"),
            "market_minus_elo_prob": target.get("market_minus_elo_prob"),
            "model_market_elo_triangle": target.get("model_market_elo_triangle"),
            "current_decision": "bet" if target.get("bet") else "skip",
            "skip_reason": target.get("skip_reason"),
            "odds_provenance": {
                "odds_source_file": target.get("odds_source_file"),
                "odds_source_line": target.get("odds_source_line"),
                "odds_source_type": target.get("odds_source_type"),
                "odds_source_row": target.get("odds_source_row"),
                "source_event_id": target.get("source_event_id"),
                "source_url": target.get("source_url"),
                "scraped_at": target.get("scraped_at"),
                "bookmaker": target.get("bookmaker"),
                "odds_timestamp": target.get("odds_timestamp"),
                "odds_is_opening_line": target.get("odds_is_opening_line"),
                "odds_is_closing_line": target.get("odds_is_closing_line"),
            },
        }
    finally:
        conn.close()


@mcp.tool()
def init_fight_analysis(
    fighter1: str,
    fighter2: str,
    fight_date: str | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
) -> dict[str, Any]:
    """Resolve fighters, normalize market odds, and run a fresh dynamic prediction."""
    return build_init_fight_analysis(
        fighter1=fighter1,
        fighter2=fighter2,
        fight_date=fight_date,
        fighter1_odds=fighter1_odds,
        fighter2_odds=fighter2_odds,
    )


@mcp.tool()
def get_deterministic_signal_filter(
    fighter1: str,
    fighter2: str,
    fight_date: str | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
) -> dict[str, Any]:
    """Run the fast deterministic ELO/cardio screening filter without a deep-dive evidence chain."""
    return build_deterministic_signal_filter(
        fighter1=fighter1,
        fighter2=fighter2,
        fight_date=fight_date,
        fighter1_odds=fighter1_odds,
        fighter2_odds=fighter2_odds,
    )


@mcp.tool()
def get_elo_market_signal(
    fighter1: str,
    fighter2: str,
    fight_date: str | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
) -> dict[str, Any]:
    """Summarize current ELO edge, price relationship, and matching historical ROI buckets."""
    return build_elo_market_signal(
        fighter1=fighter1,
        fighter2=fighter2,
        fight_date=fight_date,
        fighter1_odds=fighter1_odds,
        fighter2_odds=fighter2_odds,
    )


@mcp.tool()
def get_fight_elo_context(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
) -> dict[str, Any]:
    """Return fight-row ELO context fields when present."""
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        target = resolve_target_row(
            conn,
            fight_pool_id=fight_pool_id,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
            season=season,
        )
        pick_elo_diff = target.get("pick_elo_diff")
        return {
            **fight_locator_payload(target),
            "fighter1_elo": target.get("fighter1_elo"),
            "fighter2_elo": target.get("fighter2_elo"),
            "pick_elo": target.get("pick_elo"),
            "opponent_elo": target.get("opponent_elo"),
            "pick_elo_diff": pick_elo_diff,
            "abs_elo_diff": target.get("abs_elo_diff"),
            "support_level": support_level(pick_elo_diff),
            "model_agrees_with_elo": target.get("model_agrees_with_elo"),
            "join_status": target.get("join_status"),
            "join_method": target.get("join_method"),
        }
    finally:
        conn.close()


@mcp.tool()
def get_fight_opponent_quality(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
) -> dict[str, Any]:
    """Return opponent-quality and prior-fight context for one fight."""
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        target = resolve_target_row(
            conn,
            fight_pool_id=fight_pool_id,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
            season=season,
        )
        return {
            **fight_locator_payload(target),
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
            "pick_recent_fights": json.loads(target.get("pick_recent_fights_json") or "[]"),
            "opponent_recent_fights": json.loads(target.get("opponent_recent_fights_json") or "[]"),
        }
    finally:
        conn.close()


@mcp.tool()
def get_fight_trait_deltas(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
) -> dict[str, Any]:
    """Return structured trait-delta evidence for one fight."""
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        target = resolve_target_row(
            conn,
            fight_pool_id=fight_pool_id,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
            season=season,
        )
        trait_delta = fetch_trait_delta_evidence(conn, target)
        return {
            **fight_locator_payload(target),
            "trait_delta": trait_delta,
        }
    finally:
        conn.close()


@mcp.tool()
def get_fight_historical_patterns(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
) -> dict[str, Any]:
    """Return applicable historical pattern stats and derived pattern score."""
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        target = resolve_target_row(
            conn,
            fight_pool_id=fight_pool_id,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
            season=season,
        )
        patterns = pattern_payload(conn, target)
        pattern_score = build_pattern_score(target, patterns)
        return {
            **fight_locator_payload(target),
            "pattern_score_v0": pattern_score,
            "patterns": patterns,
        }
    finally:
        conn.close()


@mcp.tool()
def get_fight_style_flags(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
) -> dict[str, Any]:
    """Return support/risk flags derived from fight-row context and patterns."""
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        target = resolve_target_row(
            conn,
            fight_pool_id=fight_pool_id,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
            season=season,
        )
        patterns = pattern_payload(conn, target)
        trait_delta = fetch_trait_delta_evidence(conn, target)
        return {
            **fight_locator_payload(target),
            "flags": build_flags(target, patterns, trait_delta),
        }
    finally:
        conn.close()


@mcp.tool()
def get_fight_nearest_examples(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Return nearest historical examples shaped for qualitative comparison."""
    if limit <= 0 or limit > 50:
        raise ValueError("limit must be between 1 and 50.")
    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        target = resolve_target_row(
            conn,
            fight_pool_id=fight_pool_id,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
            season=season,
        )
        rows = fetch_similar_rows(conn, target, limit=limit)
        return {
            **fight_locator_payload(target),
            "warning": "Nearest examples are illustrative only; use aggregate pattern stats for stronger historical support.",
            "examples": rows,
        }
    finally:
        conn.close()


@mcp.tool()
def review_context(
    fight_pool_id: int | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
    season: int | None = None,
    llm: bool = False,
    max_evidence: int = 40,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Create a cited evidence review for one context-pool fight."""
    if max_evidence <= 0 or max_evidence > 200:
        raise ValueError("max_evidence must be between 1 and 200.")
    pool_path = resolve_database_path("context_pool")
    conn = readonly_connection(pool_path)
    try:
        resolved_id = resolve_fight_pool_id(
            conn,
            fight_pool_id=fight_pool_id,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
            season=season,
        )
        evidence = load_evidence(conn, resolved_id)
    finally:
        conn.close()

    deterministic_review = build_review(evidence)
    return (
        build_llm_review(
            deterministic_review,
            evidence,
            max_evidence=max_evidence,
            temperature=temperature,
        )
        if llm
        else deterministic_review
    )


@mcp.tool()
def validate_combined_context(
    mode: Literal["leave-one-out", "temporal", "in-sample"] = "leave-one-out",
    skips_only: bool = False,
    dedupe_main_fight: bool = True,
    min_date: str | None = None,
    max_date: str | None = None,
    compare_modes: bool = False,
    audit_rule: str | None = None,
) -> dict[str, Any]:
    """Return combined evidence validation summaries and optional audit rows."""
    if audit_rule is not None and audit_rule not in {rule[0] for rule in RULES}:
        raise ValueError(f"Unknown audit_rule: {audit_rule}")

    conn = readonly_connection(resolve_database_path("context_pool"))
    try:
        rows = fetch_combined_rows(conn)
    finally:
        conn.close()

    response: dict[str, Any] = {
        "requested_mode": mode,
        "skips_only": skips_only,
        "dedupe_main_fight": dedupe_main_fight,
        "min_date": min_date,
        "max_date": max_date,
        "rule_summaries": build_rule_rows(
            rows,
            mode=mode,
            skips_only=skips_only,
            dedupe_main_fight=dedupe_main_fight,
            min_date=min_date,
            max_date=max_date,
        ),
    }

    if compare_modes:
        response["mode_comparison"] = {
            candidate_mode: build_rule_rows(
                rows,
                mode=candidate_mode,
                skips_only=skips_only,
                dedupe_main_fight=dedupe_main_fight,
                min_date=min_date,
                max_date=max_date,
            )
            for candidate_mode in ("leave-one-out", "temporal", "in-sample")
        }

    if audit_rule:
        audit_rows = matching_rows_for_rule(
            rows,
            mode=mode,
            skips_only=skips_only,
            rule_name=audit_rule,
            min_date=min_date,
            max_date=max_date,
        )
        enrich_with_main_db(audit_rows, main_db_path())
        response["audit_rule"] = audit_rule
        response["audit_rows"] = audit_rows

    return response


@mcp.tool()
def read_backtest_file(relative_path: str, start_line: int = 1, end_line: int = 200) -> dict[str, Any]:
    """Read a whitelisted backtest or documentation file with line numbers."""
    if start_line <= 0 or end_line < start_line:
        raise ValueError("start_line must be >= 1 and end_line must be >= start_line.")
    if end_line - start_line > 500:
        raise ValueError("Read windows larger than 500 lines are not allowed.")

    path = resolve_whitelisted_file(relative_path)
    lines = path.read_text().splitlines()
    excerpt = [
        {"line": index, "text": lines[index - 1]}
        for index in range(start_line, min(end_line, len(lines)) + 1)
    ]
    return {
        "path": display_path(path),
        "start_line": start_line,
        "end_line": min(end_line, len(lines)),
        "total_lines": len(lines),
        "lines": excerpt,
    }


def _parse_elo_window(window: str | int) -> int | None:
    """Return an integer row limit, or None for 'all'.

    Accepts integers, integer strings ('3', '5'), or the sentinel 'all'
    (case-insensitive).  Raises ValueError for anything else.
    """
    if isinstance(window, int):
        if window <= 0:
            raise ValueError("window must be a positive integer or 'all'.")
        return window
    text = str(window).strip().lower()
    if text == "all":
        return None
    try:
        n = int(text)
    except ValueError:
        raise ValueError(f"window must be a positive integer or 'all', got: {window!r}")
    if n <= 0:
        raise ValueError("window must be a positive integer or 'all'.")
    return n


def _fuzzy_fighter_ids(
    conn: sqlite3.Connection,
    normalized_names: set[str],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return candidate fighters whose normalised name matches any normalized variant.

    We compare against the Sergey `fighters` table which stores mixed-case
    original names.  We normalise on the fly in Python so that accent-folding
    and punctuation rules stay consistent with the rest of the codebase.
    """
    rows = conn.execute(
        "SELECT fighter_id, full_name, elo_current, elo_peak FROM fighters "
        "WHERE full_name IS NOT NULL LIMIT 50000"
    ).fetchall()
    matches = []
    for row in rows:
        row_name = normalize_name(row["full_name"])
        if any(name and (name in row_name or row_name == name) for name in normalized_names):
            matches.append(
                {
                    "fighter_id": row["fighter_id"],
                    "full_name": row["full_name"],
                    "elo_current": row["elo_current"],
                    "elo_peak": row["elo_peak"],
                }
            )
        if len(matches) >= limit:
            break
    return matches


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _normalize_pair(name1: str, name2: str) -> tuple[str, str]:
    return tuple(sorted((normalize_name(name1), normalize_name(name2))))


def _normalize_lookup_date(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%B %d, %Y", "%b %d, %Y", "%B %d", "%b %d"):
        try:
            dt = datetime.strptime(text, fmt)
            if "%Y" not in fmt:
                dt = dt.replace(year=datetime.now().year)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        return text


def _parse_birth_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _age_at_cutoff(profile: dict[str, Any], as_of_date: str | None) -> tuple[int | None, str]:
    dob = _parse_birth_date(profile.get("date_of_birth"))
    cutoff_text = _normalize_lookup_date(as_of_date)
    cutoff = None
    if cutoff_text:
        try:
            cutoff = datetime.strptime(cutoff_text, "%Y-%m-%d")
        except ValueError:
            cutoff = None
    if dob is not None and cutoff is not None:
        age = cutoff.year - dob.year - ((cutoff.month, cutoff.day) < (dob.month, dob.day))
        return age, "date_of_birth_at_cutoff"
    return profile.get("age"), "profile_age"


def _age_bucket(age: int | None) -> str:
    if age is None:
        return "unknown_age"
    if age <= 24:
        return "young_prospect_age"
    if age <= 29:
        return "prime_age"
    if age <= 34:
        return "veteran_age"
    return "aging_veteran_age"


def _experience_bucket(fight_count: int | None) -> str:
    if fight_count is None:
        return "unknown_experience"
    if fight_count <= 2:
        return "thin_sample"
    if fight_count <= 7:
        return "developing_sample"
    if fight_count <= 14:
        return "experienced_sample"
    return "veteran_sample"


def _gap_bucket(value: int | None, *, kind: str) -> str:
    if value is None:
        return f"unknown_{kind}_gap"
    gap = abs(value)
    if kind == "age":
        if gap < 3:
            return "similar_age"
        if gap < 6:
            return "moderate_age_gap"
        if gap < 10:
            return "large_age_gap"
        return "extreme_age_gap"
    if gap < 3:
        return "similar_experience"
    if gap < 8:
        return "moderate_experience_gap"
    return "large_experience_gap"


def _prospect_vet_bucket(
    *,
    fighter1_age: int | None,
    fighter2_age: int | None,
    fighter1_count: int | None,
    fighter2_count: int | None,
) -> str:
    if None in {fighter1_age, fighter2_age, fighter1_count, fighter2_count}:
        return "insufficient_age_experience_data"

    def prospect(age: int, count: int) -> bool:
        return age <= 28 and count <= 7

    def veteran(age: int, count: int) -> bool:
        return age >= 32 and count >= 8

    if prospect(fighter1_age, fighter1_count) and veteran(fighter2_age, fighter2_count):
        return "fighter1_prospect_vs_fighter2_veteran"
    if prospect(fighter2_age, fighter2_count) and veteran(fighter1_age, fighter1_count):
        return "fighter2_prospect_vs_fighter1_veteran"

    age_delta = fighter1_age - fighter2_age
    experience_delta = fighter1_count - fighter2_count
    if abs(age_delta) >= 6 and abs(experience_delta) >= 5:
        if age_delta < 0 and experience_delta < 0:
            return "fighter1_younger_less_experienced"
        if age_delta > 0 and experience_delta > 0:
            return "fighter2_younger_less_experienced"
    return "no_clear_prospect_veteran_split"


def _fight_stats_candidates(
    conn: sqlite3.Connection,
    *,
    fight_id: str | None,
    fighter1: str | None,
    fighter2: str | None,
    date: str | None,
) -> list[sqlite3.Row]:
    if fight_id:
        return conn.execute(
            """
            SELECT
                f.id,
                f.fight_id,
                e.event_id,
                e.name AS event_name,
                e.date AS event_date,
                e.url AS event_url,
                f.fight_number,
                f.weight_class,
                f.is_title_fight,
                f.scheduled_rounds,
                f.result,
                f.method,
                f.method_detail,
                f.round_finished,
                f.time,
                f.fight_detail_url,
                f1.name AS fighter1_name,
                f2.name AS fighter2_name,
                w.name AS winner_name,
                fs.fighter_1_totals,
                fs.fighter_2_totals,
                fs.round_by_round,
                fs.significant_strikes
            FROM fights f
            JOIN events e ON e.id = f.event_id
            JOIN fighters f1 ON f1.id = f.fighter_1_id
            JOIN fighters f2 ON f2.id = f.fighter_2_id
            LEFT JOIN fighters w ON w.id = f.winner_id
            LEFT JOIN fight_stats fs ON fs.fight_id = f.id
            WHERE f.fight_id = ?
            """,
            (fight_id,),
        ).fetchall()

    if not fighter1 or not fighter2:
        raise ValueError("Pass fight_id or both fighter1 and fighter2.")

    rows = conn.execute(
        """
        SELECT
            f.id,
            f.fight_id,
            e.event_id,
            e.name AS event_name,
            e.date AS event_date,
            e.url AS event_url,
            f.fight_number,
            f.weight_class,
            f.is_title_fight,
            f.scheduled_rounds,
            f.result,
            f.method,
            f.method_detail,
            f.round_finished,
            f.time,
            f.fight_detail_url,
            f1.name AS fighter1_name,
            f2.name AS fighter2_name,
            w.name AS winner_name,
            fs.fighter_1_totals,
            fs.fighter_2_totals,
            fs.round_by_round,
            fs.significant_strikes
        FROM fights f
        JOIN events e ON e.id = f.event_id
        JOIN fighters f1 ON f1.id = f.fighter_1_id
        JOIN fighters f2 ON f2.id = f.fighter_2_id
        LEFT JOIN fighters w ON w.id = f.winner_id
        LEFT JOIN fight_stats fs ON fs.fight_id = f.id
        ORDER BY f.scraped_at DESC, f.id DESC
        """
    ).fetchall()
    target_pair = _normalize_pair(fighter1, fighter2)
    requested_date = _normalize_lookup_date(date)

    matches: list[sqlite3.Row] = []
    for row in rows:
        if _normalize_pair(row["fighter1_name"], row["fighter2_name"]) != target_pair:
            continue
        row_date = _normalize_lookup_date(row["event_date"])
        if requested_date and row_date != requested_date:
            continue
        matches.append(row)
    return matches


def _fight_stats_payload(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        "mapped": True,
        "fight_id": row["fight_id"],
        "event": {
            "event_id": row["event_id"],
            "event_name": row["event_name"],
            "event_date": row["event_date"],
            "event_url": row["event_url"],
            "fight_number": row["fight_number"],
        },
        "fighters": {
            "fighter1": row["fighter1_name"],
            "fighter2": row["fighter2_name"],
            "winner": row["winner_name"],
        },
        "result": {
            "result": row["result"],
            "method": row["method"],
            "method_detail": row["method_detail"],
            "round_finished": row["round_finished"],
            "time": row["time"],
        },
        "fight_details": {
            "weight_class": row["weight_class"],
            "is_title_fight": bool(row["is_title_fight"]),
            "scheduled_rounds": row["scheduled_rounds"],
            "fight_detail_url": row["fight_detail_url"],
            "stats_available": any(
                row[field] is not None
                for field in ("fighter_1_totals", "fighter_2_totals", "round_by_round", "significant_strikes")
            ),
        },
        "stats": {
            "fighter1_totals": _maybe_json(row["fighter_1_totals"]),
            "fighter2_totals": _maybe_json(row["fighter_2_totals"]),
            "round_by_round": _maybe_json(row["round_by_round"]),
            "significant_strikes": _maybe_json(row["significant_strikes"]),
        },
    }


def _historical_market_odds(conn: sqlite3.Connection, fight_pk: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT
            bookmaker,
            fighter_1_odds,
            fighter_2_odds,
            odds_timestamp,
            is_opening_line,
            is_closing_line
        FROM betting_odds
        WHERE fight_id = ?
        ORDER BY is_closing_line DESC, odds_timestamp DESC, id DESC
        LIMIT 1
        """,
        (fight_pk,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


@mcp.tool()
def get_fighter_elo_history(
    fighter_name: str,
    window: str | int = "all",
) -> dict[str, Any]:
    """Return a fighter's pre-fight ELO history from the Sergey sidecar database.

    Each row in the returned `fights` list represents one fight and includes:
    - fight_date
    - event_name
    - opponent_name
    - result  ("win" | "loss" | "draw" | "no_contest" | "unknown")
    - method   (e.g. "Decision", "KO/TKO", "Submission")
    - division
    - fighter_pre_elo  — ELO snapshot recorded *before* the fight
    - opponent_pre_elo — opponent's ELO snapshot recorded *before* the fight
    - elo_diff         — fighter_pre_elo minus opponent_pre_elo (positive = favourite by ELO)

    Parameters
    ----------
    fighter_name : str
        Fighter name to look up.  Accent-folding and common aliases are
        applied automatically.  If the name is ambiguous (multiple fighters
        match), the tool returns a `candidates` list so the caller can retry
        with a more specific name.
    window : str | int, default "all"
        Number of most-recent fights to return.  Pass an integer (3, 5, 10)
        or the string "all" to retrieve the full history.

    Notes
    -----
    - ELO values come exclusively from Sergey's pre-fight snapshots; fights
      where the snapshot is NULL are included but `fighter_pre_elo` will be
      None.
    - Fights without a UFC promotion tag in the sidecar are excluded.
    - Fighters absent from the sidecar return `mapped: false` with an empty
      `fights` list.
    """
    row_limit = _parse_elo_window(window)

    normalized = normalize_name(fighter_name)
    if not normalized:
        raise ValueError("fighter_name must not be empty.")
    canonical_name = aliases().get(normalized, normalized)
    normalized_variants = {normalized, canonical_name}
    normalized_variants.update(
        alias for alias, canonical in aliases().items() if canonical == canonical_name
    )

    sidecar_path = resolve_database_path("sergey_sidecar")
    conn = readonly_connection(sidecar_path)
    try:
        candidates = _fuzzy_fighter_ids(conn, normalized_variants)

        if len(candidates) == 0:
            return {
                "fighter_name": fighter_name,
                "normalized_name": normalized,
                "resolved_name": canonical_name,
                "mapped": False,
                "window": "all" if row_limit is None else row_limit,
                "returned_fights": 0,
                "total_fights_in_db": 0,
                "fights": [],
                "note": (
                    "Fighter not found in Sergey sidecar.  The sidecar covers UFC-mapped "
                    "fights only.  Try a spelling variant or check aliases."
                ),
            }

        if len(candidates) > 1:
            # Prefer an exact normalised match when possible
            exact = [c for c in candidates if normalize_name(c["full_name"]) == normalized]
            if len(exact) == 1:
                candidates = exact
            else:
                return {
                    "fighter_name": fighter_name,
                    "normalized_name": normalized,
                    "resolved_name": canonical_name,
                    "mapped": False,
                    "window": "all" if row_limit is None else row_limit,
                    "returned_fights": 0,
                    "total_fights_in_db": 0,
                    "fights": [],
                    "ambiguous": True,
                    "candidates": candidates[:20],
                    "note": (
                        f"Name '{fighter_name}' matched {len(candidates)} fighters.  "
                        "Retry with the full name from the candidates list."
                    ),
                }

        fighter = candidates[0]
        fighter_id: int = fighter["fighter_id"]
        full_name: str = fighter["full_name"]

        fight_rows = conn.execute(
            """
            SELECT
                f.fight_id,
                f.event_date,
                f.event_name,
                f.fighter_red_id,
                f.fighter_red_name,
                f.fighter_blue_id,
                f.fighter_blue_name,
                f.fighter_red_elo,
                f.fighter_blue_elo,
                f.elo_diff,
                f.winner_name,
                f.winner_id,
                f.short_method,
                f.division,
                f.fight_status
            FROM fights f
            WHERE (f.fighter_red_id = ? OR f.fighter_blue_id = ?)
              AND f.promotion LIKE '%Ultimate Fighting%'
            ORDER BY f.event_date DESC, f.fight_id DESC
            """,
            (fighter_id, fighter_id),
        ).fetchall()

        total = len(fight_rows)
        if row_limit is not None:
            fight_rows = fight_rows[:row_limit]

        history: list[dict[str, Any]] = []
        for row in fight_rows:
            is_red = row["fighter_red_id"] == fighter_id
            fighter_elo = row["fighter_red_elo"] if is_red else row["fighter_blue_elo"]
            opp_elo = row["fighter_blue_elo"] if is_red else row["fighter_red_elo"]
            opp_name = row["fighter_blue_name"] if is_red else row["fighter_red_name"]

            # Compute elo_diff from fighter's perspective (positive = ELO advantage)
            elo_diff: int | None = None
            if fighter_elo is not None and opp_elo is not None:
                elo_diff = fighter_elo - opp_elo

            # Determine result from fighter's perspective
            winner_id = row["winner_id"]
            winner_name = row["winner_name"]
            if winner_id is None and not winner_name:
                result = "unknown"
            elif winner_id == fighter_id:
                result = "win"
            elif winner_name and normalize_name(winner_name) == normalize_name(full_name):
                result = "win"
            elif winner_name and normalize_name(winner_name) in {
                normalize_name(row["fighter_red_name"]),
                normalize_name(row["fighter_blue_name"]),
            }:
                result = "loss"
            else:
                # winner_name null but fight_status can hint draw/nc
                status = (row["fight_status"] or "").lower()
                if "draw" in status:
                    result = "draw"
                elif "no_contest" in status or "no contest" in status:
                    result = "no_contest"
                else:
                    result = "unknown"

            history.append(
                {
                    "fight_id": row["fight_id"],
                    "fight_date": row["event_date"],
                    "event_name": row["event_name"],
                    "opponent_name": opp_name,
                    "result": result,
                    "method": row["short_method"],
                    "division": row["division"],
                    "fighter_pre_elo": fighter_elo,
                    "opponent_pre_elo": opp_elo,
                    "elo_diff": elo_diff,
                }
            )

    finally:
        conn.close()

    return {
        "fighter_name": fighter_name,
        "resolved_name": full_name,
        "fighter_id": fighter_id,
        "elo_current": fighter["elo_current"],
        "elo_peak": fighter["elo_peak"],
        "mapped": True,
        "window": "all" if row_limit is None else row_limit,
        "returned_fights": len(history),
        "total_fights_in_db": total,
        "fights": history,
    }


@mcp.tool()
def get_fight_stats(
    fight_id: str | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    """Return past-fight metadata plus UFCStats totals/breakdowns from the main DB."""
    conn = readonly_connection(resolve_database_path("main"))
    try:
        rows = _fight_stats_candidates(
            conn,
            fight_id=fight_id,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
        )
    finally:
        conn.close()

    if not rows:
        return {
            "mapped": False,
            "fight_id": fight_id,
            "fighter1": fighter1,
            "fighter2": fighter2,
            "date": date,
            "note": "No matching fight found in the main DB.",
        }

    if len(rows) > 1:
        return {
            "mapped": False,
            "ambiguous": True,
            "fight_id": fight_id,
            "fighter1": fighter1,
            "fighter2": fighter2,
            "date": date,
            "candidates": [
                {
                    "fight_id": row["fight_id"],
                    "event_name": row["event_name"],
                    "event_date": row["event_date"],
                    "fighter1_name": row["fighter1_name"],
                    "fighter2_name": row["fighter2_name"],
                }
                for row in rows[:20]
            ],
            "note": "Multiple fights matched. Retry with fight_id or add the exact event date.",
        }

    return _fight_stats_payload(rows[0])


@mcp.tool()
def get_historical_fight_deep_dive(
    fight_id: str | None = None,
    fighter1: str | None = None,
    fighter2: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    """Return a historical fight deep dive with pre-fight clamped analysis plus actual fight stats."""
    conn = readonly_connection(resolve_database_path("main"))
    try:
        rows = _fight_stats_candidates(
            conn,
            fight_id=fight_id,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
        )
        if not rows:
            return {
                "mapped": False,
                "fight_id": fight_id,
                "fighter1": fighter1,
                "fighter2": fighter2,
                "date": date,
                "note": "No matching fight found in the main DB.",
            }
        if len(rows) > 1:
            return {
                "mapped": False,
                "ambiguous": True,
                "fight_id": fight_id,
                "fighter1": fighter1,
                "fighter2": fighter2,
                "date": date,
                "candidates": [
                    {
                        "fight_id": row["fight_id"],
                        "event_name": row["event_name"],
                        "event_date": row["event_date"],
                        "fighter1_name": row["fighter1_name"],
                        "fighter2_name": row["fighter2_name"],
                    }
                    for row in rows[:20]
                ],
                "note": "Multiple fights matched. Retry with fight_id or add the exact event date.",
            }
        row = rows[0]
        market_odds = _historical_market_odds(conn, int(row["id"]))
    finally:
        conn.close()

    actual_fight = _fight_stats_payload(row)
    pre_fight_analysis = build_init_fight_analysis(
        fighter1=row["fighter1_name"],
        fighter2=row["fighter2_name"],
        fight_date=row["event_date"],
        fighter1_odds=market_odds.get("fighter_1_odds") if market_odds else None,
        fighter2_odds=market_odds.get("fighter_2_odds") if market_odds else None,
    )

    return {
        "mapped": True,
        "fight_id": row["fight_id"],
        "lookup": {
            "requested": {
                "fight_id": fight_id,
                "fighter1": fighter1,
                "fighter2": fighter2,
                "date": date,
            },
            "resolved": {
                "fighter1": row["fighter1_name"],
                "fighter2": row["fighter2_name"],
                "event_date": row["event_date"],
            },
        },
        "actual_fight": actual_fight,
        "pre_fight": {
            "analysis_cutoff": row["event_date"],
            "market_odds": market_odds,
            "analysis": pre_fight_analysis,
            "trait_delta": _metric_delta_payload(pre_fight_analysis),
        },
        "provenance": {
            "actual_fight": "main.fights + main.fight_stats",
            "pre_fight_analysis": "mcp_server.fight_init.init_fight_analysis",
            "point_in_time_rule": "fighter snapshots and fight counts are clamped strictly before the fight date",
        },
    }


@mcp.tool()
def get_fighter_snapshot(
    fighter_name: str,
    as_of_date: str | None = None,
    recent_elo_fights: int = 2,
) -> dict[str, Any]:
    """Return structured fighter state for MCP init flows.

    The snapshot keeps the main DB fighter name canonical while resolving sidecar
    variants behind the scenes for ELO and trait enrichment.  When `as_of_date`
    is provided, completed fight history is filtered strictly before that date to
    mirror the app's point-in-time semantics.
    """
    return build_fighter_snapshot(
        fighter_name,
        as_of=as_of_date,
        recent_elo_fights=recent_elo_fights,
    )


@mcp.tool()
def get_age_experience_context(
    fighter1: str,
    fighter2: str,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """Return direct age-gap, experience-gap, and prospect-vs-veteran buckets."""
    snapshot1 = build_fighter_snapshot(fighter1, as_of=as_of_date, recent_elo_fights=2)
    snapshot2 = build_fighter_snapshot(fighter2, as_of=as_of_date, recent_elo_fights=2)

    if not snapshot1.get("resolved") or not snapshot2.get("resolved"):
        return {
            "mapped": False,
            "as_of_date": as_of_date,
            "fighters": {
                "fighter1": {
                    "requested_name": fighter1,
                    "resolved": bool(snapshot1.get("resolved")),
                    "resolved_name": (snapshot1.get("identity") or {}).get("resolved_name"),
                },
                "fighter2": {
                    "requested_name": fighter2,
                    "resolved": bool(snapshot2.get("resolved")),
                    "resolved_name": (snapshot2.get("identity") or {}).get("resolved_name"),
                },
            },
            "note": "Both fighters must resolve before age/experience buckets can be computed.",
        }

    profile1 = snapshot1.get("profile") or {}
    profile2 = snapshot2.get("profile") or {}
    record1 = snapshot1.get("record") or {}
    record2 = snapshot2.get("record") or {}
    fighter1_age, fighter1_age_source = _age_at_cutoff(profile1, as_of_date)
    fighter2_age, fighter2_age_source = _age_at_cutoff(profile2, as_of_date)
    fighter1_count = record1.get("fight_count_as_of")
    fighter2_count = record2.get("fight_count_as_of")
    age_delta = None if fighter1_age is None or fighter2_age is None else fighter1_age - fighter2_age
    experience_delta = None if fighter1_count is None or fighter2_count is None else fighter1_count - fighter2_count

    return {
        "mapped": True,
        "as_of_date": as_of_date,
        "fighters": {
            "fighter1": {
                "requested_name": fighter1,
                "resolved_name": (snapshot1.get("identity") or {}).get("resolved_name"),
                "age": fighter1_age,
                "age_source": fighter1_age_source,
                "date_of_birth": profile1.get("date_of_birth"),
                "fight_count_as_of": fighter1_count,
                "age_bucket": _age_bucket(fighter1_age),
                "experience_bucket": _experience_bucket(fighter1_count),
            },
            "fighter2": {
                "requested_name": fighter2,
                "resolved_name": (snapshot2.get("identity") or {}).get("resolved_name"),
                "age": fighter2_age,
                "age_source": fighter2_age_source,
                "date_of_birth": profile2.get("date_of_birth"),
                "fight_count_as_of": fighter2_count,
                "age_bucket": _age_bucket(fighter2_age),
                "experience_bucket": _experience_bucket(fighter2_count),
            },
        },
        "deltas": {
            "fighter1_minus_fighter2_age": age_delta,
            "fighter1_minus_fighter2_fight_count": experience_delta,
        },
        "buckets": {
            "age_gap_bucket": _gap_bucket(age_delta, kind="age"),
            "experience_gap_bucket": _gap_bucket(experience_delta, kind="experience"),
            "prospect_vs_veteran_bucket": _prospect_vet_bucket(
                fighter1_age=fighter1_age,
                fighter2_age=fighter2_age,
                fighter1_count=fighter1_count,
                fighter2_count=fighter2_count,
            ),
        },
        "provenance": {
            "source": "fastapi_app.services.fighter_snapshot.build_fighter_snapshot",
            "point_in_time_rule": "fight_count_as_of is strictly before as_of_date; age is computed from DOB at as_of_date when DOB is available",
        },
    }


@mcp.tool()
def find_similar_fighter_profiles(
    fighter_name: str,
    as_of_date: str | None = None,
    limit: int = 8,
    min_fight_count: int = 3,
) -> dict[str, Any]:
    """Return historical fighter analogs by qualitative traits, quantitative stats, and ELO state."""
    if limit <= 0 or limit > 50:
        raise ValueError("limit must be between 1 and 50.")
    snapshot = build_fighter_snapshot(
        fighter_name,
        as_of=as_of_date,
        recent_elo_fights=2,
    )
    traits_path = resolve_database_path("trait_snapshots")
    conn = readonly_connection(traits_path)
    try:
        profile_neighbors = build_similar_fighter_profiles(
            conn,
            target_snapshot=snapshot,
            as_of_date=as_of_date,
            limit=limit,
            min_fight_count=min_fight_count,
        )
    finally:
        conn.close()

    return {
        "query": {
            "fighter_name": fighter_name,
            "as_of_date": as_of_date,
            "limit": limit,
            "min_fight_count": min_fight_count,
        },
        "target_snapshot": {
            "resolved": snapshot.get("resolved"),
            "identity": snapshot.get("identity"),
            "record": snapshot.get("record"),
            "elo": {
                key: (snapshot.get("elo") or {}).get(key)
                for key in ("available", "elo_current", "elo_peak", "elo_decline_from_peak", "elo_current_source")
            },
        },
        "profile_neighbors": profile_neighbors,
        "historical_elo_neighbors": _historical_elo_fighter_neighbors(
            target_snapshot=snapshot,
            as_of_date=as_of_date,
            limit=limit,
        ),
        "provenance": {
            "profile_neighbors": "trait_snapshots.fighter_trait_snapshots",
            "historical_elo_neighbors": "sergey_sidecar.fights",
            "target_snapshot": "fastapi_app.services.fighter_snapshot.build_fighter_snapshot",
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
