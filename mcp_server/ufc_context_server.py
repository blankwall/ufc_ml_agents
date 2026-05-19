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
        "whitelisted_roots": sorted(display_path(path) for path in WHITELISTED_FILE_ROOTS),
        "combined_evidence_rules": [rule[0] for rule in RULES],
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
    similar_limit: int = 10,
) -> dict[str, Any]:
    """Build the deterministic context packet for one fight."""
    if similar_limit <= 0 or similar_limit > 50:
        raise ValueError("similar_limit must be between 1 and 50.")
    pool_path = resolve_database_path("context_pool")
    conn = readonly_connection(pool_path)
    try:
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
            raise ValueError(str(exc)) from None
        return build_packet(
            conn,
            target=target,
            candidate_count=candidate_count,
            similar_limit=similar_limit,
            pool_path=pool_path,
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
