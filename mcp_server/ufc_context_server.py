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
from backtest.context_packet import DEFAULT_POOL, build_packet, find_target
from backtest.elo_analysis import DEFAULT_ALIAS_SOURCES, load_aliases, normalize_name
from backtest.validate_combined_evidence import (
    RULES,
    build_rule_rows,
    enrich_with_main_db,
    fetch_rows as fetch_combined_rows,
    matching_rows_for_rule,
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
    target, _ = find_target(
        conn,
        fighter1=fighter1,
        fighter2=fighter2,
        date=date,
        season=season,
        aliases=aliases(),
    )
    return int(target["id"])


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
        target, candidate_count = find_target(
            conn,
            fighter1=fighter1,
            fighter2=fighter2,
            date=date,
            season=season,
            aliases=aliases(),
        )
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
    normalized: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return candidate fighters whose normalised name contains *normalized*.

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
        if normalized in normalize_name(row["full_name"]):
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

    sidecar_path = resolve_database_path("sergey_sidecar")
    conn = readonly_connection(sidecar_path)
    try:
        candidates = _fuzzy_fighter_ids(conn, normalized)

        if len(candidates) == 0:
            return {
                "fighter_name": fighter_name,
                "normalized_name": normalized,
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
