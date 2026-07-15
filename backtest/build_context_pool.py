#!/usr/bin/env python3
"""
Build a generated SQLite context pool for evidence-packet/retrieval work.

The pool materializes ELO-enriched 2025/2026 backtest rows into a queryable DB:

  data/enrichment/context_pool.sqlite

This is the first deterministic evidence layer for future context packets. It
does not call an LLM and does not rerun model inference; it compiles historical
model/backtest outcomes plus Sergey ELO context for fast similarity queries.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.elo_analysis import (  # noqa: E402
    DEFAULT_ALIAS_SOURCES,
    DEFAULT_SIDECAR,
    EnrichedFight,
    american_implied_prob,
    canonical_name,
    enrich_results,
    load_aliases,
    names_match,
)


DEFAULT_RESULTS = [
    ROOT_DIR / "backtest" / "backtest_2025_results.csv",
    ROOT_DIR / "backtest" / "backtest_2026_results.csv",
]
DEFAULT_OUT = ROOT_DIR / "data" / "enrichment" / "context_pool.sqlite"
DEFAULT_TRAITS = ROOT_DIR / "data" / "enrichment" / "trait_snapshots.sqlite"

TRAIT_DELTA_FIELDS = [
    "experience_score_diff",
    "recent_form_score_diff",
    "cardio_score_diff",
    "durability_risk_score_diff",
    "defensive_exposure_score_diff",
    "offensive_control_score_diff",
    "anti_control_score_diff",
    "scramble_score_diff",
    "striking_pressure_score_diff",
    "striking_efficiency_score_diff",
    "grappling_threat_score_diff",
    "finishing_threat_score_diff",
    "variance_score_diff",
]

TRAIT_VALIDATION_NOTES = {
    "cardio_score_diff": {
        "label": "pace_retention",
        "assessment_corr": 0.454,
        "status": "first_pass_aligned",
    },
    "striking_efficiency_score_diff": {
        "label": "distance_control",
        "assessment_corr": 0.244,
        "status": "mild_first_pass_alignment",
    },
    "recent_form_score_diff": {
        "label": "fight_iq",
        "assessment_corr": 0.167,
        "status": "weak_first_pass_alignment",
    },
    "scramble_score_diff": {
        "label": "scramble",
        "assessment_corr": 0.155,
        "status": "weak_first_pass_alignment",
    },
    "anti_control_score_diff": {
        "label": "scramble",
        "assessment_corr": 0.020,
        "status": "refined_v0_1_monitor",
    },
    "defensive_exposure_score_diff": {
        "label": "hittability",
        "assessment_corr": -0.135,
        "status": "inverse_alignment_possible_monitor",
    },
}


POOL_COLUMNS = [
    "season",
    "source_results",
    "row_num",
    "source_row_key",
    "date",
    "main_fight_id",
    "sergey_fight_id",
    "fighter1",
    "fighter2",
    "pick",
    "winner",
    "pick_prob",
    "pick_odds",
    "pick_correct",
    "actual_pnl",
    "bet",
    "skip_reason",
    "female",
    "edge",
    "odds_source_file",
    "odds_source_line",
    "odds_source_type",
    "odds_source_row",
    "source_event_id",
    "source_url",
    "scraped_at",
    "bookmaker",
    "odds_timestamp",
    "odds_is_opening_line",
    "odds_is_closing_line",
    "join_status",
    "join_method",
    "fighter1_elo",
    "fighter2_elo",
    "pick_elo",
    "opponent_elo",
    "pick_elo_diff",
    "abs_elo_diff",
    "model_agrees_with_elo",
    "elo_pick",
    "elo_pick_odds",
    "elo_pick_correct",
    "elo_pick_pnl",
    "pick_prior_fight_count",
    "opponent_prior_fight_count",
    "pick_avg_prior_opponent_elo",
    "opponent_avg_prior_opponent_elo",
    "pick_recent3_prior_opponent_elo",
    "opponent_recent3_prior_opponent_elo",
    "pick_best_win_opponent_elo",
    "opponent_best_win_opponent_elo",
    "pick_opponent_quality_diff",
    "pick_recent_opponent_quality_diff",
    "pick_best_win_quality_diff",
    "pick_peak_elo_as_of",
    "opponent_peak_elo_as_of",
    "pick_current_vs_peak_decline",
    "opponent_current_vs_peak_decline",
    "pick_decline_diff",
    "pick_recent_fights_json",
    "opponent_recent_fights_json",
    "market_implied_prob",
    "elo_implied_prob",
    "model_minus_elo_prob",
    "market_minus_elo_prob",
    "model_market_elo_triangle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generated context_pool.sqlite from ELO-enriched backtests.")
    parser.add_argument(
        "--results",
        type=Path,
        action="append",
        default=[],
        help="Backtest results CSV. Repeatable. Defaults to 2025 and 2026 results if present.",
    )
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--traits",
        type=Path,
        default=DEFAULT_TRAITS,
        help="Optional trait_snapshots.sqlite to materialize trait-delta evidence when present.",
    )
    parser.add_argument("--date-tolerance-days", type=int, default=1)
    parser.add_argument(
        "--similar-elo",
        nargs=2,
        metavar=("PICK_PROB", "PICK_ELO_DIFF"),
        type=float,
        help="After building, print similar historical rows by pick_prob and pick_elo_diff.",
    )
    parser.add_argument("--limit", type=int, default=12, help="Limit for --similar-elo output.")
    return parser.parse_args()


def infer_season(path: Path, row: EnrichedFight | None = None) -> int | None:
    match = next((part for part in path.stem.split("_") if part.isdigit() and len(part) == 4), None)
    if match:
        return int(match)
    if row and row.date:
        return int(row.date[:4])
    return None


def bool_to_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def clean_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def avg(values: list[int | float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def max_or_none(values: list[int | float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def quality_metrics(history: list[dict[str, Any]], *, current_elo: int | None) -> dict[str, Any]:
    opponent_elos = [item["opponent_elo"] for item in history if item["opponent_elo"] is not None]
    win_opponent_elos = [
        item["opponent_elo"]
        for item in history
        if item["won"] is True and item["opponent_elo"] is not None
    ]
    own_elos = [item["own_elo"] for item in history if item["own_elo"] is not None]
    if current_elo is not None:
        own_elos.append(current_elo)
    peak_elo = max_or_none(own_elos)
    return {
        "prior_fight_count": len(history),
        "avg_prior_opponent_elo": avg(opponent_elos),
        "recent3_prior_opponent_elo": avg(opponent_elos[-3:]),
        "recent_fights": history[-3:],
        "best_win_opponent_elo": max_or_none(win_opponent_elos),
        "peak_elo_as_of": peak_elo,
        "current_vs_peak_decline": None if current_elo is None or peak_elo is None else peak_elo - current_elo,
    }


def load_opponent_quality(sidecar_path: Path) -> dict[int, dict[str, Any]]:
    conn = sqlite3.connect(sidecar_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                fight_id,
                fight_date,
                fighter_red_id,
                fighter_red_name,
                fighter_blue_id,
                fighter_blue_name,
                winner_name,
                winner_id,
                fighter_red_elo,
                fighter_blue_elo,
                short_method,
                scheduled_rounds
            FROM fights
            WHERE fight_date IS NOT NULL
              AND fighter_red_id IS NOT NULL
              AND fighter_blue_id IS NOT NULL
            ORDER BY fight_date, fight_id
            """
        ).fetchall()
    finally:
        conn.close()

    by_date: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        by_date[str(row["fight_date"])].append(row)

    histories: dict[int, list[dict[str, Any]]] = defaultdict(list)
    quality_by_fight: dict[int, dict[str, Any]] = {}

    for fight_date in sorted(by_date):
        day_rows = by_date[fight_date]
        for row in day_rows:
            red_id = int(row["fighter_red_id"])
            blue_id = int(row["fighter_blue_id"])
            red_elo = row["fighter_red_elo"]
            blue_elo = row["fighter_blue_elo"]
            quality_by_fight[int(row["fight_id"])] = {
                "red_name": row["fighter_red_name"],
                "blue_name": row["fighter_blue_name"],
                "red": quality_metrics(histories[red_id], current_elo=red_elo),
                "blue": quality_metrics(histories[blue_id], current_elo=blue_elo),
            }

        for row in day_rows:
            red_id = int(row["fighter_red_id"])
            blue_id = int(row["fighter_blue_id"])
            winner_id = int(row["winner_id"]) if row["winner_id"] is not None else None
            histories[red_id].append(
                {
                    "date": str(row["fight_date"]),
                    "opponent": row["fighter_blue_name"],
                    "own_elo": row["fighter_red_elo"],
                    "opponent_elo": row["fighter_blue_elo"],
                    "won": winner_id == red_id,
                    "result": "W" if winner_id == red_id else ("L" if winner_id == blue_id else None),
                    "method": row["short_method"],
                    "scheduled_rounds": row["scheduled_rounds"],
                    "fight_id": int(row["fight_id"]),
                }
            )
            histories[blue_id].append(
                {
                    "date": str(row["fight_date"]),
                    "opponent": row["fighter_red_name"],
                    "own_elo": row["fighter_blue_elo"],
                    "opponent_elo": row["fighter_red_elo"],
                    "won": winner_id == blue_id,
                    "result": "W" if winner_id == blue_id else ("L" if winner_id == red_id else None),
                    "method": row["short_method"],
                    "scheduled_rounds": row["scheduled_rounds"],
                    "fight_id": int(row["fight_id"]),
                }
            )

    return quality_by_fight


def oriented_quality(
    row: EnrichedFight,
    quality_by_fight: dict[int, dict[str, Any]],
    aliases: dict[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if row.sergey_fight_id is None:
        return None, None
    entry = quality_by_fight.get(row.sergey_fight_id)
    if not entry:
        return None, None

    red_name = canonical_name(entry["red_name"], aliases)
    blue_name = canonical_name(entry["blue_name"], aliases)
    pick_key = opponent_key = None
    if canonical_name(row.pick, aliases) == red_name:
        pick_key, opponent_key = "red", "blue"
    elif canonical_name(row.pick, aliases) == blue_name:
        pick_key, opponent_key = "blue", "red"
    elif names_match(row.pick, row.fighter1, aliases):
        f1_name = canonical_name(row.fighter1, aliases)
        if f1_name == red_name:
            pick_key, opponent_key = "red", "blue"
        elif f1_name == blue_name:
            pick_key, opponent_key = "blue", "red"
    elif names_match(row.pick, row.fighter2, aliases):
        f2_name = canonical_name(row.fighter2, aliases)
        if f2_name == red_name:
            pick_key, opponent_key = "red", "blue"
        elif f2_name == blue_name:
            pick_key, opponent_key = "blue", "red"

    if pick_key is None or opponent_key is None:
        return None, None
    return entry[pick_key], entry[opponent_key]


def diff(a: float | int | None, b: float | int | None) -> float | None:
    return None if a is None or b is None else float(a) - float(b)


def elo_implied_probability(elo_diff: int | float | None) -> float | None:
    if elo_diff is None:
        return None
    return 1 / (1 + 10 ** (-float(elo_diff) / 400))


def triangle_label(model_minus_elo: float | None, market_minus_elo: float | None) -> str | None:
    if model_minus_elo is None or market_minus_elo is None:
        return None
    if model_minus_elo < 0 and market_minus_elo < 0:
        return "model_and_market_under_elo"
    if model_minus_elo > 0 and market_minus_elo > 0:
        return "model_and_market_over_elo"
    if model_minus_elo >= 0 and market_minus_elo < 0:
        return "model_over_market_under_elo"
    return "model_under_market_over_elo"


def row_values(
    row: EnrichedFight,
    *,
    season: int | None,
    source_results: Path,
    quality_by_fight: dict[int, dict[str, Any]],
    aliases: dict[str, str],
) -> dict[str, Any]:
    pick_quality, opponent_quality = oriented_quality(row, quality_by_fight, aliases)
    pick_quality = pick_quality or {}
    opponent_quality = opponent_quality or {}
    market_implied = american_implied_prob(row.pick_odds)
    elo_implied = elo_implied_probability(row.pick_elo_diff)
    model_minus_elo = diff(row.pick_prob, elo_implied)
    market_minus_elo = diff(market_implied, elo_implied)
    return {
        "season": season,
        "source_results": str(source_results.relative_to(ROOT_DIR) if source_results.is_relative_to(ROOT_DIR) else source_results),
        "row_num": row.row_num,
        "source_row_key": row.source_row_key,
        "date": row.date,
        "main_fight_id": row.main_fight_id,
        "sergey_fight_id": row.sergey_fight_id,
        "fighter1": row.fighter1,
        "fighter2": row.fighter2,
        "pick": row.pick,
        "winner": row.winner,
        "pick_prob": clean_float(row.pick_prob),
        "pick_odds": row.pick_odds,
        "pick_correct": bool_to_int(row.pick_correct),
        "actual_pnl": clean_float(row.actual_pnl),
        "bet": bool_to_int(row.bet),
        "skip_reason": row.skip_reason,
        "female": bool_to_int(row.female),
        "edge": clean_float(row.edge),
        "odds_source_file": row.odds_source_file,
        "odds_source_line": row.odds_source_line,
        "odds_source_type": row.odds_source_type,
        "odds_source_row": row.odds_source_row,
        "source_event_id": row.source_event_id,
        "source_url": row.source_url,
        "scraped_at": row.scraped_at,
        "bookmaker": row.bookmaker,
        "odds_timestamp": row.odds_timestamp,
        "odds_is_opening_line": bool_to_int(row.odds_is_opening_line),
        "odds_is_closing_line": bool_to_int(row.odds_is_closing_line),
        "join_status": row.join_status,
        "join_method": row.join_method,
        "fighter1_elo": row.fighter1_elo,
        "fighter2_elo": row.fighter2_elo,
        "pick_elo": row.pick_elo,
        "opponent_elo": row.opponent_elo,
        "pick_elo_diff": row.pick_elo_diff,
        "abs_elo_diff": row.abs_elo_diff,
        "model_agrees_with_elo": bool_to_int(row.model_agrees_with_elo),
        "elo_pick": row.elo_pick,
        "elo_pick_odds": row.elo_pick_odds,
        "elo_pick_correct": bool_to_int(row.elo_pick_correct),
        "elo_pick_pnl": clean_float(row.elo_pick_pnl),
        "pick_prior_fight_count": pick_quality.get("prior_fight_count"),
        "opponent_prior_fight_count": opponent_quality.get("prior_fight_count"),
        "pick_avg_prior_opponent_elo": pick_quality.get("avg_prior_opponent_elo"),
        "opponent_avg_prior_opponent_elo": opponent_quality.get("avg_prior_opponent_elo"),
        "pick_recent3_prior_opponent_elo": pick_quality.get("recent3_prior_opponent_elo"),
        "opponent_recent3_prior_opponent_elo": opponent_quality.get("recent3_prior_opponent_elo"),
        "pick_best_win_opponent_elo": pick_quality.get("best_win_opponent_elo"),
        "opponent_best_win_opponent_elo": opponent_quality.get("best_win_opponent_elo"),
        "pick_opponent_quality_diff": diff(
            pick_quality.get("avg_prior_opponent_elo"),
            opponent_quality.get("avg_prior_opponent_elo"),
        ),
        "pick_recent_opponent_quality_diff": diff(
            pick_quality.get("recent3_prior_opponent_elo"),
            opponent_quality.get("recent3_prior_opponent_elo"),
        ),
        "pick_best_win_quality_diff": diff(
            pick_quality.get("best_win_opponent_elo"),
            opponent_quality.get("best_win_opponent_elo"),
        ),
        "pick_peak_elo_as_of": pick_quality.get("peak_elo_as_of"),
        "opponent_peak_elo_as_of": opponent_quality.get("peak_elo_as_of"),
        "pick_current_vs_peak_decline": pick_quality.get("current_vs_peak_decline"),
        "opponent_current_vs_peak_decline": opponent_quality.get("current_vs_peak_decline"),
        "pick_decline_diff": diff(
            pick_quality.get("current_vs_peak_decline"),
            opponent_quality.get("current_vs_peak_decline"),
        ),
        "pick_recent_fights_json": json.dumps(pick_quality.get("recent_fights", []), sort_keys=True),
        "opponent_recent_fights_json": json.dumps(opponent_quality.get("recent_fights", []), sort_keys=True),
        "market_implied_prob": market_implied,
        "elo_implied_prob": elo_implied,
        "model_minus_elo_prob": model_minus_elo,
        "market_minus_elo_prob": market_minus_elo,
        "model_market_elo_triangle": triangle_label(model_minus_elo, market_minus_elo),
    }


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP VIEW IF EXISTS v_agent_packet_evidence;
        DROP VIEW IF EXISTS v_recent_fight_evidence;
        DROP VIEW IF EXISTS v_pattern_evidence;
        DROP VIEW IF EXISTS v_context_targets;
        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS backtest_fight_pool;
        DROP TABLE IF EXISTS pattern_stats;
        DROP TABLE IF EXISTS evidence_items;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE backtest_fight_pool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER,
            source_results TEXT NOT NULL,
            row_num INTEGER NOT NULL,
            source_row_key TEXT NOT NULL,
            date TEXT NOT NULL,
            main_fight_id INTEGER,
            sergey_fight_id INTEGER,
            fighter1 TEXT NOT NULL,
            fighter2 TEXT NOT NULL,
            pick TEXT NOT NULL,
            winner TEXT,
            pick_prob REAL NOT NULL,
            pick_odds INTEGER,
            pick_correct INTEGER,
            actual_pnl REAL,
            bet INTEGER NOT NULL,
            skip_reason TEXT,
            female INTEGER NOT NULL,
            edge REAL,
            odds_source_file TEXT,
            odds_source_line INTEGER,
            odds_source_type TEXT,
            odds_source_row TEXT,
            source_event_id TEXT,
            source_url TEXT,
            scraped_at TEXT,
            bookmaker TEXT,
            odds_timestamp TEXT,
            odds_is_opening_line INTEGER,
            odds_is_closing_line INTEGER,
            join_status TEXT NOT NULL,
            join_method TEXT,
            fighter1_elo INTEGER,
            fighter2_elo INTEGER,
            pick_elo INTEGER,
            opponent_elo INTEGER,
            pick_elo_diff INTEGER,
            abs_elo_diff INTEGER,
            model_agrees_with_elo INTEGER,
            elo_pick TEXT,
            elo_pick_odds INTEGER,
            elo_pick_correct INTEGER,
            elo_pick_pnl REAL,
            pick_prior_fight_count INTEGER,
            opponent_prior_fight_count INTEGER,
            pick_avg_prior_opponent_elo REAL,
            opponent_avg_prior_opponent_elo REAL,
            pick_recent3_prior_opponent_elo REAL,
            opponent_recent3_prior_opponent_elo REAL,
            pick_best_win_opponent_elo REAL,
            opponent_best_win_opponent_elo REAL,
            pick_opponent_quality_diff REAL,
            pick_recent_opponent_quality_diff REAL,
            pick_best_win_quality_diff REAL,
            pick_peak_elo_as_of REAL,
            opponent_peak_elo_as_of REAL,
            pick_current_vs_peak_decline REAL,
            opponent_current_vs_peak_decline REAL,
            pick_decline_diff REAL,
            pick_recent_fights_json TEXT NOT NULL,
            opponent_recent_fights_json TEXT NOT NULL,
            market_implied_prob REAL,
            elo_implied_prob REAL,
            model_minus_elo_prob REAL,
            market_minus_elo_prob REAL,
            model_market_elo_triangle TEXT
        );

        CREATE INDEX idx_pool_date ON backtest_fight_pool(date);
        CREATE INDEX idx_pool_source_row_key ON backtest_fight_pool(source_row_key);
        CREATE INDEX idx_pool_season ON backtest_fight_pool(season);
        CREATE INDEX idx_pool_pick_prob ON backtest_fight_pool(pick_prob);
        CREATE INDEX idx_pool_pick_elo_diff ON backtest_fight_pool(pick_elo_diff);
        CREATE INDEX idx_pool_pick_odds ON backtest_fight_pool(pick_odds);
        CREATE INDEX idx_pool_bet ON backtest_fight_pool(bet);
        CREATE INDEX idx_pool_join_status ON backtest_fight_pool(join_status);
        CREATE INDEX idx_pool_pick_opp_quality_diff ON backtest_fight_pool(pick_opponent_quality_diff);
        CREATE INDEX idx_pool_market_minus_elo ON backtest_fight_pool(market_minus_elo_prob);

        CREATE TABLE pattern_stats (
            pattern_name TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            filters_json TEXT NOT NULL,
            sample_size INTEGER NOT NULL,
            graded_sample_size INTEGER NOT NULL,
            ungraded_sample_size INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            win_rate REAL,
            profit REAL NOT NULL,
            roi REAL,
            avg_confidence REAL,
            avg_edge REAL,
            avg_elo_diff REAL,
            last_graded_date TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE evidence_items (
            evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fight_pool_id INTEGER NOT NULL,
            evidence_type TEXT NOT NULL,
            evidence_role TEXT NOT NULL,
            summary TEXT NOT NULL,
            data_json TEXT NOT NULL,
            source_table TEXT NOT NULL,
            source_key TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (fight_pool_id) REFERENCES backtest_fight_pool(id)
        );

        CREATE INDEX idx_evidence_fight_pool_id ON evidence_items(fight_pool_id);
        CREATE INDEX idx_evidence_type ON evidence_items(evidence_type);
        CREATE INDEX idx_evidence_role ON evidence_items(evidence_role);
        """
    )


def insert_pool_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in POOL_COLUMNS)
    conn.executemany(
        f"INSERT INTO backtest_fight_pool ({', '.join(POOL_COLUMNS)}) VALUES ({placeholders})",
        [[row[column] for column in POOL_COLUMNS] for row in rows],
    )


PatternPredicate = Callable[[dict[str, Any]], bool]


PATTERNS: list[tuple[str, str, dict[str, Any], PatternPredicate]] = [
    (
        "all_oriented_elo",
        "All graded rows with oriented ELO.",
        {"join_status": "matched", "pick_elo_diff": "not null"},
        lambda r: r["join_status"] == "matched" and r["pick_elo_diff"] is not None,
    ),
    (
        "model_pick_higher_elo",
        "Model pick has higher pre-fight ELO than opponent.",
        {"model_agrees_with_elo": True},
        lambda r: r["model_agrees_with_elo"] == 1,
    ),
    (
        "model_pick_lower_elo",
        "Model pick has lower pre-fight ELO than opponent.",
        {"model_agrees_with_elo": False},
        lambda r: r["model_agrees_with_elo"] == 0,
    ),
    (
        "skip_50_65_elo_50_plus",
        "Skipped picks from 50-65% confidence with at least +50 ELO support.",
        {"bet": False, "pick_prob": [0.50, 0.65], "pick_elo_diff": ">= 50"},
        lambda r: r["bet"] == 0 and 0.50 <= r["pick_prob"] < 0.65 and (r["pick_elo_diff"] or -9999) >= 50,
    ),
    (
        "skip_50_65_elo_100_plus",
        "Skipped picks from 50-65% confidence with at least +100 ELO support.",
        {"bet": False, "pick_prob": [0.50, 0.65], "pick_elo_diff": ">= 100"},
        lambda r: r["bet"] == 0 and 0.50 <= r["pick_prob"] < 0.65 and (r["pick_elo_diff"] or -9999) >= 100,
    ),
    (
        "skip_50_65_elo_50_plus_not_expensive",
        "Skipped 50-65% picks with +50 ELO support and odds better than -300.",
        {"bet": False, "pick_prob": [0.50, 0.65], "pick_elo_diff": ">= 50", "pick_odds": "> -300"},
        lambda r: r["bet"] == 0
        and 0.50 <= r["pick_prob"] < 0.65
        and (r["pick_elo_diff"] or -9999) >= 50
        and r["pick_odds"] is not None
        and r["pick_odds"] > -300,
    ),
    (
        "skip_50_65_elo_50_plus_market_under_elo_10",
        "Skipped 50-65% picks with +50 ELO support where market implied probability is at least 10 points below ELO implied.",
        {"bet": False, "pick_prob": [0.50, 0.65], "pick_elo_diff": ">= 50", "market_minus_elo_prob": "<= -0.10"},
        lambda r: r["bet"] == 0
        and 0.50 <= r["pick_prob"] < 0.65
        and (r["pick_elo_diff"] or -9999) >= 50
        and r["market_minus_elo_prob"] is not None
        and r["market_minus_elo_prob"] <= -0.10,
    ),
    (
        "skip_50_65_elo_50_plus_opp_quality_support",
        "Skipped 50-65% picks with +50 ELO support and non-negative average prior opponent-quality diff.",
        {"bet": False, "pick_prob": [0.50, 0.65], "pick_elo_diff": ">= 50", "pick_opponent_quality_diff": ">= 0"},
        lambda r: r["bet"] == 0
        and 0.50 <= r["pick_prob"] < 0.65
        and (r["pick_elo_diff"] or -9999) >= 50
        and r["pick_opponent_quality_diff"] is not None
        and r["pick_opponent_quality_diff"] >= 0,
    ),
    (
        "skip_50_65_elo_50_plus_opp_quality_against",
        "Skipped 50-65% picks with +50 ELO support but negative average prior opponent-quality diff.",
        {"bet": False, "pick_prob": [0.50, 0.65], "pick_elo_diff": ">= 50", "pick_opponent_quality_diff": "< 0"},
        lambda r: r["bet"] == 0
        and 0.50 <= r["pick_prob"] < 0.65
        and (r["pick_elo_diff"] or -9999) >= 50
        and r["pick_opponent_quality_diff"] is not None
        and r["pick_opponent_quality_diff"] < 0,
    ),
    (
        "bet_elo_support",
        "Placed bets where model pick had higher ELO.",
        {"bet": True, "pick_elo_diff": "> 0"},
        lambda r: r["bet"] == 1 and (r["pick_elo_diff"] or 0) > 0,
    ),
    (
        "bet_elo_against",
        "Placed bets where model pick had lower ELO.",
        {"bet": True, "pick_elo_diff": "< 0"},
        lambda r: r["bet"] == 1 and (r["pick_elo_diff"] or 0) < 0,
    ),
    (
        "model_pick_underdog",
        "Model-picked underdogs at plus money.",
        {"pick_odds": "> 0"},
        lambda r: (r["pick_odds"] or 0) > 0,
    ),
    (
        "underdog_elo_support",
        "Model-picked underdogs with higher ELO than opponent.",
        {"pick_odds": "> 0", "pick_elo_diff": "> 0"},
        lambda r: (r["pick_odds"] or 0) > 0 and (r["pick_elo_diff"] or 0) > 0,
    ),
    (
        "underdog_elo_against",
        "Model-picked underdogs with lower ELO than opponent.",
        {"pick_odds": "> 0", "pick_elo_diff": "< 0"},
        lambda r: (r["pick_odds"] or 0) > 0 and (r["pick_elo_diff"] or 0) < 0,
    ),
]


def build_pattern_stats(pool_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats_rows: list[dict[str, Any]] = []
    created_at = datetime.now(UTC).isoformat(timespec="seconds")

    for name, description, filters, predicate in PATTERNS:
        matching_rows = [row for row in pool_rows if predicate(row)]
        graded_rows = [row for row in matching_rows if row["pick_correct"] is not None]
        ungraded_sample_size = len(matching_rows) - len(graded_rows)
        sample_size = len(graded_rows)
        wins = sum(1 for row in graded_rows if row["pick_correct"] == 1)
        losses = sum(1 for row in graded_rows if row["pick_correct"] == 0)
        profit = sum(float(row["actual_pnl"] or 0.0) for row in graded_rows)
        stats_rows.append(
            {
                "pattern_name": name,
                "description": description,
                "filters_json": json.dumps(filters, sort_keys=True),
                "sample_size": sample_size,
                "graded_sample_size": sample_size,
                "ungraded_sample_size": ungraded_sample_size,
                "wins": wins,
                "losses": losses,
                "win_rate": wins / sample_size if sample_size else None,
                "profit": profit,
                "roi": profit / sample_size if sample_size else None,
                "avg_confidence": sum(row["pick_prob"] for row in graded_rows) / sample_size if sample_size else None,
                "avg_edge": sum(float(row["edge"] or 0.0) for row in graded_rows) / sample_size if sample_size else None,
                "avg_elo_diff": sum(float(row["pick_elo_diff"] or 0.0) for row in graded_rows) / sample_size if sample_size else None,
                "last_graded_date": max((row["date"] for row in graded_rows), default=None),
                "created_at": created_at,
            }
        )
    return stats_rows


def insert_pattern_stats(conn: sqlite3.Connection, stats_rows: list[dict[str, Any]]) -> None:
    columns = [
        "pattern_name",
        "description",
        "filters_json",
        "sample_size",
        "graded_sample_size",
        "ungraded_sample_size",
        "wins",
        "losses",
        "win_rate",
        "profit",
        "roi",
        "avg_confidence",
        "avg_edge",
        "avg_elo_diff",
        "last_graded_date",
        "created_at",
    ]
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO pattern_stats ({', '.join(columns)}) VALUES ({placeholders})",
        [[row[column] for column in columns] for row in stats_rows],
    )


def pct(value: float | None) -> str:
    return "--" if value is None else f"{value:.1%}"


def signed_pct(value: float | None) -> str:
    return "--" if value is None else f"{value:+.1%}"


def num(value: float | int | None) -> str:
    return "--" if value is None else f"{value:.0f}"


def json_data(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)


def pattern_summary(stats: dict[str, Any]) -> str:
    return (
        f"{stats['pattern_name']}: N={stats['sample_size']} "
        f"W-L={stats['wins']}-{stats['losses']} "
        f"WR={pct(stats['win_rate'])} ROI={signed_pct(stats['roi'])}"
    )


def build_evidence_items(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    pattern_stats_by_name = {
        row["pattern_name"]: dict(row)
        for row in conn.execute("SELECT * FROM pattern_stats")
    }
    rows = [dict(row) for row in conn.execute("SELECT * FROM backtest_fight_pool ORDER BY date, id")]
    evidence: list[dict[str, Any]] = []

    for row in rows:
        fight_pool_id = row["id"]
        current_decision = "bet" if row["bet"] else "skip"
        evidence.append(
            {
                "fight_pool_id": fight_pool_id,
                "evidence_type": "target_context",
                "evidence_role": "target",
                "summary": (
                    f"{row['date']} {row['fighter1']} vs {row['fighter2']} | "
                    f"pick={row['pick']} prob={pct(row['pick_prob'])} odds={row['pick_odds']} "
                    f"decision={current_decision}"
                ),
                "data_json": json_data(
                    {
                        "date": row["date"],
                        "fighter1": row["fighter1"],
                        "fighter2": row["fighter2"],
                        "pick": row["pick"],
                        "pick_prob": row["pick_prob"],
                        "pick_odds": row["pick_odds"],
                        "edge": row["edge"],
                        "current_decision": current_decision,
                        "skip_reason": row["skip_reason"],
                        "source_results": row["source_results"],
                        "row_num": row["row_num"],
                        "source_row_key": row["source_row_key"],
                        "odds_source_file": row["odds_source_file"],
                        "odds_source_line": row["odds_source_line"],
                        "odds_source_type": row["odds_source_type"],
                        "odds_source_row": row["odds_source_row"],
                        "source_event_id": row["source_event_id"],
                        "source_url": row["source_url"],
                        "scraped_at": row["scraped_at"],
                        "bookmaker": row["bookmaker"],
                        "odds_timestamp": row["odds_timestamp"],
                        "odds_is_opening_line": row["odds_is_opening_line"],
                        "odds_is_closing_line": row["odds_is_closing_line"],
                        "winner": row["winner"],
                        "pick_correct": row["pick_correct"],
                        "actual_pnl": row["actual_pnl"],
                    }
                ),
                "source_table": "backtest_fight_pool",
                "source_key": str(fight_pool_id),
                "created_at": created_at,
            }
        )

        if row["pick_elo_diff"] is not None:
            evidence.append(
                {
                    "fight_pool_id": fight_pool_id,
                    "evidence_type": "elo_snapshot",
                    "evidence_role": "context_metric",
                    "summary": (
                        f"Pre-fight ELO: pick {row['pick_elo']} vs opponent {row['opponent_elo']} "
                        f"(diff={row['pick_elo_diff']:+})"
                    ),
                    "data_json": json_data(
                        {
                            "pick_elo": row["pick_elo"],
                            "opponent_elo": row["opponent_elo"],
                            "pick_elo_diff": row["pick_elo_diff"],
                            "abs_elo_diff": row["abs_elo_diff"],
                            "model_agrees_with_elo": row["model_agrees_with_elo"],
                            "join_method": row["join_method"],
                            "sergey_fight_id": row["sergey_fight_id"],
                        }
                    ),
                    "source_table": "backtest_fight_pool",
                    "source_key": str(fight_pool_id),
                    "created_at": created_at,
                }
            )

        if row["elo_implied_prob"] is not None:
            evidence.append(
                {
                    "fight_pool_id": fight_pool_id,
                    "evidence_type": "elo_triangle",
                    "evidence_role": "context_metric",
                    "summary": (
                        f"Model/market/ELO: model={pct(row['pick_prob'])}, "
                        f"market={pct(row['market_implied_prob'])}, ELO={pct(row['elo_implied_prob'])}, "
                        f"triangle={row['model_market_elo_triangle']}"
                    ),
                    "data_json": json_data(
                        {
                            "pick_prob": row["pick_prob"],
                            "market_implied_prob": row["market_implied_prob"],
                            "elo_implied_prob": row["elo_implied_prob"],
                            "model_minus_elo_prob": row["model_minus_elo_prob"],
                            "market_minus_elo_prob": row["market_minus_elo_prob"],
                            "model_market_elo_triangle": row["model_market_elo_triangle"],
                        }
                    ),
                    "source_table": "backtest_fight_pool",
                    "source_key": str(fight_pool_id),
                    "created_at": created_at,
                }
            )

        if row["pick_opponent_quality_diff"] is not None:
            evidence.append(
                {
                    "fight_pool_id": fight_pool_id,
                    "evidence_type": "opponent_quality",
                    "evidence_role": "context_metric",
                    "summary": (
                        f"Opponent quality: avg prior opponent ELO diff="
                        f"{row['pick_opponent_quality_diff']:+.0f} "
                        f"(pick {num(row['pick_avg_prior_opponent_elo'])} vs "
                        f"opponent {num(row['opponent_avg_prior_opponent_elo'])})"
                    ),
                    "data_json": json_data(
                        {
                            "pick_prior_fight_count": row["pick_prior_fight_count"],
                            "opponent_prior_fight_count": row["opponent_prior_fight_count"],
                            "pick_avg_prior_opponent_elo": row["pick_avg_prior_opponent_elo"],
                            "opponent_avg_prior_opponent_elo": row["opponent_avg_prior_opponent_elo"],
                            "pick_recent3_prior_opponent_elo": row["pick_recent3_prior_opponent_elo"],
                            "opponent_recent3_prior_opponent_elo": row["opponent_recent3_prior_opponent_elo"],
                            "pick_best_win_opponent_elo": row["pick_best_win_opponent_elo"],
                            "opponent_best_win_opponent_elo": row["opponent_best_win_opponent_elo"],
                            "pick_opponent_quality_diff": row["pick_opponent_quality_diff"],
                            "pick_recent_opponent_quality_diff": row["pick_recent_opponent_quality_diff"],
                            "pick_best_win_quality_diff": row["pick_best_win_quality_diff"],
                            "pick_current_vs_peak_decline": row["pick_current_vs_peak_decline"],
                            "opponent_current_vs_peak_decline": row["opponent_current_vs_peak_decline"],
                            "pick_decline_diff": row["pick_decline_diff"],
                        }
                    ),
                    "source_table": "backtest_fight_pool",
                    "source_key": str(fight_pool_id),
                    "created_at": created_at,
                }
            )

        for side, source_key, raw_json in (
            ("pick", "pick_recent_fights_json", row["pick_recent_fights_json"]),
            ("opponent", "opponent_recent_fights_json", row["opponent_recent_fights_json"]),
        ):
            try:
                recent_fights = json.loads(raw_json or "[]")
            except json.JSONDecodeError:
                recent_fights = []
            for recent in recent_fights:
                evidence.append(
                    {
                        "fight_pool_id": fight_pool_id,
                        "evidence_type": "recent_fight",
                        "evidence_role": "audit_detail",
                        "summary": (
                            f"{side} recent fight: {recent.get('date')} vs {recent.get('opponent')} "
                            f"oppELO={recent.get('opponent_elo')} result={recent.get('result')}"
                        ),
                        "data_json": json_data({"side": side, **recent}),
                        "source_table": "backtest_fight_pool",
                        "source_key": f"{source_key}:{recent.get('fight_id')}",
                        "created_at": created_at,
                    }
                )

        for pattern_name, _, _, predicate in PATTERNS:
            if not predicate(row):
                continue
            stats = pattern_stats_by_name.get(pattern_name)
            if not stats:
                continue
            evidence.append(
                {
                    "fight_pool_id": fight_pool_id,
                    "evidence_type": "pattern_stat",
                    "evidence_role": "aggregate_pattern",
                    "summary": pattern_summary(stats),
                    "data_json": json_data(
                        {
                            "pattern_name": pattern_name,
                            "description": stats["description"],
                            "filters": json.loads(stats["filters_json"]),
                            "sample_size": stats["sample_size"],
                            "graded_sample_size": stats["graded_sample_size"],
                            "ungraded_sample_size": stats["ungraded_sample_size"],
                            "wins": stats["wins"],
                            "losses": stats["losses"],
                            "win_rate": stats["win_rate"],
                            "profit": stats["profit"],
                            "roi": stats["roi"],
                            "avg_confidence": stats["avg_confidence"],
                            "avg_edge": stats["avg_edge"],
                            "avg_elo_diff": stats["avg_elo_diff"],
                            "last_graded_date": stats["last_graded_date"],
                        }
                    ),
                    "source_table": "pattern_stats",
                    "source_key": pattern_name,
                    "created_at": created_at,
                }
            )

    return evidence


def trait_delta_summary(row: dict[str, Any]) -> str:
    def delta(field: str) -> str:
        value = row.get(field)
        return "--" if value is None else f"{value:+.1f}"

    return (
        "Trait deltas v0: "
        f"cardio={delta('cardio_score_diff')}, "
        f"striking_eff={delta('striking_efficiency_score_diff')}, "
        f"control={delta('offensive_control_score_diff')}, "
        f"anti_control={delta('anti_control_score_diff')}, "
        f"def_exposure_risk={delta('defensive_exposure_score_diff')}"
    )


def load_trait_delta_rows(traits_path: Path) -> dict[int, list[dict[str, Any]]]:
    if not traits_path.exists():
        return {}

    trait_conn = sqlite3.connect(traits_path)
    trait_conn.row_factory = sqlite3.Row
    try:
        rows = trait_conn.execute("SELECT * FROM v_trait_pair_deltas").fetchall()
    finally:
        trait_conn.close()

    by_fight: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fight[int(row["main_fight_id"])].append(dict(row))
    return by_fight


def build_trait_evidence_items(
    conn: sqlite3.Connection,
    traits_path: Path,
    aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    trait_rows_by_fight = load_trait_delta_rows(traits_path)
    if not trait_rows_by_fight:
        return []

    conn.row_factory = sqlite3.Row
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    evidence: list[dict[str, Any]] = []
    context_rows = conn.execute(
        """
        SELECT id, main_fight_id, pick
        FROM backtest_fight_pool
        WHERE main_fight_id IS NOT NULL
        """
    ).fetchall()

    for context_row in context_rows:
        main_fight_id = int(context_row["main_fight_id"])
        pick_key = canonical_name(context_row["pick"], aliases or {})
        trait_row = next(
            (
                row
                for row in trait_rows_by_fight.get(main_fight_id, [])
                if canonical_name(row["fighter_name"], aliases or {}) == pick_key
            ),
            None,
        )
        if trait_row is None:
            continue

        deltas = {field: trait_row.get(field) for field in TRAIT_DELTA_FIELDS}
        validation = {
            field: TRAIT_VALIDATION_NOTES.get(field, {"status": "exploratory_unvalidated"})
            for field in TRAIT_DELTA_FIELDS
        }
        evidence.append(
            {
                "fight_pool_id": context_row["id"],
                "evidence_type": "trait_delta",
                "evidence_role": "context_metric",
                "summary": trait_delta_summary(trait_row),
                "data_json": json_data(
                    {
                        "trait_version": "trait_v0_1_stats_totals",
                        "fighter_name": trait_row["fighter_name"],
                        "opponent_name": trait_row["opponent_name"],
                        "fight_count": trait_row["fight_count"],
                        "opponent_fight_count": trait_row["opponent_fight_count"],
                        "trait_confidence": trait_row["trait_confidence"],
                        "opponent_trait_confidence": trait_row["opponent_trait_confidence"],
                        "deltas": deltas,
                        "validation_notes": validation,
                        "interpretation_note": (
                            "Positive ability-score deltas favor the pick; positive risk-score deltas "
                            "mean the pick carries more of that risk. Evidence only, not a recommendation."
                        ),
                    }
                ),
                "source_table": "trait_snapshots.v_trait_pair_deltas",
                "source_key": f"{main_fight_id}:{trait_row['fighter_id']}",
                "created_at": created_at,
            }
        )
    return evidence


def insert_evidence_items(conn: sqlite3.Connection, evidence_rows: list[dict[str, Any]]) -> None:
    if not evidence_rows:
        return
    columns = [
        "fight_pool_id",
        "evidence_type",
        "evidence_role",
        "summary",
        "data_json",
        "source_table",
        "source_key",
        "created_at",
    ]
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO evidence_items ({', '.join(columns)}) VALUES ({placeholders})",
        [[row[column] for column in columns] for row in evidence_rows],
    )


def create_agent_views(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE VIEW v_context_targets AS
        SELECT
            id AS fight_pool_id,
            source_results,
            row_num,
            source_row_key,
            date,
            season,
            fighter1,
            fighter2,
            pick,
            pick_prob,
            pick_odds,
            market_implied_prob,
            edge,
            CASE WHEN bet = 1 THEN 'bet' ELSE 'skip' END AS current_decision,
            skip_reason,
            odds_source_file,
            odds_source_line,
            odds_source_type,
            odds_source_row,
            source_event_id,
            source_url,
            scraped_at,
            bookmaker,
            odds_timestamp,
            odds_is_opening_line,
            odds_is_closing_line,
            pick_elo,
            opponent_elo,
            pick_elo_diff,
            elo_implied_prob,
            model_minus_elo_prob,
            market_minus_elo_prob,
            model_market_elo_triangle,
            pick_opponent_quality_diff,
            pick_recent_opponent_quality_diff,
            pick_correct,
            actual_pnl
        FROM backtest_fight_pool;

        CREATE VIEW v_pattern_evidence AS
        SELECT
            e.evidence_id,
            e.fight_pool_id,
            t.date,
            t.fighter1,
            t.fighter2,
            t.pick,
            e.source_key AS pattern_name,
            e.summary,
            e.data_json
        FROM evidence_items e
        JOIN v_context_targets t ON t.fight_pool_id = e.fight_pool_id
        WHERE e.evidence_type = 'pattern_stat';

        CREATE VIEW v_recent_fight_evidence AS
        SELECT
            e.evidence_id,
            e.fight_pool_id,
            t.date,
            t.fighter1,
            t.fighter2,
            e.summary,
            e.data_json
        FROM evidence_items e
        JOIN v_context_targets t ON t.fight_pool_id = e.fight_pool_id
        WHERE e.evidence_type = 'recent_fight';

        CREATE VIEW v_agent_packet_evidence AS
        SELECT
            e.evidence_id,
            e.fight_pool_id,
            t.date,
            t.fighter1,
            t.fighter2,
            t.pick,
            e.evidence_role,
            e.evidence_type,
            e.summary,
            e.data_json,
            e.source_table,
            e.source_key
        FROM evidence_items e
        JOIN v_context_targets t ON t.fight_pool_id = e.fight_pool_id;
        """
    )


def insert_metadata(
    conn: sqlite3.Connection,
    *,
    results_paths: list[Path],
    sidecar: Path,
    traits: Path,
    total_rows: int,
    trait_evidence_rows: int,
) -> None:
    metadata = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_results": json.dumps([str(path.relative_to(ROOT_DIR) if path.is_relative_to(ROOT_DIR) else path) for path in results_paths]),
        "source_sidecar": str(sidecar.relative_to(ROOT_DIR) if sidecar.is_relative_to(ROOT_DIR) else sidecar),
        "source_traits": str(traits.relative_to(ROOT_DIR) if traits.is_relative_to(ROOT_DIR) else traits),
        "total_rows": str(total_rows),
        "trait_evidence_rows": str(trait_evidence_rows),
        "schema_version": "8",
    }
    conn.executemany("INSERT INTO metadata (key, value) VALUES (?, ?)", metadata.items())


def print_summary(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) FROM backtest_fight_pool").fetchone()[0]
    oriented = conn.execute(
        "SELECT COUNT(*) FROM backtest_fight_pool WHERE join_status = 'matched' AND pick_elo_diff IS NOT NULL"
    ).fetchone()[0]
    print(f"Rows in pool:            {total}")
    print(f"Rows with oriented ELO:  {oriented} ({oriented / total:.1%})" if total else "Rows with oriented ELO:  0")
    print("\nBy season")
    for row in conn.execute(
        """
        SELECT season, COUNT(*) AS rows, SUM(CASE WHEN join_status = 'matched' AND pick_elo_diff IS NOT NULL THEN 1 ELSE 0 END) AS oriented
        FROM backtest_fight_pool
        GROUP BY season
        ORDER BY season
        """
    ):
        coverage = row["oriented"] / row["rows"] if row["rows"] else 0.0
        print(f"  {row['season']}: rows={row['rows']:>3} oriented={row['oriented']:>3} ({coverage:.1%})")

    print("\nPattern stats")
    for row in conn.execute(
        """
        SELECT pattern_name, sample_size, ungraded_sample_size, wins, losses, win_rate, profit, roi, last_graded_date
        FROM pattern_stats
        ORDER BY sample_size DESC
        """
    ):
        win_rate = "--" if row["win_rate"] is None else f"{row['win_rate']:.1%}"
        roi = "--" if row["roi"] is None else f"{row['roi'] * 100:+.1f}%"
        print(
            f"  {row['pattern_name']:<28} N={row['sample_size']:>3} "
            f"pending={row['ungraded_sample_size']:>2} "
            f"W-L={row['wins']}-{row['losses']} WR={win_rate:>6} "
            f"PnL={row['profit']:+.2f} ROI={roi:>7} "
            f"last={row['last_graded_date'] or '-'}"
        )

    print("\nEvidence rows")
    for row in conn.execute(
        """
        SELECT evidence_type, COUNT(*) AS rows
        FROM evidence_items
        GROUP BY evidence_type
        ORDER BY rows DESC, evidence_type
        """
    ):
        print(f"  {row['evidence_type']:<24} {row['rows']:>5}")


def print_similar_rows(conn: sqlite3.Connection, pick_prob: float, pick_elo_diff: float, *, limit: int) -> None:
    print(f"\nSimilar historical rows for pick_prob={pick_prob:.1%}, pick_elo_diff={pick_elo_diff:+.0f}")
    print("-" * 118)
    rows = conn.execute(
        """
        SELECT
            date,
            fighter1,
            fighter2,
            pick,
            pick_prob,
            pick_odds,
            edge,
            pick_elo_diff,
            pick_correct,
            actual_pnl,
            skip_reason,
            (
                ABS(pick_prob - ?)
                + ABS(COALESCE(pick_elo_diff, 0) - ?) / 300.0
            ) AS distance
        FROM backtest_fight_pool
        WHERE join_status = 'matched'
          AND pick_elo_diff IS NOT NULL
          AND pick_correct IS NOT NULL
        ORDER BY distance ASC
        LIMIT ?
        """,
        (pick_prob, pick_elo_diff, limit),
    ).fetchall()
    for row in rows:
        result = "W" if row["pick_correct"] else "L"
        edge = "--" if row["edge"] is None else f"{row['edge']:+.1%}"
        print(
            f"{row['date']} {row['fighter1']} vs {row['fighter2']} | "
            f"pick={row['pick']} @{row['pick_odds']} prob={row['pick_prob']:.1%} "
            f"edge={edge} ELO={row['pick_elo_diff']:+} {result} "
            f"pnl={row['actual_pnl']:+.2f} reason={row['skip_reason'] or '-'}"
        )


def main() -> None:
    args = parse_args()
    results_paths = args.results or [path for path in DEFAULT_RESULTS if path.exists()]
    if not results_paths:
        raise SystemExit("No results CSVs found. Pass --results at least once.")

    aliases = load_aliases(DEFAULT_ALIAS_SOURCES)
    quality_by_fight = load_opponent_quality(args.sidecar)
    pool_rows: list[dict[str, Any]] = []

    for path in results_paths:
        enriched = enrich_results(
            path,
            args.sidecar,
            aliases,
            date_tolerance_days=args.date_tolerance_days,
        )
        season = infer_season(path, enriched[0] if enriched else None)
        for row in enriched:
            pool_rows.append(
                row_values(
                    row,
                    season=season,
                    source_results=path,
                    quality_by_fight=quality_by_fight,
                    aliases=aliases,
                )
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    conn = sqlite3.connect(args.out)
    try:
        create_schema(conn)
        insert_pool_rows(conn, pool_rows)
        insert_pattern_stats(conn, build_pattern_stats(pool_rows))
        evidence_rows = build_evidence_items(conn)
        trait_evidence_rows = build_trait_evidence_items(conn, args.traits, aliases)
        if not trait_evidence_rows:
            print(f"Trait evidence rows: 0 (trait DB missing or no joinable rows at {args.traits})")
        evidence_rows.extend(trait_evidence_rows)
        insert_evidence_items(conn, evidence_rows)
        create_agent_views(conn)
        insert_metadata(
            conn,
            results_paths=results_paths,
            sidecar=args.sidecar,
            traits=args.traits,
            total_rows=len(pool_rows),
            trait_evidence_rows=len(trait_evidence_rows),
        )
        conn.commit()
        print(f"Context pool written: {args.out}")
        print_summary(conn)
        if args.similar_elo:
            print_similar_rows(conn, args.similar_elo[0], args.similar_elo[1], limit=args.limit)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
