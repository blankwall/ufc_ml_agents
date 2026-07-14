#!/usr/bin/env python3
"""
Validate the independent context-scoring pipeline on historical graded rows.

This script does not call the app, does not change betting rules, and does not
use an LLM. It reads context_pool.sqlite and evaluates pattern_score_v0 across
historical fights. By default, each target row is scored with leave-one-out
pattern aggregates so the target fight does not contribute to its own score.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.build_context_pool import PATTERNS  # noqa: E402
from backtest.context_packet import (  # noqa: E402
    DEFAULT_POOL,
    applicable_pattern_names,
    build_pattern_score,
    fmt_signed_pct,
    row_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pattern_score_v0 on historical context-pool rows.")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument(
        "--mode",
        choices=("leave-one-out", "temporal", "in-sample"),
        default="leave-one-out",
        help="How to build aggregate pattern evidence for each target row.",
    )
    parser.add_argument("--min-score", type=int, default=0, help="Only print scored rows at or above this score.")
    parser.add_argument("--skips-only", action="store_true", help="Validate only rows current rules skipped.")
    parser.add_argument("--json-only", action="store_true", help="Print validation rows as JSON.")
    return parser.parse_args()


def fetch_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM backtest_fight_pool ORDER BY date, id").fetchall()
    return [row_to_dict(row) for row in rows]


def graded_rows(rows: Iterable[dict[str, Any]], *, skips_only: bool) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("pick_correct") is not None and (not skips_only or row.get("bet") is False)
    ]


def evidence_rows_for_target(
    rows: list[dict[str, Any]],
    target: dict[str, Any],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "in-sample":
        return [row for row in rows if row.get("pick_correct") is not None]
    if mode == "temporal":
        return [
            row for row in rows
            if row.get("pick_correct") is not None
            and row["id"] != target["id"]
            and row["date"] < target["date"]
        ]
    return [
        row for row in rows
        if row.get("pick_correct") is not None and row["id"] != target["id"]
    ]


def pattern_by_name(pattern_name: str):
    pattern = next((pattern for pattern in PATTERNS if pattern[0] == pattern_name), None)
    if pattern is None:
        raise ValueError(f"Unknown pattern: {pattern_name}")
    return pattern


def aggregate_pattern(pattern_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    name, description, filters, predicate = pattern_by_name(pattern_name)
    matching_rows = [row for row in rows if predicate(row)]
    sample_size = len(matching_rows)
    wins = sum(1 for row in matching_rows if row["pick_correct"] is True)
    losses = sum(1 for row in matching_rows if row["pick_correct"] is False)
    profit = sum(float(row["actual_pnl"] or 0.0) for row in matching_rows)
    return {
        "pattern_name": name,
        "description": description,
        "filters": filters,
        "sample_size": sample_size,
        "graded_sample_size": sample_size,
        "ungraded_sample_size": 0,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / sample_size if sample_size else None,
        "profit": profit,
        "roi": profit / sample_size if sample_size else None,
        "avg_confidence": sum(row["pick_prob"] for row in matching_rows) / sample_size if sample_size else None,
        "avg_edge": sum(float(row["edge"] or 0.0) for row in matching_rows) / sample_size if sample_size else None,
        "avg_elo_diff": sum(float(row["pick_elo_diff"] or 0.0) for row in matching_rows) / sample_size
        if sample_size
        else None,
        "last_graded_date": max((row["date"] for row in matching_rows), default=None),
        "evidence_role": "decision_support",
    }


def build_validation_row(
    rows: list[dict[str, Any]],
    target: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    evidence_rows = evidence_rows_for_target(rows, target, mode=mode)
    patterns = [
        aggregate_pattern(pattern_name, evidence_rows)
        for pattern_name in applicable_pattern_names(target)
    ]
    score = build_pattern_score(target, patterns)
    return {
        "id": target["id"],
        "date": target["date"],
        "season": target["season"],
        "fight": f"{target['fighter1']} vs {target['fighter2']}",
        "pick": target["pick"],
        "pick_prob": target["pick_prob"],
        "pick_odds": target["pick_odds"],
        "edge": target["edge"],
        "pick_elo_diff": target["pick_elo_diff"],
        "bet": target["bet"],
        "skip_reason": target["skip_reason"],
        "pick_correct": target["pick_correct"],
        "actual_pnl": target["actual_pnl"],
        "score": score["score"],
        "support_level": score["support_level"],
        "source_pattern": score["source_pattern"],
        "source_n": None if score["basis"] is None else score["basis"]["sample_size"],
        "source_roi": None if score["basis"] is None else score["basis"]["roi"],
    }


def american_bucket(odds: int | None) -> str:
    if odds is None:
        return "unknown"
    if odds <= -300:
        return "favorite_300_plus"
    if odds < 0:
        return "favorite_under_300"
    if odds < 200:
        return "underdog_plus_100_199"
    return "underdog_plus_200_plus"


def edge_bucket(edge: float | None) -> str:
    if edge is None:
        return "edge_unknown"
    if edge < 0:
        return "edge_negative"
    if edge < 0.05:
        return "edge_0_5"
    if edge < 0.10:
        return "edge_5_10"
    return "edge_10_plus"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    wins = sum(1 for row in rows if row["pick_correct"] is True)
    losses = sum(1 for row in rows if row["pick_correct"] is False)
    profit = sum(float(row["actual_pnl"] or 0.0) for row in rows)
    return {
        "n": count,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / count if count else None,
        "profit": profit,
        "roi": profit / count if count else None,
    }


def grouped(rows: list[dict[str, Any]], key: str) -> list[tuple[Any, dict[str, Any]]]:
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row[key]].append(row)
    return sorted(buckets.items(), key=lambda item: str(item[0]))


def print_group(title: str, rows: list[dict[str, Any]], key: str) -> None:
    print(f"\n{title}")
    print("| Group | N | W-L | WinRate | PnL | ROI |")
    print("|---|---:|---:|---:|---:|---:|")
    for group, group_rows in grouped(rows, key):
        stats = summarize(group_rows)
        wr = "--" if stats["win_rate"] is None else f"{stats['win_rate']:.1%}"
        roi = "--" if stats["roi"] is None else fmt_signed_pct(stats["roi"])
        print(
            f"| {group} | {stats['n']} | {stats['wins']}-{stats['losses']} | "
            f"{wr} | {stats['profit']:+.2f} | {roi} |"
        )


def print_report(rows: list[dict[str, Any]], *, mode: str, skips_only: bool, min_score: int) -> None:
    filtered = [row for row in rows if row["score"] >= min_score]
    for row in filtered:
        row["edge_bucket"] = edge_bucket(row["edge"])
        row["odds_bucket"] = american_bucket(row["pick_odds"])
        row["bet_group"] = "bet" if row["bet"] else "skip"

    print("=" * 100)
    print("CONTEXT PIPELINE VALIDATION")
    print("=" * 100)
    print(f"Mode: {mode}")
    print(f"Scope: {'skips only' if skips_only else 'all graded rows'}")
    print(f"Rows scored: {len(rows)}")
    print(f"Rows shown:  {len(filtered)} (min_score={min_score})")
    overall = summarize(filtered)
    overall_wr = "--" if overall["win_rate"] is None else f"{overall['win_rate']:.1%}"
    overall_roi = "--" if overall["roi"] is None else fmt_signed_pct(overall["roi"])
    print(
        f"Overall shown: N={overall['n']} W-L={overall['wins']}-{overall['losses']} "
        f"WR={overall_wr} PnL={overall['profit']:+.2f} ROI={overall_roi}"
    )

    print_group("By score", filtered, "score")
    print_group("By source pattern", [row for row in filtered if row["source_pattern"]], "source_pattern")
    print_group("By season", filtered, "season")
    print_group("By bet/skip", filtered, "bet_group")
    print_group("By edge bucket", filtered, "edge_bucket")
    print_group("By odds bucket", filtered, "odds_bucket")


def main() -> None:
    args = parse_args()
    if not args.pool.exists():
        raise SystemExit(f"Context pool not found: {args.pool}. Run backtest/build_context_pool.py first.")

    conn = sqlite3.connect(args.pool)
    conn.row_factory = sqlite3.Row
    try:
        rows = fetch_rows(conn)
    finally:
        conn.close()

    targets = graded_rows(rows, skips_only=args.skips_only)
    validation_rows = [
        build_validation_row(rows, target, mode=args.mode)
        for target in targets
    ]

    if args.json_only:
        print(json.dumps(validation_rows, indent=2, sort_keys=True))
    else:
        print_report(validation_rows, mode=args.mode, skips_only=args.skips_only, min_score=args.min_score)


if __name__ == "__main__":
    main()
