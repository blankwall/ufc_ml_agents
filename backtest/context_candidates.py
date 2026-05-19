#!/usr/bin/env python3
"""
Rank current skipped fights that have strong historical context evidence.

This is a deterministic watchlist: it reads context_pool.sqlite, scores each
ungraded skipped row with the same pattern_score_v0 logic used by context_packet,
and prints candidates where historical aggregate evidence may deserve deeper
review. It is not a betting recommendation or rule engine.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.context_packet import (  # noqa: E402
    DEFAULT_POOL,
    build_pattern_score,
    fmt_pct,
    fmt_signed_pct,
    pattern_payload,
    row_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List current skipped fights with strong empirical context evidence.")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--as-of", default=date.today().isoformat(), help="Only include fights on/after this YYYY-MM-DD.")
    parser.add_argument("--min-score", type=int, default=7, help="Minimum pattern_score_v0 to show.")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--include-past-pending", action="store_true", help="Include ungraded rows before --as-of.")
    parser.add_argument("--json-only", action="store_true", help="Print candidates as JSON.")
    return parser.parse_args()


def fetch_ungraded_skips(conn: sqlite3.Connection, *, as_of: str, include_past_pending: bool) -> list[dict[str, Any]]:
    date_clause = "" if include_past_pending else "AND date >= ?"
    params = () if include_past_pending else (as_of,)
    rows = conn.execute(
        f"""
        SELECT *
        FROM backtest_fight_pool
        WHERE pick_correct IS NULL
          AND bet = 0
          {date_clause}
        ORDER BY date, fighter1, fighter2, id
        """,
        params,
    ).fetchall()
    return [row_to_dict(row) for row in rows]


def build_candidates(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    *,
    min_score: int,
) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        patterns = pattern_payload(conn, row)
        score = build_pattern_score(row, patterns)
        if score["score"] < min_score or score["source_pattern"] is None:
            continue
        candidates.append(
            {
                "date": row["date"],
                "fight": f"{row['fighter1']} vs {row['fighter2']}",
                "pick": row["pick"],
                "pick_prob": row["pick_prob"],
                "pick_odds": row["pick_odds"],
                "edge": row["edge"],
                "pick_elo_diff": row["pick_elo_diff"],
                "skip_reason": row["skip_reason"],
                "score": score["score"],
                "action": score["action"],
                "support_level": score["support_level"],
                "source_pattern": score["source_pattern"],
                "basis": score["basis"],
                "warnings": score["warnings"],
            }
        )

    return sorted(
        candidates,
        key=lambda c: (
            c["score"],
            (c["basis"] or {}).get("roi") or -999,
            (c["basis"] or {}).get("sample_size") or 0,
            c["date"],
        ),
        reverse=True,
    )


def fmt_odds(value: int | None) -> str:
    return "--" if value is None else str(value)


def print_candidates(candidates: list[dict[str, Any]], *, as_of: str, include_past_pending: bool) -> None:
    scope = "all pending skipped picks" if include_past_pending else f"pending skipped picks on/after {as_of}"
    print("=" * 120)
    print(f"CONTEXT EVIDENCE WATCHLIST ({scope})")
    print("=" * 120)
    if not candidates:
        print("No candidates met the score threshold.")
        return

    print(
        "| Date | Score | Fight | Pick | Prob | Odds | Edge | ELO | Source pattern | N / ROI | Skip reason |"
    )
    print("|---|---:|---|---|---:|---:|---:|---:|---|---:|---|")
    for candidate in candidates:
        basis = candidate["basis"] or {}
        n_roi = f"{basis.get('sample_size', 0)} / {fmt_signed_pct(basis.get('roi'))}"
        elo = "--" if candidate["pick_elo_diff"] is None else f"{candidate['pick_elo_diff']:+}"
        print(
            f"| {candidate['date']} | {candidate['score']} | {candidate['fight']} | {candidate['pick']} | "
            f"{fmt_pct(candidate['pick_prob'])} | {fmt_odds(candidate['pick_odds'])} | "
            f"{fmt_signed_pct(candidate['edge'])} | {elo} | {candidate['source_pattern']} | "
            f"{n_roi} | {candidate['skip_reason'] or '-'} |"
        )


def main() -> None:
    args = parse_args()
    if not args.pool.exists():
        raise SystemExit(f"Context pool not found: {args.pool}. Run backtest/build_context_pool.py first.")

    conn = sqlite3.connect(args.pool)
    conn.row_factory = sqlite3.Row
    try:
        rows = fetch_ungraded_skips(conn, as_of=args.as_of, include_past_pending=args.include_past_pending)
        candidates = build_candidates(conn, rows, min_score=args.min_score)[: args.limit]
    finally:
        conn.close()

    if args.json_only:
        print(json.dumps(candidates, indent=2, sort_keys=True))
    else:
        print_candidates(candidates, as_of=args.as_of, include_past_pending=args.include_past_pending)


if __name__ == "__main__":
    main()
