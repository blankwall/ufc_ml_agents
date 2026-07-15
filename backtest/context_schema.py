#!/usr/bin/env python3
"""
Print the agent-facing schema for the generated context sidecar database.

The database is built by backtest/build_context_pool.py and is intentionally
evidence-only: rows expose context, empirical pattern stats, and audit details
without making recommendations.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT_DIR / "data" / "enrichment" / "context_pool.sqlite"


TABLES = {
    "backtest_fight_pool": "One row per historical/pending fight pick, enriched with model, market, ELO, opponent-quality, recent-fight, source-row, and odds-provenance fields.",
    "pattern_stats": "Global empirical aggregates for named context patterns, including sample size, win rate, ROI, and filters.",
    "evidence_items": "Agent-ready evidence rows keyed by fight_pool_id. Contains summaries plus machine-readable JSON payloads, including ELO, pattern, recent-fight, and trait-delta evidence.",
    "metadata": "Build metadata, source files, and schema version.",
}


VIEWS = {
    "v_context_targets": "Compact fight-level target view for finding rows to inspect.",
    "v_pattern_evidence": "Pattern-stat evidence rows joined to target fight fields.",
    "v_recent_fight_evidence": "Recent-fight audit rows joined to target fight fields.",
    "v_agent_packet_evidence": "All target-level evidence rows joined to target fight fields for packet-style retrieval.",
}


EXAMPLE_QUERIES = [
    (
        "Find pending skipped fights with ELO support",
        """
SELECT fight_pool_id, date, fighter1, fighter2, pick, pick_prob, pick_odds, pick_elo_diff
FROM v_context_targets
WHERE current_decision = 'skip'
  AND pick_correct IS NULL
  AND pick_prob >= 0.50
  AND pick_prob < 0.65
  AND pick_elo_diff >= 50
ORDER BY date, fight_pool_id;
""".strip(),
    ),
    (
        "Retrieve all evidence for one fight",
        """
SELECT evidence_role, evidence_type, summary, data_json
FROM v_agent_packet_evidence
WHERE fight_pool_id = 443
ORDER BY evidence_role, evidence_type, evidence_id;
""".strip(),
    ),
    (
        "Audit source row provenance",
        """
SELECT fight_pool_id, source_results, row_num, source_row_key,
       odds_source_file, odds_source_line, bookmaker, odds_timestamp
FROM v_context_targets
WHERE fight_pool_id = 443;
""".strip(),
    ),
    (
        "Inspect empirical pattern evidence only",
        """
SELECT pattern_name, summary, data_json
FROM v_pattern_evidence
WHERE fight_pool_id = 443
ORDER BY pattern_name;
""".strip(),
    ),
    (
        "Compare model, market, and ELO probability gaps",
        """
SELECT fight_pool_id, date, pick, pick_prob, market_implied_prob, elo_implied_prob,
       model_minus_elo_prob, market_minus_elo_prob, model_market_elo_triangle
FROM v_context_targets
WHERE elo_implied_prob IS NOT NULL
ORDER BY ABS(model_minus_elo_prob) DESC
LIMIT 20;
""".strip(),
    ),
    (
        "Inspect trait-delta evidence",
        """
SELECT fight_pool_id, date, pick, summary, data_json
FROM v_agent_packet_evidence
WHERE evidence_type = 'trait_delta'
ORDER BY date, fight_pool_id
LIMIT 20;
""".strip(),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to context_pool.sqlite")
    return parser.parse_args()


def table_counts(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        counts: dict[str, int] = {}
        for table in TABLES:
            try:
                counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                continue
        return counts
    finally:
        conn.close()


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> None:
    args = parse_args()
    counts = table_counts(args.db)

    print(f"Context sidecar schema: {args.db}")
    if not counts:
        print("Database not found or not built yet. Build it with: .venv/bin/python backtest/build_context_pool.py")

    print_section("Tables")
    for table, description in TABLES.items():
        suffix = f" ({counts[table]} rows)" if table in counts else ""
        print(f"- {table}{suffix}: {description}")

    print_section("Views")
    for view, description in VIEWS.items():
        print(f"- {view}: {description}")

    print_section("Evidence roles")
    print("- target: fight-level context for the selected row")
    print("- context_metric: model/market/ELO/opponent-quality/trait metrics")
    print("- aggregate_pattern: historical empirical pattern stats applicable to the row")
    print("- audit_detail: recent-fight records that explain aggregate context")

    print_section("Example queries")
    for title, query in EXAMPLE_QUERIES:
        print(f"\n-- {title}\n{query}")


if __name__ == "__main__":
    main()
