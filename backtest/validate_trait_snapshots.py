#!/usr/bin/env python3
"""
Validate first-pass fighter trait snapshots.

The report is intentionally descriptive: coverage, sparse Sergey assessment
alignment, and simple backtest-context splits. It does not promote traits into
betting rules.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TRAITS = ROOT_DIR / "data" / "enrichment" / "trait_snapshots.sqlite"
DEFAULT_SIDECAR = ROOT_DIR / "data" / "enrichment" / "sergey_sidecar.sqlite"
DEFAULT_CONTEXT = ROOT_DIR / "data" / "enrichment" / "context_pool.sqlite"


LABEL_PAIRS = [
    ("pace_retention", "cardio_score"),
    ("hittability", "defensive_exposure_score"),
    ("hittability", "defensive_responsibility_score"),
    ("scramble", "anti_control_score"),
    ("scramble", "scramble_score"),
    ("distance_control", "striking_efficiency_score"),
    ("fight_iq", "recent_form_score"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traits", type=Path, default=DEFAULT_TRAITS)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    return parser.parse_args()


def correlation(pairs: Iterable[tuple[float | None, float | None]]) -> tuple[int, float | None]:
    clean = [(float(x), float(y)) for x, y in pairs if x is not None and y is not None]
    n = len(clean)
    if n < 3:
        return n, None
    xs = [x for x, _ in clean]
    ys = [y for _, y in clean]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in clean)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return n, None
    return n, cov / math.sqrt(var_x * var_y)


def pct(value: float | None) -> str:
    return "--" if value is None else f"{value:.1%}"


def signed_pct(value: float | None) -> str:
    return "--" if value is None else f"{value:+.1%}"


def print_coverage(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) FROM fighter_trait_snapshots").fetchone()[0]
    print("Coverage")
    print("--------")
    print(f"snapshots: {total}")
    for label, where in [
        ("with prior fight history", "fight_count > 0"),
        ("with 3+ prior fights", "fight_count >= 3"),
        ("with Sergey identity", "sergey_fighter_id IS NOT NULL"),
        ("with cardio proxy", "cardio_score IS NOT NULL"),
    ]:
        n = conn.execute(f"SELECT COUNT(*) FROM fighter_trait_snapshots WHERE {where}").fetchone()[0]
        print(f"{label:<26} {n:>6} ({n / total:.1%})" if total else f"{label:<26}      0")


def print_label_alignment(conn: sqlite3.Connection, sidecar: Path) -> None:
    if not sidecar.exists():
        print("\nSergey assessment alignment\n---------------------------\nsidecar missing")
        return
    conn.execute(f"ATTACH DATABASE '{sidecar}' AS sergey")
    print("\nSergey assessment alignment")
    print("---------------------------")
    for label, trait in LABEL_PAIRS:
        if trait == "defensive_responsibility_score":
            expression = "100 - s.defensive_exposure_score"
            where = "s.defensive_exposure_score IS NOT NULL"
        else:
            expression = f"s.{trait}"
            where = f"s.{trait} IS NOT NULL"
        rows = conn.execute(
            f"""
            SELECT a.{label} AS label_value, s.{trait} AS trait_value
            FROM fighter_trait_snapshots s
            JOIN sergey.assessments a
              ON a.fight_id = s.sergey_fight_id
             AND a.fighter_id = s.sergey_fighter_id
            WHERE a.{label} IS NOT NULL
              AND {where}
            """
            if trait != "defensive_responsibility_score"
            else f"""
            SELECT a.{label} AS label_value, {expression} AS trait_value
            FROM fighter_trait_snapshots s
            JOIN sergey.assessments a
              ON a.fight_id = s.sergey_fight_id
             AND a.fighter_id = s.sergey_fighter_id
            WHERE a.{label} IS NOT NULL
              AND {where}
            """
        ).fetchall()
        n, corr = correlation((row["label_value"], row["trait_value"]) for row in rows)
        corr_text = "--" if corr is None else f"{corr:+.3f}"
        print(f"{label:<18} vs {trait:<30} n={n:>3} corr={corr_text}")
    conn.execute("DETACH DATABASE sergey")


def print_backtest_splits(conn: sqlite3.Connection, context: Path) -> None:
    if not context.exists():
        print("\nBacktest context splits\n-----------------------\ncontext pool missing")
        return
    conn.execute(f"ATTACH DATABASE '{context}' AS context")
    print("\nBacktest context splits")
    print("-----------------------")
    for trait in [
        "offensive_control_score_diff",
        "anti_control_score_diff",
        "striking_efficiency_score_diff",
        "defensive_exposure_score_diff",
        "durability_risk_score_diff",
        "grappling_threat_score_diff",
    ]:
        rows = conn.execute(
            f"""
            SELECT
                CASE
                    WHEN d.{trait} >= 10 THEN 'pick +10 or more'
                    WHEN d.{trait} <= -10 THEN 'pick -10 or worse'
                    ELSE 'within +/-10'
                END AS bucket,
                COUNT(*) AS n,
                SUM(CASE WHEN c.pick_correct = 1 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN c.pick_correct = 0 THEN 1 ELSE 0 END) AS losses,
                SUM(c.actual_pnl) AS profit
            FROM context.backtest_fight_pool c
            JOIN v_trait_pair_deltas d
              ON d.main_fight_id = c.main_fight_id
             AND LOWER(d.fighter_name) = LOWER(c.pick)
            WHERE c.pick_correct IS NOT NULL
              AND d.{trait} IS NOT NULL
            GROUP BY bucket
            ORDER BY bucket
            """
        ).fetchall()
        print(f"\n{trait}")
        for row in rows:
            n = row["n"]
            wins = row["wins"] or 0
            losses = row["losses"] or 0
            profit = row["profit"] or 0.0
            roi = profit / n if n else None
            win_rate = wins / n if n else None
            print(f"  {row['bucket']:<17} N={n:>3} W-L={wins}-{losses} WR={pct(win_rate):>6} ROI={signed_pct(roi):>7}")
    conn.execute("DETACH DATABASE context")


def main() -> None:
    args = parse_args()
    if not args.traits.exists():
        raise SystemExit(f"Trait DB missing. Build it with: .venv/bin/python backtest/build_trait_snapshots.py")
    conn = sqlite3.connect(args.traits)
    conn.row_factory = sqlite3.Row
    try:
        print_coverage(conn)
        print_label_alignment(conn, args.sidecar)
        print_backtest_splits(conn, args.context)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
