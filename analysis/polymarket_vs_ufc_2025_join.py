#!/usr/bin/env python3
"""
Join Polymarket favorites with 2025 book odds to compare true vs expected ROI.

This script:
  - Loads Polymarket favorite results (with true outcomes) from
      analysis/polymarket_model_results_2.csv
  - Loads 2025 book odds + model EV from
      backtest/backtest_2025_results.csv   (derived from backtest/odds/ufc_2025_odds.csv)
  - Matches fights by fighter names (order-insensitive)
  - Outputs a joined CSV:
      analysis/polymarket_vs_ufc_2025_join.csv

Each row includes:
  - Fighters and date
  - Polymarket favorite price and realized PNL (using Polymarket share math)
  - Book American odds for both sides + implied probs
  - Model probabilities and EVs from the backtest

You can then compute:
  - True realized ROI on Polymarket for this overlapping set
  - Expected ROI implied by book odds + model
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BET_SIZE = 1_000.0


def odds_to_probability(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return -american_odds / (-american_odds + 100.0)


def _fight_key(name1: str, name2: str) -> str:
    """Order-insensitive fight key based on lowercase names."""
    a = (name1 or "").strip().lower()
    b = (name2 or "").strip().lower()
    return " vs ".join(sorted([a, b]))


def main() -> None:
    # Load Polymarket model results (new analytics-based comparison)
    poly_path = ROOT / "analysis" / "polymarket_model_results_2.csv"
    poly = pd.read_csv(poly_path)

    # Load 2025 backtest results (built from backtest/odds/ufc_2025_odds.csv)
    backtest_path = ROOT / "backtest" / "backtest_2025_results.csv"
    bt = pd.read_csv(backtest_path)
    bt = bt[~bt["error"]].copy()

    # Build fight keys using DB-resolved fighter names on Polymarket side
    poly["fight_key"] = poly.apply(
        lambda r: _fight_key(str(r["f1"]), str(r["f2"])), axis=1
    )

    # Build fight keys using fighter1/fighter2 from the odds backtest
    bt["fight_key"] = bt.apply(
        lambda r: _fight_key(str(r["fighter1"]), str(r["fighter2"])), axis=1
    )

    # Inner join on fight_key
    joined = pd.merge(poly, bt, on="fight_key", how="inner", suffixes=("_poly", "_bt"))

    print(f"Polymarket fights: {len(poly)}")
    print(f"Backtest fights:   {len(bt)}")
    print(f"Joined fights:     {len(joined)}")

    if joined.empty:
        print("No overlapping fights found; check name normalization.")
        return

    # Determine which side is the Polymarket favorite in book odds
    def _book_fav_side(row):
        f1 = str(row["fighter1"])
        f2 = str(row["fighter2"])
        fav_name = str(row["fav_name"])

        if fav_name.lower() == f1.lower():
            return 1
        if fav_name.lower() == f2.lower():
            return 2
        # Fallback: use implied prob from odds
        o1 = row["odds1"]
        o2 = row["odds2"]
        if pd.isna(o1) or pd.isna(o2):
            return None
        if o1 < 0 and o2 < 0:
            return 1 if o1 <= o2 else 2
        elif o1 < 0 and o2 >= 0:
            return 1
        elif o2 < 0 and o1 >= 0:
            return 2
        else:
            return 1 if o1 <= o2 else 2

    joined["book_fav_side"] = joined.apply(_book_fav_side, axis=1)

    # Compute book favorite odds / implied prob / EV for that side
    def _book_fav_row(row):
        side = row["book_fav_side"]
        if side == 1:
            od = row["odds1"]
            ev = row["ev1"]
        elif side == 2:
            od = row["odds2"]
            ev = row["ev2"]
        else:
            return pd.Series({"book_fav_odds": None, "book_fav_imp": None, "book_fav_ev": None})

        if pd.isna(od):
            return pd.Series({"book_fav_odds": None, "book_fav_imp": None, "book_fav_ev": None})
        imp = odds_to_probability(od)
        return pd.Series({"book_fav_odds": od, "book_fav_imp": imp, "book_fav_ev": ev})

    book_cols = joined.apply(_book_fav_row, axis=1)
    joined = pd.concat([joined, book_cols], axis=1)

    # Compute true Polymarket PNL for the favorite, per fight, using fav_odd / fav_won
    def _poly_pnl(row):
        price = float(row["fav_odd"])
        won = bool(row["fav_won"])
        if price <= 0:
            return 0.0
        if won:
            return BET_SIZE / price - BET_SIZE
        else:
            return -BET_SIZE

    joined["poly_pnl"] = joined.apply(_poly_pnl, axis=1)

    # Assemble a clean output view
    out_cols = [
        "date",
        "fighter1",
        "fighter2",
        "fav_name",
        "dog_name",
        "fav_odd",
        "fav_won",
        "poly_pnl",
        "book_fav_odds",
        "book_fav_imp",
        "book_fav_ev",
        "prob1",
        "prob2",
        "ev1",
        "ev2",
    ]
    out = joined[out_cols].copy()

    out_path = ROOT / "analysis" / "polymarket_vs_ufc_2025_join.csv"
    out.to_csv(out_path, index=False, float_format="%.4f")

    # Quick summary for sanity
    total_poly_pnl = out["poly_pnl"].sum()
    total_stake = BET_SIZE * len(out)
    roi_poly = total_poly_pnl / total_stake if total_stake else 0.0

    avg_book_ev = out["book_fav_ev"].mean()

    print(f"\nJoined CSV written to: {out_path}")
    print(f"  Overlapping fights: {len(out)}")
    print(f"  Polymarket total PNL: {total_poly_pnl:+.2f} on stake {total_stake:.0f} → ROI {roi_poly*100:+.2f}%")
    print(f"  Avg book favorite EV per $1 (model-based): {avg_book_ev:+.4f} → expected ROI {avg_book_ev*100:+.2f}%")


if __name__ == "__main__":
    main()
