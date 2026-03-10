#!/usr/bin/env python3
"""
Clean Polymarket favorites sheet → CSV with script‑matching PNL.

Reads:
  ~/Downloads/Follow Favorite Bets Analytics.xlsx

Writes:
  analysis/polymarket_clean.csv

Keeps the original key columns and recomputes:
  - GROSS_PAYOUT = stake / FAVORITE_ODD        (matches _gross_return in script)
  - PNL          = GROSS_PAYOUT - stake        on wins, -stake on losses
  - PROFIT       = PNL                         (for backwards compatibility)

This matches the blind-favorite economics used in analysis/polymarket_model_comparison.py.
"""

from __future__ import annotations

import pathlib

import pandas as pd


SRC_XLSX = pathlib.Path.home() / "Downloads" / "Follow Favorite Bets Analytics.xlsx"
OUT_CSV = pathlib.Path(__file__).resolve().parent / "polymarket_clean.csv"

STAKE_PER_FIGHT = 1_000.0


def main() -> None:
    # Load sheet with the header on row 4 (0-based index 3).
    df = pd.read_excel(SRC_XLSX, header=3)
    # Drop fully empty rows, then promote the header row that Excel repeats.
    df = df.dropna(how="all")
    df.columns = list(df.iloc[0])
    df = df[1:].reset_index(drop=True)

    # Remove the summary rows ("Total", "ROI") and keep only numeric FAVORITE_ODD.
    df = df[df["FAVORITE_ODD"].apply(lambda x: isinstance(x, (int, float)))]

    # Convert the key columns to proper types.
    df["FAVORITE_ODD"] = df["FAVORITE_ODD"].astype(float)
    df["IS_FAVORITE_A_WINNER"] = df["IS_FAVORITE_A_WINNER"].astype(float).astype(bool)

    # Recompute economics to exactly match the backtest script:
    #   gross_payout = stake / price       when the favorite wins, else 0
    #   pnl          = gross_payout - stake  (so -stake on losses)
    price = df["FAVORITE_ODD"]
    won = df["IS_FAVORITE_A_WINNER"]

    gross = pd.Series(0.0, index=df.index)
    win_mask = (price > 0) & won
    gross.loc[win_mask] = STAKE_PER_FIGHT / price.loc[win_mask]

    pnl = gross - STAKE_PER_FIGHT

    df["GROSS_PAYOUT"] = gross
    df["PNL"] = pnl
    df["PROFIT"] = pnl  # alias so it lines up with how we talk about profit elsewhere

    # Also recompute a clean ROI number.
    total_stake = STAKE_PER_FIGHT * len(df)
    total_profit = df["PNL"].sum()
    roi = total_profit / total_stake if total_stake > 0 else 0.0

    # Save only the meaningful columns + recomputed profit.
    base_cols = [
        "source_url",
        "slug",
        "event_title",
        "event_start_utc",
        "fighter_1_name",
        "fighter_2_name",
        "winner",
        "fighter_1_odds_24h_after_start",
        "fighter_2_odds_24h_after_start",
        "has_fighter_1_won",
        "has_fighter_2_won",
        "moneyline_market_slug",
        "market_question",
        "FAVORITE_ODD",
        "IS_FAVORITE_A_WINNER",
        "GROSS_PAYOUT",
        "PNL",
        "PROFIT",
    ]

    clean = df[base_cols].copy()

    # Append a human-readable summary row at the end.
    summary_text = (
        f"{len(df)} fights (full Excel universe); "
        f"Total stake = {total_stake:,.0f}; "
        f"Total PNL (PROFIT) = {total_profit:,.2f}; "
        f"ROI = {roi*100:.2f}%"
    )
    # Put the summary text into a new trailing column.
    clean["SUMMARY"] = ""
    clean.loc[len(clean)] = {**{col: "" for col in clean.columns}, "SUMMARY": summary_text}

    # Write CSV with floats formatted to 2 decimal places.
    clean.to_csv(OUT_CSV, index=False, float_format="%.2f")

    print(f"Wrote {len(clean)} fights → {OUT_CSV}")
    print(f"Total stake:  {total_stake:,.0f}")
    print(f"Total profit: {total_profit:,.2f}")
    print(f"ROI:          {roi:.4f} ({roi*100:.2f}%)")


if __name__ == "__main__":
    main()

