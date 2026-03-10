#!/usr/bin/env python3
"""
Analyze Polymarket favorites from the new CSV export and compute true ROI.

Input:
  ~/Downloads/Follow Favorite Bets Analytics - Polymarket Data analytics.csv

Assumptions (to match our other Polymarket work):
  - MAX_BET is the Polymarket implied probability / price for the favorite (0–1).
  - IS_FAV_WINNER is $1 when the favorite won, $0 when they lost.
  - We stake a fixed STAKE_PER_FIGHT on every favorite.

Economics per fight:
  price = MAX_BET
  shares = stake / price
  if favorite wins:
      gross_payout = stake / price
      pnl          = gross_payout - stake
  else:
      gross_payout = 0
      pnl          = -stake

This matches the logic used in polymarket_model_comparison.py.
"""

from __future__ import annotations

import pathlib

import pandas as pd


SRC_CSV = pathlib.Path.home() / "Downloads" / "Follow Favorite Bets Analytics - Polymarket Data analytics.csv"
OUT_CSV = pathlib.Path(__file__).resolve().parent / "polymarket_new_analytics_clean.csv"

STAKE_PER_FIGHT = 1_000.0


def main() -> None:
    # Load CSV and promote the first non-empty row as header
    # (same pattern we inspected interactively).
    df_raw = pd.read_csv(SRC_CSV)
    df_raw = df_raw.dropna(how="all")
    # First non-empty row contains the real headers.
    df_raw.columns = list(df_raw.iloc[0])
    df_raw = df_raw[1:].reset_index(drop=True)

    # Rename for clarity.
    df = df_raw.rename(
        columns={
            "MAX_BET": "fav_price",
            "IS_FAV_WINNER": "fav_won_flag",
        }
    ).copy()

    # Coerce price and win flag to usable numeric forms.
    df["fav_price"] = pd.to_numeric(df["fav_price"], errors="coerce")

    def _parse_win_flag(x) -> bool:
        # Expected formats: "$1.00", "$0.00", 1.0, 0.0, "1", "0"
        if isinstance(x, str):
            x = x.strip().replace("$", "")
            try:
                val = float(x)
            except ValueError:
                return False
            return val > 0.5
        if isinstance(x, (int, float)):
            return x > 0.5
        return False

    df["fav_won"] = df["fav_won_flag"].apply(_parse_win_flag)

    # Keep only rows with a valid favorite price.
    df = df[df["fav_price"].notna()].copy()

    price = df["fav_price"]
    won = df["fav_won"]

    # Compute true Polymarket economics using the same formula as our other analyses:
    #   shares       = stake / price
    #   gross_payout = stake / price (if favorite wins) else 0
    #   pnl          = gross_payout - stake  (so -stake on losses)
    gross = pd.Series(0.0, index=df.index)
    win_mask = (price > 0) & won
    gross.loc[win_mask] = STAKE_PER_FIGHT / price.loc[win_mask]

    pnl = gross - STAKE_PER_FIGHT

    df["GROSS_PAYOUT"] = gross
    df["PNL"] = pnl

    total_stake = STAKE_PER_FIGHT * len(df)
    total_pnl = df["PNL"].sum()
    roi = total_pnl / total_stake if total_stake > 0 else 0.0

    keep_cols = [
        "source_url",
        "event_title",
        "fighter_1_name",
        "fighter_2_name",
        "fav_price",
        "fav_won",
        "GROSS_PAYOUT",
        "PNL",
    ]
    clean = df[keep_cols].copy()

    # Add a summary row at the end.
    summary_text = (
        f"{len(df)} fights; "
        f"Total stake = {total_stake:,.0f}; "
        f"Total PNL = {total_pnl:,.2f}; "
        f"ROI = {roi*100:.2f}%"
    )
    clean["SUMMARY"] = ""
    clean.loc[len(clean)] = {**{col: "" for col in clean.columns}, "SUMMARY": summary_text}

    clean.to_csv(OUT_CSV, index=False, float_format="%.2f")

    print(f"Wrote {len(df)} fights → {OUT_CSV}")
    print(f"Total stake:  {total_stake:,.0f}")
    print(f"Total PNL:    {total_pnl:,.2f}")
    print(f"ROI:          {roi:.4f} ({roi*100:.2f}%)")


if __name__ == "__main__":
    main()

