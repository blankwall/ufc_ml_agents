#!/usr/bin/env python3
"""
Grid-search optimize backtest strategy parameters to maximize P&L.

Loads the pre-built results CSV once and evaluates every parameter combo
using pure vectorized pandas — no subprocess, no model inference, no
apply_strategy call per iteration. Runs in a few seconds.

Usage:
    python backtest/optimize_config.py
    python backtest/optimize_config.py --results backtest/backtest_results.csv
    python backtest/optimize_config.py --min-bets 30 --top 20 --sort-by roi
"""

import sys
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Parameter search grids ────────────────────────────────────────────────────
# Tightly bounded around the baseline config — no extremes.
# Order must match the unpacking in the main loop: edge_ud, conf_fav, conf_ud, fav_cap, ud_cap, female
PARAM_GRIDS = {
    "edge_underdog":       np.round(np.arange(0.00, 0.16, 0.01),  2).tolist(),  # 0.00 … 0.15, step 0.01
    "confidence_favorite": np.round(np.arange(0.50, 0.76, 0.025), 3).tolist(),  # 0.50 … 0.75, step 0.025
    "confidence_underdog": np.round(np.arange(0.50, 0.61, 0.01),  2).tolist(),  # 0.50 … 0.60, step 0.01
    "favorite_odds_cap":   list(range(-300, -525, -25)),                          # -300 … -500, step 25
    "underdog_odds_cap":   list(range(200, 410, 20)),                             # 200  … 400,  step 20
    "female":              [True, False],                                         # include women's fights?
}

# Fixed params (not tuned); "female" is now tuned in PARAM_GRIDS
BASE_CONFIG = {
    "model":       "mar_4_v2",
    "cutoff_date": "2026-03-01",
    "edge_min":    0.00,
    "min_fights":  3,
}


def prepare_arrays(results_path: Path, cutoff_date: str):
    """
    Load CSV and extract pre-computed numpy arrays for fast inner-loop evaluation.
    Returns a dict of arrays + metadata needed to score each combination.
    """
    df = pd.read_csv(results_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] < pd.to_datetime(cutoff_date)].copy()

    # Keep only successful predictions with a known outcome
    df = df[~df["error"].fillna(True)].copy()
    df = df.dropna(subset=["pick_correct", "actual_pnl", "pick_odds", "pick_prob"]).copy()
    df = df.reset_index(drop=True)

    # pick_ev: EV of whichever side the model picked
    pick_ev = np.where(
        df["pick"].values == df["fighter1"].values,
        df["ev1"].fillna(0).values,
        df["ev2"].fillna(0).values,
    )

    # female flag — present in new CSVs, False for all rows in old CSVs
    if "female" in df.columns:
        is_female = df["female"].fillna(False).values.astype(bool)
    else:
        is_female = np.zeros(len(df), dtype=bool)

    return {
        "odds":        df["pick_odds"].values.astype(float),
        "prob":        df["pick_prob"].values.astype(float),
        "ev":          pick_ev.astype(float),
        "pnl":         df["actual_pnl"].values.astype(float),
        "correct":     df["pick_correct"].values.astype(bool),
        "is_female":   is_female,
    }


def score_combo(arrays: dict,
                edge_underdog: float,
                confidence_favorite: float,
                confidence_underdog: float,
                favorite_odds_cap: int,
                underdog_odds_cap: int,
                female: bool,
                min_bets: int):
    """Evaluate one parameter combo using pre-extracted numpy arrays."""
    odds      = arrays["odds"]
    prob      = arrays["prob"]
    ev        = arrays["ev"]
    pnl       = arrays["pnl"]
    corr      = arrays["correct"]
    is_female = arrays["is_female"]

    # Exclude women's fights when female=False
    gender_ok = female | ~is_female

    is_fav = odds < 0
    is_ud  = odds > 0

    conf_ok = (
        (is_fav & (prob >= confidence_favorite)) |
        (is_ud  & (prob >= confidence_underdog))
    )
    edge_ok = (
        (is_fav & (ev >= BASE_CONFIG["edge_min"])) |
        (is_ud  & (ev >= edge_underdog))
    )
    odds_ok = (
        (is_fav & (odds >= favorite_odds_cap)) |
        (is_ud  & (odds <= underdog_odds_cap))
    )

    mask = gender_ok & conf_ok & edge_ok & odds_ok
    n = int(mask.sum())
    if n < min_bets:
        return None

    total_pnl = float(pnl[mask].sum())
    return {
        "n_bets":   n,
        "n_fav":    int((odds[mask] < 0).sum()),
        "n_ud":     int((odds[mask] > 0).sum()),
        "pnl":      total_pnl,
        "roi":      total_pnl / n * 100,
        "accuracy": float(corr[mask].mean() * 100),
    }


def main():
    parser = argparse.ArgumentParser(description="Grid-search optimize backtest strategy params")
    parser.add_argument("--results",  default="backtest/backtest_results.csv",
                        help="Pre-built results CSV")
    parser.add_argument("--cutoff",   default=None,
                        help="Override cutoff date filter (YYYY-MM-DD). Defaults to BASE_CONFIG cutoff_date.")
    parser.add_argument("--min-bets", type=int, default=20,
                        help="Minimum placed bets to consider a config valid (default: 20)")
    parser.add_argument("--min-roi",  type=float, default=None,
                        help="Minimum ROI%% required (e.g. 25 means >= 25%%). Filters before sorting.")
    parser.add_argument("--top",      type=int, default=15,
                        help="Number of top configs to print (default: 15)")
    parser.add_argument("--sort-by",  choices=["pnl", "roi", "score"], default="score",
                        help="Metric to optimize: pnl=total profit, roi=efficiency, "
                             "score=pnl*roi (balances volume + efficiency, default)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run the full backtest first:  python backtest/backtest_2025.py --quiet")
        sys.exit(1)

    cutoff = args.cutoff or BASE_CONFIG["cutoff_date"]
    print(f"Loading {results_path}...", flush=True)
    arrays = prepare_arrays(results_path, cutoff)
    n_fights = len(arrays["odds"])
    print(f"  {n_fights} successful predictions with known outcomes", flush=True)

    keys   = list(PARAM_GRIDS.keys())
    values = list(PARAM_GRIDS.values())
    combos = list(itertools.product(*values))
    total  = len(combos)
    print(f"\nSearching {total:,} combinations ({' x '.join(str(len(v)) for v in values)})...",
          flush=True)

    rows = []
    for i, combo in enumerate(combos):
        (edge_ud, conf_fav, conf_ud, fav_cap, ud_cap, female) = combo
        metrics = score_combo(arrays, edge_ud, conf_fav, conf_ud, fav_cap, ud_cap, female, args.min_bets)
        if metrics is None:
            continue
        rows.append({
            "edge_underdog":       edge_ud,
            "confidence_favorite": conf_fav,
            "confidence_underdog": conf_ud,
            "favorite_odds_cap":   fav_cap,
            "underdog_odds_cap":   ud_cap,
            "female":              female,
            **metrics,
            "score": metrics["pnl"] * metrics["roi"],  # pnl × roi rewards volume + efficiency
        })
        if (i + 1) % 25000 == 0:
            print(f"  {i+1:>6,}/{total:,}  valid configs so far: {len(rows):,}", flush=True)

    if not rows:
        print(f"\nNo configs found with >= {args.min_bets} bets. Try --min-bets lower.")
        sys.exit(0)

    df_res = pd.DataFrame(rows)

    if args.min_roi is not None:
        before = len(df_res)
        df_res = df_res[df_res["roi"] >= args.min_roi]
        print(f"\nROI filter >= {args.min_roi}%: {len(df_res):,} configs remain (removed {before - len(df_res):,})", flush=True)
        if df_res.empty:
            print(f"No configs pass the ROI filter. Try a lower --min-roi.")
            sys.exit(0)

    df_res = df_res.sort_values(args.sort_by, ascending=False).reset_index(drop=True)

    # ── Print top N ────────────────────────────────────────────────────────
    sort_label = {"pnl": "P&L", "roi": "ROI", "score": "SCORE (P&L × ROI)"}[args.sort_by]
    print(f"\n{'='*90}")
    print(f"TOP {args.top} CONFIGS BY {sort_label}  (min {args.min_bets} bets, {len(df_res):,} valid configs found)")
    print(f"{'='*90}")

    display_cols = ["edge_underdog", "confidence_favorite", "confidence_underdog",
                    "favorite_odds_cap", "underdog_odds_cap", "female",
                    "n_bets", "n_fav", "n_ud", "pnl", "roi", "accuracy", "score"]
    print(df_res[display_cols].head(args.top).to_string(index=True))

    # ── Print best config ──────────────────────────────────────────────────
    best = df_res.iloc[0]
    print(f"\n{'='*90}")
    print("BEST CONFIG")
    print(f"{'='*90}")
    for k in keys:
        print(f"  {k:30s}: {best[k]}")
    print(f"  {'n_bets':30s}: {int(best['n_bets'])}  ({int(best['n_fav'])} fav / {int(best['n_ud'])} ud)")
    print(f"  {'pnl':30s}: {best['pnl']:+.2f} units")
    print(f"  {'roi':30s}: {best['roi']:+.1f}%")
    print(f"  {'accuracy':30s}: {best['accuracy']:.1f}%")

    best_cfg = {**BASE_CONFIG}
    for k in keys:
        v = best[k]
        if k == "female":
            best_cfg[k] = bool(v)
        elif isinstance(v, (float, np.floating)):
            best_cfg[k] = float(v)
        else:
            best_cfg[k] = int(v)

    import json
    print(f"\nFull config JSON:")
    print(json.dumps(best_cfg, indent=4))

    # ── Save results ───────────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "backtest" / "optimize_results.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\nFull results ({len(df_res):,} valid configs) saved to {out_path}")


if __name__ == "__main__":
    main()
