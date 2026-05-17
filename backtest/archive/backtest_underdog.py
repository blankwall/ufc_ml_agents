#!/usr/bin/env python3
"""
Underdog Model Backtest
=======================
Properly evaluates four strategies on the 2025 holdout data:

  A) General model   — ALL fights (one row per fight, f1=winner perspective)
  B) General model   — underdog fights only (f1 is underdog, market_prob_f1 < threshold)
  C) Blended model   — underdog fights only (65% underdog_v1 + 35% general)
  D) Blended + edge  — only bet when blended edge >= MIN_EDGE

KEY POINT: For underdog fights we keep ALL rows where market_prob_f1 < threshold
(not just upsets). This gives the real distribution: ~28% upsets, ~72% fav wins.

Usage:
  python backtest_underdog.py
  python backtest_underdog.py --edge 0.12 --detail
  python backtest_underdog.py --threshold 0.35 --blend 0.70
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path

EVAL_CSV   = "reports_mar_4_v2/eval_data_20260304_165004.csv"
MODEL_DIR  = Path("models/saved")
OUTPUT_CSV = "backtest_underdog_results.csv"

UD_THRESHOLD = 0.40
BLEND_WEIGHT = 0.65
MIN_EDGE     = 0.08
FLAT_BET     = 100   # dollars per bet


# ── helpers ───────────────────────────────────────────────────────────────────

def prob_to_american(p):
    p = min(max(p, 0.001), 0.999)
    return int(-p / (1 - p) * 100) if p >= 0.5 else int((1 - p) / p * 100)

def roi(profit, staked):
    return profit / staked * 100 if staked > 0 else 0.0

def flat_pnl(row, bet_on_f1: bool):
    """Profit/loss on a $FLAT_BET wager."""
    won = (row["target"] == 1) if bet_on_f1 else (row["target"] == 0)
    if bet_on_f1:
        odds = row["price_f1"]
    else:
        # derive f2 price from market prob
        odds = prob_to_american(1.0 - row["market_prob_f1"])
    if won:
        return FLAT_BET * (odds / 100 if odds > 0 else 100 / abs(odds))
    return -FLAT_BET


# ── model loading ─────────────────────────────────────────────────────────────

def load_underdog_model():
    m = xgb.XGBClassifier()
    m.load_model(MODEL_DIR / "underdog_v1.json")
    scaler   = joblib.load(MODEL_DIR / "underdog_v1_feature_scaler.pkl")
    features = joblib.load(MODEL_DIR / "underdog_v1_feature_names.pkl")
    return m, scaler, features


# ── simulation ────────────────────────────────────────────────────────────────

def simulate(subset: pd.DataFrame, prob_col: str) -> dict:
    """Flat-bet simulation: always bet on whoever the model favours."""
    if len(subset) == 0:
        return {"n": 0, "accuracy": 0, "roi": 0, "profit": 0,
                "upset_det": 0, "actual_upset_rate": 0}

    df = subset.copy()
    df["bet_on_f1"] = df[prob_col] > 0.5
    df["correct"]   = ((df["bet_on_f1"]) & (df["target"] == 1)) | \
                      ((~df["bet_on_f1"]) & (df["target"] == 0))
    df["pnl"]       = df.apply(lambda r: flat_pnl(r, r["bet_on_f1"]), axis=1)

    actual_upsets   = (df["target"] == 1).sum()          # f1 won = upset (since f1 is underdog)
    detected_upsets = ((df[prob_col] > 0.5) & (df["target"] == 1)).sum()

    return {
        "n":                 len(df),
        "accuracy":          df["correct"].mean(),
        "profit":            df["pnl"].sum(),
        "roi":               roi(df["pnl"].sum(), len(df) * FLAT_BET),
        "actual_upset_rate": actual_upsets / len(df),
        "upset_det":         detected_upsets / actual_upsets if actual_upsets > 0 else 0,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=UD_THRESHOLD)
    parser.add_argument("--blend",     type=float, default=BLEND_WEIGHT)
    parser.add_argument("--edge",      type=float, default=MIN_EDGE)
    parser.add_argument("--detail",    action="store_true")
    args = parser.parse_args()

    print(f"\nLoading eval data …")
    df_full = pd.read_csv(EVAL_CSV)
    print(f"  Full eval rows  : {len(df_full)}  (both perspectives)")

    # ── Strategy A: general model on ALL fights (target=1 = one row/fight)
    df_all = df_full[df_full["target"] == 1].copy().reset_index(drop=True)
    print(f"  All fights (A)  : {len(df_all)}")

    # ── Underdog fights: rows where f1 is the underdog (full distribution)
    df_under = df_full[df_full["market_prob_f1"] < args.threshold].copy().reset_index(drop=True)
    print(f"  Underdog fights : {len(df_under)}  "
          f"({(df_under['target']==1).sum()} upsets / {(df_under['target']==0).sum()} fav wins, "
          f"{(df_under['target']==1).mean()*100:.1f}% upset rate)")

    # ── Load underdog model and score underdog fights ───────────────────────
    print("\nLoading underdog_v1 model …")
    ud_model, scaler, ud_features = load_underdog_model()

    X = df_under.reindex(columns=ud_features, fill_value=0).fillna(0)
    X_sc = pd.DataFrame(scaler.transform(X), columns=ud_features, index=df_under.index)
    df_under["ud_prob_f1"] = ud_model.predict_proba(X_sc)[:, 1]
    df_under["gen_prob_f1"] = df_under["model_prob_f1_symmetric"]
    df_under["blended_prob_f1"] = (
        args.blend * df_under["ud_prob_f1"] +
        (1 - args.blend) * df_under["gen_prob_f1"]
    )
    df_under["blended_edge"] = df_under["blended_prob_f1"] - df_under["market_prob_f1"]

    # ── Run simulations ────────────────────────────────────────────────────
    df_edge = df_under[df_under["blended_edge"] >= args.edge]

    strats = {
        "A_general_all":         simulate(df_all,   "model_prob_f1_symmetric"),
        "B_general_underdogs":   simulate(df_under, "gen_prob_f1"),
        "C_blended_underdogs":   simulate(df_under, "blended_prob_f1"),
        "D_blended_edge_gate":   simulate(df_edge,  "blended_prob_f1"),
    }

    # ── Print results ──────────────────────────────────────────────────────
    print()
    print("=" * 76)
    print(f"  UNDERDOG MODEL BACKTEST  |  2025 holdout  |  flat bet ${FLAT_BET}")
    print(f"  Underdog threshold : market_prob < {args.threshold}")
    print(f"  Blend              : {args.blend*100:.0f}% underdog + {(1-args.blend)*100:.0f}% general")
    print(f"  Edge gate (D)      : blended edge >= {args.edge*100:.0f}%")
    print("=" * 76)

    labels = {
        "A_general_all":       "A  General model   — ALL fights",
        "B_general_underdogs": "B  General model   — underdog fights",
        "C_blended_underdogs": "C  Blended model   — underdog fights",
        "D_blended_edge_gate": f"D  Blended + {args.edge*100:.0f}% edge gate",
    }
    hdr = f"  {'Strategy':<42} {'Bets':>5} {'Acc%':>6} {'UpDet%':>8} {'Profit$':>9} {'ROI%':>7}"
    print(hdr)
    print("  " + "-" * 74)

    for k, label in labels.items():
        r = strats[k]
        if r["n"] == 0:
            print(f"  {label:<42}   {'—':>5}")
            continue
        print(
            f"  {label:<42} "
            f"{r['n']:>5} "
            f"{r['accuracy']*100:>6.1f} "
            f"{r['upset_det']*100:>8.1f} "
            f"{r['profit']:>+9.0f} "
            f"{r['roi']:>+7.1f}%"
        )

    print("=" * 76)

    # Summary
    b_roi = strats["C_blended_underdogs"]["roi"]
    g_roi = strats["B_general_underdogs"]["roi"]
    d_roi = strats["D_blended_edge_gate"]["roi"]
    d_n   = strats["D_blended_edge_gate"]["n"]

    print()
    print("TAKEAWAY:")
    diff = b_roi - g_roi
    if diff > 0:
        print(f"  + Blended model improves underdog ROI by {diff:+.1f}pp over general model")
    else:
        print(f"  - Blended model did NOT beat general on underdog fights ({diff:+.1f}pp)")

    if d_n > 0:
        print(f"  + Edge gate ({args.edge*100:.0f}%+) selects {d_n} bets at {d_roi:+.1f}% ROI")

    upset_base = strats["B_general_underdogs"]["upset_det"]
    upset_bld  = strats["C_blended_underdogs"]["upset_det"]
    print(f"  + Upset detection: general {upset_base*100:.1f}% → blended {upset_bld*100:.1f}%")
    print()

    # ── Edge sensitivity sweep ────────────────────────────────────────────
    print("EDGE GATE SENSITIVITY (blended, underdog fights):")
    print(f"  {'Gate':>6}  {'Bets':>5}  {'Acc%':>6}  {'Upset%':>7}  {'Profit$':>9}  {'ROI%':>7}")
    print(f"  {'-'*52}")
    for gate in [0.00, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        sub = df_under[df_under["blended_edge"] >= gate]
        if len(sub) == 0:
            break
        r = simulate(sub, "blended_prob_f1")
        print(f"  {gate*100:>5.0f}%  {r['n']:>5}  {r['accuracy']*100:>6.1f}  "
              f"{r['upset_det']*100:>7.1f}  {r['profit']:>+9.0f}  {r['roi']:>+7.1f}%")
    print()

    # ── Fight-level detail ────────────────────────────────────────────────
    if args.detail:
        detail = df_under[[
            "f1_name", "f2_name", "market_prob_f1",
            "gen_prob_f1", "ud_prob_f1", "blended_prob_f1",
            "blended_edge", "target"
        ]].copy()
        detail["gen_pred"]     = np.where(detail["gen_prob_f1"] > 0.5,     "f1-wins", "f2-wins")
        detail["blended_pred"] = np.where(detail["blended_prob_f1"] > 0.5, "f1-wins", "f2-wins")
        detail["actual"]       = np.where(detail["target"] == 1, "f1-wins", "f2-wins")
        detail["upset"]        = detail["target"] == 1
        print("UNDERDOG FIGHT DETAIL:")
        print(detail.sort_values("market_prob_f1").to_string(index=False))
        print()

    # ── Save ──────────────────────────────────────────────────────────────
    rows = [{"strategy": k, **v} for k, v in strats.items()]
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    fight_path = OUTPUT_CSV.replace(".csv", "_fights.csv")
    df_under[[
        "f1_name", "f2_name", "market_prob_f1", "gen_prob_f1", "ud_prob_f1",
        "blended_prob_f1", "blended_edge", "target", "price_f1"
    ]].to_csv(fight_path, index=False)

    print(f"Results saved : {OUTPUT_CSV}")
    print(f"Fights detail : {fight_path}")


if __name__ == "__main__":
    main()
