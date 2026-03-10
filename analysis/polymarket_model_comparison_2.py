#!/usr/bin/env python3
"""
Polymarket Favorite Bets – XGBoost Model Comparison (New Analytics CSV)
========================================================================

This is a variant of `polymarket_model_comparison.py` that uses the newer
Polymarket analytics export:

  ~/Downloads/Follow Favorite Bets Analytics - Polymarket Data analytics.csv

It:
  1. Loads all fights with a valid favorite price (`MAX_BET`)
  2. Runs the `mar_4_v2` model on each matchup
  3. Compares the same strategies as before:
       - Blind Favorite
       - Model-Confirmed Favorite
       - Model-Only
       - Contrarian (fade favorite when model disagrees)

Odds interpretation:
  - `MAX_BET` is treated as the Polymarket YES price for the favorite (0–1)
  - Gross return per winning favorite bet (stake = BET_SIZE) is:
        BET_SIZE / MAX_BET
    which matches our other Polymarket analyses.
"""

from __future__ import annotations

import sys
import re
import warnings
from pathlib import Path

import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database.db_manager import DatabaseManager  # type: ignore
from features.matchup_features import MatchupFeatureExtractor  # type: ignore
from features.feature_pipeline import FeaturePipeline  # type: ignore
from models.xgboost_model import XGBoostModel  # type: ignore
from sqlalchemy import or_  # type: ignore
from database.schema import Fighter, Fight  # type: ignore

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
CSV_PATH   = Path.home() / "Downloads" / "Follow Favorite Bets Analytics - Polymarket Data analytics.csv"
MODEL_NAME = "mar_4_v2"
BET_SIZE   = 1_000
CONF_THRESH = 0.55


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_f1_name(raw: str) -> str:
    """Strip event prefix: 'UFC 322: Edwards' → 'Edwards'."""
    if not isinstance(raw, str):
        return str(raw)
    parts = raw.split(": ", 1)
    return parts[-1].strip()


def _clean_f2_name(raw: str) -> str:
    """Strip weight-class suffix: 'Prates (Welterweight Main Card)' → 'Prates'."""
    if not isinstance(raw, str):
        return str(raw)
    return re.sub(r"\s*\(.*\)\s*$", "", raw).strip()


def _parse_win_flag(x) -> bool:
    """
    Parse IS_FAV_WINNER from the new analytics CSV.

    Expected formats:
      - "$1.00" / "$0.00"
      - "1" / "0"
      - 1.0 / 0.0
    """
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


def _resolve_fighter(session, name: str):
    """Fuzzy name lookup; returns best Fighter or None."""
    candidates = session.query(Fighter).filter(Fighter.name.ilike(f"%{name}%")).all()
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Multiple → pick the one with the most DB fights (active UFC roster)
    scored = []
    for f in candidates:
        cnt = session.query(Fight).filter(
            or_(Fight.fighter_1_id == f.id, Fight.fighter_2_id == f.id)
        ).count()
        scored.append((cnt, f.id, f))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored[0][2]


def _model_predict(extractor, pipeline, xgb_model, f1_id: int, f2_id: int) -> float:
    """Return symmetric P(fighter_1 wins) ∈ [0, 1]."""
    feat1 = extractor.extract_matchup_features(f1_id, f2_id)
    feat1["is_title_fight"] = 0
    feat2 = extractor.extract_matchup_features(f2_id, f1_id)
    feat2["is_title_fight"] = 0

    X1, _ = pipeline.prepare_features(pd.DataFrame([feat1]), fit_scaler=False)
    X2, _ = pipeline.prepare_features(pd.DataFrame([feat2]), fit_scaler=False)

    p1_raw = float(xgb_model.predict(X1, use_calibrated=False)[0])
    p2_raw = float(xgb_model.predict(X2, use_calibrated=False)[0])

    return 0.5 * (p1_raw + (1.0 - p2_raw))   # symmetric average


def _gross_return(price: float, outcome_won: bool) -> float:
    """
    Gross return on a $BET_SIZE YES position at Polymarket-style price.

    price in [0, 1] is the YES share price; if it wins, payout is stake/price.
    """
    if not outcome_won:
        return 0.0
    price = float(price)
    if price <= 0:
        return 0.0
    return BET_SIZE / price


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. load & clean analytics CSV ─────────────────────────────────────────
    print(f"Loading {CSV_PATH} …")
    df_raw = pd.read_csv(CSV_PATH)
    df_raw = df_raw.dropna(how="all")
    # First real row after empty padding contains the header labels.
    df_raw.columns = list(df_raw.iloc[0])
    df_raw = df_raw[1:].reset_index(drop=True)

    # Keep only rows that have real fighter names and a numeric favorite price.
    df_raw = df_raw.dropna(subset=["fighter_1_name"])
    df_raw["fav_price"] = pd.to_numeric(df_raw["MAX_BET"], errors="coerce")
    df_raw = df_raw[df_raw["fav_price"].notna()].copy()

    df_raw["fav_won"] = df_raw["IS_FAV_WINNER"].apply(_parse_win_flag)

    df_raw["f1_clean"] = df_raw["fighter_1_name"].apply(_clean_f1_name)
    df_raw["f2_clean"] = df_raw["fighter_2_name"].apply(_clean_f2_name)

    df_raw["fav_odd"] = df_raw["fav_price"].astype(float)

    # Correct Polymarket-style economics for the favorite
    df_raw["fav_edge"]                    = (1.0 - df_raw["fav_odd"]) / df_raw["fav_odd"]
    df_raw["fav_profit_per_share_if_win"] = df_raw["fav_edge"]
    df_raw["fav_profit_per_1000_if_win"]  = df_raw["fav_edge"] * BET_SIZE
    df_raw["gross_ret"]                   = df_raw.apply(
        lambda r: _gross_return(r["fav_odd"], r["fav_won"]), axis=1
    )

    # Determine which side is the favorite using Polymarket implied probs per side.
    df_raw["f1_prob"] = pd.to_numeric(df_raw["fighter_1_odds_24h_after_start"], errors="coerce")
    df_raw["f2_prob"] = pd.to_numeric(df_raw["fighter_2_odds_24h_after_start"], errors="coerce")
    df_raw["fav_is_f1"] = df_raw["f1_prob"] >= df_raw["f2_prob"]
    df_raw["fav_name"]  = df_raw.apply(
        lambda r: r["f1_clean"] if r["fav_is_f1"] else r["f2_clean"], axis=1
    )
    df_raw["dog_name"]  = df_raw.apply(
        lambda r: r["f2_clean"] if r["fav_is_f1"] else r["f1_clean"], axis=1
    )

    print(f"  {len(df_raw)} fights loaded\n")

    # ── 2. load model & pipeline (once) ───────────────────────────────────────
    print("Loading XGBoost model …")
    xgb_model = XGBoostModel()
    xgb_model.load_model(MODEL_NAME)

    pipeline = FeaturePipeline(initialize_db=False)
    pipeline.load_pipeline(model_name=MODEL_NAME)

    db      = DatabaseManager()
    session = db.get_session()
    extractor = MatchupFeatureExtractor(session)
    print("  Model ready.\n")

    # ── 3. iterate over fights ────────────────────────────────────────────────
    results = []
    skipped = []

    for idx, row in df_raw.iterrows():
        f1_name = row["f1_clean"]
        f2_name = row["f2_clean"]

        f1 = _resolve_fighter(session, f1_name)
        f2 = _resolve_fighter(session, f2_name)

        if f1 is None or f2 is None:
            missing = []
            if f1 is None:
                missing.append(f1_name)
            if f2 is None:
                missing.append(f2_name)
            skipped.append({"row": idx, "missing": missing, "f1": f1_name, "f2": f2_name})
            continue

        try:
            p_f1 = _model_predict(extractor, pipeline, xgb_model, f1.id, f2.id)
        except Exception as e:
            skipped.append({"row": idx, "missing": [f"model_error: {e}"], "f1": f1_name, "f2": f2_name})
            continue

        p_f2 = 1.0 - p_f1

        # which fighter does the model pick?
        model_pick_is_f1   = p_f1 >= p_f2
        model_pick_prob    = max(p_f1, p_f2)

        # does the market favorite = f1?
        fav_is_f1          = row["fav_is_f1"]

        # does model agree with the market favorite?
        model_agrees_fav   = (model_pick_is_f1 == fav_is_f1)

        # probabilities from model for fav / dog
        model_prob_fav = p_f1 if fav_is_f1 else p_f2
        model_prob_dog = 1.0 - model_prob_fav

        results.append(
            {
                "f1":                         f1.name,
                "f2":                         f2.name,
                "fav_name":                   row["fav_name"],
                "dog_name":                   row["dog_name"],
                "fav_odd":                    row["fav_odd"],
                "fav_won":                    row["fav_won"],
                "fav_edge":                   row["fav_edge"],
                "fav_profit_per_share_if_win": row["fav_profit_per_share_if_win"],
                "fav_profit_per_1000_if_win":  row["fav_profit_per_1000_if_win"],
                "gross_ret_fav":              row["gross_ret"],
                "p_f1_model":                 round(p_f1, 4),
                "p_f2_model":                 round(p_f2, 4),
                "model_prob_fav":             round(model_prob_fav, 4),
                "model_prob_dog":             round(model_prob_dog, 4),
                "model_pick_is_f1":           model_pick_is_f1,
                "model_agrees_fav":           model_agrees_fav,
                "model_pick_prob":            round(model_pick_prob, 4),
            }
        )

        marker = "✓" if row["fav_won"] else "✗"
        agree  = "AGREE  " if model_agrees_fav else "FADE   "
        print(
            f"  [{marker}] {f1.name[:22]:22s} vs {f2.name[:22]:22s}  "
            f"fav={row['fav_odd']:.2f}  model_fav={model_prob_fav:.2f}  {agree}"
        )

    session.close()

    res = pd.DataFrame(results)
    n   = len(res)

    # ── 4. strategy analysis ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"RESULTS SUMMARY  ({n} fights analysed by model, {len(skipped)} skipped)")
    print("=" * 72)

    def _roi(bets_df: pd.DataFrame, gross_col: str = "gross_ret_fav") -> tuple[int, int, float, float]:
        """ROI for a set of bets, all using BET_SIZE stake."""
        invested = len(bets_df) * BET_SIZE
        if invested == 0:
            return 0, 0, 0.0, 0.0
        gross = bets_df[gross_col].sum()
        net   = gross - invested
        wins  = int((bets_df[gross_col] > 0).sum())
        roi_v = net / invested
        return wins, len(bets_df), net, roi_v

    # Strategy 1: blind favorite on the FULL analytics universe (no model requirement)
    w1_all, t1_all, net1_all, roi1_all = _roi(df_raw, "gross_ret")
    print("\n" + "─" * 72)
    print(f"  [1] BLIND FAVORITE (ALL)    – bet ALL {t1_all} favorites in analytics CSV")
    print(f"      Wins: {w1_all}/{t1_all} ({w1_all/t1_all*100:.1f}%)   Net P&L: ${net1_all:+,.0f}   ROI: {roi1_all*100:+.1f}%")

    # Also report blind favorite restricted to fights the model could analyse
    w1_sub, t1_sub, net1_sub, roi1_sub = _roi(res)
    print(f"\n  [1b] BLIND FAVORITE (model subset) – {t1_sub} favorites with model features")
    print(f"       Wins: {w1_sub}/{t1_sub} ({w1_sub/t1_sub*100:.1f}%)   Net P&L: ${net1_sub:+,.0f}   ROI: {roi1_sub*100:+.1f}%")

    # Strategy 2: model-confirmed favorite
    confirmed = res[res["model_agrees_fav"]].copy()
    w2, t2, net2, roi2 = _roi(confirmed)
    print(f"\n  [2] MODEL-CONFIRMED   – bet only when model ALSO picks the favorite ({t2} fights)")
    print(f"      Wins: {w2}/{t2} ({w2/t2*100:.1f}%)   Net P&L: ${net2:+,.0f}   ROI: {roi2*100:+.1f}%")

    # Strategy 2b: model-confirmed + confidence
    conf_hi = confirmed[confirmed["model_prob_fav"] >= CONF_THRESH].copy()
    if len(conf_hi):
        w3, t3, net3, roi3 = _roi(conf_hi)
        print(f"\n  [2b] MODEL-CONFIRMED (≥{CONF_THRESH:.0%} conf) – {t3} fights")
        print(f"       Wins: {w3}/{t3} ({w3/t3*100:.1f}%)   Net P&L: ${net3:+,.0f}   ROI: {roi3*100:+.1f}%")

    # Strategy 3: model-only
    def _gross_return_model_row(r) -> float:
        if r["model_agrees_fav"]:
            return _gross_return(r["fav_odd"], r["fav_won"])
        else:
            dog_price = 1.0 - r["fav_odd"]
            dog_won   = not r["fav_won"]
            return _gross_return(dog_price, dog_won)

    res["model_wins"] = (res["model_agrees_fav"] & res["fav_won"]) | (~res["model_agrees_fav"] & ~res["fav_won"])
    res["gross_ret_model"] = res.apply(_gross_return_model_row, axis=1)

    w4, t4, net4, roi4 = _roi(res, "gross_ret_model")
    print(f"\n  [3] MODEL-ONLY        – bet whoever model picks ({t4} fights)")
    print(f"      Wins: {w4}/{t4} ({w4/t4*100:.1f}%)   Net P&L: ${net4:+,.0f}   ROI: {roi4*100:+.1f}%")

    # Strategy 3b: model-only with confidence filter
    res_hi = res[res["model_pick_prob"] >= CONF_THRESH].copy()
    if len(res_hi):
        w5, t5, net5, roi5 = _roi(res_hi, "gross_ret_model")
        print(f"\n  [3b] MODEL-ONLY (≥{CONF_THRESH:.0%} conf) – {t5} fights")
        print(f"       Wins: {w5}/{t5} ({w5/t5*100:.1f}%)   Net P&L: ${net5:+,.0f}   ROI: {roi5*100:+.1f}%")

    # Strategy 4: contrarian (fade favorite when model disagrees)
    contrarian = res[~res["model_agrees_fav"]].copy()
    contrarian["gross_ret_dog"] = contrarian.apply(
        lambda r: _gross_return(1.0 - r["fav_odd"], not r["fav_won"]),
        axis=1,
    )
    if len(contrarian):
        w6, t6, net6, roi6 = _roi(contrarian, "gross_ret_dog")
        print(f"\n  [4] CONTRARIAN (fade fav) – bet underdog when model disagrees ({t6} fights)")
        print(f"      Wins: {w6}/{t6} ({w6/t6*100:.1f}%)   Net P&L: ${net6:+,.0f}   ROI: {roi6*100:+.1f}%")

    print("\n" + "─" * 72)

    # ── 5. favorite size sensitivity + worst upsets ───────────────────────────
    print("\n[FAVORITE SIZE SENSITIVITY – BLIND FAVORITES ONLY]")
    for cap in [0.9, 0.8, 0.7, 0.6]:
        capped = df_raw[df_raw["fav_odd"] <= cap].copy()
        if len(capped) == 0:
            continue
        w_cap, t_cap, net_cap, roi_cap = _roi(capped, "gross_ret")
        print(
            f"  Max fav price ≤ {cap:.2f}: bets={t_cap}, "
            f"wins={w_cap}/{t_cap} ({w_cap/t_cap*100:.1f}%), "
            f"net=${net_cap:+,.0f}, ROI={roi_cap*100:+.1f}%"
        )

    # Biggest losing favorites by implied Polymarket price
    print("\n[BIGGEST LOSING FAVORITES (BLIND)]")
    losers = df_raw[df_raw["fav_won"] == False].copy()
    if len(losers) > 0:
        losers = losers.sort_values("fav_odd", ascending=False).head(10)
        for _, r in losers.iterrows():
            print(
                f"  fav_price={r['fav_odd']:.3f} | "
                f"{r['fighter_1_name']} vs {r['fighter_2_name']} | "
                f"market_question='{r['market_question']}'"
            )

    # Edge analysis summary
    print("\n[MODEL EDGE ANALYSIS]")
    print(f"  Fights model agrees with favorite: {len(confirmed)}/{n} ({len(confirmed)/n*100:.0f}%)")
    print(
        f"  When model agrees, favorite wins:  "
        f"{int(confirmed['fav_won'].sum())}/{len(confirmed)} "
        f"({confirmed['fav_won'].mean()*100:.1f}%)"
    )
    disagree = res[~res["model_agrees_fav"]]
    if len(disagree):
        print(
            f"  When model DISAGREES, favorite wins: "
            f"{int(disagree['fav_won'].sum())}/{len(disagree)} "
            f"({disagree['fav_won'].mean()*100:.1f}%)"
        )

    # Save detailed results
    out_path = ROOT / "analysis" / "polymarket_model_results_2.csv"
    res.to_csv(out_path, index=False)
    print(f"\nDetailed results saved → {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

