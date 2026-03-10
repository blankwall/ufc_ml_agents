#!/usr/bin/env python3
"""
Polymarket Favorite Bets – XGBoost Model Comparison
=====================================================
Takes the 87-fight dataset from "Follow Favorite Bets Analytics.xlsx",
runs the XGBoost model on each matchup, and compares four betting strategies:

  1. Blind Favorite  – bet every time, always on the market favorite
  2. Model-Confirmed – only bet when model ALSO picks the favorite
  3. Model-Only      – always bet, but pick whoever the model predicts wins
  4. Model-Contrarian– only bet when model picks the UNDERDOG (fade the market)

Odds format in the xlsx:  FAVORITE_ODD is the Polymarket implied probability
  of the favorite (0-1).  Gross return per winning bet = 1000 * (1 + FAVORITE_ODD),
  which matches the observed total of 109 775 on 87 × $1 000 bets → 26.2 % ROI.
"""

import sys, re, warnings
from pathlib import Path

import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database.db_manager import DatabaseManager
from features.matchup_features import MatchupFeatureExtractor
from features.feature_pipeline import FeaturePipeline
from models.xgboost_model import XGBoostModel
from sqlalchemy import or_
from database.schema import Fighter, Fight

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
EXCEL_PATH  = Path.home() / "Downloads" / "Follow Favorite Bets Analytics.xlsx"
MODEL_NAME  = "mar_4_v2"
BET_SIZE    = 1_000   # dollars per bet (matches the spreadsheet)
CONF_THRESH = 0.55    # model must assign ≥ this probability to its pick

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_f1_name(raw: str) -> str:
    """Strip event prefix: 'UFC 326: Charles Oliveira' → 'Charles Oliveira'."""
    if not isinstance(raw, str):
        return str(raw)
    parts = raw.split(": ", 1)
    return parts[-1].strip()


def _clean_f2_name(raw: str) -> str:
    """Strip weight-class suffix: 'Max Holloway (Lightweight\t Main Card)' → 'Max Holloway'."""
    if not isinstance(raw, str):
        return str(raw)
    return re.sub(r"\s*\(.*\)\s*$", "", raw).strip()


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

    Let:
      - price  = cost per YES share in [0, 1]
      - stake  = BET_SIZE

    You buy stake/price shares.
      - Payout if YES wins  = (stake / price) * 1
      - Profit if YES wins  = stake * (1 - price) / price

    This helper returns the *payout* (gross), not the profit:
      gross_if_win = stake / price
      gross_if_lose = 0
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
    # ── 1. load & clean Excel ─────────────────────────────────────────────────
    print(f"Loading {EXCEL_PATH} …")
    df_raw = pd.read_excel(EXCEL_PATH, header=3)
    df_raw = df_raw.dropna(how="all")
    df_raw.columns = list(df_raw.iloc[0])
    df_raw = df_raw[1:].reset_index(drop=True)
    df_raw = df_raw.dropna(subset=["fighter_1_name"])

    # keep only real fight rows (numeric FAVORITE_ODD)
    df_raw = df_raw[
        df_raw["FAVORITE_ODD"].apply(
            lambda x: isinstance(x, (int, float)) and not pd.isna(x)
        )
    ].copy()

    df_raw["f1_clean"]   = df_raw["fighter_1_name"].apply(_clean_f1_name)
    df_raw["f2_clean"]   = df_raw["fighter_2_name"].apply(_clean_f2_name)
    df_raw["fav_odd"]    = df_raw["FAVORITE_ODD"].astype(float)
    df_raw["fav_won"]    = df_raw["IS_FAVORITE_A_WINNER"].astype(float).astype(bool)

    # Correct Polymarket-style economics for the favorite
    # edge_per_share = (1 - price) / price
    df_raw["fav_edge"]                    = (1.0 - df_raw["fav_odd"]) / df_raw["fav_odd"]
    df_raw["fav_profit_per_share_if_win"] = df_raw["fav_edge"]                  # e.g. (1-0.66)/0.66
    df_raw["fav_profit_per_1000_if_win"]  = df_raw["fav_edge"] * BET_SIZE       # stake × (1-price)/price
    df_raw["gross_ret"]                   = df_raw.apply(
        lambda r: _gross_return(r["fav_odd"], r["fav_won"]), axis=1
    )

    # which fighter IS the favorite?
    df_raw["f1_prob"]    = df_raw["fighter_1_odds_24h_after_start"].astype(float)
    df_raw["f2_prob"]    = df_raw["fighter_2_odds_24h_after_start"].astype(float)
    df_raw["fav_is_f1"]  = df_raw["f1_prob"] >= df_raw["f2_prob"]
    df_raw["fav_name"]   = df_raw.apply(
        lambda r: r["f1_clean"] if r["fav_is_f1"] else r["f2_clean"], axis=1
    )
    df_raw["dog_name"]   = df_raw.apply(
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
            if f1 is None: missing.append(f1_name)
            if f2 is None: missing.append(f2_name)
            skipped.append({"row": idx, "missing": missing,
                            "f1": f1_name, "f2": f2_name})
            continue

        try:
            p_f1 = _model_predict(extractor, pipeline, xgb_model, f1.id, f2.id)
        except Exception as e:
            skipped.append({"row": idx, "missing": [f"model_error: {e}"],
                            "f1": f1_name, "f2": f2_name})
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

        results.append({
            "f1":                         f1.name,
            "f2":                         f2.name,
            "fav_name":                   row["fav_name"],
            "dog_name":                   row["dog_name"],
            "fav_odd":                    row["fav_odd"],
            "fav_won":                    row["fav_won"],
            "fav_edge":                   row["fav_edge"],
            "fav_profit_per_share_if_win": row["fav_profit_per_share_if_win"],
            "fav_profit_per_1000_if_win":  row["fav_profit_per_1000_if_win"],
            "gross_ret_fav":              row["gross_ret"],      # corrected payout if we bet on fav
            "p_f1_model":                 round(p_f1, 4),
            "p_f2_model":                 round(p_f2, 4),
            "model_prob_fav":             round(model_prob_fav, 4),
            "model_prob_dog":             round(model_prob_dog, 4),
            "model_pick_is_f1":           model_pick_is_f1,
            "model_agrees_fav":           model_agrees_fav,
            "model_pick_prob":            round(model_pick_prob, 4),
        })

        marker = "✓" if row["fav_won"] else "✗"
        agree  = "AGREE  " if model_agrees_fav else "FADE   "
        print(
            f"  [{marker}] {f1.name[:22]:22s} vs {f2.name[:22]:22s}  "
            f"fav={row['fav_odd']:.2f}  model_fav={model_prob_fav:.2f}  {agree}"
        )

    session.close()

    res  = pd.DataFrame(results)
    n    = len(res)
    cost = n * BET_SIZE

    # ── 4. strategy analysis ──────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"RESULTS SUMMARY  ({n} fights analysed, {len(skipped)} skipped)")
    print("="*72)

    def _roi(bets_df, gross_col="gross_ret_fav"):
        """ROI for a set of bets, all using BET_SIZE stake."""
        invested = len(bets_df) * BET_SIZE
        if invested == 0:
            return 0, 0, 0, 0
        gross = bets_df[gross_col].sum()
        net   = gross - invested
        wins  = int((bets_df[gross_col] > 0).sum())
        roi   = net / invested
        return wins, len(bets_df), net, roi

    # --- Strategy 1: blind favorite (all 87 fights)
    w, t, net, roi = _roi(res)
    print(f"\n{'─'*72}")
    print(f"  [1] BLIND FAVORITE    – bet ALL {t} favorites")
    print(f"      Wins: {w}/{t} ({w/t*100:.1f}%)   Net P&L: ${net:+,.0f}   ROI: {roi*100:+.1f}%")

    # --- Strategy 2: model-confirmed (only bet when model agrees with favorite)
    confirmed = res[res["model_agrees_fav"]]
    w2, t2, net2, roi2 = _roi(confirmed)
    print(f"\n  [2] MODEL-CONFIRMED   – bet only when model ALSO picks the favorite ({t2} fights)")
    print(f"      Wins: {w2}/{t2} ({w2/t2*100:.1f}%)   Net P&L: ${net2:+,.0f}   ROI: {roi2*100:+.1f}%")

    # --- Strategy 2b: high-confidence confirmed (model prob >= CONF_THRESH)
    conf_hi = confirmed[confirmed["model_prob_fav"] >= CONF_THRESH]
    if len(conf_hi):
        w3, t3, net3, roi3 = _roi(conf_hi)
        print(f"\n  [2b] MODEL-CONFIRMED (≥{CONF_THRESH:.0%} conf) – {t3} fights")
        print(f"       Wins: {w3}/{t3} ({w3/t3*100:.1f}%)   Net P&L: ${net3:+,.0f}   ROI: {roi3*100:+.1f}%")

    # --- Strategy 3: model-only (bet on whoever model picks, regardless of being fav/dog)
    # Use the same Polymarket-style profit logic for whichever side we back.
    def _gross_return_model_row(r) -> float:
        if r["model_agrees_fav"]:
            # model also bets the favorite
            return _gross_return(r["fav_odd"], r["fav_won"])
        else:
            # model bets the underdog; its price is 1 - fav_odd
            dog_price = 1.0 - r["fav_odd"]
            dog_won   = not r["fav_won"]
            return _gross_return(dog_price, dog_won)

    res["model_wins"] = (
        (res["model_agrees_fav"] & res["fav_won"]) |
        (~res["model_agrees_fav"] & ~res["fav_won"])
    )
    res["gross_ret_model"] = res.apply(_gross_return_model_row, axis=1)
    w4, t4, net4, roi4 = _roi(res, "gross_ret_model")
    print(f"\n  [3] MODEL-ONLY        – bet whoever model picks ({t4} fights)")
    print(f"      Wins: {w4}/{t4} ({w4/t4*100:.1f}%)   Net P&L: ${net4:+,.0f}   ROI: {roi4*100:+.1f}%")

    # --- Strategy 3b: model-only with confidence filter
    res_hi = res[res["model_pick_prob"] >= CONF_THRESH].copy()
    if len(res_hi):
        w5, t5, net5, roi5 = _roi(res_hi, "gross_ret_model")
        print(f"\n  [3b] MODEL-ONLY (≥{CONF_THRESH:.0%} conf) – {t5} fights")
        print(f"       Wins: {w5}/{t5} ({w5/t5*100:.1f}%)   Net P&L: ${net5:+,.0f}   ROI: {roi5*100:+.1f}%")

    # --- Strategy 4: contrarian (fade the fav when model disagrees strongly)
    contrarian = res[~res["model_agrees_fav"]].copy()
    contrarian["gross_ret_dog"] = contrarian.apply(
        lambda r: _gross_return(1.0 - r["fav_odd"], not r["fav_won"]),
        axis=1,
    )
    if len(contrarian):
        w6, t6, net6, roi6 = _roi(contrarian, "gross_ret_dog")
        print(f"\n  [4] CONTRARIAN (fade fav) – bet underdog when model disagrees ({t6} fights)")
        print(f"      Wins: {w6}/{t6} ({w6/t6*100:.1f}%)   Net P&L: ${net6:+,.0f}   ROI: {roi6*100:+.1f}%")

    print(f"\n{'─'*72}")

    # ── 5. where model adds most value ────────────────────────────────────────
    print("\n[MODEL EDGE ANALYSIS]")
    print(f"  Fights model agrees with favorite: {len(confirmed)}/{n} ({len(confirmed)/n*100:.0f}%)")
    print(f"  When model agrees, favorite wins:  {int(confirmed['fav_won'].sum())}/{len(confirmed)} "
          f"({confirmed['fav_won'].mean()*100:.1f}%)")
    print(f"  When model DISAGREES, favorite wins: "
          f"{int(res[~res['model_agrees_fav']]['fav_won'].sum())}/{len(res[~res['model_agrees_fav']])} "
          f"({res[~res['model_agrees_fav']]['fav_won'].mean()*100:.1f}%)")

    print("\n[WHERE THE MODEL ADDED / REMOVED VALUE]")
    # Fights where model correctly faded the favorite (saved us from a loss)
    saved = res[~res["model_agrees_fav"] & ~res["fav_won"]]
    print(f"  Correct fades (model said underdog, underdog won): {len(saved)}")

    # Fights where model wrongly faded a winning favorite (missed profits)
    missed = res[~res["model_agrees_fav"] & res["fav_won"]]
    print(f"  Wrong fades  (model said underdog, favorite still won): {len(missed)}")

    # Fights where model confirmed a winning favorite
    hits = res[res["model_agrees_fav"] & res["fav_won"]]
    print(f"  Confirmed wins (model + market both right): {len(hits)}")

    # Fights where both model and market were wrong
    both_wrong = res[res["model_agrees_fav"] & ~res["fav_won"]]
    print(f"  Both wrong (model confirmed fav, underdog won): {len(both_wrong)}")

    # ── 6. skipped fights ────────────────────────────────────────────────────
    if skipped:
        print(f"\n[SKIPPED – {len(skipped)} fights not found in DB or model error]")
        for s in skipped:
            print(f"  row {s['row']}: {s['f1']} vs {s['f2']}  →  {s['missing']}")

    # ── 7. save results ───────────────────────────────────────────────────────
    out_path = ROOT / "analysis" / "polymarket_model_results.csv"
    res.to_csv(out_path, index=False)
    print(f"\nDetailed results saved → {out_path}")
    print("="*72)


if __name__ == "__main__":
    main()
