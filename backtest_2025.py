#!/usr/bin/env python3
"""
Backtest UFC predictions against betting odds.
Run predictions on all fights and calculate expected value.

Odds can come from:
  - A CSV (e.g. ufc_2025_odds.csv or export from DB)
  - Export from DB: python scripts/export_odds_from_db.py --year 2025 -o data/odds/db_2025.csv

Usage:
  python backtest_2025.py
  python backtest_2025.py --odds data/odds/db_odds_2025.csv --model mar_4_v2
  python backtest_2025.py --odds ufc_2025_odds.csv --quiet
"""

import subprocess
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

# Default paths (overridable by --odds / --model)
DEFAULT_ODDS_CSV = Path(__file__).parent / "ufc_2025_odds.csv"
DEFAULT_MODEL = "mar_4_v2"

def run_prediction(fighter1, fighter2, model_name: str):
    """Run xgboost_predict and return the prediction."""
    root = Path(__file__).resolve().parent
    cmd = [
        str(root / ".venv/bin/python"), str(root / "xgboost_predict.py"),
        "--fighter-1", fighter1,
        "--fighter-2", fighter2,
        "--model-name", model_name,
        "--allow-ambiguous",
        "--symmetric",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=60,
            env={"PYTHONPATH": str(root)},
        )

        output = result.stdout + result.stderr

        # Extract prediction percentage
        match = re.search(rf"{fighter1}:\s+([\d.]+)%\s+chance to win", output)
        if match:
            prob_f1 = float(match.group(1)) / 100
            prob_f2 = 1 - prob_f1
            return prob_f1, prob_f2

        return None, None

    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"Error predicting {fighter1} vs {fighter2}: {e}")
        return None, None


def odds_to_probability(american_odds):
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return -american_odds / (-american_odds + 100)


def calculate_ev(model_prob, american_odds):
    """Calculate expected value for a bet."""
    implied_prob = odds_to_probability(american_odds)

    if american_odds > 0:
        profit = american_odds / 100
    else:
        profit = 100 / -american_odds

    ev = (model_prob * profit) - ((1 - model_prob) * 1)
    return ev, implied_prob


def main():
    parser = argparse.ArgumentParser(description="Backtest model predictions vs odds")
    parser.add_argument("--odds", type=str, default=str(DEFAULT_ODDS_CSV), help="Path to odds CSV (date, fighter1, fighter2, fighter1_odds, fighter2_odds)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name for xgboost_predict")
    parser.add_argument("--quiet", action="store_true", help="Only print summary, not per-fight")
    parser.add_argument("--cutoff", type=str, default="2026-03-01", help="Only include fights before this date (YYYY-MM-DD)")
    args = parser.parse_args()

    csv_path = Path(args.odds)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent / csv_path
    if not csv_path.exists():
        print(f"Odds file not found: {csv_path}")
        print("Export from DB:  python scripts/export_odds_from_db.py --year 2025 -o data/odds/db_2025.csv")
        return

    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        print("CSV must have columns: date, fighter1, fighter2, fighter1_odds, fighter2_odds")
        return

    print(f"Loaded {len(df)} fights from {csv_path}  (model: {args.model})")
    print("=" * 80)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    cutoff_date = pd.to_datetime(args.cutoff)
    past_fights = df[df["date"] < cutoff_date].copy()

    print(f"Fights before {args.cutoff}: {len(past_fights)}")
    print("=" * 80)

    results = []

    for idx, row in past_fights.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        f1 = row["fighter1"]
        f2 = row["fighter2"]
        odds1 = int(row["fighter1_odds"]) if pd.notna(row.get("fighter1_odds")) else None
        odds2 = int(row["fighter2_odds"]) if pd.notna(row.get("fighter2_odds")) else None

        if not args.quiet:
            print(f"\n[{date}] {f1} vs {f2}")
            print(f"Odds: {f1} {odds1}, {f2} {odds2}")

        prob_f1, prob_f2 = run_prediction(f1, f2, args.model)

        if prob_f1 is None:
            if not args.quiet:
                print("  ❌ Prediction failed")
            results.append({
                'date': date,
                'fighter1': f1,
                'fighter2': f2,
                'odds1': odds1,
                'odds2': odds2,
                'prob1': None,
                'prob2': None,
                'pick': None,
                'ev1': None,
                'ev2': None,
                'error': True
            })
            continue

        if not args.quiet:
            print(f"  Model: {f1} {prob_f1*100:.1f}%, {f2} {prob_f2*100:.1f}%")

        ev1, imp1 = calculate_ev(prob_f1, odds1) if odds1 is not None else (None, None)
        ev2, imp2 = calculate_ev(prob_f2, odds2) if odds2 is not None else (None, None)

        if not args.quiet and ev1 is not None:
            print(f"  EV: {f1} ${ev1:.2f} (implied {imp1*100:.1f}%), {f2} ${ev2:.2f} (implied {imp2*100:.1f}%)")

        if prob_f1 > prob_f2:
            pick = f1
            pick_odds = odds1
            pick_prob = prob_f1
            pick_ev = ev1
        else:
            pick = f2
            pick_odds = odds2
            pick_prob = prob_f2
            pick_ev = ev2

        results.append({
            'date': date,
            'fighter1': f1,
            'fighter2': f2,
            'odds1': odds1,
            'odds2': odds2,
            'prob1': prob_f1,
            'prob2': prob_f2,
            'pick': pick,
            'pick_odds': pick_odds,
            'pick_prob': pick_prob,
            'ev1': ev1,
            'ev2': ev2,
            'error': False
        })

    results_df = pd.DataFrame(results)

    out_path = Path(__file__).resolve().parent / "backtest_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")

    print("\n" + "=" * 80)
    print("BETTING ANALYSIS SUMMARY")
    print("=" * 80)

    # Filter successful predictions
    successful = results_df[~results_df['error']].copy()

    print(f"\nTotal fights: {len(results_df)}")
    print(f"Successful predictions: {len(successful)}")
    print(f"Failed predictions: {len(results_df[results_df['error']])}")

    # Value bet analysis
    print("\n--- VALUE BETS ---")
    value_bets = successful[(successful['ev1'] > 0.05) | (successful['ev2'] > 0.05)].copy()
    print(f"Value bets (+EV > 5%): {len(value_bets)}")

    if len(value_bets) > 0:
        # Determine which side has +EV
        value_bets['best_ev'] = value_bets[['ev1', 'ev2']].max(axis=1)
        value_bets['best_side'] = value_bets.apply(
            lambda row: row['fighter1'] if row['ev1'] > row['ev2'] else row['fighter2'],
            axis=1
        )

        print("\nTop 10 Value Bets:")
        top_value = value_bets.nlargest(10, 'best_ev')
        for _, row in top_value.iterrows():
            side = row['fighter1'] if row['ev1'] > row['ev2'] else row['fighter2']
            ev = row['ev1'] if row['ev1'] > row['ev2'] else row['ev2']
            print(f"  {row['date']}: {side} - EV: ${ev:.2f}")

    # Underdog analysis
    print("\n--- UNDERDOG ANALYSIS ---")
    underdog_favs = successful[
        ((successful['odds1'] > 100) & (successful['prob1'] > 0.60)) |
        ((successful['odds2'] > 100) & (successful['prob2'] > 0.60))
    ].copy()
    print(f"Model favorites on underdogs (+odds, >60% prob): {len(underdog_favs)}")

    if len(underdog_favs) > 0:
        print("\nModel likes these underdogs:")
        for _, row in underdog_favs.head(10).iterrows():
            if row['odds1'] > 100 and row['prob1'] > 0.60:
                print(f"  {row['date']}: {row['fighter1']} {row['prob1']*100:.1f}% (odds: {row['odds1']:+})")
            elif row['odds2'] > 100 and row['prob2'] > 0.60:
                print(f"  {row['date']}: {row['fighter2']} {row['prob2']*100:.1f}% (odds: {row['odds2']:+})")

    # High confidence analysis
    print("\n--- HIGH CONFIDENCE ANALYSIS ---")
    high_conf = successful[
        (successful['prob1'] > 0.65) | (successful['prob2'] > 0.65)
    ].copy()
    print(f"High confidence picks (>65%): {len(high_conf)}")

    # ------------------------------------------------------------------
    # Strategy comparison: blind favorite vs model‑confirmed favorite
    # ------------------------------------------------------------------
    print("\n--- FAVORITE STRATEGY COMPARISON (EXPECTED VALUE) ---")
    # Work only with rows where we have full odds and probabilities.
    strat_df = successful.dropna(subset=['odds1', 'odds2', 'prob1', 'prob2', 'ev1', 'ev2']).copy()
    if len(strat_df) == 0:
        print("No rows with complete odds/probabilities for strategy comparison.")
    else:
        # Implied probabilities using the same helper as EV.
        strat_df['imp1'] = strat_df['odds1'].apply(odds_to_probability)
        strat_df['imp2'] = strat_df['odds2'].apply(odds_to_probability)

        # Market favorite is the side with higher implied probability.
        strat_df['fav_side'] = strat_df.apply(
            lambda r: 1 if r['imp1'] >= r['imp2'] else 2,
            axis=1,
        )
        strat_df['fav_ev'] = strat_df.apply(
            lambda r: r['ev1'] if r['fav_side'] == 1 else r['ev2'],
            axis=1,
        )
        strat_df['fav_model_prob'] = strat_df.apply(
            lambda r: r['prob1'] if r['fav_side'] == 1 else r['prob2'],
            axis=1,
        )
        strat_df['fav_name'] = strat_df.apply(
            lambda r: r['fighter1'] if r['fav_side'] == 1 else r['fighter2'],
            axis=1,
        )

        # Does the model's chosen side match the market favorite?
        strat_df['model_picks_fav'] = strat_df['pick'] == strat_df['fav_name']

        # --- Strategy [1]: Blind favorite – bet the market favorite in every fight.
        blind = strat_df.copy()
        n_blind = len(blind)
        total_ev_blind = blind['fav_ev'].sum()
        roi_blind = total_ev_blind / n_blind if n_blind > 0 else 0.0

        print(f"\n[1] BLIND FAVORITE – bet market favorite every fight")
        print(f"    Fights: {n_blind}")
        print(f"    Total expected P&L (units per 1 stake): {total_ev_blind:+.2f}")
        print(f"    Expected ROI per bet: {roi_blind*100:+.2f}%")

        # --- Strategy [2b]: Model‑confirmed favorite (≥55% confidence).
        mask_conf = strat_df['model_picks_fav'] & (strat_df['fav_model_prob'] >= 0.55)
        confirmed = strat_df[mask_conf].copy()
        n_conf = len(confirmed)
        total_ev_conf = confirmed['fav_ev'].sum()
        roi_conf = total_ev_conf / n_conf if n_conf > 0 else 0.0

        print(f"\n[2b] MODEL‑CONFIRMED FAVORITE (≥55% model prob)")
        print(f"     Fights bet: {n_conf}")
        print(f"     Total expected P&L (units per 1 stake): {total_ev_conf:+.2f}")
        print(f"     Expected ROI per bet: {roi_conf*100:+.2f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
