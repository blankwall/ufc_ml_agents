#!/usr/bin/env python3
"""
Backtest UFC 2025 predictions against betting odds.
Run predictions on all fights and calculate expected value.
"""

import subprocess
import json
import re
from datetime import datetime
from pathlib import Path
import pandas as pd

# Paths
CSV_PATH = Path("/Users/tylerbohan/code/ufc_ml_agents/ufc_2025_odds.csv")
MODEL_NAME = "power_veteran_v2_test"

def run_prediction(fighter1, fighter2):
    """Run xgboost_predict and return the prediction."""
    cmd = [
        ".venv/bin/python", "xgboost_predict.py",
        "--fighter-1", fighter1,
        "--fighter-2", fighter2,
        "--model-name", MODEL_NAME,
        "--symmetric"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env={"PYTHONPATH": "/Users/tylerbohan/code/ufc_ml_agents"}
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
    # Load odds data
    df = pd.read_csv(CSV_PATH)

    print(f"Loaded {len(df)} fights from {CSV_PATH}")
    print("=" * 80)

    # Filter to fights before March 2026 (already happened)
    df['date'] = pd.to_datetime(df['date'])
    cutoff_date = pd.to_datetime("March 1, 2026")
    past_fights = df[df['date'] < cutoff_date].copy()

    print(f"Fights that have occurred: {len(past_fights)}")
    print("=" * 80)

    results = []

    for idx, row in past_fights.iterrows():
        date = row['date'].strftime("%Y-%m-%d")
        f1 = row['fighter1']
        f2 = row['fighter2']
        odds1 = row['fighter1_odds']
        odds2 = row['fighter2_odds']

        print(f"\n[{date}] {f1} vs {f2}")
        print(f"Odds: {f1} {odds1}, {f2} {odds2}")

        # Run prediction
        prob_f1, prob_f2 = run_prediction(f1, f2)

        if prob_f1 is None:
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

        print(f"  Model: {f1} {prob_f1*100:.1f}%, {f2} {prob_f2*100:.1f}%")

        # Calculate EV for both sides
        ev1, imp1 = calculate_ev(prob_f1, odds1)
        ev2, imp2 = calculate_ev(prob_f2, odds2)

        print(f"  EV: {f1} ${ev1:.2f} (implied {imp1*100:.1f}%), {f2} ${ev2:.2f} (implied {imp2*100:.1f}%)")

        # Determine model pick
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

    # Save results
    results_df = pd.DataFrame(results)

    # Print summary statistics
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

    # Save to CSV
    output_path = Path("/Users/tylerbohan/code/ufc_ml_agents/backtest_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 80)
    print("NEXT: Provide actual fight results to calculate ROI")
    print("=" * 80)


if __name__ == "__main__":
    main()
