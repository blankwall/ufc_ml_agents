#!/usr/bin/env python3
"""
Validate model predictions against actual fight results.
Calculate ROI for different betting strategies.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Paths
FIGHT_DETAILS_PATH = Path("/Users/tylerbohan/code/ufc_ml_agents/data/fight_details.json")
BACKTEST_RESULTS_PATH = Path("/Users/tylerbohan/code/ufc_ml_agents/backtest/backtest_2025_results.csv")
ODDS_PATH = Path("/Users/tylerbohan/code/ufc_ml_agents/backtest/odds/ufc_2025_odds.csv")

def load_fight_results():
    """Load actual fight results from fight_details.json."""
    with open(FIGHT_DETAILS_PATH, 'r') as f:
        events = json.load(f)

    results = {}
    for event in events:
        event_date = pd.to_datetime(event['date'])
        for fight in event['fights']:
            # Create match key (sorted names to handle order-agnostic matching)
            f1_name = fight['fighter_1_name']
            f2_name = fight['fighter_2_name']

            # Winner: 1 if fighter_1 won, 2 if fighter_2 won
            winner = 1 if fight['result'] == 'fighter_1' else 2

            # Store multiple possible keys for matching
            key1 = f"{f1_name}|{f2_name}"
            key2 = f"{f2_name}|{f1_name}"

            results[key1] = {
                'winner': winner,
                'f1_name': f1_name,
                'f2_name': f2_name,
                'date': event_date,
                'method': fight['method'],
                'round': fight['round'],
                'time': fight['time']
            }
            results[key2] = results[key1]  # Same result for both keys

    return results

def load_backtest_predictions():
    """Load backtest predictions."""
    df = pd.read_csv(BACKTEST_RESULTS_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

def american_odds_to_decimal(odds):
    """Convert American odds to decimal."""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def calculate_roi(df, fight_results, filter_fn=None):
    """Calculate ROI for a given betting strategy."""
    total_bets = 0
    total_wins = 0
    total_staked = 0
    total_returned = 0
    correct_predictions = 0
    wrong_predictions = 0
    missed_fights = []

    for _, row in df.iterrows():
        # Apply filter if provided
        if filter_fn and not filter_fn(row):
            continue

        # Skip failed predictions
        if row['error']:
            continue

        f1_name = row['fighter1']
        f2_name = row['fighter2']
        key = f"{f1_name}|{f2_name}"

        # Check if we have the result
        if key not in fight_results:
            missed_fights.append(f"{f1_name} vs {f2_name}")
            continue

        result = fight_results[key]
        winner = result['winner']

        # Model's pick
        model_pick = row['pick']
        pick_odds = row['pick_odds']

        # Determine if model picked fighter 1 or 2
        if model_pick == f1_name:
            model_picked = 1
        elif model_pick == f2_name:
            model_picked = 2
        else:
            # Name matching failed
            missed_fights.append(f"{f1_name} vs {f2_name} (name match failed)")
            continue

        # Calculate stake and return
        stake = 100  # Standard $100 bet

        # Convert American odds to profit
        if pick_odds > 0:
            profit = (pick_odds / 100) * stake
        else:
            profit = (100 / abs(pick_odds)) * stake

        total_bets += 1
        total_staked += stake

        if model_picked == winner:
            # Model won!
            total_wins += 1
            total_returned += stake + profit
            correct_predictions += 1
        else:
            # Model lost
            wrong_predictions += 1

    # Calculate metrics
    if total_bets == 0:
        return {
            'total_bets': 0,
            'win_rate': 0,
            'roi': 0,
            'total_profit': 0,
            'correct': 0,
            'wrong': 0
        }

    win_rate = total_wins / total_bets if total_bets > 0 else 0
    roi = ((total_returned - total_staked) / total_staked * 100) if total_staked > 0 else 0
    total_profit = total_returned - total_staked

    return {
        'total_bets': total_bets,
        'wins': total_wins,
        'losses': total_bets - total_wins,
        'win_rate': win_rate,
        'roi': roi,
        'total_profit': total_profit,
        'total_staked': total_staked,
        'total_returned': total_returned,
        'correct': correct_predictions,
        'wrong': wrong_predictions,
        'missed_count': len(missed_fights),
        'missed_fights': missed_fights[:10]  # First 10 missed
    }

def main():
    print("=" * 80)
    print("MODEL VALIDATION AGAINST ACTUAL FIGHT RESULTS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    fight_results = load_fight_results()
    predictions_df = load_backtest_predictions()

    print(f"Loaded {len(fight_results)//2} fight results (deduped)")
    print(f"Loaded {len(predictions_df)} predictions")

    # Calculate ROI for different strategies
    strategies = [
        ("ALL PICKS", None),
        ("HIGH CONFIDENCE (>65%)", lambda row: row.get('pick_prob', 0) > 0.65),
        ("VERY HIGH CONFIDENCE (>70%)", lambda row: row.get('pick_prob', 0) > 0.70),
        ("VALUE BETS (+EV > 10%)", lambda row: max(row.get('ev1', -999), row.get('ev2', -999)) > 0.10),
        ("UNDERDOGS MODEL LIKES (+odds, >60% prob)",
            lambda row: row.get('pick_odds', 0) > 0 and row.get('pick_prob', 0) > 0.60),
        ("FADE HEAVY FAVORITES (model picks underdog vs -200+ fav)",
            lambda row: row.get('pick_odds', 0) > 0 and
                       ((row.get('odds1', 0) < -200 and row['pick'] == row['fighter2']) or
                        (row.get('odds2', 0) < -200 and row['pick'] == row['fighter1']))),
    ]

    print("\n" + "=" * 80)
    print("BETTING STRATEGY RESULTS")
    print("=" * 80)

    results_summary = []

    for name, filter_fn in strategies:
        result = calculate_roi(predictions_df, fight_results, filter_fn)
        results_summary.append((name, result))

        print(f"\n--- {name} ---")
        if result['total_bets'] == 0:
            print("  No bets matched this strategy")
            continue

        print(f"  Total Bets: {result['total_bets']}")
        print(f"  Record: {result['wins']}-{result['losses']}")
        print(f"  Win Rate: {result['win_rate']*100:.1f}%")
        print(f"  ROI: {result['roi']:+.1f}%")
        print(f"  Total Profit: ${result['total_profit']:+,.2f}")
        print(f"  Total Staked: ${result['total_staked']:,.2f}")
        print(f"  Total Returned: ${result['total_returned']:,.2f}")

        if result['missed_count'] > 0:
            print(f"  Missed (no result): {result['missed_count']}")
            if result['missed_fights']:
                print(f"  Sample missed: {result['missed_fights'][:3]}")

    # Find best strategy
    print("\n" + "=" * 80)
    print("STRATEGY RANKING BY ROI")
    print("=" * 80)

    valid_strategies = [(name, r) for name, r in results_summary if r['total_bets'] >= 5]
    valid_strategies.sort(key=lambda x: x[1]['roi'], reverse=True)

    for i, (name, result) in enumerate(valid_strategies, 1):
        print(f"{i}. {name}: ROI {result['roi']:+.1f}% ({result['wins']}-{result['losses']}, {result['total_bets']} bets)")

    # Specific analysis: Underdog performance
    print("\n" + "=" * 80)
    print("UNDERDOG VALUE ANALYSIS")
    print("=" * 80)

    underdog_value = calculate_roi(
        predictions_df,
        fight_results,
        lambda row: row.get('pick_odds', 0) > 150 and row.get('pick_prob', 0) > 0.55
    )

    print(f"\nModel likes underdogs (+150 or higher, >55% probability):")
    print(f"  Bets: {underdog_value['total_bets']}")
    print(f"  Win Rate: {underdog_value['win_rate']*100:.1f}%")
    print(f"  ROI: {underdog_value['roi']:+.1f}%")

    # Statistical significance test
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 80)

    best_strategy = valid_strategies[0] if valid_strategies else None
    if best_strategy:
        name, result = best_strategy
        n = result['total_bets']
        win_rate = result['win_rate']

        # Binomial test approximate p-value (vs 50% random)
        import math
        if n > 0:
            # Standard error for proportion
            se = math.sqrt(0.5 * 0.5 / n)
            z_score = (win_rate - 0.5) / se
            # Rough p-value (two-tailed)
            p_value = 2 * (1 - 0.5 * (1 + abs(math.erf(z_score / math.sqrt(2)))))

            print(f"\nBest Strategy: {name}")
            print(f"  Bets: {n}")
            print(f"  Win Rate: {win_rate*100:.1f}%")
            print(f"  vs Random (50%): Z-score = {z_score:.2f}")
            print(f"  Approximate p-value: {p_value:.4f}")

            if p_value < 0.05:
                print(f"  ✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
            elif p_value < 0.10:
                print(f"  ⚠️  TRENDING SIGNIFICANCE (p < 0.10)")
            else:
                print(f"  ❌ NOT STATISTICALLY SIGNIFICANT (p >= 0.10)")


if __name__ == "__main__":
    main()
