#!/usr/bin/env python3
"""
Deep Analysis of Model Performance
-----------------------------------

Rigorous analysis to understand if 90%+ accuracy is legitimate or due to:
  - Data leakage
  - Overfitting
  - Selection bias
  - Or if the model is actually that good
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from database.db_manager import DatabaseManager
from database.schema import Fighter, Event, Fight
from features.feature_pipeline import FeaturePipeline
from models.xgboost_model import XGBoostModel


def analyze_prediction_distribution(
    df: pd.DataFrame,
    title: str = "Prediction Distribution"
) -> Dict:
    """Analyze the distribution of model probabilities."""
    probs = df["model_prob_f1"].values
    
    # Bin predictions
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    binned = pd.cut(probs, bins=bins, include_lowest=True)
    counts = binned.value_counts().sort_index()
    
    # Statistics
    stats = {
        "mean": float(probs.mean()),
        "median": float(np.median(probs)),
        "std": float(probs.std()),
        "min": float(probs.min()),
        "max": float(probs.max()),
        "confident_predictions_pct": float((np.abs(probs - 0.5) > 0.2).mean() * 100),
        "very_confident_pct": float((np.abs(probs - 0.5) > 0.3).mean() * 100),
        "coin_flip_pct": float((np.abs(probs - 0.5) < 0.1).mean() * 100),
    }
    
    print(f"\n{title}:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std Dev: {stats['std']:.3f}")
    print(f"  Confident (>70% or <30%): {stats['confident_predictions_pct']:.1f}%")
    print(f"  Very Confident (>80% or <20%): {stats['very_confident_pct']:.1f}%")
    print(f"  Near 50-50 (40-60%): {stats['coin_flip_pct']:.1f}%")
    
    print("\n  Distribution by bin:")
    for interval, count in counts.items():
        pct = (count / len(probs)) * 100
        print(f"    {interval}: {count} ({pct:.1f}%)")
    
    return stats


def compare_to_market(df: pd.DataFrame) -> Dict:
    """Compare model accuracy to market accuracy."""
    y_true = df["target"].values
    model_probs = df["model_prob_f1"].values
    market_probs = df["market_prob_f1"].values
    
    # Model picks
    model_picks = (model_probs >= 0.5).astype(int)
    model_correct = (model_picks == y_true).sum()
    model_acc = model_correct / len(y_true)
    
    # Market picks
    market_picks = (market_probs >= 0.5).astype(int)
    market_correct = (market_picks == y_true).sum()
    market_acc = market_correct / len(y_true)
    
    # Agreement rate
    agreement = (model_picks == market_picks).sum()
    agreement_rate = agreement / len(y_true)
    
    # When they disagree, who's right?
    disagree_mask = model_picks != market_picks
    if disagree_mask.sum() > 0:
        model_right_when_disagree = (
            model_picks[disagree_mask] == y_true[disagree_mask]
        ).sum()
        market_right_when_disagree = (
            market_picks[disagree_mask] == y_true[disagree_mask]
        ).sum()
        disagree_count = disagree_mask.sum()
    else:
        model_right_when_disagree = 0
        market_right_when_disagree = 0
        disagree_count = 0
    
    stats = {
        "model_accuracy": float(model_acc),
        "market_accuracy": float(market_acc),
        "agreement_rate": float(agreement_rate),
        "disagreements": int(disagree_count),
        "model_right_when_disagree": int(model_right_when_disagree),
        "market_right_when_disagree": int(market_right_when_disagree),
    }
    
    print("\n" + "=" * 80)
    print("MODEL VS MARKET COMPARISON")
    print("=" * 80)
    print(f"\nModel Accuracy: {model_acc:.3f} ({model_correct}/{len(y_true)})")
    print(f"Market Accuracy: {market_acc:.3f} ({market_correct}/{len(y_true)})")
    print(f"Agreement Rate: {agreement_rate:.3f} ({agreement}/{len(y_true)} fights)")
    
    if disagree_count > 0:
        print(f"\nWhen Model and Market Disagree ({disagree_count} fights):")
        print(f"  Model was right: {model_right_when_disagree} times")
        print(f"  Market was right: {market_right_when_disagree} times")
        model_edge_pct = (model_right_when_disagree / disagree_count) * 100
        print(f"  Model win rate on disagreements: {model_edge_pct:.1f}%")
    else:
        print("\nModel and Market never disagree (100% agreement)")
    
    return stats


def analyze_incorrect_predictions(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Analyze fights where the model was most wrong."""
    # Get incorrect predictions
    df = df.copy()
    df["model_pick"] = (df["model_prob_f1"] >= 0.5).astype(int)
    df["correct"] = df["model_pick"] == df["target"]
    
    incorrect = df[~df["correct"]].copy()
    
    if len(incorrect) == 0:
        print("\n⚠️  Model got ALL predictions correct! (This is suspicious)")
        return pd.DataFrame()
    
    # Calculate "surprise" - how confident the model was when wrong
    incorrect["confidence"] = np.abs(incorrect["model_prob_f1"] - 0.5)
    incorrect = incorrect.sort_values("confidence", ascending=False)
    
    print("\n" + "=" * 80)
    print(f"TOP {min(top_n, len(incorrect))} MOST SURPRISING INCORRECT PREDICTIONS")
    print("=" * 80)
    print("(Where model was most confident but WRONG)\n")
    
    for idx, row in incorrect.head(top_n).iterrows():
        f1_name = row.get("f1_name", "Unknown")
        f2_name = row.get("f2_name", "Unknown")
        model_prob = row["model_prob_f1"]
        target = row["target"]
        market_prob = row.get("market_prob_f1", 0.5)
        
        winner = f1_name if target == 1 else f2_name
        loser = f2_name if target == 1 else f1_name
        model_predicted = f1_name if model_prob >= 0.5 else f2_name
        
        print(f"Fight: {f1_name} vs {f2_name}")
        print(f"  Actual Winner: {winner}")
        print(f"  Model Predicted: {model_predicted} ({model_prob:.1%} for {f1_name})")
        print(f"  Market Probability: {market_prob:.1%} for {f1_name}")
        print(f"  Model Confidence: {row['confidence']:.1%}")
        print()
    
    return incorrect


def check_for_patterns_in_errors(df: pd.DataFrame, session) -> Dict:
    """Check if errors cluster by weight class, age, experience, etc."""
    df = df.copy()
    df["model_pick"] = (df["model_prob_f1"] >= 0.5).astype(int)
    df["correct"] = df["model_pick"] == df["target"]
    
    print("\n" + "=" * 80)
    print("ERROR PATTERN ANALYSIS")
    print("=" * 80)
    
    patterns = {}
    
    # By weight class
    if "weight_class" in df.columns:
        wc_stats = df.groupby("weight_class")["correct"].agg(["sum", "count", "mean"])
        print("\nAccuracy by Weight Class:")
        for wc, row in wc_stats.iterrows():
            print(f"  {wc}: {row['mean']:.3f} ({int(row['sum'])}/{int(row['count'])})")
        patterns["weight_class"] = wc_stats.to_dict()
    
    # Upsets (market underdog wins)
    df["market_pick"] = (df["market_prob_f1"] >= 0.5).astype(int)
    df["was_upset"] = df["market_pick"] != df["target"]
    
    upset_acc = df[df["was_upset"]]["correct"].mean()
    favorite_acc = df[~df["was_upset"]]["correct"].mean()
    
    print(f"\nAccuracy on Upsets (market underdog won):")
    print(f"  Upsets: {upset_acc:.3f} ({df[df['was_upset']]['correct'].sum()}/{df['was_upset'].sum()})")
    print(f"  Favorites: {favorite_acc:.3f} ({df[~df['was_upset']]['correct'].sum()}/{(~df['was_upset']).sum()})")
    
    patterns["upset_accuracy"] = float(upset_acc)
    patterns["favorite_accuracy"] = float(favorite_acc)
    
    # By model confidence level
    df["confidence_level"] = pd.cut(
        np.abs(df["model_prob_f1"] - 0.5),
        bins=[0, 0.1, 0.2, 0.3, 0.5],
        labels=["Low (50-60%)", "Med (60-70%)", "High (70-80%)", "Very High (>80%)"]
    )
    
    conf_stats = df.groupby("confidence_level", observed=True)["correct"].agg(["sum", "count", "mean"])
    print("\nAccuracy by Model Confidence:")
    for conf, row in conf_stats.iterrows():
        print(f"  {conf}: {row['mean']:.3f} ({int(row['sum'])}/{int(row['count'])})")
    
    return patterns


def check_feature_distributions(df: pd.DataFrame, top_n: int = 20) -> None:
    """Check if any features are suspiciously predictive."""
    feature_cols = [col for col in df.columns if col.startswith("f1_") or col.startswith("f2_") or col.endswith("_diff")]
    
    if not feature_cols:
        print("\n⚠️  No feature columns found in dataframe")
        return
    
    print("\n" + "=" * 80)
    print("FEATURE CORRELATION WITH ACTUAL OUTCOMES")
    print("=" * 80)
    print("(Checking for suspiciously high correlations that might indicate leakage)\n")
    
    correlations = []
    y = df["target"].values
    
    for col in feature_cols:
        try:
            x = df[col].fillna(0).values
            if x.std() > 0:  # Skip constant features
                corr = np.corrcoef(x, y)[0, 1]
                correlations.append((col, abs(corr)))
        except:
            continue
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top {top_n} features most correlated with outcome:\n")
    for i, (feat, corr) in enumerate(correlations[:top_n], 1):
        flag = "⚠️ " if corr > 0.5 else ""
        print(f"  {i:2d}. {flag}{feat}: {corr:.3f}")
    
    # Flag suspiciously high correlations
    suspicious = [f for f, c in correlations if c > 0.5]
    if suspicious:
        print(f"\n⚠️  WARNING: {len(suspicious)} features have >0.5 correlation with outcome!")
        print("This might indicate data leakage. Investigate these features:")
        for feat in suspicious[:10]:
            print(f"  • {feat}")


def analyze_holdout_verification(df: pd.DataFrame, session, min_year: int) -> None:
    """Verify that holdout fights are truly unseen."""
    print("\n" + "=" * 80)
    print("HOLDOUT VERIFICATION")
    print("=" * 80)
    
    # Check event dates
    if "event_id" in df.columns:
        event_ids = df["event_id"].dropna().astype(int).unique()
        events = session.query(Event).filter(Event.id.in_(event_ids.tolist())).all()
        
        # Handle both datetime objects and strings
        event_years = []
        for e in events:
            if e.date:
                if isinstance(e.date, str):
                    try:
                        year = pd.to_datetime(e.date).year
                        event_years.append(year)
                    except:
                        pass
                else:
                    event_years.append(e.date.year)
        
        if event_years:
            print(f"\nEvent Year Range in Evaluation Set:")
            print(f"  Min: {min(event_years)}")
            print(f"  Max: {max(event_years)}")
            print(f"  Expected Min: {min_year}")
            
            if min(event_years) < min_year:
                print(f"\n⚠️  WARNING: Found events before {min_year}! Possible data leakage.")
                pre_cutoff = [y for y in event_years if y < min_year]
                print(f"  {len(pre_cutoff)} events before cutoff year")
            else:
                print(f"\n✓ All events are from {min_year}+ (holdout is valid)")


def main():
    parser = argparse.ArgumentParser(
        description="Deep analysis of model performance on holdout set"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        required=True,
        help="Path to evaluation results CSV (merged predictions + outcomes)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2025,
        help="Expected minimum year for holdout set",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/deep_analysis",
        help="Directory to save analysis outputs",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation data
    logger.info(f"Loading evaluation data from {args.eval_data}...")
    df = pd.read_csv(args.eval_data)
    logger.info(f"Loaded {len(df)} evaluation rows")
    
    # Initialize DB for pattern analysis
    db = DatabaseManager()
    session = db.get_session()
    
    try:
        # Run all analyses
        results = {}
        
        # 1. Prediction distribution
        print("\n" + "=" * 80)
        print("1. PREDICTION DISTRIBUTION ANALYSIS")
        print("=" * 80)
        dist_stats = analyze_prediction_distribution(df, "Model Predictions")
        results["prediction_distribution"] = dist_stats
        
        # Market distribution for comparison
        if "market_prob_f1" in df.columns:
            df_market = df.copy()
            df_market["model_prob_f1"] = df_market["market_prob_f1"]
            market_dist = analyze_prediction_distribution(df_market, "Market Odds")
            results["market_distribution"] = market_dist
        
        # 2. Model vs Market
        if "market_prob_f1" in df.columns:
            market_stats = compare_to_market(df)
            results["model_vs_market"] = market_stats
        
        # 3. Analyze errors
        incorrect_df = analyze_incorrect_predictions(df, top_n=15)
        
        # 4. Pattern analysis
        patterns = check_for_patterns_in_errors(df, session)
        results["error_patterns"] = patterns
        
        # 5. Feature correlation check
        check_feature_distributions(df, top_n=25)
        
        # 6. Holdout verification
        analyze_holdout_verification(df, session, args.min_year)
        
        # Save results
        results_path = output_dir / "analysis_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.success(f"Saved analysis results to {results_path}")
        
        # Final verdict
        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)
        
        model_acc = results.get("model_vs_market", {}).get("model_accuracy", 0)
        market_acc = results.get("model_vs_market", {}).get("market_accuracy", 0)
        agreement = results.get("model_vs_market", {}).get("agreement_rate", 0)
        
        print(f"\nModel Accuracy: {model_acc:.1%}")
        print(f"Market Accuracy: {market_acc:.1%}")
        print(f"Agreement Rate: {agreement:.1%}")
        
        if agreement > 0.95:
            print("\n⚠️  CONCERN: Model agrees with market >95% of the time")
            print("The model might just be learning to match market odds, not finding real edge.")
        
        if model_acc > 0.85 and market_acc > 0.85:
            print("\n⚠️  CONCERN: Both model and market are >85% accurate")
            print("This suggests the holdout set might have easy/obvious fights.")
            print("Consider testing on a more challenging subset (closer fights, upsets, etc.)")
        
        if model_acc > market_acc + 0.05:
            print(f"\n✓ Model is outperforming market by {(model_acc - market_acc)*100:.1f}%")
            print("This suggests the model has learned something valuable.")
        elif model_acc < market_acc - 0.05:
            print(f"\n⚠️  Model is underperforming market by {(market_acc - model_acc)*100:.1f}%")
            print("The model is worse than just following market odds.")
        else:
            print("\n⚠️  Model and market have similar accuracy")
            print("Model is not adding significant value over market consensus.")
        
    finally:
        session.close()


if __name__ == "__main__":
    main()

