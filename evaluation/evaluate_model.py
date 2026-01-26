#!/usr/bin/env python3
"""
Model-vs-Market Evaluation for UFC Fights
----------------------------------------

Phase 4 from `todo_1.md`:
  - Load a holdout set of fights (e.g. all 2025 events)
  - Generate / load features using the schema builder
  - Compare model probabilities vs betting market
  - Track:
      * Brier score
      * Log-loss
      * Calibration curves
      * Win / loss edges
      * Expected value vs closing lines
      * ROI vs Kelly / flat stakes
  - Save results into:
      reports/model_eval_<timestamp>.json
      reports/calibration_<timestamp>.png
      reports/roc_<timestamp>.png
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from database.db_manager import DatabaseManager
from database.schema import Fighter, Event
from features.feature_pipeline import FeaturePipeline
from models.xgboost_model import XGBoostModel


def normalize_name(name: str) -> str:
    """Simple name normalizer for joining odds to DB fighter names."""
    if not isinstance(name, str):
        return ""
    name = name.lower()
    # Remove punctuation and extra whitespace
    name = re.sub(r"[^a-z0-9\s]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability (no vig removal)."""
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return np.nan
    if o == 0:
        return np.nan
    if o < 0:
        return (-o) / ((-o) + 100.0)
    else:
        return 100.0 / (o + 100.0)


def build_odds_index(odds_path: Path, date_tolerance_days: int = 0) -> pd.DataFrame:
    """
    Load odds CSV and build normalized keys for joining.

    Expected columns:
      - date (parseable to datetime)
      - fighter1
      - fighter2
      - fighter1_odds
      - fighter2_odds
    """
    df = pd.read_csv(odds_path)

    if "date" not in df.columns:
        raise ValueError("Odds file must have a 'date' column.")

    df["event_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["event_date"])

    df["f1_name_norm"] = df["fighter1"].apply(normalize_name)
    df["f2_name_norm"] = df["fighter2"].apply(normalize_name)

    # Sorted name pair so we can join irrespective of ordering
    def _sorted_pair(row) -> Tuple[str, str]:
        n1, n2 = row["f1_name_norm"], row["f2_name_norm"]
        return tuple(sorted([n1, n2]))

    df[["name_a", "name_b"]] = df.apply(
        lambda r: pd.Series(_sorted_pair(r)), axis=1
    )

    # Normalize to midnight so we can safely shift by whole days if needed
    df["event_day"] = df["event_date"].dt.normalize()

    # Optional tolerance: if odds dates are off by +/- 1 day (common timezone issue),
    # we can expand each odds row to multiple candidate event days.
    try:
        date_tolerance_days = int(date_tolerance_days)
    except (TypeError, ValueError):
        date_tolerance_days = 0

    if date_tolerance_days > 0:
        expanded = []
        for delta in range(-date_tolerance_days, date_tolerance_days + 1):
            tmp = df.copy()
            tmp["event_day"] = tmp["event_day"] + pd.Timedelta(days=delta)
            tmp["odds_date_shift_days"] = delta
            expanded.append(tmp)
        df = pd.concat(expanded, ignore_index=True)
    else:
        df["odds_date_shift_days"] = 0

    # Key: YYYY-MM-DD | name_a | name_b
    df["fight_key"] = df.apply(
        lambda r: f"{r['event_day'].date()}|{r['name_a']}|{r['name_b']}", axis=1
    )

    # Precompute implied probabilities
    df["fighter1_prob"] = df["fighter1_odds"].apply(american_to_prob)
    df["fighter2_prob"] = df["fighter2_odds"].apply(american_to_prob)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model vs betting market on holdout fights"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgboost_model",
        help=(
            "Saved XGBoost model name in models/saved/ (default: xgboost_model). "
            "Tip: keep a dedicated pre-2025 snapshot (e.g. xgboost_model_pre2025) "
            "so evaluation on 2025+ is stable and reproducible."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/training_data.csv",
        help="Path to full training dataset (with all fights)",
    )
    parser.add_argument(
        "--odds-path",
        type=str,
        default="ufc_2025_odds.csv",
        help="Path to odds CSV (American odds, one row per fight)",
    )
    parser.add_argument(
        "--odds-date-tolerance-days",
        type=int,
        default=0,
        help=(
            "Allow matching odds rows to events even if the odds 'date' is off by +/- N days "
            "(e.g., timezone mismatch where UFCStats shows Jan 18 but odds file uses Jan 19). "
            "Default: 0 (exact date match)."
        ),
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2025,
        help="Minimum event year to include in evaluation (e.g. 2025)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save evaluation reports/plots",
    )
    parser.add_argument(
        "--strict-point-in-time",
        action="store_true",
        help=(
            "Leakage-audit mode: zero out base striking/grappling features that are sourced from "
            "current fighter profile stats and/or unfiltered fight relationships (not point-in-time safe). "
            "This helps quantify how much optimistic lift those features provide in backtests."
        ),
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        default=True,
        help=(
            "Use symmetric probabilities by averaging both fighter orders (DEFAULT: True). "
            "Makes prediction order-invariant (flipping fighters gives same result). "
            "This matches the behavior of xgboost_predict.py with --symmetric flag."
        ),
    )
    parser.add_argument(
        "--no-symmetric",
        dest="symmetric",
        action="store_false",
        help="Disable symmetric mode (use raw prediction from single fighter order).",
    )
    parser.add_argument(
        "--compare-to-baseline",
        action="store_true",
        help="Automatically run baseline comparison after generating evaluation report.",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="models/baseline.json",
        help="Path to baseline JSON file for comparison (default: models/baseline.json)",
    )
    parser.add_argument(
        "--underdog",
        action="store_true",
        help="Print detailed analysis of underdog fights (market prob ≤ 45% for winner)",
    )
    parser.add_argument(
        "--highest-confidence-per-card",
        action="store_true",
        help="Evaluate highest confidence pick per card (event) and report wins/losses",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_short = datetime.now().strftime("%Y%m%d")

    # ------------------------------------------------------------------
    # 1) Load odds and build index
    # ------------------------------------------------------------------
    logger.info(f"Loading odds from {args.odds_path}...")
    odds_df = build_odds_index(
        Path(args.odds_path),
        date_tolerance_days=args.odds_date_tolerance_days,
    )

    logger.info(f"Loaded {len(odds_df)} odds rows.")

    # ------------------------------------------------------------------
    # 2) Load dataset and restrict to holdout year (e.g. 2025)
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset from {args.data_path}...")
    raw_df = pd.read_csv(args.data_path)

    if "event_id" not in raw_df.columns or "fighter_1_id" not in raw_df.columns:
        raise ValueError(
            "Dataset must contain 'event_id', 'fighter_1_id', and 'fighter_2_id' columns."
        )

    # Map event_id -> date, fighter_id -> name
    db = DatabaseManager()
    session = db.get_session()
    try:
        event_ids = (
            raw_df["event_id"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        events = (
            session.query(Event)
            .filter(Event.id.in_(event_ids))
            .all()
        )
        id_to_date = {e.id: e.date for e in events}

        fighter_ids = pd.concat(
            [raw_df["fighter_1_id"], raw_df["fighter_2_id"]]
        ).dropna().astype(int).unique().tolist()

        fighters = (
            session.query(Fighter)
            .filter(Fighter.id.in_(fighter_ids))
            .all()
        )
        id_to_name = {f.id: f.name for f in fighters}
    finally:
        session.close()

    raw_df["event_date"] = pd.to_datetime(
        raw_df["event_id"].map(id_to_date), errors="coerce"
    )
    raw_df["event_year"] = raw_df["event_date"].dt.year
    raw_df["event_day"] = raw_df["event_date"].dt.date

    # Restrict to holdout year+
    eval_df = raw_df[raw_df["event_year"] >= args.min_year].copy()
    logger.info(
        f"Holdout selection: {len(eval_df)} rows with event_year >= {args.min_year}."
    )

    if eval_df.empty:
        logger.warning("No evaluation rows found for specified year range.")
        return

    # Attach fighter names (normalized) for joining to odds
    eval_df["f1_name"] = eval_df["fighter_1_id"].map(id_to_name)
    eval_df["f2_name"] = eval_df["fighter_2_id"].map(id_to_name)
    eval_df["f1_name_norm"] = eval_df["f1_name"].apply(normalize_name)
    eval_df["f2_name_norm"] = eval_df["f2_name"].apply(normalize_name)

    def _sorted_pair_eval(row) -> Tuple[str, str]:
        n1, n2 = row["f1_name_norm"], row["f2_name_norm"]
        return tuple(sorted([n1, n2]))

    eval_df[["name_a", "name_b"]] = eval_df.apply(
        lambda r: pd.Series(_sorted_pair_eval(r)), axis=1
    )

    eval_df["fight_key"] = eval_df.apply(
        lambda r: f"{r['event_day']}|{r['name_a']}|{r['name_b']}", axis=1
    )

    # ------------------------------------------------------------------
    # 3) Join evaluation rows to odds
    # ------------------------------------------------------------------
    merged = eval_df.merge(
        odds_df[
            [
                "fight_key",
                "fighter1",
                "fighter2",
                "fighter1_odds",
                "fighter2_odds",
                "fighter1_prob",
                "fighter2_prob",
                "event_date",
            ]
        ],
        on="fight_key",
        how="inner",
        suffixes=("", "_odds"),
    )

    logger.info(
        f"Matched {len(merged)} evaluation rows to odds "
        f"(out of {len(eval_df)} potential rows)."
    )

    if merged.empty:
        logger.warning("No evaluation rows matched to odds; nothing to do.")
        return

    # Determine which side of the odds corresponds to this row's fighter_1
    merged["fighter1_norm_odds"] = merged["fighter1"].apply(normalize_name)
    merged["fighter2_norm_odds"] = merged["fighter2"].apply(normalize_name)

    def _pick_market_prob(row) -> float:
        f1_norm = row["f1_name_norm"]
        if f1_norm == row["fighter1_norm_odds"]:
            return row["fighter1_prob"]
        elif f1_norm == row["fighter2_norm_odds"]:
            return row["fighter2_prob"]
        else:
            # Name mismatch – we will drop this row later
            return np.nan

    merged["market_prob_f1"] = merged.apply(_pick_market_prob, axis=1)
    before_drop = len(merged)
    merged = merged.dropna(subset=["market_prob_f1"])
    logger.info(
        f"Dropped {before_drop - len(merged)} rows due to name mismatch between DB and odds."
    )

    if merged.empty:
        logger.warning("After name alignment, no rows remain; nothing to evaluate.")
        return

    # ------------------------------------------------------------------
    # Optional leakage-audit: remove non-point-in-time-safe feature groups
    # ------------------------------------------------------------------
    if args.strict_point_in_time:
        # These are the feature keys produced by the base "striking" and base "grappling" groups.
        # We keep *recent_* FightStats-derived features (they already use fight_history filtered by as_of_date).
        base_striking_keys = [
            "sig_strikes_landed_per_min",
            "striking_accuracy",
            "striking_defense",
            "striking_differential",
            "defensive_efficiency",
            "striking_volume_control",
            "distance_accuracy_last_3",
            "clinch_accuracy_last_3",
            "ground_output_per_min_last_3",
            "leg_strike_rate_last_3",
            "knockdowns_last_3",
            "striking_accuracy_last_3",
            "sig_strikes_landed_per_min_last_3",
            "head_strike_rate_last_3",
            "body_strike_rate_last_3",
            "ground_strike_rate_last_3",
            "distance_strike_rate_last_3",
            "distance_accuracy_lifetime",
            "clinch_accuracy_lifetime",
            "ground_output_per_min_lifetime",
            "leg_strike_rate_lifetime",
            "knockdowns_lifetime",
            "head_strike_rate_lifetime",
            "body_strike_rate_lifetime",
            "ground_strike_rate_lifetime",
            "distance_strike_rate_lifetime",
            "sig_strikes_landed_per_min_lifetime",
            "striking_accuracy_lifetime",
        ]
        base_grappling_keys = [
            "takedown_avg_per_15min",
            "takedown_accuracy",
            "takedown_defense",
            "submission_avg_per_15min",
        ]

        cols_to_zero = []
        for prefix in ("f1_", "f2_"):
            for k in base_striking_keys + base_grappling_keys:
                c = f"{prefix}{k}"
                if c in merged.columns:
                    cols_to_zero.append(c)

        if cols_to_zero:
            merged.loc[:, cols_to_zero] = 0.0
            logger.warning(
                f"[strict-point-in-time] Zeroed {len(cols_to_zero)} base striking/grappling columns "
                f"to reduce leakage risk for backtests."
            )
        else:
            logger.warning(
                "[strict-point-in-time] No matching striking/grappling columns found to zero; "
                "check feature schema or dataset columns."
            )

    # ------------------------------------------------------------------
    # 4) Prepare features (using saved pipeline) and get model predictions
    # ------------------------------------------------------------------
    logger.info("Loading feature pipeline and model...")
    model_name = args.model_name
    pipeline = FeaturePipeline(initialize_db=False)
    pipeline.load_pipeline(model_name=model_name)

    xgb_model = XGBoostModel()
    xgb_model.load_model(model_name)

    # Prepare features for evaluation (inference mode: fit_scaler=False)
    X_eval, y_eval = pipeline.prepare_features(merged, fit_scaler=False)

    logger.info(f"Prepared evaluation features: {X_eval.shape[0]} rows, {X_eval.shape[1]} cols.")

    # Predict probabilities for fighter_1 in each row
    model_probs = xgb_model.predict(X_eval, use_calibrated=False)

    merged["model_prob_f1"] = model_probs
    merged["model_name"] = model_name
    merged["target"] = merged.get("target", y_eval)

    # ------------------------------------------------------------------
    # 4b) Apply symmetric averaging if requested (matches xgboost_predict.py behavior)
    # ------------------------------------------------------------------
    if args.symmetric:
        logger.info("Applying symmetric probability averaging (matching xgboost_predict.py --symmetric)...")
        # For each unique fight (identified by fight_key), we have 2 rows (both perspectives)
        # We need to compute symmetric probability: p_f1 = 0.5 * (p_f1_raw + (1.0 - p_f2_raw))
        
        # Create a symmetric probability column
        merged["model_prob_f1_symmetric"] = np.nan
        
        # Group by fight_key to get both perspectives of the same fight
        for fight_key, group in merged.groupby("fight_key"):
            if len(group) < 2:
                # Only one perspective available, use raw prediction
                merged.loc[group.index, "model_prob_f1_symmetric"] = group["model_prob_f1"].values
                continue
            
            # Get both rows - they should be the two perspectives (f1, f2) and (f2, f1)
            if len(group) != 2:
                # Unexpected number of rows, use raw predictions
                merged.loc[group.index, "model_prob_f1_symmetric"] = group["model_prob_f1"].values
                continue
            
            # Get the two rows as Series
            idx_list = list(group.index)
            row1_idx = idx_list[0]
            row2_idx = idx_list[1]
            row1 = group.loc[row1_idx]
            row2 = group.loc[row2_idx]
            
            # Verify these are the two perspectives of the same fight
            # (fighter_1_id and fighter_2_id should be swapped)
            f1_id_1 = int(row1["fighter_1_id"]) if "fighter_1_id" in row1 else None
            f2_id_1 = int(row1["fighter_2_id"]) if "fighter_2_id" in row1 else None
            f1_id_2 = int(row2["fighter_1_id"]) if "fighter_1_id" in row2 else None
            f2_id_2 = int(row2["fighter_2_id"]) if "fighter_2_id" in row2 else None
            
            if (f1_id_1 is not None and f2_id_1 is not None and 
                f1_id_2 is not None and f2_id_2 is not None and
                f1_id_1 == f2_id_2 and f2_id_1 == f1_id_2):
                # These are the two perspectives: (f1, f2) and (f2, f1)
                p_f1_raw = float(row1["model_prob_f1"])  # P(f1 wins | f1, f2)
                p_f2_raw = float(row2["model_prob_f1"])  # P(f2 wins | f2, f1)
                
                # Symmetric probability for fighter_1 (from row1's perspective)
                p_f1_symmetric = 0.5 * (p_f1_raw + (1.0 - p_f2_raw))
                p_f2_symmetric = 1.0 - p_f1_symmetric
                
                # Apply to both rows
                merged.loc[row1_idx, "model_prob_f1_symmetric"] = p_f1_symmetric
                merged.loc[row2_idx, "model_prob_f1_symmetric"] = p_f2_symmetric
            else:
                # Not matching perspectives, use raw predictions
                merged.loc[group.index, "model_prob_f1_symmetric"] = group["model_prob_f1"].values
        
        # Use symmetric probabilities if available, otherwise fall back to raw
        merged["model_prob_f1"] = merged["model_prob_f1_symmetric"].fillna(merged["model_prob_f1"])
        logger.info("Applied symmetric probability averaging.")
    else:
        logger.info("Using raw predictions (symmetric mode disabled).")

    # ------------------------------------------------------------------
    # 5) Compute metrics vs market
    # ------------------------------------------------------------------
    y_true = merged["target"].astype(int).values
    p_model = merged["model_prob_f1"].values
    p_market = merged["market_prob_f1"].values

    # Core metrics
    # Clip probabilities to avoid log(0) without relying on sklearn's deprecated eps arg
    p_model_clipped = np.clip(p_model, 1e-15, 1 - 1e-15)
    brier = brier_score_loss(y_true, p_model_clipped)
    ll = log_loss(y_true, p_model_clipped)
    try:
        auc = roc_auc_score(y_true, p_model)
    except ValueError:
        auc = float("nan")

    # Edge and simple ROI (flat 1-unit stakes, bet if edge > 0)
    edge = p_model - p_market
    merged["edge"] = edge

    # Flat betting: 1 unit on fighter_1 whenever edge > 0
    bets_mask = edge > 0
    n_bets = bets_mask.sum()
    if n_bets > 0:
        # Determine actual price you would get for fighter_1 in each row
        def _pick_price(row) -> float:
            f1_norm = row["f1_name_norm"]
            if f1_norm == row["fighter1_norm_odds"]:
                return row["fighter1_odds"]
            else:
                return row["fighter2_odds"]

        merged["price_f1"] = merged.apply(_pick_price, axis=1)

        prices = merged.loc[bets_mask, "price_f1"].values
        outcomes = y_true[bets_mask]

        # Profit per bet in units
        profits = []
        for o, price in zip(outcomes, prices):
            if price > 0:
                payout = price / 100.0  # profit on 1 unit stake if win
            else:
                payout = 100.0 / (-price)
            profits.append(payout if o == 1 else -1.0)

        profits = np.array(profits)
        roi_flat = profits.mean()
    else:
        roi_flat = float("nan")

    # ------------------------------------------------------------------
    # 5b) Simple winner/loser accuracy statistics
    # ------------------------------------------------------------------
    # Fighter-row accuracy: does model_prob_f1 > 0.5 agree with target?
    row_preds = (p_model >= 0.5).astype(int)
    row_correct = (row_preds == y_true).sum()
    row_total = len(y_true)
    row_acc = row_correct / row_total if row_total > 0 else float("nan")

    print("\n" + "=" * 80)
    print("Fighter-row prediction accuracy (model_prob_f1 >= 0.5 as pick):")
    print(f"  Correct rows: {row_correct}/{row_total}  (accuracy={row_acc:.3f})")

    # ------------------------------------------------------------------
    # 6) Human-readable diagnostics: fight-level top/bottom edges
    # ------------------------------------------------------------------
    # Collapse to one row per fight_key:
    #   - winner / loser names
    #   - model & market probs for each side
    #   - edge_best: max(edge_winner, edge_loser)
    #   - edge_worst: min(edge_winner, edge_loser)
    fight_rows = []
    for fk, g in merged.groupby("fight_key"):
        if g.empty:
            continue

        # Sort so winner row (target==1) comes first if present
        g_sorted = g.sort_values("target", ascending=False)
        winner_row = g_sorted.iloc[0]
        loser_row = g_sorted.iloc[1] if len(g_sorted) > 1 else None

        event_date_fight = pd.to_datetime(
            winner_row.get("event_date"), errors="coerce"
        ).date()

        winner_name = winner_row.get("f1_name")
        winner_model_prob = winner_row.get("model_prob_f1")
        winner_market_prob = winner_row.get("market_prob_f1")
        winner_edge = winner_row.get("edge")

        if loser_row is not None:
            loser_name = loser_row.get("f1_name")
            loser_model_prob = loser_row.get("model_prob_f1")
            loser_market_prob = loser_row.get("market_prob_f1")
            loser_edge = loser_row.get("edge")
        else:
            # Fallback if we only have one row for some reason
            loser_name = winner_row.get("f2_name")
            loser_model_prob = np.nan
            loser_market_prob = np.nan
            loser_edge = np.nan

        edges = [
            e for e in [winner_edge, loser_edge] if isinstance(e, (int, float, np.floating))
        ]
        edge_best = max(edges) if edges else np.nan
        edge_worst = min(edges) if edges else np.nan

        fight_rows.append(
            {
                "fight_key": fk,
                "event_date": event_date_fight,
                "fighter_winner": winner_name if winner_row.get("target", 0) == 1 else loser_name,
                "fighter_loser": loser_name if winner_row.get("target", 0) == 1 else winner_name,
                "winner_model_prob": winner_model_prob,
                "winner_market_prob": winner_market_prob,
                "loser_model_prob": loser_model_prob,
                "loser_market_prob": loser_market_prob,
                "edge_winner": winner_edge,
                "edge_loser": loser_edge,
                "edge_best": edge_best,
                "edge_worst": edge_worst,
            }
        )

    fight_df = pd.DataFrame(fight_rows)

    def _print_fight_table(df: pd.DataFrame, title: str) -> None:
        if df.empty:
            return
        
        # Create a simplified dataframe focused on the winner
        display_df = df.copy()
        
        # Calculate edge as percentage difference (model - market)
        display_df["edge_pct"] = (
            (display_df["winner_model_prob"] - display_df["winner_market_prob"]) * 100
        )
        
        # Format probabilities as percentages
        display_df["model_pct"] = (display_df["winner_model_prob"] * 100).round(1)
        display_df["market_pct"] = (display_df["winner_market_prob"] * 100).round(1)
        display_df["edge_pct"] = display_df["edge_pct"].round(1)
        
        # Select only the columns we want to show
        cols = [
            "event_date",
            "fighter_winner",
            "model_pct",
            "market_pct",
            "edge_pct",
        ]
        safe_cols = [c for c in cols if c in display_df.columns]
        
        # Rename columns for better display
        display_df = display_df[safe_cols].copy()
        display_df.columns = [
            "Date",
            "Winner",
            "Model %",
            "Market %",
            "Edge %",
        ]
        
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(display_df.to_string(index=False))

    if not fight_df.empty:
        # Top 10 fights where the best side (by edge) looks most underpriced
        top_fights = fight_df.sort_values("edge_best", ascending=False).head(10)
        _print_fight_table(
            top_fights,
            "Top 10 fights by max positive edge (edge_best)",
        )

        # Bottom 10 fights where even the best side has the most negative edge
        bottom_fights = fight_df.sort_values("edge_best", ascending=True).head(10)
        _print_fight_table(
            bottom_fights,
            "Bottom 10 fights by max edge (most negative edge_best)",
        )

        # Fights where the model gave higher probability to the loser than the winner
        wrong_mask = (
            fight_df["loser_model_prob"].notna()
            & fight_df["winner_model_prob"].notna()
            & (fight_df["loser_model_prob"] > fight_df["winner_model_prob"])
        )
        wrong_fights = fight_df[wrong_mask].copy()
        if not wrong_fights.empty:
            wrong_fights["prob_diff"] = (
                wrong_fights["loser_model_prob"] - wrong_fights["winner_model_prob"]
            )
            worst_misranked = wrong_fights.sort_values(
                "prob_diff", ascending=False
            ).head(10)
            _print_fight_table(
                worst_misranked,
                "Top 10 fights where model favored the LOSER over the WINNER "
                "(loser_model_prob > winner_model_prob)",
            )

            # Fight-level accuracy: did the model give higher prob to the winner?
            valid_mask = (
                fight_df["winner_model_prob"].notna()
                & fight_df["loser_model_prob"].notna()
            )
            valid_fights = fight_df[valid_mask].copy()
            n_fights_total = len(valid_fights)
            if n_fights_total > 0:
                correct_mask = (
                    valid_fights["winner_model_prob"]
                    > valid_fights["loser_model_prob"]
                )
                n_correct = int(correct_mask.sum())
                n_incorrect = int(n_fights_total - n_correct)
                fight_acc = n_correct / n_fights_total

                print("\n" + "=" * 80)
                print("Fight-level accuracy (winner_model_prob > loser_model_prob):")
                print(
                    f"  Correct fights: {n_correct}/{n_fights_total}  "
                    f"(accuracy={fight_acc:.3f})"
                )

    # ------------------------------------------------------------------
    # 6a) Highest confidence per card analysis (if --highest-confidence-per-card flag is set)
    # ------------------------------------------------------------------
    if args.highest_confidence_per_card and not merged.empty:
        print("\n" + "=" * 80)
        print("TOP 2 HIGHEST CONFIDENCE PICKS PER CARD ANALYSIS")
        print("=" * 80)
        print("For each event (card), find the top 2 fights with highest model confidence")
        print("and check if those picks won.")
        print()
        
        # Calculate confidence for each row (distance from 50%)
        merged["confidence"] = abs(merged["model_prob_f1"] - 0.5) * 2
        
        # Determine which fighter the model picks (prob > 50%)
        merged["model_pick_f1"] = merged["model_prob_f1"] >= 0.5
        merged["model_pick_correct"] = (
            (merged["model_pick_f1"] & (merged["target"] == 1)) |
            (~merged["model_pick_f1"] & (merged["target"] == 0))
        )
        
        # Group by event_id (or event_date if event_id not available)
        if "event_id" in merged.columns:
            event_group_key = "event_id"
        else:
            # Fallback to event_date if event_id not available
            event_group_key = "event_date"
            logger.warning("event_id not found, using event_date for grouping")
        
        # Get event names if available
        event_id_to_name = {}
        if "event_id" in merged.columns:
            db = DatabaseManager()
            session = db.get_session()
            try:
                event_ids = merged["event_id"].dropna().astype(int).unique().tolist()
                events = session.query(Event).filter(Event.id.in_(event_ids)).all()
                event_id_to_name = {e.id: e.name for e in events}
            except Exception as e:
                logger.warning(f"Could not fetch event names: {e}")
            finally:
                session.close()
        
        highest_confidence_picks = []
        seen_fights = set()
        
        for event_key, event_group in merged.groupby(event_group_key):
            # Skip if no valid rows
            if event_group.empty:
                continue
            
            # Remove duplicates by fight_key (in case we have both perspectives)
            # Keep the row with higher confidence
            event_group_dedup = event_group.sort_values("confidence", ascending=False)
            event_group_dedup = event_group_dedup.drop_duplicates(subset=["fight_key"], keep="first")
            
            if event_group_dedup.empty:
                continue
            
            # Get event info (same for all picks from this event)
            if event_group_key == "event_id":
                event_id = int(event_key)
                event_name = event_id_to_name.get(event_id, f"Event {event_id}")
            else:
                event_id = None
                event_name = f"Event {event_key}"
            
            # Get top 2 highest confidence picks for this event
            top_picks = event_group_dedup.nlargest(2, "confidence")
            
            for pick_rank, (idx, pick_row) in enumerate(top_picks.iterrows(), 1):
                fight_key = pick_row.get("fight_key", "")
                
                if not fight_key or fight_key in seen_fights:
                    continue
                
                seen_fights.add(fight_key)
                
                event_date = pd.to_datetime(pick_row.get("event_date"), errors="coerce")
                event_date_str = event_date.strftime("%Y-%m-%d") if pd.notna(event_date) else "Unknown"
                
                # Get fighter names
                f1_name = pick_row.get("f1_name", "Unknown")
                f2_name = pick_row.get("f2_name", "Unknown")
                
                # Determine which fighter the model picked
                model_prob_f1 = pick_row.get("model_prob_f1", np.nan)
                if pd.isna(model_prob_f1):
                    continue
                
                if model_prob_f1 > 0.5:
                    predicted_winner = f1_name
                    predicted_loser = f2_name
                    predicted_winner_prob = model_prob_f1
                else:
                    predicted_winner = f2_name
                    predicted_loser = f1_name
                    predicted_winner_prob = 1.0 - model_prob_f1
                
                # Check if the pick was correct
                target = pick_row.get("target", np.nan)
                if pd.isna(target):
                    continue
                
                if model_prob_f1 > 0.5:
                    pick_correct = (target == 1)
                else:
                    pick_correct = (target == 0)
                
                confidence = pick_row.get("confidence", 0.0)
                
                # Get odds for the predicted winner to calculate ROI
                # First determine which fighter is the predicted winner and get their normalized name
                if model_prob_f1 > 0.5:
                    predicted_winner_norm = pick_row.get("f1_name_norm", "")
                else:
                    predicted_winner_norm = pick_row.get("f2_name_norm", "")
                
                fighter1_norm_odds = pick_row.get("fighter1_norm_odds", "")
                fighter2_norm_odds = pick_row.get("fighter2_norm_odds", "")
                
                # Match predicted winner to the correct odds side
                predicted_winner_odds = np.nan
                if predicted_winner_norm == fighter1_norm_odds:
                    predicted_winner_odds = pick_row.get("fighter1_odds", np.nan)
                elif predicted_winner_norm == fighter2_norm_odds:
                    predicted_winner_odds = pick_row.get("fighter2_odds", np.nan)
                
                # Calculate profit/loss for 1 unit bet
                profit_loss = 0.0
                if pd.notna(predicted_winner_odds) and pd.notna(pick_correct):
                    if pick_correct:
                        # Bet won: calculate profit from American odds
                        if predicted_winner_odds > 0:
                            # Positive odds: profit = (odds / 100) * stake
                            profit_loss = (predicted_winner_odds / 100.0) * 1.0
                        else:
                            # Negative odds: profit = (100 / abs(odds)) * stake
                            profit_loss = (100.0 / abs(predicted_winner_odds)) * 1.0
                    else:
                        # Bet lost: lose the stake
                        profit_loss = -1.0
                
                highest_confidence_picks.append({
                    "Event": event_name,
                    "Event Date": event_date_str,
                    "Fight": f"{f1_name} vs {f2_name}",
                    "Predicted Winner": predicted_winner,
                    "Opponent": predicted_loser,
                    "Model Prob %": round(predicted_winner_prob * 100, 1),
                    "Confidence": round(confidence, 4),
                    "Odds": int(predicted_winner_odds) if pd.notna(predicted_winner_odds) else "N/A",
                    "Won?": "Yes" if pick_correct else "No",
                    "Profit/Loss": round(profit_loss, 2) if pd.notna(profit_loss) else "N/A",
                })
        
        if highest_confidence_picks:
            picks_df = pd.DataFrame(highest_confidence_picks)
            
            # Sort by wins/losses (wins first, then losses), then by confidence within each group
            # Convert "Won?" to a sortable value (Yes=0, No=1 so Yes comes first)
            picks_df["_sort_won"] = picks_df["Won?"].apply(lambda x: 0 if x == "Yes" else 1)
            picks_df = picks_df.sort_values(["_sort_won", "Confidence"], ascending=[True, False])
            picks_df = picks_df.drop(columns=["_sort_won"])
            
            # Format Event Date for display (keep as string)
            picks_df["Event Date"] = pd.to_datetime(picks_df["Event Date"], errors="coerce")
            picks_df["Event Date"] = picks_df["Event Date"].dt.strftime("%Y-%m-%d")
            
            # Count unique events
            unique_events = picks_df["Event"].nunique()
            print(f"Found {len(picks_df)} picks from {unique_events} events (top 2 per card):\n")
            print(picks_df.to_string(index=False))
            
            # Summary statistics
            total_picks = len(picks_df)
            correct_picks = (picks_df["Won?"] == "Yes").sum()
            incorrect_picks = total_picks - correct_picks
            accuracy = correct_picks / total_picks if total_picks > 0 else 0.0
            
            # Calculate ROI (1 unit per bet) - ALL BETS
            # Filter out rows where Profit/Loss is "N/A"
            valid_bets = picks_df[picks_df["Profit/Loss"] != "N/A"].copy()
            if len(valid_bets) > 0:
                # Convert Profit/Loss to numeric (in case it's stored as string)
                valid_bets["Profit/Loss"] = pd.to_numeric(valid_bets["Profit/Loss"], errors="coerce")
                total_profit_loss = valid_bets["Profit/Loss"].sum()
                total_staked = len(valid_bets) * 1.0  # 1 unit per bet
                roi_pct = (total_profit_loss / total_staked) * 100 if total_staked > 0 else 0.0
            else:
                total_profit_loss = 0.0
                total_staked = 0.0
                roi_pct = 0.0
            
            # Calculate filtered ROI (Model Prob > 71% AND Favorite odds >= -300)
            filtered_bets = picks_df[picks_df["Profit/Loss"] != "N/A"].copy()
            filtered_bets_subset = pd.DataFrame()  # Initialize empty DataFrame
            if len(filtered_bets) > 0:
                # Convert columns to numeric
                filtered_bets["Profit/Loss"] = pd.to_numeric(filtered_bets["Profit/Loss"], errors="coerce")
                filtered_bets["Model Prob %"] = pd.to_numeric(filtered_bets["Model Prob %"], errors="coerce")
                filtered_bets["Odds"] = pd.to_numeric(filtered_bets["Odds"], errors="coerce")
                
                # Filter: Model Prob > 71% AND Odds >= -300 (favorite odds >= -300 means odds >= -300)
                filtered_mask = (
                    (filtered_bets["Model Prob %"] > 71.0) &
                    (filtered_bets["Odds"] >= -300)
                )
                filtered_bets_subset = filtered_bets[filtered_mask].copy()
                
                if len(filtered_bets_subset) > 0:
                    filtered_profit_loss = filtered_bets_subset["Profit/Loss"].sum()
                    filtered_staked = len(filtered_bets_subset) * 1.0
                    filtered_roi_pct = (filtered_profit_loss / filtered_staked) * 100 if filtered_staked > 0 else 0.0
                    filtered_correct = (filtered_bets_subset["Won?"] == "Yes").sum()
                    filtered_accuracy = filtered_correct / len(filtered_bets_subset) if len(filtered_bets_subset) > 0 else 0.0
                else:
                    filtered_profit_loss = 0.0
                    filtered_staked = 0.0
                    filtered_roi_pct = 0.0
                    filtered_correct = 0
                    filtered_accuracy = 0.0
            else:
                filtered_profit_loss = 0.0
                filtered_staked = 0.0
                filtered_roi_pct = 0.0
                filtered_correct = 0
                filtered_accuracy = 0.0
            
            print("\n" + "-" * 80)
            print("TOP 2 HIGHEST CONFIDENCE PER CARD SUMMARY:")
            print(f"  Total picks (top 2 per card): {total_picks}")
            print(f"  Unique events: {unique_events}")
            print(f"  Correct picks: {correct_picks}")
            print(f"  Incorrect picks: {incorrect_picks}")
            print(f"  Accuracy: {accuracy:.1%} ({correct_picks}/{total_picks})")
            
            if total_staked > 0:
                print(f"\n  FINANCIAL SUMMARY - ALL BETS (1 unit per bet):")
                print(f"  Total staked: {total_staked:.2f} units")
                print(f"  Total profit/loss: {total_profit_loss:+.2f} units")
                print(f"  ROI: {roi_pct:+.2f}%")
            else:
                print(f"\n  FINANCIAL SUMMARY - ALL BETS: No valid odds data available for ROI calculation")
            
            if filtered_staked > 0:
                print(f"\n  FINANCIAL SUMMARY - FILTERED BETS (Model Prob > 71% AND Favorite Odds >= -300):")
                print(f"  Total bets: {len(filtered_bets_subset)}")
                print(f"  Correct picks: {filtered_correct}/{len(filtered_bets_subset)} ({filtered_accuracy:.1%})")
                print(f"  Total staked: {filtered_staked:.2f} units")
                print(f"  Total profit/loss: {filtered_profit_loss:+.2f} units")
                print(f"  ROI: {filtered_roi_pct:+.2f}%")
            else:
                print(f"\n  FINANCIAL SUMMARY - FILTERED BETS: No bets meet criteria (Model Prob > 71% AND Favorite Odds >= -300)")
            
            # Save to CSV
            csv_path = output_dir / f"highest_confidence_per_card_{timestamp}.csv"
            picks_df.to_csv(csv_path, index=False)
            logger.success(f"Saved highest confidence per card analysis to {csv_path}")
            print(f"\n  CSV saved to: {csv_path}")
        else:
            print("No highest confidence picks found.")
    
    # ------------------------------------------------------------------
    # 6b) Underdog analysis (if --underdog flag is set)
    # ------------------------------------------------------------------
    # Initialize underdog summary for JSON report
    underdog_summary = {
        "total_bets": 0,
        "underdog_won": {"count": 0, "percentage": None},
        "correct_predictions": {"count": 0, "percentage": None},
        "all_bets": {
            "total_staked": None,
            "total_profit_loss": None,
            "roi": None,
        },
        "top_25_pct": {
            "total_staked": None,
            "total_profit_loss": None,
            "roi": None,
            "correct_predictions": {"count": 0, "percentage": None},
            "confidence_threshold": None,
        },
    }
    
    if args.underdog and not merged.empty:
        print("\n" + "=" * 80)
        print("UNDERDOG BETTING ANALYSIS")
        print("=" * 80)
        print("Analysis of all fights where we bet on underdogs")
        print("(Underdog = fighter with lower market probability)")
        print("(We bet when: model prob > 50% AND edge > 0)")
        print()
        
        # Work with merged dataframe to get all perspectives
        # Identify underdogs in each row and check if we bet on them
        def _get_underdog_info(row):
            """Determine which fighter is the underdog and get their info."""
            f1_norm = row.get("f1_name_norm", "")
            fighter1_norm_odds = row.get("fighter1_norm_odds", "")
            fighter2_norm_odds = row.get("fighter2_norm_odds", "")
            
            # Get market probabilities
            if f1_norm == fighter1_norm_odds:
                f1_market_prob = row.get("fighter1_prob", np.nan)
                f2_market_prob = row.get("fighter2_prob", np.nan)
                f1_odds = row.get("fighter1_odds", np.nan)
                f2_odds = row.get("fighter2_odds", np.nan)
            elif f1_norm == fighter2_norm_odds:
                f1_market_prob = row.get("fighter2_prob", np.nan)
                f2_market_prob = row.get("fighter1_prob", np.nan)
                f1_odds = row.get("fighter2_odds", np.nan)
                f2_odds = row.get("fighter1_odds", np.nan)
            else:
                return None, None, None, None, None, None, None, None
            
            # Determine underdog (lower market prob)
            if f1_market_prob < f2_market_prob:
                underdog_id = "f1"
                underdog_name = row.get("f1_name", "")
                opponent_name = row.get("f2_name", "")
                underdog_market_prob = f1_market_prob
                underdog_model_prob = row.get("model_prob_f1", np.nan)
                underdog_edge = row.get("edge", np.nan)  # edge for f1
                underdog_odds = f1_odds
            elif f2_market_prob < f1_market_prob:
                underdog_id = "f2"
                underdog_name = row.get("f2_name", "")
                opponent_name = row.get("f1_name", "")
                underdog_market_prob = f2_market_prob
                underdog_model_prob = 1.0 - row.get("model_prob_f1", np.nan)  # f2 prob = 1 - f1 prob
                # Edge for f2 = model_prob_f2 - market_prob_f2
                underdog_edge = underdog_model_prob - underdog_market_prob
                underdog_odds = f2_odds
            else:
                return None, None, None, None, None, None, None, None
            
            return underdog_id, underdog_name, opponent_name, underdog_market_prob, underdog_model_prob, underdog_edge, underdog_odds
        
        # Process all rows to find underdog bets
        underdog_bets = []
        seen_fights = set()
        
        for idx, row in merged.iterrows():
            fight_key = row.get("fight_key", "")
            if not fight_key or fight_key in seen_fights:
                continue
            
            result = _get_underdog_info(row)
            if result[0] is None:
                continue
            
            underdog_id, underdog_name, opponent_name, underdog_market_prob, underdog_model_prob, underdog_edge, underdog_odds = result
            
            # Check if we bet on this underdog (model prob > 50% AND edge > 0)
            we_bet = (pd.notna(underdog_model_prob) and pd.notna(underdog_edge) and 
                     underdog_model_prob > 0.5 and underdog_edge > 0)
            
            # Only include if we bet on them
            if we_bet:
                # Check if underdog won (target == 1 means f1 won)
                if underdog_id == "f1":
                    underdog_won = (row.get("target", 0) == 1)
                else:
                    underdog_won = (row.get("target", 0) == 0)
                
                # Get event date
                event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
                event_date_str = event_date.strftime("%Y-%m-%d") if pd.notna(event_date) else "Unknown"
                
                # Calculate profit/loss for flat 1 unit bet
                # If underdog wins: profit = odds payout - 1 unit stake
                # If underdog loses: profit = -1 unit (lost stake)
                profit_loss = 0.0
                if pd.notna(underdog_odds) and pd.notna(underdog_won):
                    if underdog_won:
                        # Calculate profit from American odds
                        if underdog_odds > 0:
                            # Positive odds: profit = (odds / 100) * stake
                            profit_loss = (underdog_odds / 100.0) * 1.0
                        else:
                            # Negative odds: profit = (100 / abs(odds)) * stake
                            profit_loss = (100.0 / abs(underdog_odds)) * 1.0
                    else:
                        # Lost bet: lose the stake
                        profit_loss = -1.0
                
                # Calculate confidence (distance from 50%)
                confidence = abs(underdog_model_prob - 0.5) * 2 if pd.notna(underdog_model_prob) else 0.0
                
                underdog_bets.append({
                    "Date": event_date_str,
                    "Underdog": underdog_name,
                    "Favorite": opponent_name,
                    "Market Prob %": round(underdog_market_prob * 100, 1) if pd.notna(underdog_market_prob) else "N/A",
                    "Model Prob %": round(underdog_model_prob * 100, 1) if pd.notna(underdog_model_prob) else "N/A",
                    "Edge %": round(underdog_edge * 100, 1) if pd.notna(underdog_edge) else "N/A",
                    "Confidence": round(confidence, 4) if pd.notna(confidence) else 0.0,
                    "Odds": int(underdog_odds) if pd.notna(underdog_odds) else "N/A",
                    "Underdog Won?": "Yes" if underdog_won else "No",
                    "Correct?": "Yes" if underdog_won else "No",  # Correct if underdog won (since we bet on them)
                    "Profit/Loss": round(profit_loss, 2),
                })
                
                seen_fights.add(fight_key)
        
        if underdog_bets:
            underdog_df = pd.DataFrame(underdog_bets)
            # Sort by Edge % (highest first)
            if "Edge %" in underdog_df.columns:
                # Convert Edge % to numeric for proper sorting (handles "N/A" values)
                underdog_df["_sort_edge"] = pd.to_numeric(underdog_df["Edge %"], errors="coerce")
                underdog_df = underdog_df.sort_values("_sort_edge", ascending=False, na_position="last")
                underdog_df = underdog_df.drop(columns=["_sort_edge"])
            print(f"Found {len(underdog_df)} underdog bets:\n")
            print(underdog_df.to_string(index=False))
            
            # Save to CSV
            csv_path = output_dir / f"underdog_bets_{timestamp}.csv"
            underdog_df.to_csv(csv_path, index=False)
            logger.success(f"Saved underdog bets analysis to {csv_path}")
            
            # Summary statistics
            total_bets = len(underdog_df)
            correct_bets = (underdog_df["Correct?"] == "Yes").sum()
            underdog_wins = (underdog_df["Underdog Won?"] == "Yes").sum()
            
            # Calculate total profit/loss and ROI for all underdog bets
            total_profit_loss = underdog_df["Profit/Loss"].sum()
            total_staked = total_bets * 1.0  # 1 unit per bet
            roi_pct = (total_profit_loss / total_staked) * 100 if total_staked > 0 else 0.0
            
            # Calculate top 25% confidence threshold and filter
            top_25_threshold = None
            if "Confidence" in underdog_df.columns and len(underdog_df) > 0:
                top_25_threshold = underdog_df["Confidence"].quantile(0.75)
                top_25_mask = underdog_df["Confidence"] >= top_25_threshold
                top_25_df = underdog_df[top_25_mask].copy()
                
                # Calculate ROI for top 25%
                top_25_bets = len(top_25_df)
                top_25_profit_loss = top_25_df["Profit/Loss"].sum()
                top_25_staked = top_25_bets * 1.0
                top_25_roi = (top_25_profit_loss / top_25_staked) * 100 if top_25_staked > 0 else 0.0
                top_25_correct = (top_25_df["Correct?"] == "Yes").sum()
            else:
                top_25_bets = 0
                top_25_profit_loss = 0.0
                top_25_staked = 0.0
                top_25_roi = 0.0
                top_25_correct = 0
            
            # Store summary for JSON report
            underdog_summary["total_bets"] = total_bets
            underdog_summary["underdog_won"] = {
                "count": int(underdog_wins),
                "percentage": round(underdog_wins / total_bets * 100, 1) if total_bets > 0 else None,
            }
            underdog_summary["correct_predictions"] = {
                "count": int(correct_bets),
                "percentage": round(correct_bets / total_bets * 100, 1) if total_bets > 0 else None,
            }
            underdog_summary["all_bets"] = {
                "total_staked": round(total_staked, 2),
                "total_profit_loss": round(total_profit_loss, 2),
                "roi": round(roi_pct, 2),
            }
            
            if top_25_bets > 0 and top_25_threshold is not None:
                underdog_summary["top_25_pct"] = {
                    "total_staked": round(top_25_staked, 2),
                    "total_profit_loss": round(top_25_profit_loss, 2),
                    "roi": round(top_25_roi, 2),
                    "correct_predictions": {
                        "count": int(top_25_correct),
                        "percentage": round(top_25_correct / top_25_bets * 100, 1) if top_25_bets > 0 else None,
                    },
                    "confidence_threshold": round(float(top_25_threshold), 4),
                }
            
            print("\n" + "-" * 80)
            print("UNDERDOG BETTING SUMMARY:")
            print(f"  Total underdog bets: {total_bets}")
            print(f"  Underdog won: {underdog_wins}/{total_bets} ({underdog_wins/total_bets*100:.1f}%)")
            print(f"  Correct predictions: {correct_bets}/{total_bets} ({correct_bets/total_bets*100:.1f}%)")
            print(f"\n  FINANCIAL SUMMARY - ALL UNDERDOG BETS (Flat 1 unit bets):")
            print(f"  Total staked: {total_staked:.2f} units")
            print(f"  Total profit/loss: {total_profit_loss:+.2f} units")
            print(f"  ROI: {roi_pct:+.2f}%")
            if top_25_bets > 0:
                print(f"\n  FINANCIAL SUMMARY - TOP 25% CONFIDENCE UNDERDOG BETS:")
                print(f"  Total staked: {top_25_staked:.2f} units")
                print(f"  Total profit/loss: {top_25_profit_loss:+.2f} units")
                print(f"  ROI: {top_25_roi:+.2f}%")
                print(f"  Correct predictions: {top_25_correct}/{top_25_bets} ({top_25_correct/top_25_bets*100:.1f}%)")
                print(f"  Confidence threshold: {top_25_threshold:.4f}")
            print(f"\n  CSV saved to: {csv_path}")
        else:
            print("No underdog bets found (model prob > 50% AND edge > 0).")
        
        # Also show underdog winners (for comparison with HTML report)
        if not fight_df.empty:
            underdog_mask = fight_df["winner_market_prob"] < fight_df["loser_market_prob"]
            underdog_fights = fight_df[underdog_mask].copy()
            
            if not underdog_fights.empty:
                print("\n" + "-" * 80)
                print("UNDERDOG WINNERS (for comparison with HTML report):")
                print(f"  Total underdog winners: {len(underdog_fights)}")
                print("  (This matches 'UNDERDOG WINS (UPSETS)' in HTML report)")
    
    # ------------------------------------------------------------------
    # 6c) All bets ROI analysis (always runs)
    # ------------------------------------------------------------------
    if not merged.empty:
        print("\n" + "=" * 80)
        print("ALL BETS ROI ANALYSIS")
        print("=" * 80)
        print("Analysis of all bets where: model prob > 50% AND edge > 0")
        print("(Includes both favorites and underdogs)")
        print()
        
        all_bets = []
        seen_fights_all = set()
        
        for idx, row in merged.iterrows():
            fight_key = row.get("fight_key", "")
            if not fight_key or fight_key in seen_fights_all:
                continue
            
            # Get model probabilities
            model_prob_f1 = row.get("model_prob_f1", np.nan)
            model_prob_f2 = 1.0 - model_prob_f1 if pd.notna(model_prob_f1) else np.nan
            
            # Determine which fighter the model picks (prob > 50%)
            if pd.isna(model_prob_f1) or model_prob_f1 == 0.5:
                continue
            
            if model_prob_f1 > 0.5:
                predicted_winner_id = "f1"
                predicted_winner_name = row.get("f1_name", "")
                predicted_loser_name = row.get("f2_name", "")
                predicted_winner_model_prob = model_prob_f1
                predicted_winner_edge = row.get("edge", np.nan)  # edge for f1
            else:
                predicted_winner_id = "f2"
                predicted_winner_name = row.get("f2_name", "")
                predicted_loser_name = row.get("f1_name", "")
                predicted_winner_model_prob = model_prob_f2
                # Edge for f2 = model_prob_f2 - market_prob_f2
                market_prob_f2 = row.get("market_prob_f2", np.nan)
                if pd.isna(market_prob_f2):
                    # Try to get from odds
                    f1_norm = row.get("f1_name_norm", "")
                    fighter1_norm_odds = row.get("fighter1_norm_odds", "")
                    if f1_norm == fighter1_norm_odds:
                        market_prob_f2 = row.get("fighter2_prob", np.nan)
                    elif f1_norm == row.get("fighter2_norm_odds", ""):
                        market_prob_f2 = row.get("fighter1_prob", np.nan)
                predicted_winner_edge = predicted_winner_model_prob - market_prob_f2 if pd.notna(market_prob_f2) else np.nan
            
            # Check if we bet (model prob > 50% AND edge > 0)
            we_bet = (pd.notna(predicted_winner_model_prob) and pd.notna(predicted_winner_edge) and 
                     predicted_winner_model_prob > 0.5 and predicted_winner_edge > 0)
            
            if we_bet:
                # Get odds for predicted winner
                f1_norm = row.get("f1_name_norm", "")
                fighter1_norm_odds = row.get("fighter1_norm_odds", "")
                fighter2_norm_odds = row.get("fighter2_norm_odds", "")
                
                if predicted_winner_id == "f1":
                    if f1_norm == fighter1_norm_odds:
                        predicted_winner_odds = row.get("fighter1_odds", np.nan)
                        predicted_winner_market_prob = row.get("fighter1_prob", np.nan)
                    elif f1_norm == fighter2_norm_odds:
                        predicted_winner_odds = row.get("fighter2_odds", np.nan)
                        predicted_winner_market_prob = row.get("fighter2_prob", np.nan)
                    else:
                        predicted_winner_odds = np.nan
                        predicted_winner_market_prob = np.nan
                    
                    predicted_winner_won = (row.get("target", 0) == 1)
                else:  # f2
                    if f1_norm == fighter2_norm_odds:
                        predicted_winner_odds = row.get("fighter2_odds", np.nan)
                        predicted_winner_market_prob = row.get("fighter2_prob", np.nan)
                    elif f1_norm == fighter1_norm_odds:
                        predicted_winner_odds = row.get("fighter1_odds", np.nan)
                        predicted_winner_market_prob = row.get("fighter1_prob", np.nan)
                    else:
                        predicted_winner_odds = np.nan
                        predicted_winner_market_prob = np.nan
                    
                    predicted_winner_won = (row.get("target", 0) == 0)
                
                # Calculate profit/loss for flat 1 unit bet
                profit_loss = 0.0
                if pd.notna(predicted_winner_odds) and pd.notna(predicted_winner_won):
                    if predicted_winner_won:
                        # Calculate profit from American odds
                        if predicted_winner_odds > 0:
                            # Positive odds: profit = (odds / 100) * stake
                            profit_loss = (predicted_winner_odds / 100.0) * 1.0
                        else:
                            # Negative odds: profit = (100 / abs(odds)) * stake
                            profit_loss = (100.0 / abs(predicted_winner_odds)) * 1.0
                    else:
                        # Lost bet: lose the stake
                        profit_loss = -1.0
                
                # Calculate confidence (distance from 50%)
                confidence = abs(predicted_winner_model_prob - 0.5) * 2 if pd.notna(predicted_winner_model_prob) else 0.0
                
                # Determine if underdog (lower market prob than opponent)
                # Use the same logic as --underdog section for consistency
                is_underdog = False
                if pd.notna(predicted_winner_market_prob):
                    # Get opponent's market probability
                    # We already have predicted_winner_market_prob, now get the opponent's
                    if predicted_winner_id == "f1":
                        # Predicted winner is f1, opponent is f2
                        if f1_norm == fighter1_norm_odds:
                            opponent_market_prob = row.get("fighter2_prob", np.nan)
                        elif f1_norm == fighter2_norm_odds:
                            opponent_market_prob = row.get("fighter1_prob", np.nan)
                        else:
                            opponent_market_prob = np.nan
                    else:  # f2
                        # Predicted winner is f2, opponent is f1
                        if f1_norm == fighter2_norm_odds:
                            opponent_market_prob = row.get("fighter1_prob", np.nan)
                        elif f1_norm == fighter1_norm_odds:
                            opponent_market_prob = row.get("fighter2_prob", np.nan)
                        else:
                            opponent_market_prob = np.nan
                    
                    if pd.notna(opponent_market_prob):
                        is_underdog = predicted_winner_market_prob < opponent_market_prob
                
                # Get event date
                event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
                event_date_str = event_date.strftime("%Y-%m-%d") if pd.notna(event_date) else "Unknown"
                
                all_bets.append({
                    "Date": event_date_str,
                    "Predicted Winner": predicted_winner_name,
                    "Opponent": predicted_loser_name,
                    "Is Underdog": "Yes" if is_underdog else "No",
                    "Market Prob %": round(predicted_winner_market_prob * 100, 1) if pd.notna(predicted_winner_market_prob) else "N/A",
                    "Model Prob %": round(predicted_winner_model_prob * 100, 1) if pd.notna(predicted_winner_model_prob) else "N/A",
                    "Edge %": round(predicted_winner_edge * 100, 1) if pd.notna(predicted_winner_edge) else "N/A",
                    "Confidence": round(confidence, 4) if pd.notna(confidence) else 0.0,
                    "Odds": int(predicted_winner_odds) if pd.notna(predicted_winner_odds) else "N/A",
                    "Won?": "Yes" if predicted_winner_won else "No",
                    "Correct?": "Yes" if predicted_winner_won else "No",
                    "Profit/Loss": round(profit_loss, 2),
                })
                
                seen_fights_all.add(fight_key)
        
        if all_bets:
            all_bets_df = pd.DataFrame(all_bets)
            
            # Summary statistics
            total_bets_all = len(all_bets_df)
            correct_bets_all = (all_bets_df["Correct?"] == "Yes").sum()
            
            # Calculate total profit/loss and ROI for all bets
            total_profit_loss_all = all_bets_df["Profit/Loss"].sum()
            total_staked_all = total_bets_all * 1.0  # 1 unit per bet
            roi_pct_all = (total_profit_loss_all / total_staked_all) * 100 if total_staked_all > 0 else 0.0
            
            # Calculate top 25% confidence threshold and filter
            if "Confidence" in all_bets_df.columns and len(all_bets_df) > 0:
                top_25_threshold_all = all_bets_df["Confidence"].quantile(0.75)
                top_25_mask_all = all_bets_df["Confidence"] >= top_25_threshold_all
                top_25_df_all = all_bets_df[top_25_mask_all].copy()
                
                # Calculate ROI for top 25%
                top_25_bets_all = len(top_25_df_all)
                top_25_profit_loss_all = top_25_df_all["Profit/Loss"].sum()
                top_25_staked_all = top_25_bets_all * 1.0
                top_25_roi_all = (top_25_profit_loss_all / top_25_staked_all) * 100 if top_25_staked_all > 0 else 0.0
                top_25_correct_all = (top_25_df_all["Correct?"] == "Yes").sum()
            else:
                top_25_bets_all = 0
                top_25_profit_loss_all = 0.0
                top_25_staked_all = 0.0
                top_25_roi_all = 0.0
                top_25_correct_all = 0
            
            # Breakdown by underdog vs favorite
            underdog_bets_all = all_bets_df[all_bets_df["Is Underdog"] == "Yes"]
            favorite_bets_all = all_bets_df[all_bets_df["Is Underdog"] == "No"]
            
            print("\n" + "-" * 80)
            print("ALL BETS SUMMARY:")
            print(f"  Total bets: {total_bets_all}")
            print(f"  Correct predictions: {correct_bets_all}/{total_bets_all} ({correct_bets_all/total_bets_all*100:.1f}%)")
            print(f"  Underdog bets: {len(underdog_bets_all)}")
            print(f"  Favorite bets: {len(favorite_bets_all)}")
            print(f"\n  FINANCIAL SUMMARY - ALL BETS (Flat 1 unit bets):")
            print(f"  Total staked: {total_staked_all:.2f} units")
            print(f"  Total profit/loss: {total_profit_loss_all:+.2f} units")
            print(f"  ROI: {roi_pct_all:+.2f}%")
            
            if len(underdog_bets_all) > 0:
                underdog_profit_all = underdog_bets_all["Profit/Loss"].sum()
                underdog_staked_all = len(underdog_bets_all) * 1.0
                underdog_roi_all = (underdog_profit_all / underdog_staked_all) * 100 if underdog_staked_all > 0 else 0.0
                print(f"\n  UNDERDOG BETS:")
                print(f"  Total staked: {underdog_staked_all:.2f} units")
                print(f"  Total profit/loss: {underdog_profit_all:+.2f} units")
                print(f"  ROI: {underdog_roi_all:+.2f}%")
            
            if len(favorite_bets_all) > 0:
                favorite_profit_all = favorite_bets_all["Profit/Loss"].sum()
                favorite_staked_all = len(favorite_bets_all) * 1.0
                favorite_roi_all = (favorite_profit_all / favorite_staked_all) * 100 if favorite_staked_all > 0 else 0.0
                print(f"\n  FAVORITE BETS:")
                print(f"  Total staked: {favorite_staked_all:.2f} units")
                print(f"  Total profit/loss: {favorite_profit_all:+.2f} units")
                print(f"  ROI: {favorite_roi_all:+.2f}%")
            
            if top_25_bets_all > 0:
                print(f"\n  FINANCIAL SUMMARY - TOP 25% CONFIDENCE BETS:")
                print(f"  Total staked: {top_25_staked_all:.2f} units")
                print(f"  Total profit/loss: {top_25_profit_loss_all:+.2f} units")
                print(f"  ROI: {top_25_roi_all:+.2f}%")
                print(f"  Correct predictions: {top_25_correct_all}/{top_25_bets_all} ({top_25_correct_all/top_25_bets_all*100:.1f}%)")
                print(f"  Confidence threshold: {top_25_threshold_all:.4f}")
            
            # Save to CSV
            csv_path_all = output_dir / f"all_bets_{timestamp}.csv"
            all_bets_df.to_csv(csv_path_all, index=False)
            logger.success(f"Saved all bets analysis to {csv_path_all}")
            print(f"\n  CSV saved to: {csv_path_all}")
        else:
            print("No bets found (model prob > 50% AND edge > 0).")
    
    if args.underdog:
        return  # Early return to skip the old code below
        
    # Old code below (kept for reference but won't execute if --underdog is set)
    if False and args.underdog and not fight_df.empty:
        print("\n" + "=" * 80)
        print("UNDERDOG FIGHTS ANALYSIS")
        print("=" * 80)
        print("Underdog fights: Fights where the winner had lower market probability than the loser")
        print("(Matches the definition used in the HTML report)")
        print()
        
        # Filter to underdog fights (winner had lower market prob than loser)
        # This matches the HTML report definition: underdog = fighter with lower market prob
        underdog_mask = fight_df["winner_market_prob"] < fight_df["loser_market_prob"]
        underdog_fights = fight_df[underdog_mask].copy()
        
        if underdog_fights.empty:
            print("No underdog fights found (market prob ≤ 45% for winner).")
        else:
            # Get odds information from merged dataframe
            # We need to match fight_key back to merged to get odds
            underdog_data = []
            for _, fight_row in underdog_fights.iterrows():
                fight_key = fight_row["fight_key"]
                # Find matching rows in merged
                fight_rows_merged = merged[merged["fight_key"] == fight_key]
                
                if not fight_rows_merged.empty:
                    # Get the winner row
                    winner_row_merged = fight_rows_merged[fight_rows_merged["target"] == 1].iloc[0] if (fight_rows_merged["target"] == 1).any() else fight_rows_merged.iloc[0]
                    
                    # Determine which fighter is the winner and get their odds
                    winner_name = fight_row["fighter_winner"]
                    winner_f1_name = winner_row_merged.get("f1_name", "")
                    winner_f2_name = winner_row_merged.get("f2_name", "")
                    
                    # Get odds for winner by matching normalized names
                    winner_f1_norm = normalize_name(winner_f1_name) if winner_f1_name else ""
                    winner_f2_norm = normalize_name(winner_f2_name) if winner_f2_name else ""
                    fighter1_norm_odds = winner_row_merged.get("fighter1_norm_odds", "")
                    fighter2_norm_odds = winner_row_merged.get("fighter2_norm_odds", "")
                    
                    # Match winner to odds side
                    if winner_f1_norm == fighter1_norm_odds:
                        winner_odds = winner_row_merged.get("fighter1_odds", np.nan)
                    elif winner_f1_norm == fighter2_norm_odds:
                        winner_odds = winner_row_merged.get("fighter2_odds", np.nan)
                    elif winner_f2_norm == fighter1_norm_odds:
                        winner_odds = winner_row_merged.get("fighter1_odds", np.nan)
                    elif winner_f2_norm == fighter2_norm_odds:
                        winner_odds = winner_row_merged.get("fighter2_odds", np.nan)
                    else:
                        winner_odds = np.nan
                    
                    # Check if prediction was correct (model picked winner)
                    # Model picked winner if winner_model_prob > loser_model_prob
                    model_picked_winner = False
                    if pd.notna(fight_row["winner_model_prob"]) and pd.notna(fight_row["loser_model_prob"]):
                        model_picked_winner = fight_row["winner_model_prob"] > fight_row["loser_model_prob"]
                    elif pd.notna(fight_row["winner_model_prob"]):
                        model_picked_winner = fight_row["winner_model_prob"] > 0.5
                    
                    # Check if we bet on this underdog
                    # We only bet when: model thinks underdog will win (model_prob > 0.5) AND edge > 0
                    winner_edge = fight_row["edge_winner"]
                    model_thinks_underdog_wins = fight_row["winner_model_prob"] > 0.5
                    we_bet = model_thinks_underdog_wins and winner_edge > 0
                    
                    underdog_data.append({
                        "Date": fight_row["event_date"],
                        "Winner": fight_row["fighter_winner"],
                        "Loser": fight_row["fighter_loser"],
                        "Market Prob %": round(fight_row["winner_market_prob"] * 100, 1),
                        "Model Prob %": round(fight_row["winner_model_prob"] * 100, 1),
                        "Edge %": round(winner_edge * 100, 1),
                        "Odds": winner_odds if not pd.isna(winner_odds) else "N/A",
                        "We Bet?": "Yes" if we_bet else "No",
                        "Correct?": "Yes" if model_picked_winner else "No",
                    })
            
            if underdog_data:
                underdog_df = pd.DataFrame(underdog_data)
                print(f"Found {len(underdog_df)} underdog fights:\n")
                print(underdog_df.to_string(index=False))
                
                # Save to CSV
                csv_path = output_dir / f"underdog_fights_{timestamp}.csv"
                underdog_df.to_csv(csv_path, index=False)
                logger.success(f"Saved underdog fights analysis to {csv_path}")
                
                # Summary statistics
                total_underdogs = len(underdog_df)
                we_bet_count = (underdog_df["We Bet?"] == "Yes").sum()
                correct_count = (underdog_df["Correct?"] == "Yes").sum()
                we_bet_correct = ((underdog_df["We Bet?"] == "Yes") & (underdog_df["Correct?"] == "Yes")).sum()
                
                print("\n" + "-" * 80)
                print("UNDERDOG SUMMARY:")
                print(f"  Total underdog fights (underdog won): {total_underdogs}")
                print(f"  We bet on underdog winner: {we_bet_count} ({we_bet_count/total_underdogs*100:.1f}%)")
                print(f"  Model correctly predicted underdog winner: {correct_count}/{total_underdogs} ({correct_count/total_underdogs*100:.1f}%)")
                if we_bet_count > 0:
                    print(f"  Correct when we bet on underdog winner: {we_bet_correct}/{we_bet_count} ({we_bet_correct/we_bet_count*100:.1f}%)")
                print()
                print("NOTE: This differs from 'UNDERDOG PICKS (MODEL)' in HTML report:")
                print("  - HTML counts ALL model picks of underdogs (winner or loser perspective)")
                print("  - This CLI output counts only underdog WINNERS where we bet (edge > 0)")
                print("  - The 44/83 in HTML means: model picked underdog 83 times, correct 44 times")
                print("  - The 43/67 here means: we bet on underdog winner 67 times, correct 43 times")
                print(f"\n  CSV saved to: {csv_path}")
            else:
                print("Could not retrieve odds information for underdog fights.")

    # ------------------------------------------------------------------
    # 6c) Print specific fighter matchup: Jalin Turner vs Edson Barboza
    # ------------------------------------------------------------------
    target_fighters = ["Jack Della Maddalena", "Belal Muhammed"]
    target_fighters_norm = [normalize_name(f) for f in target_fighters]
    
    # Search in merged dataframe for rows containing these fighters
    fighter_mask = (
        merged["f1_name_norm"].isin(target_fighters_norm)
        | merged["f2_name_norm"].isin(target_fighters_norm)
    )
    target_rows = merged[fighter_mask].copy()
    
    if not target_rows.empty:
        # Further filter to only rows where BOTH fighters are in the target list
        both_fighters_mask = (
            target_rows["f1_name_norm"].isin(target_fighters_norm)
            & target_rows["f2_name_norm"].isin(target_fighters_norm)
        )
        target_fight = target_rows[both_fighters_mask]
        
        if not target_fight.empty:
            # Take the first matching row
            row = target_fight.iloc[0]
            
            # Determine which fighter is which
            f1_name = row["f1_name"]
            f2_name = row["f2_name"]
            f1_model_prob = row["model_prob_f1"]
            f2_model_prob = 1.0 - f1_model_prob
            f1_market_prob = row["market_prob_f1"]
            
            # Get market prob for fighter_2
            f1_norm = row["f1_name_norm"]
            if f1_norm == row["fighter1_norm_odds"]:
                f2_market_prob = row["fighter2_prob"]
            else:
                f2_market_prob = row["fighter1_prob"]
            
            f1_edge = row["edge"]
            f2_edge = f2_model_prob - f2_market_prob
            
            event_date_str = pd.to_datetime(row.get("event_date"), errors="coerce")
            event_date_display = event_date_str.strftime("%Y-%m-%d") if pd.notna(event_date_str) else "Unknown"
            
            print("\n" + "=" * 80)
            print("SPECIFIC FIGHT: Jalin Turner vs Edson Barboza")
            print("=" * 80)
            print(f"Event Date: {event_date_display}")
            print(f"\n{f1_name}:")
            print(f"  Model Probability: {f1_model_prob:.1%}")
            print(f"  Market Probability: {f1_market_prob:.1%}")
            print(f"  Edge: {f1_edge:+.1%}")
            print(f"\n{f2_name}:")
            print(f"  Model Probability: {f2_model_prob:.1%}")
            print(f"  Market Probability: {f2_market_prob:.1%}")
            print(f"  Edge: {f2_edge:+.1%}")
            
            # Get odds
            if f1_norm == row["fighter1_norm_odds"]:
                f1_odds = row["fighter1_odds"]
                f2_odds = row["fighter2_odds"]
            else:
                f1_odds = row["fighter2_odds"]
                f2_odds = row["fighter1_odds"]
            
            print(f"\nMarket Odds:")
            print(f"  {f1_name}: {f1_odds:+.0f}")
            print(f"  {f2_name}: {f2_odds:+.0f}")
            
            # Determine recommended bet based on edge
            if f1_edge > 0.05:  # 5% edge threshold
                print(f"\n⭐ Recommended Bet: {f1_name} (edge: {f1_edge:+.1%})")
            elif f2_edge > 0.05:
                print(f"\n⭐ Recommended Bet: {f2_name} (edge: {f2_edge:+.1%})")
            else:
                print(f"\n⚠️  No strong edge detected (both edges < 5%)")
        else:
            # Check if fight exists in odds but not in evaluation data (future fight)
            odds_fighter_mask = (
                odds_df["fighter1"].str.contains("Turner|Barboza", case=False, na=False)
                | odds_df["fighter2"].str.contains("Turner|Barboza", case=False, na=False)
            )
            odds_match = odds_df[odds_fighter_mask]
            
            if not odds_match.empty:
                # Check if both fighters are present
                both_in_odds = odds_match.apply(
                    lambda r: (
                        normalize_name(str(r["fighter1"])) in target_fighters_norm
                        and normalize_name(str(r["fighter2"])) in target_fighters_norm
                    ),
                    axis=1
                )
                odds_fight = odds_match[both_in_odds]
                
                if not odds_fight.empty:
                    row = odds_fight.iloc[0]
                    event_date_str = pd.to_datetime(row.get("event_date"), errors="coerce")
                    event_date_display = event_date_str.strftime("%Y-%m-%d") if pd.notna(event_date_str) else "Unknown"
                    
                    print("\n" + "=" * 80)
                    print("SPECIFIC FIGHT: Jalin Turner vs Edson Barboza")
                    print("=" * 80)
                    print(f"Event Date: {event_date_display}")
                    print("\n⚠️  Fight found in odds file but not in evaluation data (likely future fight)")
                    print(f"\nMarket Odds:")
                    print(f"  {row['fighter1']}: {row['fighter1_odds']:+.0f} (implied prob: {row['fighter1_prob']:.1%})")
                    print(f"  {row['fighter2']}: {row['fighter2_odds']:+.0f} (implied prob: {row['fighter2_prob']:.1%})")
                    print("\n⚠️  Model prediction not available (fight not in training data)")
                else:
                    print("\n" + "=" * 80)
                    print("SPECIFIC FIGHT: Jalin Turner vs Edson Barboza")
                    print("=" * 80)
                    print("⚠️  Fight not found in evaluation data or odds file")
            else:
                print("\n" + "=" * 80)
                print("SPECIFIC FIGHT: Jalin Turner vs Edson Barboza")
                print("=" * 80)
                print("⚠️  Fight not found in evaluation data or odds file")
    else:
        # Check odds file directly
        odds_fighter_mask = (
            odds_df["fighter1"].str.contains("Turner|Barboza", case=False, na=False)
            | odds_df["fighter2"].str.contains("Turner|Barboza", case=False, na=False)
        )
        odds_match = odds_df[odds_fighter_mask]
        
        if not odds_match.empty:
            both_in_odds = odds_match.apply(
                lambda r: (
                    normalize_name(str(r["fighter1"])) in target_fighters_norm
                    and normalize_name(str(r["fighter2"])) in target_fighters_norm
                ),
                axis=1
            )
            odds_fight = odds_match[both_in_odds]
            
            if not odds_fight.empty:
                row = odds_fight.iloc[0]
                event_date_str = pd.to_datetime(row.get("event_date"), errors="coerce")
                event_date_display = event_date_str.strftime("%Y-%m-%d") if pd.notna(event_date_str) else "Unknown"
                
                print("\n" + "=" * 80)
                print("SPECIFIC FIGHT: Jalin Turner vs Edson Barboza")
                print("=" * 80)
                print(f"Event Date: {event_date_display}")
                print("\n⚠️  Fight found in odds file but not in evaluation data (likely future fight)")
                print(f"\nMarket Odds:")
                print(f"  {row['fighter1']}: {row['fighter1_odds']:+.0f} (implied prob: {row['fighter1_prob']:.1%})")
                print(f"  {row['fighter2']}: {row['fighter2_odds']:+.0f} (implied prob: {row['fighter2_prob']:.1%})")
                print("\n⚠️  Model prediction not available (fight not in training data)")
            else:
                print("\n" + "=" * 80)
                print("SPECIFIC FIGHT: Jalin Turner vs Edson Barboza")
                print("=" * 80)
                print("⚠️  Fight not found in evaluation data or odds file")
        else:
            print("\n" + "=" * 80)
            print("SPECIFIC FIGHT: Jalin Turner vs Edson Barboza")
            print("=" * 80)
            print("⚠️  Fight not found in evaluation data or odds file")

    # ------------------------------------------------------------------
    # 6c) Save evaluation data for deep analysis
    # ------------------------------------------------------------------
    eval_data_path = output_dir / f"eval_data_{timestamp}.csv"
    merged.to_csv(eval_data_path, index=False)
    logger.success(f"Saved evaluation data to {eval_data_path}")
    
    # Generate HTML report
    logger.info("Generating interactive HTML report...")
    try:
        from evaluation.generate_html_report import generate_html_report
        html_path = output_dir / f"model_evaluation_{timestamp}.html"
        generate_html_report(eval_data_path, html_path, min_year=args.min_year)
        logger.success(f"✓ Open report in browser: file://{html_path.absolute()}")
    except Exception as e:
        logger.warning(f"Failed to generate HTML report: {e}")

    # ------------------------------------------------------------------
    # 7) Compute curves and save JSON report / plots
    # ------------------------------------------------------------------
    # Calibration curve (for plotting)
    frac_pos, mean_pred = calibration_curve(y_true, p_model, n_bins=10)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, p_model)

    # ------------------------------------------------------------------
    # Save JSON report and plots
    # ------------------------------------------------------------------
    
    # Calculate accuracy metrics
    row_preds = (p_model >= 0.5).astype(int)
    row_correct = (row_preds == y_true).sum()
    row_total = len(y_true)
    overall_accuracy = row_correct / row_total if row_total > 0 else float("nan")
    
    # Calculate accuracy by favorites/underdogs
    # Determine if fighter_1 is favorite based on market odds
    merged["is_favorite"] = merged["market_prob_f1"] >= 0.5
    merged["prediction_correct"] = (row_preds == y_true)
    
    favorites_mask = merged["is_favorite"]
    underdogs_mask = ~favorites_mask
    
    favorites_correct = merged.loc[favorites_mask, "prediction_correct"].sum()
    favorites_total = favorites_mask.sum()
    favorites_accuracy = favorites_correct / favorites_total if favorites_total > 0 else float("nan")
    
    underdogs_correct = merged.loc[underdogs_mask, "prediction_correct"].sum()
    underdogs_total = underdogs_mask.sum()
    underdogs_accuracy = underdogs_correct / underdogs_total if underdogs_total > 0 else float("nan")
    
    # Calculate accuracy by confidence buckets (percentile-based)
    # Base confidence: abs(model_prob_f1 - 0.5) * 2, which maps [0.5, 1.0] -> [0, 1]
    # This measures distance from 50/50 (larger = more confident)
    base_conf = np.abs(p_model - 0.5) * 2  # [0, 1]
    
    # Odds-aware confidence adjustment (for ranking only, not for training)
    adjusted_conf = np.full_like(base_conf, np.nan)
    p_market = merged["market_prob_f1"].values
    
    # Calculate finish rate for volatility gate
    # Check if either fighter has high finish rate (> 0.55)
    high_finish_rate = np.zeros(len(merged), dtype=bool)
    if "f1_finish_rate" in merged.columns and "f2_finish_rate" in merged.columns:
        f1_fr = merged["f1_finish_rate"].fillna(0).values
        f2_fr = merged["f2_finish_rate"].fillna(0).values
        # High finish rate if either fighter has > 0.55
        high_finish_rate = (f1_fr > 0.55) | (f2_fr > 0.55)
    elif "finish_rate_diff" in merged.columns:
        # Fallback: use finish_rate_diff if individual rates not available
        # This is less precise but better than nothing
        finish_rate_abs = np.abs(merged["finish_rate_diff"].fillna(0).values)
        high_finish_rate = finish_rate_abs > 0.3  # Threshold for high finish rate differential
    # If no finish rate data available, high_finish_rate remains False
    
    for i in range(len(merged)):
        if pd.notna(p_market[i]):
            # Odds available: calculate market confidence and edge
            market_conf = np.abs(p_market[i] - 0.5) * 2  # [0, 1]
            market_edge = base_conf[i] - market_conf
            
            # Penalize ONLY when model is more confident than market
            if market_edge > 0:
                # Penalty: reduce confidence when model is overconfident vs market
                # Cap penalty at 0.6 (60% reduction max)
                penalty = min(market_edge * 1.5, 0.6)
                adjusted_conf[i] = base_conf[i] * (1.0 - penalty)
            else:
                # Model is less confident than market: no penalty
                adjusted_conf[i] = base_conf[i]
        else:
            # No odds available: use base confidence
            adjusted_conf[i] = base_conf[i]
        
        # Volatility gate: penalize high-confidence predictions in high-finish-rate matchups
        # This accounts for increased volatility/uncertainty in finisher vs finisher fights
        if adjusted_conf[i] > 0.7 and high_finish_rate[i]:
            adjusted_conf[i] *= 0.85  # Reduce by 15% (was 0.75 = 25% reduction)
    
    # For top 10%: use base confidence (no odds adjustment)
    top_10_threshold = np.percentile(base_conf, 90)  # Top 10% = 90th percentile
    top_10_mask = base_conf >= top_10_threshold
    
    # For top 25%: use odds-adjusted confidence (if available)
    # This identifies predictions where model is both confident AND agrees with market
    # Filter out NaN values for percentile calculation
    valid_adjusted_conf = adjusted_conf[~np.isnan(adjusted_conf)]
    if len(valid_adjusted_conf) > 0:
        top_25_threshold = np.percentile(valid_adjusted_conf, 75)  # Top 25% = 75th percentile
        top_25_mask = adjusted_conf >= top_25_threshold
    else:
        # Fallback to base confidence if no valid adjusted confidence
        top_25_threshold = np.percentile(base_conf, 75)
        top_25_mask = base_conf >= top_25_threshold
    
    by_confidence = {}
    
    # Top 10% bucket
    if top_10_mask.sum() > 0:
        top_10_correct = (row_preds[top_10_mask] == y_true[top_10_mask]).sum()
        top_10_total = top_10_mask.sum()
        top_10_accuracy = top_10_correct / top_10_total if top_10_total > 0 else None
        by_confidence["top_10_pct"] = {
            "min_p": round(float(top_10_threshold), 4),
            "accuracy": round(top_10_accuracy, 4) if top_10_accuracy is not None else None,
            "n": int(top_10_total)
        }
    else:
        by_confidence["top_10_pct"] = {
            "min_p": None,
            "accuracy": None,
            "n": 0
        }
    
    # Top 25% bucket
    if top_25_mask.sum() > 0:
        top_25_correct = (row_preds[top_25_mask] == y_true[top_25_mask]).sum()
        top_25_total = top_25_mask.sum()
        top_25_accuracy = top_25_correct / top_25_total if top_25_total > 0 else None
        by_confidence["top_25_pct"] = {
            "min_p": round(float(top_25_threshold), 4),
            "accuracy": round(top_25_accuracy, 4) if top_25_accuracy is not None else None,
            "n": int(top_25_total)
        }
        
        # Export detailed top 25% predictions for investigation
        top_25_df = merged[top_25_mask].copy()
        top_25_df["base_confidence"] = base_conf[top_25_mask]
        top_25_df["adjusted_confidence"] = adjusted_conf[top_25_mask]
        top_25_df["prediction_correct"] = (row_preds[top_25_mask] == y_true[top_25_mask])
        top_25_df["model_prob_f1"] = p_model[top_25_mask]
        top_25_df["model_prob_f2"] = 1.0 - p_model[top_25_mask]
        
        # Get top 3 features for f1 and f2 from feature importance
        try:
            feature_importance = xgb_model.get_feature_importance(importance_type='weight', top_n=None)
            # Get top 3 f1_ features
            f1_features = feature_importance[feature_importance['feature'].str.startswith('f1_')].head(3)
            f2_features = feature_importance[feature_importance['feature'].str.startswith('f2_')].head(3)
            
            # Add top 3 feature values for f1
            for rank, (_, row) in enumerate(f1_features.iterrows(), 1):
                feat_name = row['feature']
                if feat_name in top_25_df.columns:
                    feat_short_name = feat_name.replace('f1_', '')
                    top_25_df[f"f1_top{rank}_{feat_short_name}"] = top_25_df[feat_name]
            
            # Add top 3 feature values for f2
            for rank, (_, row) in enumerate(f2_features.iterrows(), 1):
                feat_name = row['feature']
                if feat_name in top_25_df.columns:
                    feat_short_name = feat_name.replace('f2_', '')
                    top_25_df[f"f2_top{rank}_{feat_short_name}"] = top_25_df[feat_name]
        except Exception as e:
            logger.warning(f"Could not extract top features: {e}")
        
        # Select relevant columns for investigation (excluding removed columns)
        investigation_cols = [
            "event_date", "f1_name", "f2_name", "weight_class",
            "model_prob_f1", "model_prob_f2", "base_confidence", "adjusted_confidence",
            "prediction_correct", "market_prob_f2", "fighter1_odds", "fighter2_odds",
            "is_favorite"
        ]
        
        # Add top feature columns if they exist
        top_feature_cols = [col for col in top_25_df.columns if col.startswith('f1_top') or col.startswith('f2_top')]
        investigation_cols.extend(top_feature_cols)
        
        # Only include columns that exist
        available_cols = [col for col in investigation_cols if col in top_25_df.columns]
        top_25_export = top_25_df[available_cols].copy()
        
        # Deduplicate: keep only one row per fight (f1 vs f2, not f2 vs f1)
        # Create a canonical fight key from sorted fighter names
        top_25_export["canonical_fight_key"] = top_25_export.apply(
            lambda row: "|".join(sorted([str(row.get("f1_name", "")), str(row.get("f2_name", ""))])),
            axis=1
        )
        
        # Keep only rows where f1_name < f2_name (alphabetically) to avoid duplicates
        # This ensures we keep (A vs B) but not (B vs A)
        def _should_keep(row):
            f1 = str(row.get("f1_name", ""))
            f2 = str(row.get("f2_name", ""))
            try:
                return f1 < f2
            except:
                # If comparison fails, keep the row (better than dropping)
                return True
        
        keep_mask = top_25_export.apply(_should_keep, axis=1)
        top_25_export = top_25_export[keep_mask].copy()
        
        # Also drop duplicates by canonical key (in case there are still duplicates)
        top_25_export = top_25_export.drop_duplicates(subset=["canonical_fight_key"], keep="first")
        
        # Drop the canonical key column (it was just for deduplication)
        if "canonical_fight_key" in top_25_export.columns:
            top_25_export = top_25_export.drop(columns=["canonical_fight_key"])
        
        # Sort by adjusted confidence (highest first) then by correctness (wrong predictions first)
        top_25_export = top_25_export.sort_values(
            ["prediction_correct", "adjusted_confidence"],
            ascending=[True, False]  # Wrong predictions first, then by adjusted confidence descending
        )
        
        # Save to CSV
        top_25_export_path = output_dir / f"top_25_pct_investigation_{timestamp}.csv"
        top_25_export.to_csv(top_25_export_path, index=False)
        logger.success(f"Saved top 25% confidence bucket details to {top_25_export_path}")
        logger.info(f"  Total predictions in top 25%: {top_25_total}")
        logger.info(f"  After deduplication: {len(top_25_export)}")
        logger.info(f"  Correct: {top_25_correct} ({top_25_accuracy:.1%})")
        logger.info(f"  Incorrect: {top_25_total - top_25_correct} ({1.0 - top_25_accuracy:.1%})")
        logger.info(f"  Confidence threshold: {top_25_threshold:.4f}")
    else:
        by_confidence["top_25_pct"] = {
            "min_p": None,
            "accuracy": None,
            "n": 0
        }
    
    # ------------------------------------------------------------------
    # Calculate underdog odds bands analysis
    # ------------------------------------------------------------------
    # Initialize with empty bands
    underdog_odds_bands = {
        "+100_to_+150": {"n_bets": 0, "accuracy": None, "roi": None},
        "+150_to_+250": {"n_bets": 0, "accuracy": None, "roi": None},
        "+250_plus": {"n_bets": 0, "accuracy": None, "roi": None},
    }
    
    if not merged.empty:
        # Identify underdog bets: model prob > 50% for underdog AND edge > 0
        underdog_bets_data = []
        seen_fights_bands = set()
        
        for idx, row in merged.iterrows():
            fight_key = row.get("fight_key", "")
            if not fight_key or fight_key in seen_fights_bands:
                continue
            
            # Get market probabilities
            f1_norm = row.get("f1_name_norm", "")
            fighter1_norm_odds = row.get("fighter1_norm_odds", "")
            fighter2_norm_odds = row.get("fighter2_norm_odds", "")
            
            # Get market probabilities
            if f1_norm == fighter1_norm_odds:
                f1_market_prob = row.get("fighter1_prob", np.nan)
                f2_market_prob = row.get("fighter2_prob", np.nan)
                f1_odds = row.get("fighter1_odds", np.nan)
                f2_odds = row.get("fighter2_odds", np.nan)
            elif f1_norm == fighter2_norm_odds:
                f1_market_prob = row.get("fighter2_prob", np.nan)
                f2_market_prob = row.get("fighter1_prob", np.nan)
                f1_odds = row.get("fighter2_odds", np.nan)
                f2_odds = row.get("fighter1_odds", np.nan)
            else:
                continue
            
            # Determine underdog (lower market prob)
            if pd.isna(f1_market_prob) or pd.isna(f2_market_prob):
                continue
            
            if f1_market_prob < f2_market_prob:
                # f1 is underdog
                underdog_market_prob = f1_market_prob
                underdog_model_prob = row.get("model_prob_f1", np.nan)
                underdog_edge = row.get("edge", np.nan)  # edge for f1
                underdog_odds = f1_odds
                underdog_won = (row.get("target", 0) == 1)
            elif f2_market_prob < f1_market_prob:
                # f2 is underdog
                underdog_market_prob = f2_market_prob
                underdog_model_prob = 1.0 - row.get("model_prob_f1", np.nan)  # f2 prob
                # Edge for f2 = model_prob_f2 - market_prob_f2
                underdog_edge = underdog_model_prob - underdog_market_prob
                underdog_odds = f2_odds
                underdog_won = (row.get("target", 0) == 0)
            else:
                continue  # Equal market probs, skip
            
            # Check if we bet (model prob > 50% AND edge > 0)
            if (pd.notna(underdog_model_prob) and pd.notna(underdog_edge) and 
                underdog_model_prob > 0.5 and underdog_edge > 0 and pd.notna(underdog_odds)):
                
                # Calculate profit/loss for flat 1 unit bet
                profit_loss = 0.0
                if underdog_won:
                    if underdog_odds > 0:
                        profit_loss = (underdog_odds / 100.0) * 1.0
                    else:
                        profit_loss = (100.0 / abs(underdog_odds)) * 1.0
                else:
                    profit_loss = -1.0
                
                # Calculate confidence (distance from 50%)
                confidence = abs(underdog_model_prob - 0.5) * 2 if pd.notna(underdog_model_prob) else 0.0
                
                underdog_bets_data.append({
                    "odds": underdog_odds,
                    "won": underdog_won,
                    "profit_loss": profit_loss,
                    "confidence": confidence,
                })
                
                seen_fights_bands.add(fight_key)
        
        if underdog_bets_data:
            # Group by odds bands
            bands = {
                "+100_to_+150": {"min": 100, "max": 150, "bets": []},
                "+150_to_+250": {"min": 150, "max": 250, "bets": []},
                "+250_plus": {"min": 250, "max": float("inf"), "bets": []},
            }
            
            for bet in underdog_bets_data:
                odds = bet["odds"]
                if pd.isna(odds):
                    continue
                
                # Only positive odds for underdogs
                if odds < 100:
                    continue
                
                # Assign to band
                if 100 <= odds < 150:
                    bands["+100_to_+150"]["bets"].append(bet)
                elif 150 <= odds < 250:
                    bands["+150_to_+250"]["bets"].append(bet)
                elif odds >= 250:
                    bands["+250_plus"]["bets"].append(bet)
            
            # Calculate metrics for each band
            for band_name, band_data in bands.items():
                bets = band_data["bets"]
                if len(bets) > 0:
                    n_bets = len(bets)
                    correct = sum(1 for b in bets if b["won"])
                    accuracy = correct / n_bets if n_bets > 0 else 0.0
                    total_profit = sum(b["profit_loss"] for b in bets)
                    total_staked = n_bets * 1.0
                    roi = (total_profit / total_staked) if total_staked > 0 else 0.0
                    
                    underdog_odds_bands[band_name] = {
                        "n_bets": int(n_bets),
                        "accuracy": round(accuracy, 3),
                        "roi": round(roi, 2),
                    }
                else:
                    underdog_odds_bands[band_name] = {
                        "n_bets": 0,
                        "accuracy": None,
                        "roi": None,
                    }
    
    # ------------------------------------------------------------------
    # Calculate underdog confidence breakdown
    # ------------------------------------------------------------------
    # Initialize with empty breakdown
    underdog_confidence_breakdown = {
        "top_10_pct": {"confidence_min": None, "n_bets": 0, "accuracy": None, "profit_units": None, "roi": None},
        "top_25_pct": {"confidence_min": None, "n_bets": 0, "accuracy": None, "profit_units": None, "roi": None},
        "bottom_75_pct": {"n_bets": 0, "accuracy": None, "profit_units": None, "roi": None},
    }
    
    if underdog_bets_data:
        # Sort by confidence (highest first)
        underdog_bets_sorted = sorted(underdog_bets_data, key=lambda x: x["confidence"], reverse=True)
        
        total_bets = len(underdog_bets_sorted)
        
        if total_bets > 0:
            # Calculate thresholds for top 10% and top 25%
            top_10_count = max(1, int(np.ceil(total_bets * 0.10)))  # Top 10%
            top_25_count = max(1, int(np.ceil(total_bets * 0.25)))  # Top 25%
            
            # Get confidence thresholds
            if top_10_count <= total_bets:
                top_10_threshold = underdog_bets_sorted[top_10_count - 1]["confidence"]
            else:
                top_10_threshold = 0.0
            
            if top_25_count <= total_bets:
                top_25_threshold = underdog_bets_sorted[top_25_count - 1]["confidence"]
            else:
                top_25_threshold = 0.0
            
            # Split into groups
            top_10_bets = underdog_bets_sorted[:top_10_count]
            top_25_bets = underdog_bets_sorted[:top_25_count]
            bottom_75_bets = underdog_bets_sorted[top_25_count:]
            
            # Calculate metrics for top 10%
            if len(top_10_bets) > 0:
                top_10_n = len(top_10_bets)
                top_10_correct = sum(1 for b in top_10_bets if b["won"])
                top_10_accuracy = top_10_correct / top_10_n if top_10_n > 0 else 0.0
                top_10_profit = sum(b["profit_loss"] for b in top_10_bets)
                top_10_staked = top_10_n * 1.0
                top_10_roi = (top_10_profit / top_10_staked) if top_10_staked > 0 else 0.0
                
                underdog_confidence_breakdown["top_10_pct"] = {
                    "confidence_min": round(float(top_10_threshold), 2),
                    "n_bets": int(top_10_n),
                    "accuracy": round(top_10_accuracy, 3),
                    "profit_units": round(top_10_profit, 1),
                    "roi": round(top_10_roi, 2),
                }
            
            # Calculate metrics for top 25%
            if len(top_25_bets) > 0:
                top_25_n = len(top_25_bets)
                top_25_correct = sum(1 for b in top_25_bets if b["won"])
                top_25_accuracy = top_25_correct / top_25_n if top_25_n > 0 else 0.0
                top_25_profit = sum(b["profit_loss"] for b in top_25_bets)
                top_25_staked = top_25_n * 1.0
                top_25_roi = (top_25_profit / top_25_staked) if top_25_staked > 0 else 0.0
                
                underdog_confidence_breakdown["top_25_pct"] = {
                    "confidence_min": round(float(top_25_threshold), 2),
                    "n_bets": int(top_25_n),
                    "accuracy": round(top_25_accuracy, 3),
                    "profit_units": round(top_25_profit, 1),
                    "roi": round(top_25_roi, 2),
                }
            
            # Calculate metrics for bottom 75%
            if len(bottom_75_bets) > 0:
                bottom_75_n = len(bottom_75_bets)
                bottom_75_correct = sum(1 for b in bottom_75_bets if b["won"])
                bottom_75_accuracy = bottom_75_correct / bottom_75_n if bottom_75_n > 0 else 0.0
                bottom_75_profit = sum(b["profit_loss"] for b in bottom_75_bets)
                bottom_75_staked = bottom_75_n * 1.0
                bottom_75_roi = (bottom_75_profit / bottom_75_staked) if bottom_75_staked > 0 else 0.0
                
                underdog_confidence_breakdown["bottom_75_pct"] = {
                    "n_bets": int(bottom_75_n),
                    "accuracy": round(bottom_75_accuracy, 3),
                    "profit_units": round(bottom_75_profit, 1),
                    "roi": round(bottom_75_roi, 2),
                }
    
    # Build comprehensive report
    report = {
        "model_name": model_name,
        "holdout_from_year": args.min_year,
        "timestamp": timestamp_short,
        "overall": {
            "accuracy": round(overall_accuracy, 4) if not np.isnan(overall_accuracy) else None,
            "n_correct": int(row_correct),
            "n_total": int(row_total),
            "brier": round(float(brier), 4) if not np.isnan(brier) else None,
            "auc": round(float(auc), 4) if not np.isnan(auc) else None,
            "log_loss": round(float(ll), 4) if not np.isnan(ll) else None,
        },
        "by_bucket": {
            "favorites": {
                "accuracy": round(favorites_accuracy, 4) if not np.isnan(favorites_accuracy) else None,
                "n": int(favorites_total)
            },
            "underdogs": {
                "accuracy": round(underdogs_accuracy, 4) if not np.isnan(underdogs_accuracy) else None,
                "n": int(underdogs_total)
            }
        },
        "by_confidence": by_confidence,
        # Additional metrics for backward compatibility
        "n_eval_rows": int(len(merged)),
        "n_odds_rows": int(len(odds_df)),
        "n_bets_edge_gt_0": int(n_bets),
        "roi_flat_edge_gt_0": round(float(roi_flat), 4) if not np.isnan(roi_flat) else None,
        "underdog_odds_bands": underdog_odds_bands,
        "underdog_confidence_breakdown": underdog_confidence_breakdown,
    }
    
    # Add underdog summary if --underdog flag was used
    if args.underdog and underdog_summary["total_bets"] > 0:
        report["underdog_betting_summary"] = underdog_summary

    report_path = output_dir / f"model_eval_{timestamp}.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    logger.success(f"Saved evaluation report to {report_path}")

    # Calibration plot
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, "s-", label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve (holdout)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    calib_path = output_dir / f"calibration_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(calib_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"Saved calibration plot to {calib_path}")

    # ROC plot
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.plot(fpr, tpr, label=f"Model (AUC={auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (holdout)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    roc_path = output_dir / f"roc_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"Saved ROC plot to {roc_path}")

    # ------------------------------------------------------------------
    # 8) Optionally run baseline comparison
    # ------------------------------------------------------------------
    if args.compare_to_baseline:
        logger.info("Running baseline comparison...")
        baseline_path = Path(args.baseline_path)
        
        if not baseline_path.exists():
            logger.warning(f"Baseline file not found: {baseline_path}. Skipping comparison.")
        else:
            # Generate comparison report path
            comparison_output = output_dir / f"comparison_report_{timestamp}.json"
            
            # Run compare_to_baseline as a subprocess
            cmd = [
                sys.executable,
                "-m",
                "evaluation.compare_to_baseline",
                "--baseline",
                str(baseline_path),
                "--current",
                str(report_path),
                "--output",
                str(comparison_output),
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=False,  # Let output go to stdout/stderr
                    check=False,  # Don't raise on non-zero exit
                )
                
                if result.returncode == 0:
                    logger.success(f"Baseline comparison completed successfully")
                elif result.returncode == 1:
                    logger.warning("Baseline comparison returned REJECT verdict")
                elif result.returncode == 2:
                    logger.warning("Baseline comparison returned REVIEW verdict")
                else:
                    logger.warning(f"Baseline comparison exited with code {result.returncode}")
            except Exception as e:
                logger.error(f"Failed to run baseline comparison: {e}")


if __name__ == "__main__":
    main()


