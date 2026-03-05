#!/usr/bin/env python3
"""
Walk-Forward Validation for UFC Fight Prediction Model
-------------------------------------------------------

Trains the model on all fights BEFORE year N and evaluates on fights IN year N.
Repeats for each test year (default: 2020 through 2024) and aggregates results.

This is the gold standard for time-series model evaluation because:
  1. The test set is always strictly in the future relative to training data.
  2. There is no temporal leakage (the model cannot have "seen" test fights).
  3. Results are reported per year so we can see if performance degrades over time.

Usage:
    python -m models.walk_forward_eval \
        --data-path data/processed/training_data.csv \
        --test-years 2020 2021 2022 2023 2024 \
        --n-estimators 200 \
        --max-depth 4 \
        --learning-rate 0.05

Output:
    - Per-year metrics table (accuracy, AUC, Brier, calibration gap)
    - Confidence-vs-accuracy correlation check (should be positive)
    - Combined out-of-sample metrics across all test years
    - Saves results to models/saved/walk_forward_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from database.db_manager import DatabaseManager
from database.schema import Event
from features.feature_pipeline import FeaturePipeline
from models.xgboost_model import XGBoostModel


# ---------------------------------------------------------------------------
# Helper: get event year map from the database
# ---------------------------------------------------------------------------

def _get_event_year_map(raw_df: pd.DataFrame) -> pd.Series:
    """Return a Series mapping row index → event year (NaN if unknown)."""
    if "event_id" not in raw_df.columns:
        return pd.Series(np.nan, index=raw_df.index)

    db = DatabaseManager()
    session = db.get_session()
    try:
        event_ids = raw_df["event_id"].dropna().astype(int).unique().tolist()
        events = session.query(Event).filter(Event.id.in_(event_ids)).all()
        id_to_date = {e.id: e.date for e in events}
    finally:
        session.close()

    dates = raw_df["event_id"].map(id_to_date)
    dates_parsed = pd.to_datetime(dates, errors="coerce")
    return dates_parsed.dt.year


# ---------------------------------------------------------------------------
# Helper: group train/test split by fight_id to prevent mirror-row leakage
# ---------------------------------------------------------------------------

def _group_split_by_fight_id(
    raw_df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Apply pre-computed boolean masks to X/y, respecting the raw_df index."""
    X_train = X.loc[raw_df.index[train_mask]]
    X_test = X.loc[raw_df.index[test_mask]]
    y_train = y.loc[raw_df.index[train_mask]]
    y_test = y.loc[raw_df.index[test_mask]]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Helper: recency sample weights
# ---------------------------------------------------------------------------

def _compute_recency_weights(raw_df: pd.DataFrame, event_years: pd.Series, reference_year: int, lambda_: float = 0.3) -> pd.Series:
    """Exponential recency weight based on years before reference_year."""
    years_ago = reference_year - event_years
    weights = np.exp(-lambda_ * years_ago.clip(lower=0))
    weights = weights.fillna(1.0)
    return weights


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    test_year: int,
    n_bins: int = 5,
) -> Dict:
    """Compute metrics for one walk-forward fold."""
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true, y_pred_proba)
    ll = log_loss(y_true, y_pred_proba)

    # Calibration: mean |predicted_prob_bin - actual_win_rate_bin|
    if len(np.unique(y_true)) > 1:
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
        calibration_gap = float(np.mean(np.abs(prob_true - prob_pred)))
    else:
        calibration_gap = float("nan")

    # Confidence vs accuracy correlation
    # Bin predictions into 5 confidence buckets and check if higher conf = higher accuracy
    bins = np.linspace(0.5, 1.0, 6)  # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bucket_accuracies = []
    bucket_counts = []
    high_conf_mask = y_pred_proba > 0.5
    for lo, hi in zip(bins[:-1], bins[1:]):
        conf = np.maximum(y_pred_proba, 1 - y_pred_proba)  # confidence = max(p, 1-p)
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() >= 5:
            bucket_accuracies.append(accuracy_score(y_true[mask], y_pred[mask]))
            bucket_counts.append(int(mask.sum()))
        else:
            bucket_accuracies.append(float("nan"))
            bucket_counts.append(0)

    # Spearman correlation: confidence bucket → accuracy (should be positive)
    valid = [(b, a) for b, a in zip(range(len(bucket_accuracies)), bucket_accuracies) if not np.isnan(a)]
    if len(valid) >= 3:
        xs, ys = zip(*valid)
        conf_accuracy_corr = float(np.corrcoef(xs, ys)[0, 1])
    else:
        conf_accuracy_corr = float("nan")

    metrics = {
        "test_year": test_year,
        "n_test_samples": int(len(y_true)),
        "accuracy": float(accuracy),
        "auc": float(auc),
        "brier_score": float(brier),
        "log_loss": float(ll),
        "calibration_gap": float(calibration_gap),
        "confidence_accuracy_correlation": conf_accuracy_corr,
        "confidence_buckets": {
            "bins": [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(bins[:-1], bins[1:])],
            "accuracies": [round(a, 4) if not np.isnan(a) else None for a in bucket_accuracies],
            "counts": bucket_counts,
        },
    }
    return metrics


# ---------------------------------------------------------------------------
# Main walk-forward loop
# ---------------------------------------------------------------------------

def run_walk_forward(
    data_path: str = "data/processed/training_data.csv",
    test_years: List[int] = None,
    min_train_years: int = 3,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    output_path: str = "models/saved/walk_forward_results.json",
) -> Dict:
    if test_years is None:
        test_years = list(range(2020, 2025))

    pipeline = FeaturePipeline(initialize_db=False)
    raw_df = pipeline.load_dataset(data_path)

    logger.info(f"Loaded {len(raw_df)} rows from {data_path}")

    # Get event years for every row
    event_years = _get_event_year_map(raw_df)
    raw_df = raw_df.copy()
    raw_df["_event_year"] = event_years.values

    all_fold_metrics = []

    for test_year in test_years:
        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-forward fold: test_year={test_year}")

        train_mask = raw_df["_event_year"] < test_year
        test_mask = raw_df["_event_year"] == test_year

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train < 100:
            logger.warning(f"  Skipping year {test_year}: only {n_train} training rows")
            continue
        if n_test < 20:
            logger.warning(f"  Skipping year {test_year}: only {n_test} test rows")
            continue

        logger.info(f"  Train: {n_train} rows | Test: {n_test} rows")

        df_train_raw = raw_df[train_mask].copy()
        df_test_raw = raw_df[test_mask].copy()

        # Prepare features: fit scaler only on train, then transform test
        X_train_raw, y_train = pipeline.prepare_features(df_train_raw, fit_scaler=True)
        X_test_raw, y_test = pipeline.prepare_features(df_test_raw, fit_scaler=False)

        # Recency weights
        weights = _compute_recency_weights(df_train_raw, df_train_raw["_event_year"], test_year)
        w_train = weights.values

        # Train a fresh XGBoost model for this fold
        fold_model = XGBoostModel.__new__(XGBoostModel)
        fold_model.model = None
        fold_model.calibrated_model = None
        fold_model.feature_names = None
        fold_model.training_metrics = {}
        fold_model.model_dir = Path("models/saved")

        fold_model.create_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        )

        fold_model.train(X_train_raw, y_train, sample_weight=w_train)

        # Predict on test set
        y_pred_proba = fold_model.model.predict_proba(X_test_raw)[:, 1]

        # Evaluate
        fold_metrics = evaluate_fold(y_test, y_pred_proba, test_year)
        fold_metrics["n_train_samples"] = int(n_train)

        logger.info(f"  Accuracy:  {fold_metrics['accuracy']:.3f}")
        logger.info(f"  AUC:       {fold_metrics['auc']:.3f}")
        logger.info(f"  Brier:     {fold_metrics['brier_score']:.4f}")
        logger.info(f"  Calib gap: {fold_metrics['calibration_gap']:.4f}")
        conf_corr = fold_metrics['confidence_accuracy_correlation']
        corr_str = f"{conf_corr:.3f}" if not np.isnan(conf_corr) else "N/A"
        logger.info(f"  Conf-Acc corr: {corr_str}  (positive = model is well-calibrated)")

        all_fold_metrics.append(fold_metrics)

    # ---------------------------------------------------------------------------
    # Aggregate across all folds
    # ---------------------------------------------------------------------------
    if not all_fold_metrics:
        logger.error("No folds were evaluated. Check data and test_years.")
        return {}

    acc_vals = [m["accuracy"] for m in all_fold_metrics]
    auc_vals = [m["auc"] for m in all_fold_metrics if not np.isnan(m["auc"])]
    brier_vals = [m["brier_score"] for m in all_fold_metrics]
    corr_vals = [m["confidence_accuracy_correlation"] for m in all_fold_metrics if not np.isnan(m["confidence_accuracy_correlation"])]

    summary = {
        "test_years": test_years,
        "overall": {
            "mean_accuracy": float(np.mean(acc_vals)),
            "std_accuracy": float(np.std(acc_vals)),
            "mean_auc": float(np.mean(auc_vals)) if auc_vals else float("nan"),
            "mean_brier": float(np.mean(brier_vals)),
            "mean_conf_accuracy_corr": float(np.mean(corr_vals)) if corr_vals else float("nan"),
            "beats_random_50pct": bool(np.mean(acc_vals) > 0.50),
            "beats_random_52pct": bool(np.mean(acc_vals) > 0.52),
        },
        "per_year": all_fold_metrics,
    }

    logger.info(f"\n{'='*60}")
    logger.info("WALK-FORWARD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Mean accuracy:   {summary['overall']['mean_accuracy']:.3f} ± {summary['overall']['std_accuracy']:.3f}")
    logger.info(f"Mean AUC:        {summary['overall']['mean_auc']:.3f}")
    logger.info(f"Mean Brier:      {summary['overall']['mean_brier']:.4f}")
    logger.info(f"Mean Conf-Corr:  {summary['overall']['mean_conf_accuracy_corr']:.3f}")
    logger.info(f"Beats random (>50%): {summary['overall']['beats_random_50pct']}")
    logger.info(f"Beats 52% target:    {summary['overall']['beats_random_52pct']}")

    if summary["overall"]["mean_conf_accuracy_corr"] > 0:
        logger.success("CONFIDENCE-ACCURACY CORRELATION IS POSITIVE — model confidence is meaningful!")
    else:
        logger.warning("CONFIDENCE-ACCURACY CORRELATION IS NEGATIVE — model is still miscalibrated.")

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"Walk-forward results saved to {output_file}")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Walk-forward evaluation of UFC prediction model")
    parser.add_argument("--data-path", default="data/processed/training_data.csv")
    parser.add_argument("--test-years", type=int, nargs="+", default=list(range(2020, 2025)))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--output-path", default="models/saved/walk_forward_results.json")
    args = parser.parse_args()

    run_walk_forward(
        data_path=args.data_path,
        test_years=args.test_years,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
