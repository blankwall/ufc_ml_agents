"""
Underdog Specialist Model — Training Script
File: models/train_underdog_model.py

Trains a NEW model (underdog_v1) on fights where f1 is the market underdog.
Does NOT modify or retrain the general model (mar_4_v2).
Does NOT modify training_data.csv.

Run: python models/train_underdog_model.py
"""

import sys
import re
import json
import argparse
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Constants ─────────────────────────────────────────────────────────────────
TRAINING_DATA_PATH  = "data/processed/training_data.csv"
ODDS_DATA_PATH      = "data/odds/historical_odds.csv"   # ← historical BFO odds
GENERAL_MODEL_NAME  = "mar_4_v2"
UNDERDOG_MODEL_NAME = "underdog_v1"
MODEL_SAVE_DIR      = Path("models/saved")
UNDERDOG_THRESHOLD  = 0.40   # market_prob_f1 below this → underdog context
UPSET_UPSAMPLE      = 3      # repeat upset-win rows this many times

# Columns that are metadata, never passed to XGBoost as features
NON_FEATURE_COLS = {
    "target", "fight_id", "event_id", "fighter_1_id", "fighter_2_id",
    "weight_class", "is_title_fight", "method", "event_date", "event_year",
    "event_day", "f1_name", "f2_name", "f1_name_norm", "f2_name_norm",
    "name_a", "name_b", "fight_key", "fighter1", "fighter2",
    "fighter1_odds", "fighter2_odds", "fighter1_prob", "fighter2_prob",
    "event_date_odds", "fighter1_norm_odds", "fighter2_norm_odds",
    "market_prob_f1",   # added explicitly below; excluded from auto-detection
    "model_prob_f1", "model_name", "model_prob_f1_symmetric",
    "edge", "price_f1", "event_name", "event_url",
}
# ─────────────────────────────────────────────────────────────────────────────


def _norm_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace. Used for fight_key join."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_s = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_s).strip().lower()


def _fight_key(f1: str, f2: str) -> str:
    """Stable fight key regardless of which fighter is listed first."""
    a, b = sorted([_norm_name(f1), _norm_name(f2)])
    return f"{a}_vs_{b}"


def load_and_merge_odds(training_path: str, odds_path: str) -> pd.DataFrame:
    """
    Load training_data.csv and left-join market odds from historical_odds.csv.

    IMPORTANT: training_data.csv has NO fighter name columns.
    It only has fighter_1_id and fighter_2_id (integers) which correspond to
    the 'id' column (auto-increment integer) in the fighters table of
    data/ufc_database.db.

    Join strategy:
    1. Load fighters table from the SQLite DB to get id -> name mapping
    2. Add f1_name / f2_name to training rows via that mapping
    3. Compute fight_key (normalised, alphabetically sorted) in both files
    4. Left-merge on fight_key
    5. Assign market_prob_f1 correctly (flip if training f1 ≠ odds fighter1)

    Expected result: ~3,800-4,500 training rows match odds (~23%)
    Rows with no match get market_prob_f1 = NaN and are dropped later.
    """
    import sqlite3

    train = pd.read_csv(training_path)
    odds  = pd.read_csv(odds_path)

    logger.info(f"Training rows loaded:   {len(train)}")
    logger.info(f"Odds rows loaded:       {len(odds)}")

    # ── Step 1: Look up fighter names from the SQLite DB ─────────────────────
    # fighters.id (integer) is referenced by training_data fighter_1_id/fighter_2_id
    db_path = Path(training_path).parent.parent / "data" / "ufc_database.db"
    if not db_path.exists():
        db_path = Path("data/ufc_database.db")  # fallback: relative to project root
    if not db_path.exists():
        raise FileNotFoundError(
            f"Cannot find ufc_database.db. Looked at: {db_path}. "
            "Run from the project root directory."
        )

    conn = sqlite3.connect(db_path)
    fighters_df = pd.read_sql("SELECT id, name FROM fighters", conn)
    conn.close()

    # Join names for f1 and f2
    train = train.merge(
        fighters_df.rename(columns={"id": "fighter_1_id", "name": "f1_name"}),
        on="fighter_1_id", how="left"
    )
    train = train.merge(
        fighters_df.rename(columns={"id": "fighter_2_id", "name": "f2_name"}),
        on="fighter_2_id", how="left"
    )

    missing_names = train["f1_name"].isna().sum()
    if missing_names > 0:
        logger.warning(f"{missing_names} training rows have no fighter name match from DB")
    else:
        logger.info("All training rows matched to fighter names from DB")

    # ── Step 2: Build fight_key in both files ─────────────────────────────────
    train["fight_key"] = train.apply(
        lambda r: _fight_key(str(r["f1_name"]), str(r["f2_name"])), axis=1
    )
    odds["fight_key"] = odds.apply(
        lambda r: _fight_key(r["fighter1"], r["fighter2"]), axis=1
    )

    # ── Step 3: Merge ─────────────────────────────────────────────────────────
    odds_slim = (
        odds[["fight_key", "fighter1", "fighter1_prob"]]
        .drop_duplicates("fight_key")
        .rename(columns={"fighter1": "odds_fighter1", "fighter1_prob": "odds_fighter1_prob"})
    )
    merged = train.merge(odds_slim, on="fight_key", how="left")

    # ── Step 4: Assign market_prob_f1 (flip if f1 ordering differs) ──────────
    # The odds file listed fighter1_prob for whoever BFO called "fighter1".
    # If training row's f1 is a different person than odds fighter1, flip it.
    def assign_market_prob(row):
        if pd.isna(row.get("odds_fighter1_prob")):
            return np.nan
        if _norm_name(str(row["f1_name"])) == _norm_name(str(row.get("odds_fighter1", ""))):
            return float(row["odds_fighter1_prob"])
        return 1.0 - float(row["odds_fighter1_prob"])

    merged["market_prob_f1"] = merged.apply(assign_market_prob, axis=1)
    merged = merged.drop(columns=["odds_fighter1", "odds_fighter1_prob"], errors="ignore")

    n_matched = merged["market_prob_f1"].notna().sum()
    logger.info(
        f"Rows with matched odds: {n_matched} / {len(merged)} "
        f"({n_matched / len(merged) * 100:.1f}%)"
    )
    if n_matched < 1000:
        logger.warning(
            f"Only {n_matched} rows matched odds. Expected ~3,800+. "
            "Check that data/ufc_database.db is accessible and "
            "fighter IDs in training_data.csv match fighters.id in the DB."
        )

    return merged


def filter_underdog_context(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Keep only rows where the training f1 is the market underdog.

    Training data has TWO rows per fight (double-sampling):
      - Row A: f1 = actual winner (target=1), market_prob_f1 = winner's prob
      - Row B: f1 = actual loser  (target=0), market_prob_f1 = loser's prob

    After filtering market_prob_f1 < threshold:
      - Underdog WINS  → target=1  (upset)
      - Underdog LOSES → target=0  (expected result)

    The model learns: "given f1 is a market underdog, will they pull the upset?"
    """
    df_with_odds = df.dropna(subset=["market_prob_f1"]).copy()
    df_under = df_with_odds[df_with_odds["market_prob_f1"] < threshold].copy()

    n_upsets = (df_under["target"] == 1).sum()
    n_losses = (df_under["target"] == 0).sum()
    upset_rate = n_upsets / len(df_under) * 100 if len(df_under) > 0 else 0

    logger.info(
        f"Underdog context rows: {len(df_under)} | "
        f"upsets={n_upsets} | losses={n_losses} | upset_rate={upset_rate:.1f}%"
    )

    if len(df_under) < 300:
        logger.warning(
            f"Only {len(df_under)} underdog rows — very small training set. "
            "Check fight_key join above. The join must produce >= 300 rows."
        )

    return df_under


def upsample_upsets(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """
    Repeat upset-win rows `factor` times to fix class imbalance.

    Without this the model sees ~3x more losses than wins and learns to always
    predict 'underdog loses' — high accuracy but useless for betting.
    Target: roughly 50/50 class balance after upsampling.
    """
    upsets = df[df["target"] == 1].copy()
    losses = df[df["target"] == 0].copy()

    upsets_repeated = pd.concat([upsets] * factor, ignore_index=True)
    balanced = pd.concat([upsets_repeated, losses], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        f"After {factor}x upsampling: {len(balanced)} rows | "
        f"upsets={int((balanced['target']==1).sum())} | "
        f"losses={int((balanced['target']==0).sum())}"
    )
    return balanced


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Build the feature list for the underdog model.

    1. Load feature names from the general model (mar_4_v2_feature_names.pkl).
       This reuses the exact same 251 features — no new feature engineering needed.
    2. Filter to columns that exist in df.
    3. Append 'market_prob_f1' as the 252nd feature (market context signal).

    DO NOT load or use the mar_4_v2 scaler — we fit a new one below.
    """
    general_features_path = MODEL_SAVE_DIR / f"{GENERAL_MODEL_NAME}_feature_names.pkl"

    if general_features_path.exists():
        base_features = joblib.load(general_features_path)
        logger.info(f"Loaded {len(base_features)} feature names from {general_features_path}")
    else:
        logger.warning(
            f"Could not find {general_features_path}. "
            "Make sure mar_4_v2 model files exist in models/saved/. "
            "Falling back to inferring features from dataframe columns."
        )
        base_features = [c for c in df.columns if c not in NON_FEATURE_COLS]

    available = [f for f in base_features if f in df.columns]
    missing   = [f for f in base_features if f not in df.columns]
    if missing:
        logger.warning(f"{len(missing)} features missing from training data: {missing[:5]}...")

    available.append("market_prob_f1")
    logger.info(f"Final feature count: {len(available)}")
    return available


def fight_id_split(df: pd.DataFrame, test_frac: float = 0.20):
    """
    Split by fight_id groups to prevent train/test leakage.

    Each fight produces 2 rows (double-sampling). A naive row-level split lets
    the mirror of a training row appear in the test set, inflating accuracy by
    ~10-15 points. Always split by fight_id groups.

    Returns (train_index, test_index).
    """
    if "fight_id" not in df.columns:
        logger.warning("No fight_id column — using random split (leakage risk!)")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=test_frac, random_state=42, stratify=df["target"]
        )
        return train_df.index, test_df.index

    unique_ids = df["fight_id"].dropna().unique()
    rng = np.random.default_rng(42)
    rng.shuffle(unique_ids)

    n_test    = max(1, int(len(unique_ids) * test_frac))
    test_ids  = set(unique_ids[:n_test])
    train_ids = set(unique_ids[n_test:])

    train_idx = df.index[df["fight_id"].isin(train_ids)]
    test_idx  = df.index[df["fight_id"].isin(test_ids)]

    logger.info(
        f"Fight-grouped split: {len(train_ids)} train fights ({len(train_idx)} rows) | "
        f"{len(test_ids)} test fights ({len(test_idx)} rows)"
    )
    return train_idx, test_idx


def train_underdog_model(
    training_data_path: str   = TRAINING_DATA_PATH,
    odds_data_path: str       = ODDS_DATA_PATH,
    model_name: str           = UNDERDOG_MODEL_NAME,
    underdog_threshold: float = UNDERDOG_THRESHOLD,
    upset_upsample: int       = UPSET_UPSAMPLE,
    n_estimators: int         = 300,
    max_depth: int            = 3,
    learning_rate: float      = 0.03,
    subsample: float          = 0.70,
    colsample_bytree: float   = 0.60,
    reg_alpha: float          = 0.30,
    reg_lambda: float         = 2.00,
) -> dict:
    """
    Full training pipeline. Returns metrics dict.
    Saves model, scaler, feature list, and metrics to models/saved/.
    Does NOT touch any existing model files.
    """

    # ── 1. Load and filter ──────────────────────────────────────────────────
    df       = load_and_merge_odds(training_data_path, odds_data_path)
    df_under = filter_underdog_context(df, underdog_threshold)
    df_bal   = upsample_upsets(df_under, upset_upsample)

    # ── 2. Feature preparation ──────────────────────────────────────────────
    feature_cols = get_feature_columns(df_bal)

    df_bal = df_bal.copy()
    df_bal["market_prob_f1"] = df_bal["market_prob_f1"].fillna(
        df_bal["market_prob_f1"].median()
    )

    X_raw = df_bal[feature_cols].copy().fillna(0)
    y     = df_bal["target"].copy()

    # Fit a BRAND NEW StandardScaler.
    # DO NOT load or reuse mar_4_v2_feature_scaler.pkl — it was fit on different
    # columns (no market_prob_f1) and different row distributions.
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw),
        columns=feature_cols,
        index=X_raw.index,
    )

    # ── 3. Train/test split ─────────────────────────────────────────────────
    train_idx, test_idx = fight_id_split(df_bal, test_frac=0.20)
    X_train = X_scaled.loc[train_idx]
    X_test  = X_scaled.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test  = y.loc[test_idx]

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(f"scale_pos_weight = {n_neg}/{n_pos} = {scale_pos_weight:.2f}")

    # ── 4. Train ────────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators     = n_estimators,
        max_depth        = max_depth,
        learning_rate    = learning_rate,
        subsample        = subsample,
        colsample_bytree = colsample_bytree,
        min_child_weight = 5,
        gamma            = 0.20,
        reg_alpha        = reg_alpha,
        reg_lambda       = reg_lambda,
        scale_pos_weight = scale_pos_weight,
        objective        = "binary:logistic",
        eval_metric      = "logloss",
        random_state     = 42,
        n_jobs           = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── 5. Evaluate ─────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    true_upsets     = int((y_test == 1).sum())
    detected_upsets = int(((y_pred == 1) & (y_test == 1)).sum())
    upset_detection = detected_upsets / true_upsets if true_upsets > 0 else 0.0

    metrics = {
        "model_name"         : model_name,
        "n_train"            : int(len(X_train)),
        "n_test"             : int(len(X_test)),
        "accuracy"           : float(accuracy_score(y_test, y_pred)),
        "auc"                : float(roc_auc_score(y_test, y_prob)),
        "brier_score"        : float(brier_score_loss(y_test, y_prob)),
        "upset_detection"    : float(upset_detection),
        "underdog_threshold" : underdog_threshold,
        "upset_upsample"     : upset_upsample,
        "n_features"         : len(feature_cols),
    }

    logger.info("=== UNDERDOG MODEL RESULTS ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    if metrics["upset_detection"] >= 0.40:
        logger.success(f"GATE PASSED: upset_detection = {metrics['upset_detection']:.3f} >= 0.40")
    else:
        logger.warning(
            f"GATE FAILED: upset_detection = {metrics['upset_detection']:.3f} < 0.40. "
            "See troubleshooting section."
        )

    # ── 6. Save NEW model files only ────────────────────────────────────────
    # These are all new files. Nothing in models/saved/ that already exists
    # will be overwritten (the names all start with 'underdog_v1').
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model_path   = MODEL_SAVE_DIR / f"{model_name}.json"
    scaler_path  = MODEL_SAVE_DIR / f"{model_name}_feature_scaler.pkl"
    feature_path = MODEL_SAVE_DIR / f"{model_name}_feature_names.pkl"
    metrics_path = MODEL_SAVE_DIR / f"{model_name}_metrics.json"

    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, feature_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.success(f"Saved: {model_path}")
    logger.success(f"Saved: {scaler_path}")
    logger.success(f"Saved: {feature_path}")
    logger.success(f"Saved: {metrics_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train underdog specialist model")
    parser.add_argument("--training-data",  default=TRAINING_DATA_PATH)
    parser.add_argument("--odds-data",      default=ODDS_DATA_PATH)
    parser.add_argument("--model-name",     default=UNDERDOG_MODEL_NAME)
    parser.add_argument("--threshold",      type=float, default=UNDERDOG_THRESHOLD)
    parser.add_argument("--upsample",       type=int,   default=UPSET_UPSAMPLE)
    parser.add_argument("--n-estimators",   type=int,   default=300)
    parser.add_argument("--max-depth",      type=int,   default=3)
    parser.add_argument("--learning-rate",  type=float, default=0.03)
    args = parser.parse_args()

    train_underdog_model(
        training_data_path = args.training_data,
        odds_data_path     = args.odds_data,
        model_name         = args.model_name,
        underdog_threshold = args.threshold,
        upset_upsample     = args.upsample,
        n_estimators       = args.n_estimators,
        max_depth          = args.max_depth,
        learning_rate      = args.learning_rate,
    )
