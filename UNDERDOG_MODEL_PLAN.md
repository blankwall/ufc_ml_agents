# Underdog Model — Implementation Plan

> Pass this document to the implementing agent in full.
> Do not skip any step. Do not change file paths without updating all references.

---

## Goal

Train a second XGBoost model called `underdog_v1` that specialises in fights where
one fighter is a significant market underdog (priced at +150 or longer, market implied
probability < 40%). The current general model (`mar_4_v2`) only correctly identifies
34.2% of upsets. This model targets 42%+.

At inference time, when odds are provided, route underdog-context fights through
`underdog_v1` instead of (or blended with) the general model.

---

## Why This Works

The general model was trained on all fights with no knowledge of market odds. It has
to learn one decision boundary for every context. An underdog specialist can learn
the specific patterns that predict upsets:

- Favourite has a recently cracked chin (finish losses)
- Underdog has elite finish rate vs durability mismatches
- Underdog's form is improving but market hasn't repriced
- Style volatility mismatch favours the underdog

Backtest data confirms the gap: when an underdog won in 2025, the general model
only spotted it 34.2% of the time and had -1.0% ROI on those fights. There is
real room to improve.

---

## Codebase Map — Read These Files Before Starting

```
ufc_ml_agents/
├── data/processed/training_data.csv     # 16,562 rows, 258 features, NO market odds
├── ufc_2025_odds.csv                    # Market odds for 2025 fights, HAS market_prob_f1
├── models/
│   ├── xgboost_model.py                 # MAIN reference — understand train() and save_model()
│   ├── walk_forward_eval.py             # Reference for validation approach
│   └── saved/
│       ├── mar_4_v2.json                # General model weights
│       ├── mar_4_v2_feature_names.pkl   # List of 315 feature names (pickle)
│       ├── mar_4_v2_feature_scaler.pkl  # Fitted StandardScaler (pickle)
│       └── mar_4_v2_metrics.json        # Benchmark metrics to beat
├── features/feature_pipeline.py         # prepare_features(), load_pipeline()
├── evaluation/evaluate_model.py         # Evaluation CLI
└── xgboost_predict.py                   # Inference — modify for routing
```

---

## Step 1 — Create Training Script

Create **`models/train_underdog_model.py`** with the exact content below.
Do not rename it. The file must be runnable as `python models/train_underdog_model.py`.

```python
"""
Underdog Specialist Model — Training Script
File: models/train_underdog_model.py

Trains a model on fights where f1 is the market underdog (market_prob_f1 < 0.40).
Adds market_prob_f1 as an explicit input feature.
Upsamples upset wins to balance the class imbalance.
Uses tighter regularisation than the general model (smaller dataset).
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── constants ────────────────────────────────────────────────────────────────
TRAINING_DATA_PATH  = "data/processed/training_data.csv"
ODDS_DATA_PATH      = "ufc_2025_odds.csv"
GENERAL_MODEL_NAME  = "mar_4_v2"
UNDERDOG_MODEL_NAME = "underdog_v1"
MODEL_SAVE_DIR      = Path("models/saved")
UNDERDOG_THRESHOLD  = 0.40   # market_prob_f1 below this → underdog context
UPSET_UPSAMPLE      = 3      # repeat upset-win rows this many times

# Columns that are metadata, not features. Never pass these to XGBoost.
NON_FEATURE_COLS = {
    "target", "fight_id", "event_id", "fighter_1_id", "fighter_2_id",
    "weight_class", "is_title_fight", "method", "event_date", "event_year",
    "event_day", "f1_name", "f2_name", "f1_name_norm", "f2_name_norm",
    "name_a", "name_b", "fight_key", "fighter1", "fighter2",
    "fighter1_odds", "fighter2_odds", "fighter1_prob", "fighter2_prob",
    "event_date_odds", "fighter1_norm_odds", "fighter2_norm_odds",
    "market_prob_f1",   # added explicitly below — excluded from auto-detect
    "model_prob_f1", "model_name", "model_prob_f1_symmetric",
    "edge", "price_f1",
}
# ─────────────────────────────────────────────────────────────────────────────


def load_and_merge_odds(training_path: str, odds_path: str) -> pd.DataFrame:
    """
    Load training_data.csv and left-join market_prob_f1 from odds CSV.

    The odds CSV must have columns: fight_key, market_prob_f1
    The training CSV must have column: fight_key

    Rows with no matching odds are kept but market_prob_f1 = NaN.
    These rows are dropped in the next step (filter_underdog_context).
    """
    train = pd.read_csv(training_path)
    odds  = pd.read_csv(odds_path)

    logger.info(f"Training rows loaded: {len(train)}")
    logger.info(f"Odds rows loaded: {len(odds)}")

    # Only keep the columns we need from odds; drop duplicates
    if "market_prob_f1" not in odds.columns:
        raise ValueError(
            f"'market_prob_f1' column not found in {odds_path}. "
            f"Available columns: {list(odds.columns)}"
        )
    if "fight_key" not in odds.columns:
        raise ValueError(
            f"'fight_key' column not found in {odds_path}. "
            f"Available columns: {list(odds.columns)}"
        )

    odds_slim = odds[["fight_key", "market_prob_f1"]].drop_duplicates("fight_key")
    merged = train.merge(odds_slim, on="fight_key", how="left")

    n_matched = merged["market_prob_f1"].notna().sum()
    logger.info(f"Rows with matched odds: {n_matched} / {len(merged)} "
                f"({n_matched/len(merged)*100:.1f}%)")

    return merged


def filter_underdog_context(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Keep only rows where f1 is the market underdog.

    The training data has TWO rows per fight (double-sampling):
      - Row A: f1 = actual winner, target = 1
      - Row B: f1 = actual loser,  target = 0

    After filtering market_prob_f1 < threshold:
      - Some rows: underdog (f1) actually WON  → target = 1  (upset)
      - Some rows: underdog (f1) actually LOST → target = 0  (expected loss)

    The model learns: "given f1 is an underdog, will they pull the upset?"
    """
    df_with_odds = df.dropna(subset=["market_prob_f1"]).copy()
    df_under = df_with_odds[df_with_odds["market_prob_f1"] < threshold].copy()

    n_upsets = (df_under["target"] == 1).sum()
    n_losses = (df_under["target"] == 0).sum()
    upset_rate = n_upsets / len(df_under) * 100 if len(df_under) > 0 else 0

    logger.info(
        f"Underdog context: {len(df_under)} rows | "
        f"upsets={n_upsets} | losses={n_losses} | upset_rate={upset_rate:.1f}%"
    )

    if len(df_under) < 1000:
        logger.warning(
            "Very few underdog training rows. Check fight_key join is working. "
            "Print df_under.head() and odds.head() to debug."
        )

    return df_under


def upsample_upsets(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """
    Repeat upset-win rows `factor` times to balance class imbalance.

    Without this, the model sees ~3x more underdog losses than wins and learns
    to always predict the underdog loses. It would get high accuracy but be
    useless for betting (we need it to DETECT upsets).

    After upsampling, shuffle so upset and loss rows are interleaved.
    """
    upsets = df[df["target"] == 1].copy()
    losses = df[df["target"] == 0].copy()

    upsets_repeated = pd.concat([upsets] * factor, ignore_index=True)
    balanced = pd.concat([upsets_repeated, losses], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        f"After upsampling (factor={factor}): {len(balanced)} rows | "
        f"upsets={( balanced['target']==1).sum()} | "
        f"losses={(balanced['target']==0).sum()}"
    )
    return balanced


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Build the feature list for the underdog model.

    Strategy:
    1. Load the feature names saved by the general model (mar_4_v2_feature_names.pkl).
       This ensures we use the same 315 features as Model 1.
    2. Filter to columns that actually exist in df (safety check).
    3. Append 'market_prob_f1' as the context feature.

    Returns a list of column names to use as X for training.
    """
    general_features_path = MODEL_SAVE_DIR / f"{GENERAL_MODEL_NAME}_feature_names.pkl"

    if general_features_path.exists():
        base_features = joblib.load(general_features_path)
        logger.info(f"Loaded {len(base_features)} features from {general_features_path}")
    else:
        logger.warning(
            f"Could not find {general_features_path}. "
            "Inferring features from dataframe columns. "
            "Make sure mar_4_v2 model files exist in models/saved/"
        )
        base_features = [c for c in df.columns if c not in NON_FEATURE_COLS]

    # Only keep features that exist in the current dataframe
    available = [f for f in base_features if f in df.columns]
    missing   = [f for f in base_features if f not in df.columns]
    if missing:
        logger.warning(f"Missing {len(missing)} features from training data: {missing[:5]}...")

    # Add the market context feature
    available.append("market_prob_f1")

    logger.info(f"Final feature count: {len(available)}")
    return available


def fight_id_split(df: pd.DataFrame, test_frac: float = 0.20):
    """
    Split dataset by fight_id groups to prevent train/test leakage.

    Each fight produces 2 rows (double-sampling). A random row-level split
    lets the mirror of a training row appear in the test set, inflating
    validation accuracy by ~10-15 points. Always split by fight_id.

    Returns (train_index, test_index) as pandas Index objects.
    """
    if "fight_id" not in df.columns:
        logger.warning("No fight_id column — using random split (leakage risk!)")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=test_frac,
                                             random_state=42, stratify=df["target"])
        return train_df.index, test_df.index

    unique_ids = df["fight_id"].dropna().unique()
    rng = np.random.default_rng(42)
    rng.shuffle(unique_ids)

    n_test     = max(1, int(len(unique_ids) * test_frac))
    test_ids   = set(unique_ids[:n_test])
    train_ids  = set(unique_ids[n_test:])

    train_idx  = df.index[df["fight_id"].isin(train_ids)]
    test_idx   = df.index[df["fight_id"].isin(test_ids)]

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
    """

    # ── 1. Load and filter ──────────────────────────────────────────────────
    df        = load_and_merge_odds(training_data_path, odds_data_path)
    df_under  = filter_underdog_context(df, underdog_threshold)
    df_bal    = upsample_upsets(df_under, upset_upsample)

    # ── 2. Feature preparation ──────────────────────────────────────────────
    feature_cols = get_feature_columns(df_bal)

    # Fill NaN in market_prob_f1 (safety — should not be NaN after filter)
    df_bal["market_prob_f1"] = df_bal["market_prob_f1"].fillna(
        df_bal["market_prob_f1"].median()
    )

    X_raw = df_bal[feature_cols].copy().fillna(0)
    y     = df_bal["target"].copy()

    # Fit a NEW StandardScaler — do not reuse mar_4_v2 scaler because
    # market_prob_f1 is a new column the old scaler has never seen.
    scaler  = StandardScaler()
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

    # Class weight: tell XGBoost how imbalanced the classes still are
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(f"scale_pos_weight = {n_neg}/{n_pos} = {scale_pos_weight:.2f}")

    # ── 4. Train ────────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators     = n_estimators,
        max_depth        = max_depth,        # 3 (vs 4 for general model)
        learning_rate    = learning_rate,    # 0.03 (vs 0.05)
        subsample        = subsample,        # 0.70 (vs 0.80)
        colsample_bytree = colsample_bytree, # 0.60 (vs 0.80)
        min_child_weight = 5,               # 5 (vs 3) — needs more samples per leaf
        gamma            = 0.20,            # higher split penalty
        reg_alpha        = reg_alpha,        # 0.30 (vs 0.10)
        reg_lambda       = reg_lambda,       # 2.00 (vs 1.00)
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

    # upset_detection = recall on class 1 (how often we correctly spot upsets)
    true_upsets      = (y_test == 1).sum()
    detected_upsets  = ((y_pred == 1) & (y_test == 1)).sum()
    upset_detection  = detected_upsets / true_upsets if true_upsets > 0 else 0.0

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

    # Gate check — must beat general model's 34.2% upset detection
    if metrics["upset_detection"] >= 0.40:
        logger.success(
            f"GATE PASSED: upset_detection = {metrics['upset_detection']:.3f} >= 0.40"
        )
    else:
        logger.warning(
            f"GATE FAILED: upset_detection = {metrics['upset_detection']:.3f} < 0.40. "
            "Try increasing upset_upsample or reducing max_depth."
        )

    # ── 6. Save ─────────────────────────────────────────────────────────────
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
    parser.add_argument("--threshold",      type=float, default=UNDERDOG_THRESHOLD,
                        help="Market prob below which f1 = underdog (default 0.40 = +150)")
    parser.add_argument("--upsample",       type=int,   default=UPSET_UPSAMPLE,
                        help="Repeat upset-win rows this many times (default 3)")
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
```

---

## Step 2 — Run Training

```bash
# From project root
source .venv/bin/activate

python models/train_underdog_model.py \
  --training-data data/processed/training_data.csv \
  --odds-data ufc_2025_odds.csv \
  --model-name underdog_v1 \
  --threshold 0.40 \
  --upsample 3 \
  --n-estimators 300 \
  --max-depth 3 \
  --learning-rate 0.03
```

### Expected log output

```
Training rows loaded: 16562
Odds rows loaded: ~600
Rows with matched odds: 3000-6000 / 16562  ← varies based on historical odds coverage
Underdog context: 2000-4000 rows | upsets=X | losses=Y | upset_rate=30-35%
After upsampling (factor=3): 5000-9000 rows
Final feature count: 316  ← 315 general features + market_prob_f1
Fight-grouped split: ~train_fights / ~test_fights
scale_pos_weight = N/M = ~1.0 after upsampling
[50 rounds] ... [300 rounds]
=== UNDERDOG MODEL RESULTS ===
  accuracy: 0.55-0.70
  auc: 0.55-0.70
  upset_detection: X.XXX   ← must be >= 0.40
GATE PASSED / GATE FAILED
Saved: models/saved/underdog_v1.json
```

### If upset_detection < 0.40 (gate failed)

Try in this order, re-running after each change:

1. Increase `--upsample 5` (more upset rows)
2. Reduce `--max-depth 2` (simpler trees, less overfitting)
3. Increase `--n-estimators 500`
4. Reduce `--threshold 0.35` (stricter underdog filter — cleaner signal)

---

## Step 3 — Add Routing to Inference (`xgboost_predict.py`)

Open `xgboost_predict.py`. Make three targeted edits:

### Edit A — Add helper function near top of file (after imports)

Add this function after the existing `_resolve_fighter` function (~line 90):

```python
def _load_underdog_model(model_name: str = "underdog_v1"):
    """
    Load underdog specialist model, its scaler, and feature list.
    Returns (xgb_model, scaler, feature_names_list) or raises FileNotFoundError.
    """
    import joblib
    model_dir = Path("models/saved")
    model_path   = model_dir / f"{model_name}.json"
    scaler_path  = model_dir / f"{model_name}_feature_scaler.pkl"
    feature_path = model_dir / f"{model_name}_feature_names.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Underdog model not found: {model_path}")

    ud_model = xgb.XGBClassifier()
    ud_model.load_model(model_path)
    ud_scaler   = joblib.load(scaler_path)
    ud_features = joblib.load(feature_path)
    return ud_model, ud_scaler, ud_features


def _american_to_prob(odds: int) -> float:
    """Convert American odds integer to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)
```

### Edit B — Add `--odds-f1` and `--odds-f2` to the argument parser

Find the `if __name__ == '__main__':` block and add these two lines to the parser:

```python
parser.add_argument('--odds-f1', type=int, default=None,
    help='American odds for fighter 1 (e.g. -260 or +215). Enables model routing.')
parser.add_argument('--odds-f2', type=int, default=None,
    help='American odds for fighter 2. Used for routing if --odds-f1 not provided.')
```

Pass them through to `xgboost_predict()`:

```python
xgboost_predict(
    ...
    odds_f1=args.odds_f1,
    odds_f2=args.odds_f2,
)
```

Also add `odds_f1: int | None = None, odds_f2: int | None = None` to the
`xgboost_predict()` function signature.

### Edit C — Add routing block inside `xgboost_predict()`

Find the line `prediction = 1 if p_f1 > 0.5 else 0` and insert this block
**immediately before it**:

```python
# ── Underdog model routing ───────────────────────────────────────────────
# If odds are provided, check if either fighter is a significant underdog.
# If so, blend the underdog specialist model with the general model output.
UNDERDOG_THRESHOLD  = 0.40   # market prob below this = underdog context
UNDERDOG_BLEND      = 0.65   # specialist weight; general model gets 1 - this

if odds_f1 is not None or odds_f2 is not None:
    # Determine market prob for f1
    if odds_f1 is not None:
        market_prob_f1 = _american_to_prob(odds_f1)
    elif odds_f2 is not None:
        market_prob_f1 = 1.0 - _american_to_prob(odds_f2)
    else:
        market_prob_f1 = None

    if market_prob_f1 is not None and market_prob_f1 < UNDERDOG_THRESHOLD:
        # f1 is the underdog — run specialist
        try:
            ud_model, ud_scaler, ud_features = _load_underdog_model("underdog_v1")

            # Build feature dict with market_prob_f1 appended
            features_with_market = dict(features)
            features_with_market["market_prob_f1"] = market_prob_f1

            X_ud = pd.DataFrame(
                [{f: features_with_market.get(f, 0.0) for f in ud_features}]
            ).fillna(0)
            X_ud_scaled = pd.DataFrame(
                ud_scaler.transform(X_ud),
                columns=ud_features,
            )

            p_ud_f1 = float(ud_model.predict_proba(X_ud_scaled)[0, 1])

            # Weighted blend: specialist takes priority
            p_f1_blended = UNDERDOG_BLEND * p_ud_f1 + (1 - UNDERDOG_BLEND) * p_f1
            p_f2_blended = 1.0 - p_f1_blended

            print(f"\n[UNDERDOG SPECIALIST MODEL ACTIVE]")
            print(f"  market_prob_f1 = {market_prob_f1:.3f} (below {UNDERDOG_THRESHOLD} threshold)")
            print(f"  general model:    {fighter_1.name} {p_f1*100:.1f}%")
            print(f"  underdog model:   {fighter_1.name} {p_ud_f1*100:.1f}%")
            print(f"  blended output:   {fighter_1.name} {p_f1_blended*100:.1f}%")

            p_f1 = p_f1_blended
            p_f2 = p_f2_blended

        except FileNotFoundError:
            print("\n[NOTE] underdog_v1 model not found — using general model only")

    elif market_prob_f1 is not None and (1.0 - market_prob_f1) < UNDERDOG_THRESHOLD:
        # f2 is the underdog — run specialist with fighters swapped
        market_prob_f2 = 1.0 - market_prob_f1
        try:
            ud_model, ud_scaler, ud_features = _load_underdog_model("underdog_v1")

            # Use features_2 (the reversed features, f2 as f1)
            features_with_market = dict(features_2)
            features_with_market["market_prob_f1"] = market_prob_f2

            X_ud = pd.DataFrame(
                [{f: features_with_market.get(f, 0.0) for f in ud_features}]
            ).fillna(0)
            X_ud_scaled = pd.DataFrame(
                ud_scaler.transform(X_ud),
                columns=ud_features,
            )

            # p_ud_f2 = probability that f2 wins (in the swapped feature view)
            p_ud_f2 = float(ud_model.predict_proba(X_ud_scaled)[0, 1])
            p_ud_f1 = 1.0 - p_ud_f2

            p_f1_blended = UNDERDOG_BLEND * p_ud_f1 + (1 - UNDERDOG_BLEND) * p_f1
            p_f2_blended = 1.0 - p_f1_blended

            print(f"\n[UNDERDOG SPECIALIST MODEL ACTIVE]")
            print(f"  market_prob_f2 = {market_prob_f2:.3f} (below {UNDERDOG_THRESHOLD} threshold)")
            print(f"  general model:    {fighter_2.name} {p_f2*100:.1f}%")
            print(f"  underdog model:   {fighter_2.name} {p_ud_f2*100:.1f}%")
            print(f"  blended output:   {fighter_2.name} {p_f2_blended*100:.1f}%")

            p_f1 = p_f1_blended
            p_f2 = p_f2_blended

        except FileNotFoundError:
            print("\n[NOTE] underdog_v1 model not found — using general model only")
# ── End underdog routing ─────────────────────────────────────────────────
```

**Important:** The routing block references `features_2` which is the feature dict
computed for the reversed order `(fighter_2, fighter_1)`. This already exists in the
`symmetric=True` code path as `features_2 = extractor.extract_matchup_features(fighter_2.id, fighter_1.id)`.
Make sure that variable is still in scope when the routing block runs.

---

## Step 4 — Test the Routing

```bash
source .venv/bin/activate

# Test 1: Borralho (f1) vs De Ridder (f2)
# De Ridder is underdog at +215 → f2 underdog branch should activate
python xgboost_predict.py \
  --fighter-1 "Caio Borralho" \
  --fighter-2 "Reinier De Ridder" \
  --model-name mar_4_v2 \
  --allow-ambiguous \
  --quiet \
  --odds-f1 -260 \
  --odds-f2 215

# Expected output includes:
# [UNDERDOG SPECIALIST MODEL ACTIVE]
#   market_prob_f2 = 0.317 (below 0.400 threshold)
#   general model:    Reinier de Ridder XX.X%
#   underdog model:   Reinier de Ridder XX.X%
#   blended output:   Reinier de Ridder XX.X%

# Test 2: Evenly matched fight (no routing)
python xgboost_predict.py \
  --fighter-1 "Movsar Evloev" \
  --fighter-2 "Lerone Murphy" \
  --model-name mar_4_v2 \
  --allow-ambiguous \
  --quiet \
  --odds-f1 -250 \
  --odds-f2 210

# Expected: NO [UNDERDOG SPECIALIST MODEL ACTIVE] line
# Both fighters are above 40% implied prob so general model only
```

---

## Step 5 — Validate on 2025 Holdout

Run the existing evaluation CLI but for the underdog model.
Create a simple comparison script at `models/validate_underdog.py`:

```python
"""
Compare general model vs underdog model on 2025 underdog fights.
File: models/validate_underdog.py

Run: python models/validate_underdog.py
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

EVAL_CSV   = "reports_mar_4_v2/eval_data_20260304_165004.csv"
MODEL_DIR  = Path("models/saved")
THRESHOLD  = 0.40


def main():
    df = pd.read_csv(EVAL_CSV)

    # Use only target=1 rows (f1 is actual winner) to avoid double-counting
    df = df[df["target"] == 1].copy()

    # Filter to underdog context: f1 was the underdog (mkt < 0.40)
    df_under = df[df["market_prob_f1"] < THRESHOLD].copy()
    print(f"Underdog fights in 2025 holdout: {len(df_under)}")
    print(f"Actual upsets (f1 underdog won):  {len(df_under)}")

    # General model predictions
    gen_prob_col = "model_prob_f1_symmetric"
    df_under["gen_pred"] = (df_under[gen_prob_col] > 0.5).astype(int)
    gen_acc = accuracy_score(df_under["target"], df_under["gen_pred"])

    # Upset detection for general model
    # Here target=1 means f1 (the underdog) actually won
    gen_detected = df_under["gen_pred"].sum()
    gen_detection_rate = gen_detected / len(df_under)

    print(f"\n=== GENERAL MODEL (mar_4_v2) ===")
    print(f"  Accuracy on underdog fights: {gen_acc*100:.1f}%")
    print(f"  Upset detection rate:        {gen_detection_rate*100:.1f}%  ← baseline (34.2%)")

    # Underdog model predictions
    ud_model_path   = MODEL_DIR / "underdog_v1.json"
    ud_scaler_path  = MODEL_DIR / "underdog_v1_feature_scaler.pkl"
    ud_feature_path = MODEL_DIR / "underdog_v1_feature_names.pkl"

    if not ud_model_path.exists():
        print("\nUnderdog model not found. Train it first with:")
        print("  python models/train_underdog_model.py")
        return

    ud_model    = xgb.XGBClassifier()
    ud_model.load_model(ud_model_path)
    ud_scaler   = joblib.load(ud_scaler_path)
    ud_features = joblib.load(ud_feature_path)

    # Build feature matrix for underdog model
    X_ud = df_under[[f for f in ud_features if f in df_under.columns]].copy()
    for f in ud_features:
        if f not in X_ud.columns:
            X_ud[f] = 0.0
    X_ud = X_ud[ud_features].fillna(0)

    X_ud_scaled = pd.DataFrame(
        ud_scaler.transform(X_ud),
        columns=ud_features,
        index=X_ud.index,
    )

    ud_prob = ud_model.predict_proba(X_ud_scaled)[:, 1]
    ud_pred = (ud_prob > 0.5).astype(int)
    ud_acc  = accuracy_score(df_under["target"], ud_pred)
    ud_detection_rate = ud_pred.sum() / len(df_under)

    print(f"\n=== UNDERDOG MODEL (underdog_v1) ===")
    print(f"  Accuracy on underdog fights: {ud_acc*100:.1f}%")
    print(f"  Upset detection rate:        {ud_detection_rate*100:.1f}%  ← target >= 40%")

    # Blended predictions (same weights as routing code)
    BLEND = 0.65
    blended_prob = BLEND * ud_prob + (1 - BLEND) * df_under[gen_prob_col].values
    blended_pred = (blended_prob > 0.5).astype(int)
    blended_acc  = accuracy_score(df_under["target"], blended_pred)
    blended_detection = blended_pred.sum() / len(df_under)

    print(f"\n=== BLENDED (65% underdog + 35% general) ===")
    print(f"  Accuracy on underdog fights: {blended_acc*100:.1f}%")
    print(f"  Upset detection rate:        {blended_detection*100:.1f}%")

    # Decision
    print("\n=== VERDICT ===")
    if ud_detection_rate >= 0.40:
        print(f"  GATE PASSED: Deploy underdog_v1 for underdog routing")
    else:
        print(f"  GATE FAILED: upset_detection={ud_detection_rate:.3f} < 0.40")
        print(f"  Do not deploy. See troubleshooting section in UNDERDOG_MODEL_PLAN.md")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python models/validate_underdog.py
```

---

## Step 6 — What Success Looks Like

After completing all steps, running `python models/validate_underdog.py` should show:

```
Underdog fights in 2025 holdout: ~76

=== GENERAL MODEL (mar_4_v2) ===
  Accuracy on underdog fights: XX.X%
  Upset detection rate:        34.2%   ← baseline

=== UNDERDOG MODEL (underdog_v1) ===
  Accuracy on underdog fights: XX.X%
  Upset detection rate:        42.0%+  ← must beat this

=== BLENDED (65% underdog + 35% general) ===
  Upset detection rate:        40.0%+

=== VERDICT ===
  GATE PASSED: Deploy underdog_v1 for underdog routing
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ValueError: 'market_prob_f1' not in odds CSV` | Column name is different | Print `pd.read_csv('ufc_2025_odds.csv').columns` and update the `odds_slim` line in `load_and_merge_odds()` |
| `Rows with matched odds: 0` | `fight_key` format mismatch between the two CSVs | Print `train['fight_key'].head()` and `odds['fight_key'].head()` and align the format |
| `upset_detection` stays at ~0.34 | Upsampling not enough, model collapses to "always predict loss" | Set `--upsample 5`, check `y_train.value_counts()` after upsampling — should be near 50/50 |
| `Underdog model not found` at inference | Forgot to run training or wrong model name | Run Step 2, confirm `models/saved/underdog_v1.json` exists |
| `features_2` not in scope in routing block | Variable only exists inside symmetric mode | Ensure `--symmetric` is True (it is by default) or hoist `features_2` assignment out of the if-block |
| High accuracy but low upset_detection | Model is predicting "loss" for every underdog | This is the class imbalance problem — increase `--upsample` and reduce `max_depth` |
| Training accuracy 100% | Duplicate rows from upsampling leaked into test set | Confirm `fight_id_split()` runs on `df_bal` (after upsampling), and that fight_ids are unique per group |

---

## Files Created by This Implementation

```
models/
├── train_underdog_model.py      ← NEW: training script
├── validate_underdog.py         ← NEW: validation comparison script
└── saved/
    ├── underdog_v1.json         ← NEW: model weights
    ├── underdog_v1_feature_scaler.pkl  ← NEW
    ├── underdog_v1_feature_names.pkl   ← NEW
    └── underdog_v1_metrics.json        ← NEW

xgboost_predict.py               ← MODIFIED: routing logic + new args
```

No other files should need to change.
