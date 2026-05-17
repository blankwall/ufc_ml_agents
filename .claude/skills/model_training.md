# Model Training Skill

## What This Is

This skill documents the canonical flow for rebuilding the feature set, training a new XGBoost UFC model, evaluating it against market odds, and then running the formal backtest pipeline.

The safe candidate workflow is:

```text
build features -> train candidate with 2025 holdout -> evaluate vs odds -> backtest -> compare to baseline -> only then promote
```

Do **not** overwrite `mar_4_v2` during experiments. Use a new model name such as `may_17_v1`, `candidate_20260517`, or another clearly timestamped name.

---

## Step 0 — Choose a Model Name

Use one shell variable so commands stay consistent:

```bash
MODEL_NAME=may_17_v1
```

Model names map directly to files in `models/saved/`.

---

## Step 1 — Rebuild the Feature Dataset

The training dataset is built from the database through `features.feature_pipeline`.

```bash
python -m features.feature_pipeline --create
```

Default output:

| File | Purpose |
|---|---|
| `data/processed/training_data.csv` | Full training dataset used by model training/evaluation |

Useful variants:

```bash
# Disable progress output
python -m features.feature_pipeline --create --no-progress

# Disable CachedFeatureBuilder if debugging feature extraction
python -m features.feature_pipeline --create --no-cache

# Use a smaller feature registry subset
python -m features.feature_pipeline --create --feature-set advanced
```

Important implementation notes:
- `FeaturePipeline.create_dataset()` calls `features.matchup_features.create_training_dataset()`.
- The default feature set is `FeatureRegistry.FEATURE_SET_FULL`.
- Dataset rows include metadata columns like `fight_id`, `event_id`, `fighter_1_id`, `fighter_2_id`, `weight_class`, `method`, and `target`.
- Model feature columns are every non-metadata column, sorted deterministically in `FeaturePipeline.prepare_features()`.

---

## Step 2 — Train a Candidate Model

Canonical candidate training command from `flow.txt`:

```bash
python -m models.xgboost_model \
  --train \
  --evaluate \
  --check-calibration \
  --save-plots \
  --export-schema \
  --data-path data/processed/training_data.csv \
  --n-estimators 200 \
  --max-depth 4 \
  --holdout-from-year 2025 \
  --learning-rate 0.05 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --model-name "$MODEL_NAME"
```

What this does:
- Loads `data/processed/training_data.csv`
- Excludes events with year `>= 2025` from training because of `--holdout-from-year 2025`
- Uses a fight-grouped train/test split so mirrored rows from the same fight do not leak across train/test
- Applies recency-based sample weights
- Loads monotone constraints from `schema/monotone_constraints.json` when present
- Trains an `xgboost.XGBClassifier`
- Saves model-specific artifacts under `models/saved/`
- Exports `schema/feature_schema.json`

Generated artifacts:

| File | Purpose |
|---|---|
| `models/saved/${MODEL_NAME}.json` | XGBoost model |
| `models/saved/${MODEL_NAME}_feature_scaler.pkl` | Model-specific scaler |
| `models/saved/${MODEL_NAME}_feature_names.pkl` | Model-specific feature ordering |
| `models/saved/${MODEL_NAME}_features.json` | Feature list saved with model |
| `models/saved/${MODEL_NAME}_metrics.json` | Training/validation metrics |
| `models/saved/${MODEL_NAME}_feature_importance.csv` | Feature importance |
| `schema/feature_schema.json` | Canonical feature contract exported from training |

Why `--holdout-from-year 2025` matters:
- It keeps 2025 as a true out-of-sample holdout.
- The 2025 odds file and backtest results are then valid for model comparison.
- If you train on 2025, do **not** use 2025 backtest/evaluation as proof of performance.

---

## Step 3 — Evaluate Candidate vs Market Odds

Canonical evaluation command from `flow.txt`:

```bash
python -m evaluation.evaluate_model \
  --data-path data/processed/training_data.csv \
  --odds-path backtest/odds/ufc_2025_odds.csv \
  --min-year 2025 \
  --output-dir reports_strict \
  --odds-date-tolerance-days 5 \
  --model-name "$MODEL_NAME" \
  --symmetric \
  --compare-to-baseline
```

What evaluation does:
- Loads the holdout rows from `data/processed/training_data.csv`
- Joins those fights to `backtest/odds/ufc_2025_odds.csv`
- Loads the model and matching model-specific scaler/features
- Applies symmetric probability averaging by default
- Computes accuracy, Brier score, log loss, AUC, calibration, confidence buckets, market edge, and flat-bet ROI diagnostics
- Compares to `models/baseline.json` when `--compare-to-baseline` is passed

Useful flags:

```bash
# Write a JSON summary to a stable path
python -m evaluation.evaluate_model \
  --model-name "$MODEL_NAME" \
  --data-path data/processed/training_data.csv \
  --odds-path backtest/odds/ufc_2025_odds.csv \
  --min-year 2025 \
  --output-dir reports_strict \
  --output-json models/${MODEL_NAME}_eval.json \
  --compare-to-baseline

# Add underdog diagnostics
python -m evaluation.evaluate_model \
  --model-name "$MODEL_NAME" \
  --data-path data/processed/training_data.csv \
  --odds-path backtest/odds/ufc_2025_odds.csv \
  --min-year 2025 \
  --output-dir reports_strict \
  --underdog

# Evaluate highest-confidence picks per card
python -m evaluation.evaluate_model \
  --model-name "$MODEL_NAME" \
  --data-path data/processed/training_data.csv \
  --odds-path backtest/odds/ufc_2025_odds.csv \
  --min-year 2025 \
  --output-dir reports_strict \
  --highest-confidence-per-card
```

Baseline reference:

| File | Purpose |
|---|---|
| `models/baseline.json` | Current baseline metrics used by `--compare-to-baseline` |

---

## Step 4 — Backtest the Candidate

Use the formal backtest runner, not archived legacy scripts. Every newly trained candidate should be backtested on both:
- `backtest/odds/ufc_2025_odds.csv` for the frozen holdout check
- `backtest/odds/ufc_2026_odds.csv` for the current forward/live-year check

### 2025 holdout backtest

```bash
python backtest/backtest_2025.py \
  --odds backtest/odds/ufc_2025_odds.csv \
  --model "$MODEL_NAME" \
  --cutoff 2026-01-01 \
  --quiet

python backtest/bucket_analysis.py \
  --results backtest/backtest_2025_results.csv \
  --bets backtest/bets_2025.txt
```

Because the odds filename contains `2025`, `backtest_2025.py` writes to:

```text
backtest/backtest_2025_results.csv
```

### 2026 forward/live-year backtest

First rebuild the generated 2026 odds input:

```bash
python backtest/rebuild_2026_odds.py
```

Then run the same formal runner:

```bash
python backtest/backtest_2025.py \
  --odds backtest/odds/ufc_2026_odds.csv \
  --model "$MODEL_NAME" \
  --cutoff 2027-01-01 \
  --quiet

python backtest/bucket_analysis.py \
  --results backtest/backtest_2026_results.csv \
  --bets backtest/bets.txt
```

Because the odds filename contains `2026`, `backtest_2025.py` writes to:

```text
backtest/backtest_2026_results.csv
```

Important:
- If the model was trained with `--holdout-from-year 2025`, both 2025 evaluation and 2025 backtest are out-of-sample.
- If the model was trained on all available data including 2025, use later unseen fights only for validation.
- `backtest/archive/backtest_live.py` is archived legacy/prototype code and is not used for formal backtesting.

---

## Step 5 — Optional Strategy Optimization

After candidate results exist, grid-search betting thresholds:

```bash
python backtest/optimize_config.py \
  --results backtest/backtest_2026_results.csv \
  --top 20 \
  --sort-by roi
```

This writes `backtest/optimize_results.csv`, which is generated/ignored and should not be committed.

---

## Promotion Checklist

Before replacing a production model or changing app config:

1. Rebuild `data/processed/training_data.csv`.
2. Train with a new `MODEL_NAME`; do not overwrite `mar_4_v2`.
3. Confirm all model-specific artifacts exist in `models/saved/`.
4. Evaluate with `evaluation.evaluate_model` against holdout odds.
5. Backtest 2025 with `backtest/odds/ufc_2025_odds.csv`.
6. Rebuild 2026 odds with `backtest/rebuild_2026_odds.py`, then backtest `backtest/odds/ufc_2026_odds.csv`.
7. Run `bucket_analysis.py` on both `backtest/backtest_2025_results.csv` and `backtest/backtest_2026_results.csv`.
8. Compare to `models/baseline.json` and current `mar_4_v2` results.
9. Only promote if accuracy, calibration, ROI, and bucket behavior are better or intentionally traded off.

Key comparison surfaces:
- Overall accuracy
- Brier score and log loss
- Confidence calibration
- Favorite vs underdog split
- Odds bucket ROI, especially +200 underdog bucket and heavy favorite buckets
- Real placed bets via `--bets`
- Weighted ROI using `config/betting_config.json`

---

## Gotchas

- Training currently exports `schema/feature_schema.json`; review schema diffs before committing.
- Always keep model-specific scaler/feature files with the model. Missing or mismatched `{MODEL_NAME}_feature_scaler.pkl` and `{MODEL_NAME}_feature_names.pkl` can cause feature mismatch errors.
- The feature pipeline still has legacy fallback loading for `feature_scaler.pkl` and `feature_names.pkl`; prefer model-scoped artifacts.
- `--holdout-from-year` is the guardrail against training/evaluation leakage.
- `evaluation.evaluate_model` uses symmetric predictions by default; pass `--no-symmetric` only for debugging.
- Rebuilding 2026 odds creates `backtest/odds/ufc_2026_odds.csv`, which is generated/ignored.
- `backtest/backtest_2025.py` is the formal runner for both years despite the filename.

---

## Quick End-to-End Candidate Flow

```bash
MODEL_NAME=may_17_v1

python -m features.feature_pipeline --create

python -m models.xgboost_model \
  --train \
  --evaluate \
  --check-calibration \
  --save-plots \
  --export-schema \
  --data-path data/processed/training_data.csv \
  --n-estimators 200 \
  --max-depth 4 \
  --holdout-from-year 2025 \
  --learning-rate 0.05 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --model-name "$MODEL_NAME"

python -m evaluation.evaluate_model \
  --data-path data/processed/training_data.csv \
  --odds-path backtest/odds/ufc_2025_odds.csv \
  --min-year 2025 \
  --output-dir reports_strict \
  --odds-date-tolerance-days 5 \
  --model-name "$MODEL_NAME" \
  --symmetric \
  --compare-to-baseline

python backtest/backtest_2025.py \
  --odds backtest/odds/ufc_2025_odds.csv \
  --model "$MODEL_NAME" \
  --cutoff 2026-01-01 \
  --quiet

python backtest/bucket_analysis.py \
  --results backtest/backtest_2025_results.csv \
  --bets backtest/bets_2025.txt

python backtest/rebuild_2026_odds.py

python backtest/backtest_2025.py \
  --odds backtest/odds/ufc_2026_odds.csv \
  --model "$MODEL_NAME" \
  --cutoff 2027-01-01 \
  --quiet

python backtest/bucket_analysis.py \
  --results backtest/backtest_2026_results.csv \
  --bets backtest/bets.txt
```
