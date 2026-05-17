# Model System Skill

## What This Is

This skill documents the production UFC prediction model itself: what `mar_4_v2` is, how its features are built from the database, where it is strong/weak, and which architectural decisions have already been made.

Use this when changing prediction behavior, model features, feature schema, confidence scoring, or model-vs-market strategy.

---

## Current Production Model

| Item | Current value |
|---|---|
| Production model | `mar_4_v2` |
| Model type | General-purpose XGBoost binary classifier |
| Objective | `binary:logistic` |
| Output | Probability that row-level `fighter_1` beats `fighter_2` |
| Feature count | 251 model features |
| Training boundary | Trained on fights before 2025 |
| 2025+ status | True out-of-sample if not retrained on 2025+ |
| Prediction mode | Symmetric scoring by default |
| Underdog model | Exists as `underdog_v1`, but disabled in production |

Key artifacts:

| File | Purpose |
|---|---|
| `models/saved/mar_4_v2.json` | XGBoost model |
| `models/saved/mar_4_v2_feature_scaler.pkl` | Model-specific `StandardScaler` |
| `models/saved/mar_4_v2_feature_names.pkl` | Model-specific feature order for inference |
| `models/saved/mar_4_v2_features.json` | Feature names saved with the model |
| `models/saved/mar_4_v2_metrics.json` | Train/validation metrics from training |
| `models/saved/mar_4_v2_feature_importance.csv` | Feature importance |
| `schema/feature_schema.json` | Current canonical feature schema contract |

Do not infer the active production feature set from the current feature code alone. The model-specific feature names/scaler are the source of truth for inference with `mar_4_v2`.

---

## XGBoost Configuration

The canonical candidate settings used around `mar_4_v2` are:

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

Model defaults/patterns in code:

| Parameter | Value/pattern |
|---|---|
| `n_estimators` | 200 for canonical candidates |
| `max_depth` | 4 for current general model |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 3 |
| `gamma` | 0.1 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |
| `eval_metric` | `logloss` |
| `random_state` | 42 |

Training also applies:
- Model-specific pipeline artifact saving.
- Recency-based sample weights so newer fights count more.
- Fight-grouped train/test split so mirrored rows from the same fight do not leak between train and validation.
- Monotone constraints from `schema/monotone_constraints.json` where matching feature names exist.

---

## How DB Fights Become Model Rows

The model does not predict directly from raw fighters. It predicts from matchup rows produced by:

```text
database tables -> MatchupFeatureExtractor -> training_data.csv -> FeaturePipeline.prepare_features() -> XGBoost
```

Core DB tables:

| Table | Model relevance |
|---|---|
| `fighters` | Physical attributes, stance, current profile stat snapshots |
| `events` | Event dates for point-in-time filtering and holdout splits |
| `fights` | Fighter pair, result, method, weight class, title-fight flag |
| `fight_stats` | Detailed totals/recent striking/grappling features |

Critical schema rule:

```text
Use fights.id for joins.
Do not join on fights.fight_id except when matching UFCStats fight-detail hashes.
```

Training dataset construction:
- Reads completed fights where `Fight.result` is not null.
- Skips draws and no contests.
- Converts each fight into two rows:
  - winner as `fighter_1`, `target=1`
  - loser as `fighter_1`, `target=0`
- Uses `fight.event.date` as `as_of_date`.
- Feature history must use fights strictly before `as_of_date`, not `<=`.

This mirrored-row design is why random row splitting is unsafe. Always split/evaluate by `fight_id` or by event year.

---

## Feature System

Default feature extraction uses `FeatureRegistry.FEATURE_SET_FULL`, which combines:

| Feature family | Examples |
|---|---|
| Physical | age, height, reach, weight, stance |
| Striking | volume control, striking differential, accuracy/output features |
| Grappling | takedown ability, takedown matchup, submission rate |
| Career/history | total fights, finish rates, win streaks, title-fight experience |
| Time-based | recent form, layoffs, decline, activity, time-decayed rates |
| Opponent quality | average opponent win rates, beaten-opponent quality, quality-adjusted form |
| Recent FightStats | recent striking/control/knockdown differentials |
| Interactions | age x quality, striking x opponent quality, power-striker and style-volatility features |

Important interaction features already added:

| Feature | Intent |
|---|---|
| `striking_volume_control_x_opponent_quality_diff` | Separate real striking dominance from padded stats |
| `striking_differential_x_opponent_quality_diff` | Reward striking edges earned against better opposition |
| `time_decayed_win_rate_x_opponent_quality_diff` | Weight recent wins by opponent quality |
| `age_x_elite_experience_diff` | Treat age as positive when paired with elite experience |
| `age_x_quality_continuous_diff` | Continuous version of veteran/quality interaction |
| `opponent_quality_score_diff_squared` | Emphasize large strength-of-schedule gaps |
| `recent_finish_losses_last_2_diff` | Capture recent durability/chin problems |
| `style_volatility_mismatch_diff` | Capture finish-style mismatch and upset volatility |
| `power_striker_score_diff` | Capture KO-power differential |
| `durability_collapse_diff` | Penalize older fighters with recent damage/finish losses |

`feature_exclusions.py` controls columns dropped from future training datasets. Changing exclusions requires rebuilding the dataset and retraining; it can make the current code diverge from `mar_4_v2` artifacts.

---

## Top Signals in `mar_4_v2`

Top feature importances from `models/saved/mar_4_v2_feature_importance.csv` include:

| Rank | Feature | Importance |
|---|---|---:|
| 1 | `age_difference` | 73 |
| 2 | `f2_striking_volume_control` | 71 |
| 3 | `f1_striking_volume_control` | 54 |
| 4 | `takedown_ability_diff` | 45 |
| 5 | `recent_sig_strike_diff_last_3_diff` | 42 |
| 6 | `takedown_matchup` | 41 |
| 7 | `f2_sig_strikes_landed_per_min` | 38 |
| 8 | `recent_control_time_sec_last_3_diff` | 38 |
| 9 | `f1_striking_accuracy` | 38 |
| 10 | `f1_striking_differential` | 34 |
| 11 | `striking_volume_control_x_opponent_quality_diff` | 34 |
| 12 | `avg_beaten_opponent_win_rate_diff` | 33 |

Interpretation:
- The model leans heavily on age/experience, striking volume/control, takedown matchups, and recent detailed-stat deltas.
- Opponent-quality interaction features are now meaningful signals, not just speculative ideas.
- Feature importance should be used directionally only; always validate with holdout/backtest behavior.

---

## Prediction Flow

Production prediction uses `fastapi_app/services/predict_service.py`.

For each fight:

1. Resolve fighter names to DB fighter IDs.
2. Extract matchup features from the DB as of the event date when available.
3. Add fight-specific features such as `is_title_fight`.
4. Reindex feature DataFrame to `mar_4_v2_feature_names.pkl`.
5. Fill missing features with `0`.
6. Scale with `mar_4_v2_feature_scaler.pkl`.
7. Predict both directions:
   - `P(A | A as fighter_1)`
   - `P(B | B as fighter_1)`
8. Symmetrize:

```text
P_sym(A) = (P(A | A,B) + 1 - P(B | B,A)) / 2
```

Symmetric scoring is a production decision. It prevents prediction ordering from changing the result and is also used by evaluation/backtesting when comparing model candidates.

`UNDERDOG_BLEND = False`. The `underdog_v1` model and blend path exist but are intentionally disabled because the general model currently performs better alone in production.

---

## Strengths

The current general model is strongest when:

| Situation | Why it works |
|---|---|
| High-confidence picks | Calibration is much better above the uncertain middle range |
| Undervalued underdogs/value spots | Model can diverge from market with low market correlation |
| Matchups with clear recent-form or style deltas | Recent FightStats and matchup differentials give signal |
| Quality-adjusted veteran cases | Age x opponent-quality interactions help avoid simple youth bias |
| Explicit edge analysis | Model output is most useful when compared against market implied probability |

Validated observations from prior analysis:
- Baseline overall accuracy is roughly mid/high 60s depending on evaluation slice.
- Underdog/value analysis showed strong ROI in 2025 holdout-style checks.
- The model is not simply copying market odds; model-vs-market correlation has been low in underdog validation.

---

## Known Weaknesses

The major known weakness is calibration in the uncertain middle.

Prior analysis identified a "death zone" around 50-65%/50-70% confidence, especially for favorites:
- Low-confidence favorites can look good on paper but fail against real opposition.
- The model historically overweighted paper stats relative to opponent quality.
- High-confidence favorites behave much better than marginal-confidence favorites.

Practical consequences:
- Do not treat raw 55-60% probabilities as strong edges without market and bucket analysis.
- Favorite picks need stricter confidence/odds filters than underdog value spots.
- Use bucket analysis and confidence bands, not only overall accuracy.

The model also has feature-dependency risks:
- Missing `fight_stats` weakens recent striking/grappling features.
- Bad event dates can break point-in-time history.
- Name mismatches can cause missing predictions or incorrect odds joins.
- Rebuilding features with changed exclusions can drift away from the deployed schema.

---

## Model-vs-Market Decisions

The model is an outcome predictor, not a bet placer. Betting decisions are downstream and config-driven.

Current strategy decisions:
- Keep `mar_4_v2` as the production general model.
- Use market odds only after prediction to compute edge, not as a general model input.
- Use separate CSV odds files for formal backtests.
- Use `config/betting_config.json` / `backtest/backtest_config.json` for thresholds and bet sizing.
- Require higher confidence for favorites.
- Preserve WMMA-specific rules: cap multiplier and require larger edge.
- Do not enable underdog blending unless it beats the general model in fresh validation.

Current high-level betting logic:

| Context | Conservative rule |
|---|---|
| Favorites | Need stronger confidence and respect favorite odds cap |
| Underdogs | Need minimum edge and underdog confidence; odds compensate for lower raw win rate |
| 0-5% edge | Skip |
| 5-10% edge | 1x stake |
| 10-20% edge | 1.5x stake |
| 20%+ edge | 2x stake |
| WMMA | Cap at 1x and require at least 10% edge |

---

## Specialist Model Decision

A multi-model architecture has been proposed but is not the current production architecture.

Proposed idea:
- General model always runs.
- Favorites specialist handles market favorites.
- Underdogs specialist handles market underdogs.
- Market probability would be an explicit feature for specialist models only.

Current decision:
- Keep the general model as the production anchor.
- Treat `underdog_v1` as disabled/experimental.
- Only promote specialist models if they beat `mar_4_v2` on their segment in true out-of-sample validation and backtests.

---

## Validation Surfaces

Use all three layers before trusting model changes:

| Layer | Command |
|---|---|
| Feature dataset rebuild | `python -m features.feature_pipeline --create` |
| Model-vs-market evaluation | `python -m evaluation.evaluate_model --model-name "$MODEL_NAME" --data-path data/processed/training_data.csv --odds-path backtest/odds/ufc_2025_odds.csv --min-year 2025 --symmetric --compare-to-baseline` |
| Formal backtest | `python backtest/backtest_2025.py --odds backtest/odds/ufc_2025_odds.csv --model "$MODEL_NAME"` |
| Bucket analysis | `python backtest/bucket_analysis.py --results backtest/backtest_2025_results.csv` |
| Tests | `.venv/bin/pytest -q tests/` |

Key diagnostics:
- Accuracy, AUC, Brier score, log loss.
- Confidence-band calibration.
- Favorite/underdog split.
- Odds-bucket ROI.
- Edge-tier ROI.
- Skip-reason behavior.
- Real placed bets via `--bets`.
- Weighted ROI with `config/betting_config.json`.

---

## Change Rules

When modifying the model system:

1. Do not overwrite `mar_4_v2` during experiments.
2. Use a new model name for every candidate.
3. Rebuild `data/processed/training_data.csv` after ingestion or feature changes.
4. Export schema only through training/model export.
5. Keep model, scaler, and feature-name files together.
6. Validate symmetric prediction behavior.
7. Compare to `models/baseline.json` and current `mar_4_v2`.
8. Review `schema/feature_schema.json` diffs before committing.
9. Do not promote a model based only on in-sample/pre-2025 results.
10. Do not enable specialist/underdog blending without fresh segment-level evidence.

---

## Quick Mental Model

```text
DB fight schema
  -> point-in-time fight histories
  -> mirrored matchup rows
  -> 251-feature schema
  -> XGBoost general model
  -> symmetric fight probability
  -> market edge
  -> config-driven bet/no-bet + bet sizing
```

The model's job is to estimate fight outcome probability. Profitability comes from combining that estimate with market odds, confidence calibration, and conservative filters.
