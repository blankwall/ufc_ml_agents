# Future Retraining — Challenges & Runbook

This is a snapshot of what to watch out for the next time you retrain `mar_4_v2`
with new features (Elo, trend, etc.). Captured before nuking the working tree.

You already have research scaffolding sitting dirty:
- `features/ratings.py` (untracked) — Elo
- `features/trend.py`   (untracked) — recent-form features
- `features/matchup_features.py` (+96 lines) — Elo/trend differentials
- `features/registry.py` (+94 lines) — schema additions
- `models/xgboost_model.py` (+244 lines) — likely sweep/CV/cal helpers
- `schema/feature_schema.json` (modified) — bumped feature list
- `models/saved/mar_4_v2_plus_elo*` etc. — prior experiment outputs
- `scripts/` — dozens of one-off ablation/sweep scripts

---

## 1. Reproducibility — the #1 blocker

Right now training is **NOT deterministic**. Two runs same data → different models.

Causes:
- XGBoost histogram building is non-deterministic with `n_jobs > 1`
- No `--seed` flag wired through
- `train_test_split` / `StratifiedKFold` need `random_state` everywhere
- Feature ordering is fixed in CPython 3.7+ but only because dicts preserve
  insertion order. Any `set()` or unordered iteration drifts.

**Fix before retraining:**
- Add `--seed N` flag to `models.xgboost_model`
- Set `numpy.random.seed`, `random.seed`, XGB `seed=N`, `tree_method='hist'`,
  `n_jobs=1` (or accept some non-determinism for speed)
- Stamp `seed`, git SHA, and SHA256 of `training_data.csv` into a sidecar
  `models/saved/<name>_run_metadata.json`

---

## 2. Schema mismatch in production

`mar_4_v2_feature_names.pkl` is an ordered list of 251 features. Serving code
(`predict_service._score_row`) builds a vector in that exact order. **If you
change feature count or order and overwrite mar_4_v2 in place**, the running
uvicorn process serves wrong predictions until restarted.

**Always rename on retrain:** `mar_4_v3`, `mar_4_v3_elo`, etc. Switch via
config, never in place. Restart uvicorn explicitly after deploy.

---

## 3. Lookahead leakage in new features

`tests/test_no_lookahead_leakage.py` covers existing features only. Elo and
trend are easy to leak with:

- **Elo update on date D must use pre-fight ratings**, not post.
- **Trend windows ("win % last 5")** must use strict `<` filter on date —
  same as existing rolling stats in `FighterFeatureExtractor._get_fight_history`.
- For training, per-row Elo state must be reconstructed from history strictly
  before the row's `as_of_date`. Don't persist Elo state to disk and update
  incrementally — re-runs won't match.

**Add new feature names to the leakage test before merging:**
```python
LEAK_INDICATORS = ['f1_elo', 'f2_elo', 'elo_diff',
                   'f1_winrate_l5', 'f2_winrate_l5',
                   'f1_recent_form_delta', ...]
```
Then assert these change between as_of=D and as_of=D+1, but NOT between
as_of=D and as_of=D-1.

---

## 4. Elo-specific gotchas

- **Cold start**: first ~5 fights for a fighter, rating is meaningless.
  Initialize at 1500 and accept noise, OR add `f1_elo_n_priors` so the model
  can learn to discount young ratings.
- **K-factor** is a hyperparameter — sweep it (10/20/40) on the 2025 holdout.
- **Weight class crossover**: standard Elo says one rating per fighter.
  A heavyweight beating a flyweight shouldn't tank his rating, but you'll
  rarely see it. Document the choice; default to global is fine.
- **WMMA segregation**: separate Elo pool for women (they never fight men).
  Tag with `wmma_elo` if you do this.
- **Recompute from scratch every training run.** Persisted Elo defeats
  reproducibility (challenge #1).

---

## 5. Trend feature design traps

- "Last 3 fights" is biased toward recently active fighters. Pair with
  `days_since_last_fight` (already in your features) so the model can decay.
- Win/loss streak counts dominate when sample is small (3-fight streak = 100%
  rate). Mix in continuous metrics (sig strikes, takedown defense) over the
  same window.
- **Trajectory delta is more useful than each in isolation:**
  `f1_winrate_l5 - f1_winrate_career`, `f1_sigstrikes_l3 - f1_sigstrikes_career`

---

## 6. Hyperparameter risk on a richer feature set

Current command:
```
--n-estimators 200 --max-depth 4 --learning-rate 0.05
--subsample 0.8 --colsample-bytree 0.8
```
was tuned for 251 features. Adding 10–30 new features (Elo + trend) probably
wants:
- More regularization: `reg_alpha`, `reg_lambda` non-zero
- Possibly higher `colsample_bytree` to reduce noise from correlated trend variants
- Maybe deeper trees to capture Elo × matchup-style interactions

**Run a small `colsample × max_depth × reg_alpha` sweep** before declaring
victory. `scripts/run_xgb_config_sweep.py` was built for this — verify it
still works.

---

## 7. Calibration

`mar_4_v2` runs with `--check-calibration`. New features can push probabilities
apart (better AUC, worse calibration). Your edge-based bet sizing
(5/10/20% edge buckets) is **highly sensitive to calibration drift** — a 5pp
predicted-probability shift moves fights across multiple bet tiers.

**Always re-validate after retraining:**
- Reliability diagram on 2025 holdout
- Re-tune the 65% favorite / 53% underdog confidence floors — they were tuned
  to `mar_4_v2`'s specific calibration curve, not constants of nature
- Re-tune edge bucket boundaries if calibration shape changes

---

## 8. Comparing fairly to the current model

You have these helpers from your research scratch:
- `scripts/run_ablation_matrix.py`
- `scripts/run_strategy_sweep.py`
- `scripts/compare_matched_volume.py`
- `backtest/event_stability_*.json`
- `backtest/matched_volume_*.json`

**Use matched-volume comparison** (same fights bet by both models) — raw ROI
comparisons are misleading because models bet different cards. Already proven
necessary in checkpoint 003.

---

## 9. The thing that'll actually break first

`features/feature_pipeline.py --create` has been tracked-modified for a while.
**Before adding new features, prove `mar_4_v2` is reproducible from current
master.**

```bash
git stash    # park research scratch
git checkout master    # or wherever mar_4_v2 was trained from
.venv/bin/python -m features.feature_pipeline --create
sha256sum data/processed/training_data.csv
# Compare to whatever you trained mar_4_v2 from (if you saved it)
.venv/bin/python -m models.xgboost_model --train --model-name mar_4_v2_repro
# Diff predictions on 2025 holdout vs mar_4_v2 — should be identical (or near it)
```

If `mar_4_v2_repro` doesn't match `mar_4_v2`, **stop and fix challenge #1
before doing anything else.** Adding new features without a reproducible
baseline means you can never tell if a change helped or you just got lucky.

---

## Recommended order of operations

1. **Pin seed + write training metadata** (challenge #1)
2. **Re-run `--create` on master**, confirm `training_data.csv` matches what
   `mar_4_v2` was trained on (challenge #9)
3. **Train `mar_4_v2_repro`**, diff against `mar_4_v2` — should be identical
4. *Then* unstash research scratch, add Elo/trend, **name the new model
   `mar_4_v3_elo`** (challenge #2)
5. **Add new features to the leakage test** (challenge #3)
6. **Sweep hyperparams** on new feature set (challenge #6)
7. **Matched-volume backtest** vs `mar_4_v2` (challenge #8)
8. **Re-validate 65/53 confidence floors and edge buckets** on new calibration
   (challenge #7)
9. Ship as `mar_4_v3_elo`, update `config/betting_config.json` model field,
   restart uvicorn

---

## Files to revisit / un-stash

When you come back:
```bash
# See what was dirty when this doc was written
git stash list                   # if you stashed
git diff origin/HEAD -- features/ models/ schema/

# Untracked research that survived:
ls features/ratings.py features/trend.py
ls models/saved/mar_4_v2_plus_elo*
ls backtest/blend_w*.csv         # underdog blend experiments
ls backtest/edge_sweep_*.json    # already analysed in checkpoints
```

Existing checkpoints in `~/.copilot/session-state/<session-id>/checkpoints/`
have the prior history (parallel ablations, matched volume, edge sweeps,
Kelly analysis, WMMA deep dive). Read those before re-doing analysis you've
already done.

---

## Existing test suite (already in repo)

`tests/` covers:
- vig + edge-sign + market-prob consistency between `/events` and `/api/predict`
- predict order-symmetry (model invariant to fighter1/fighter2 swap)
- no-lookahead leakage (strict `<` boundary)
- all 8 skip codes
- bet-sizing buckets (continuity, boundaries, WMMA cap)
- FIGHTER_ALIASES integrity vs DB

Run before AND after retraining:
```bash
.venv/bin/python -m pytest tests/ --base-url http://localhost:8001 -v
```

If `test_predict_swap_symmetry` regresses on a new model, the symmetric
scoring path in `_score_row` was bypassed.
If `test_no_lookahead_leakage` regresses, somebody broke the `<` filter
or a new feature reads from the future.
