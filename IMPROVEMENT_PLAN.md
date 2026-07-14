# UFC ML Model Improvement Plan
## Baseline: March 3, 2026

### Current Performance
| Metric | Value |
|--------|-------|
| Overall Accuracy | 66.3% |
| AUC | 0.7253 |
| Brier Score | 0.2135 |
| Overall ROI | +22.94% |

### Breakdown
| Category | Accuracy | ROI |
|----------|----------|-----|
| **Favorites** | 47.6% ❌ | **-25.2%** ❌ |
| **Underdogs** | 71.7% ✅ | **+90.4%** ✅ |
| Top 25% Confidence | 84.6% | +32.2% |

---

## 1. FAVORITES PLAN (Highest Priority)

### Problem
- Model gives favorites 50-70% confidence but actual win rate is 48-53%
- High edge favorites (>15% edge) have terrible 18.2% accuracy
- Model is overconfident on "paper favorites"

### Root Causes
1. Over-weighting recent form vs established fighter quality
2. Missing "favorite legitimacy" features
3. Not accounting for market's "name value" bias
4. Poor calibration in 50-70% probability range

### Solutions

#### Phase 1: Calibration Fix (Quick Win)
- [ ] Add probability clipping for 50-70% range (dampen overconfidence)
- [ ] Implement Platt scaling or isotonic regression
- [ ] Test: Re-run evaluation with calibrated probabilities

#### Phase 2: Feature Engineering
- [ ] Add `market_prob` as input feature (market signal)
- [ ] Add `historical_favorite_performance` (win rate when favored)
- [ ] Add `championship_experience` flag
- [ ] Add `name_value_score` (recognition, social following)

#### Phase 3: Feature Adjustment
- [ ] Reduce weight of recent form for established fighters
- [ ] Add interaction: recent_form × fighter_tier
- [ ] Increase weight of career-long metrics for favorites

#### Phase 4: Validation
- [ ] Re-train model with new features
- [ ] Evaluate favorite ROI improvement
- [ ] Ensure underdog performance preserved

---

## 2. UNDERDOGS PLAN (Preserve & Refine)

### Problem
- Underdogs working well (+90% ROI) but need understanding
- High confidence underdogs (>0.3) underperform (43.5% accuracy)

### Solutions
- [ ] **DO NOT CHANGE** existing underdog features (working!)
- [ ] Analyze feature importance for underdog predictions specifically
- [ ] Add "upset validation" features for high confidence underdogs
- [ ] Refine underdog confidence calibration

---

## 3. AUC/ACCURACY PLAN

### Current
- Accuracy: 66.3%, AUC: 0.7253

### Hyperparameter Tuning
```
Current: max_depth=4, n_estimators=200, learning_rate=0.05

Try:
- max_depth=3 (shallower = better calibration)
- n_estimators=500 with early stopping
- learning_rate=0.03 (slower = better generalization)
- scale_pos_weight (address class imbalance)
```

### Feature Selection
- [ ] Remove features with importance < 2 (~20 features)
- [ ] Focus on top 50 features
- [ ] Re-evaluate monotonic constraints

---

## 4. NEW FEATURES PLAN

### Market-Aware Features
- [ ] `market_prob` - Market's implied probability
- [ ] `model_vs_market_diff` - Edge amount
- [ ] `line_movement` - If historical odds available

### Style Matchup Refinements
- [ ] Wrestling vs striking specialist
- [ ] Camp/gym quality (training team)
- [ ] Cardio/pace indicators

### Performance Trajectory
- [ ] Improvement slope (last 3 vs 5 vs 10)
- [ ] "Peaked" indicator (decline after prime)
- [ ] Weight class movement history

### Contextual Factors
- [ ] Home/country advantage
- [ ] Short notice indicator
- [ ] Ring rust vs active fighter

---

## EXECUTION ORDER

1. **Favorites calibration fix** (immediate - biggest leak)
2. **Feature pruning** (remove importance < 2)
3. **Hyperparameter tuning** (max_depth, learning_rate)
4. **Add market-aware features**
5. **Validate underdog performance preserved**

---

## TEST COMMANDS (from flow.txt)

```bash
# Step 1: Create features
python -m features.feature_pipeline --create

# Step 2: Train model
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
  --model-name jan_29

# Step 3: Evaluate
python -m evaluation.evaluate_model \
  --data-path data/processed/training_data.csv \
  --odds-path backtest/odds/ufc_2025_odds.csv \
  --min-year 2025 \
  --output-dir reports_strict \
  --odds-date-tolerance-days 5 \
  --model-name jan_29 \
  --symmetric \
  --compare-to-baseline
```
