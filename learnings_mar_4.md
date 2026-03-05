# UFC ML Model Validation - Learnings & Issues

**Date:** March 4, 2026
**Model:** power_veteran_v2_test
**Purpose:** Document findings from comprehensive model validation

---

## Executive Summary

After running comprehensive backtesting against 272 actual UFC fight results with real betting odds, **the model does not work for betting purposes**. The model performs worse than random chance (48.2% win rate) and loses money across all straightforward betting strategies (-15.9% ROI).

---

## Tests Performed

### 1. Small Sample Validation (19 predictions)
**Initial promising results that led to false confidence:**

| Event | Prediction | Result |
|-------|-----------|--------|
| UFC Fight Night: Moreno vs Kavanagh | 6/7 correct | 85.7% accuracy |
| UFC Fight Night: Feb 21 | 2/2 correct | 100% accuracy |

**Flaw:** Small sample size, survivorship bias, cherry-picking

### 2. Full Backtest (358 fights, 2025 odds data)
Ran predictions on all 2025 UFC fights with available betting odds:
- 308 successful predictions (86%)
- 50 failed predictions (14%) due to KeyError bug
- 240 "value bets" identified (+EV > 5%)

**Flaw:** No actual results to validate predictions

### 3. **Critical Test: Against Actual Fight Results (272 bets)**
Compared model predictions against real fight outcomes from `fight_details.json`:

| Strategy | Bets | Win Rate | ROI | Profit |
|----------|------|----------|-----|--------|
| ALL PICKS | 272 | 48.2% | **-15.9%** | -$4,335 |
| HIGH CONFIDENCE (>65%) | 140 | 45.0% | **-29.5%** | -$4,126 |
| VERY HIGH CONFIDENCE (>70%) | 95 | 42.1% | **-37.1%** | -$3,523 |
| VALUE BETS (+EV > 10%) | 187 | 48.7% | **-9.2%** | -$1,727 |
| Underdogs model likes (+odds, >60%) | 34 | 41.2% | +11.9% | +$403 |
| Fade heavy favorites | 35 | 34.3% | +12.5% | +$438 |

---

## What's Wrong with the ML Code

### 1. **Train-Val-Test Performance Gap (Overfitting)**

```
Training accuracy:     71.1%
Validation accuracy:   64.7%
Real world accuracy:   48.2%
```

**The model performs significantly worse on unseen data**, indicating severe overfitting to training patterns that don't generalize.

**Location:** Model training metrics in `models/saved/power_veteran_v2_test_metrics.json`

### 2. **Inverse Confidence Problem**

The model exhibits **negative correlation** between confidence and accuracy:

| Confidence Level | Actual Win Rate |
|-----------------|-----------------|
| >70% confidence | 42.1% |
| >65% confidence | 45.0% |
| All predictions | 48.2% |

**This is the opposite of what a working model should produce.** Higher confidence should correlate with higher accuracy, not lower.

### 3. **Double Sampling Artifact**

**Location:** `features/matchup_features.py:777-811`

```python
# Perspective 1: winner as fighter_1 (positive class)
features_win = matchup_extractor.extract_matchup_features(
    winner_id, loser_id, as_of_date=event_date
)

# Perspective 2: loser as fighter_1 (negative class)
features_lose = matchup_extractor.extract_matchup_features(
    loser_id, winner_id, as_of_date=event_date
)
```

Each fight creates **TWO training samples**:
- Winner as f1 (target=1)
- Loser as f1 (target=0)

**Issue:** This artificial symmetry may cause the model to learn patterns that don't reflect real prediction scenarios.

### 4. **Insufficient Validation**

**Missing elements:**
- No cross-validation on training data
- Holdout validation uses only year-based split (`holdout_from_year=2025`)
- No walk-forward validation for time-series data
- No statistical significance testing during training

### 5. **Feature Engineering Issues**

#### Age Difference Feature
**Location:** `features/matchup_features.py`

The model has NO monotone constraint on `age_difference`, allowing it to learn non-linear relationships. This could be:
- Learning real patterns (veteran advantage)
- Overfitting to spurious correlations

#### Opponent Quality Score
**Location:** `features/time_based.py`

Calculates opponent strength based on opponent win rates. **Critical question:** Does this properly handle point-in-time data, or does it leak future information about opponent performance?

```python
# Need to verify: is opponent_quality_score calculated with point-in-time data?
opponent_quality_diff = (
    f1_features.get('opponent_quality_score', 0) -
    f2_features.get('opponent_quality_score', 0)
)
```

### 6. **Prediction Failures**

**14% failure rate (50/358 fights)** due to:

**Location:** `features/matchup_features.py:518`

```python
def calculate_finish_reliance(fight_history):
    # BUG: Crashes when fight_history is empty or has no 'result' column
    wins = fight_history[fight_history['result'] == 'win']  # KeyError!
```

**Impact:** Fighters with 0-1 fights in database cause crashes, preventing predictions.

### 7. **Model Retracking Without Clear Best Model Selection**

**Git status shows dozens of model iterations:**
```
agent_loop_model_20260129_190829_iter1.json
agent_loop_model_20260221_223123_iter1.json
agent_loop_model_20260222_131624_iter5.json
```

**Issue:** No clear process for:
- Selecting the best iteration
- Avoiding cherry-picking
- Tracking which model should be used

---

## What We Learned

### 1. Small Sample Sizes Are Deceiving
- 19 predictions at 85% accuracy seemed promising
- 272 predictions at 48% accuracy revealed the truth
- **Lesson:** Always validate on large, out-of-sample datasets

### 2. Survivorship Bias Is Powerful
- We highlighted the 2 successful underdog picks (Strickland +235, Medic +175)
- Ignored the dozens of failed predictions
- **Lesson:** Track ALL predictions, not just the wins

### 3. Theoretical EV ≠ Actual Profit
- Model identified 240 "value bets" with +EV > 5%
- These actually lost 9.2% ROI in real testing
- **Lesson:** EV calculations are only as good as the probability estimates

### 4. Higher Confidence ≠ Better Performance
- Model's highest confidence picks (>70%) performed worst (42% win rate)
- **Lesson:** Model calibration is broken; model is overconfident

### 5. Train/Val Metrics Don't Tell the Full Story
- Training: 71.1%, Validation: 64.7% seemed reasonable
- Real world: 48.2% was catastrophic
- **Lesson:** Need true out-of-sample test with actual outcomes

### 6. Underdog Arbitrage ≠ Prediction Skill
- The only "profitable" strategies involved betting on underdogs
- These won only 34-41% of the time but paid well when they won
- **Lesson:** This is betting market inefficiency, not fighter prediction

### 7. Point-in-Time Data Safety Is Critical
- Code includes `as_of_date` parameters to prevent data leakage
- But opponent quality features may still leak future info
- **Lesson:** Need rigorous testing for temporal data integrity

---

## Statistical Significance Results

**Best Strategy:** Fade Heavy Favorites (model picks underdog vs -200+ favorite)
- Bets: 35
- Win Rate: 34.3%
- Z-score: -1.86
- **p-value: 0.063**

**Conclusion:** The model is NOT statistically significantly better than random (p > 0.05). Even the "best" strategy only shows "trending significance" at p < 0.10, and in the wrong direction (worse than random).

---

## Recommended Fixes

### High Priority
1. **Fix the KeyError bug** for fighters with 0-1 fights
2. **Implement proper cross-validation** (time-series aware)
3. **Add calibration** to ensure 70% confidence ≈ 70% win rate
4. **Reduce overfitting** (simpler model, more regularization)

### Medium Priority
5. **Audit opponent_quality_score** for data leakage
6. **Implement walk-forward validation** for time-series
7. **Add feature importance analysis** to identify meaningful features
8. **Statistical significance testing** during model selection

### Low Priority
9. **Consider single sampling** instead of double (one row per fight)
10. **Add bankroll management** and Kelly Criterion sizing
11. **Implement ensemble methods** to reduce variance

---

## Conclusion

The model **does not work for betting purposes**. The apparent edge seen in small samples was variance, not skill. The model loses money across all straightforward betting strategies and performs worse than random chance.

**Key takeaways:**
- Always validate on large out-of-sample datasets
- Be skeptical of small sample results
- Higher confidence should correlate with higher accuracy (if it doesn't, model is broken)
- Theoretical EV is meaningless without accurate probability estimates

**Recommendation:** Do not use this model for betting decisions. Significant rework needed to make it viable.

---

## Files Referenced

- `/Users/tylerbohan/code/ufc_ml_agents/models/saved/power_veteran_v2_test_metrics.json` - Model metrics
- `/Users/tylerbohan/code/ufc_ml_agents/features/matchup_features.py` - Feature calculation (line 518 bug)
- `/Users/tylerbohan/code/ufc_ml_agents/features/time_based.py` - Opponent quality features
- `/Users/tylerbohan/code/ufc_ml_agents/data/fight_details.json` - Actual fight results
- `/Users/tylerbohan/code/ufc_ml_agents/backtest_results.csv` - Prediction results
- `/Users/tylerbohan/code/ufc_ml_agents/validate_predictions.py` - Validation script
