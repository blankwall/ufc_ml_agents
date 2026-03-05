# Engineering Hypotheses for UFC ML Model Improvements

## Confidence Threshold Analysis Results

| Threshold | Total Bets | Total ROI | Fav ROI | Fav Accuracy | Insight |
|-----------|------------|-----------|---------|--------------|---------|
| 50% | 144 | +22.9% | -25.2% | 47.6% | **Baseline - favorites lose** |
| 60% | 114 | +55.3% | +16.3% | 74.1% | **Tipping point - favorites turn profitable** |
| 65% | 97 | +71.9% | +41.9% | 91.9% | Strong favorite performance |
| **70%** | **84** | **+78.5%** | **+48.7%** | **100%** | **Sweet spot - favorites perfect** |
| 75% | 74 | +82.1% | +46.4% | 100% | Peak ROI but lower volume |

**Key Discovery**: The 50-60% confidence range is the "death zone" where favorites consistently lose.

---

## Hypothesis 1: Probability Calibration via Feature Interactions

### Problem Statement
The model is overconfident in the 50-70% range because it doesn't properly account for opponent quality when assessing "on paper" fighter advantages.

### Hypothesis
Adding interaction features between raw metrics and opponent quality will help the model distinguish between:
- **Legit favorites**: Good metrics vs high-quality opponents
- **Paper favorites**: Good metrics vs low-quality opponents

### Proposed Features
```python
# 1. Striking Volume × Opponent Quality
striking_volume_x_opp_quality = (
    striking_volume_control *
    opponent_quality_score
)

# 2. Fighter Tier × Recent Form
tier_x_recent_form = (
    fighter_tier_score *
    recent_win_rate_last_3
)

# 3. Age Advantage × Fighter Tier
age_advantage_x_tier = (
    age_difference *
    fighter_tier_score_diff
)
```

### Expected Impact
- Model will reduce confidence for "paper favorites" (good stats, weak opposition)
- Favorites in 50-70% range should decrease or be eliminated
- Overall calibration should improve

### Success Metrics
- 50-60% favorites: ROI > -20% (currently -100%)
- 60-70% favorites: ROI > +20% (currently +16%)
- Overall ROI: Maintain > +70%

### Risk Assessment
- **Low risk**: Adding features won't hurt existing predictions
- **Medium effort**: Need to implement in matchup_features.py
- **Medium testing**: Need to rebuild training data

---

## Hypothesis 2: Market-Aware Confidence Adjustment

### Problem Statement
The model's raw probabilities don't align with market probability in the 50-70% range. When the model is 55-60% confident but the market is equally uncertain, the model should be more conservative.

### Hypothesis
The market knows something the model doesn't. When model confidence (50-70%) differs significantly from market confidence, adjust toward market.

### Proposed Approach
```python
# Market-aware confidence adjustment
def adjust_confidence(model_prob, market_prob, min_conf=0.5):
    """
    Adjust model confidence toward market when in uncertain range.
    """
    if 0.5 <= model_prob <= 0.7:
        # In uncertain range, blend with market
        weight = 0.3  # 30% market, 70% model
        adjusted = (model_prob * (1 - weight) + market_prob * weight)
        return max(adjusted, min_conf)  # Don't drop below 50%
    return model_prob
```

### Expected Impact
- Reduces false confidence in death zone
- Preserves high-confidence predictions
- Leverages market wisdom when uncertain

### Success Metrics
- 50-60% favorites: Win rate > 30% (currently 0%)
- Overall favorite ROI: > +20%
- Underdog ROI: Preserve > +70%

### Risk Assessment
- **Medium risk**: Could hurt underdog performance if over-adjusted
- **Low effort**: Can implement in evaluation layer
- **Low testing**: No retraining needed

---

## Hypothesis 3: Opponent Quality Feature Weighting

### Problem Statement
Opponent quality features have low importance (33 vs 83 for age) despite being critical for distinguishing legit vs paper favorites.

### Hypothesis
Increasing the weight of opponent quality features through monotone constraints or custom loss weighting will reduce "paper favorite" predictions.

### Proposed Approaches

#### Option A: Monotone Constraints
```python
# Force opponent quality to have positive relationship with win prob
monotone_constraints = {
    'avg_opponent_win_rate_diff': 1,  # Higher = better
    'opponent_quality_score_diff': 1,
    'fighter_tier_score_diff': 1,
}
```

#### Option B: Sample Weighting
```python
# Weight samples by opponent quality delta
sample_weight = 1 + abs(opp_quality_diff) * 2
# Paper favorites (bad opp quality) get lower weight
```

#### Option C: Feature Scaling
```python
# Amplify opponent quality differences
scaled_opp_quality = opp_quality_diff ** 2
```

### Expected Impact
- Model will rely more on opponent quality when making predictions
- Reduces overconfidence on fighters with good stats vs weak opposition
- Should shift some 50-60% predictions to <50% (no bet)

### Success Metrics
- Opponent quality feature importance: > 50 (currently 33)
- 50-60% favorites: Reduced by > 50%
- Overall calibration: Improved in 50-70% range

### Risk Assessment
- **Medium risk**: Changing feature relationships could have unintended effects
- **Medium effort**: Requires model retraining
- **Medium testing**: Need to verify no negative impact

---

## Hypothesis 4: Separate Models for Confidence Ranges

### Problem Statement
A single model struggles with different prediction regimes:
- High confidence (70%+): Elite vs elite, predictable
- Medium confidence (50-70%): Mismatches, hard to predict
- Low confidence (<50%): Upsets, very hard to predict

### Hypothesis
Training separate models for different confidence ranges will improve overall performance by specializing in each regime.

### Proposed Architecture
```
                    Input Features
                          |
                          v
              +-----------------------+
              |  Confidence Classifier  |
              |  (Predicts model_prob)   |
              +-----------------------+
                          |
              +-------+-------+-------+
              |       |       |       |
              v       v       v       v
         High (70+) Med (50-70) Low (<50)
              |       |       |       |
          Model A   Model B   Model C   (Ensemble)
              |       |       |       |
              +-------+-------+-------+
                          |
                          v
                   Final Prediction
```

### Model Specializations
- **Model A (High confidence)**: Focus on elite fighter attributes, championship experience
- **Model B (Medium confidence)**: Focus on style matchups, preparation quality
- **Model C (Low confidence)**: Focus on upset indicators, variance factors

### Expected Impact
- Each model optimized for its regime
- Overall calibration improved
- Specialized features for each regime

### Success Metrics
- Overall ROI: > +85% (vs current +78% at 70% threshold)
- Total bets: > 100 (increase volume while maintaining ROI)
- Each regime ROI: All positive

### Risk Assessment
- **High risk**: Complex architecture, more failure points
- **High effort**: Need to train and maintain 3 models
- **High testing**: Complex validation needed

---

## Hypothesis 5: Paper Favorite Detection Model

### Problem Statement
The current model cannot explicitly identify "paper favorites" - fighters with good stats who are actually overvalued.

### Hypothesis
Training a binary classifier to detect paper favorites will help filter bad bets before they happen.

### Target Variable
```python
# Paper favorite = Good stats BUT loses
is_paper_favorite = (
    model_prob > 0.5 AND  # Model likes them
    model_prob < 0.7 AND  # But not very confident
    opponent_quality < 0.3 AND  # Weak opposition
    actual_result == 'Loss'  # They lose
)
```

### Features for Detection
- Striking volume vs recent opponents' average quality
- Win rate vs strength of schedule
- Recent form vs career consistency
- Age × experience mismatch (old but inexperienced)

### Implementation
```python
# 1. Train paper favorite detector
paper_detector = train_classifier(
    features=fighter_features,
    target=is_paper_favorite
)

# 2. Use as filter
if paper_detector.predict(fighter) > 0.5:
    # Skip this bet
    pass
```

### Expected Impact
- Directly filters death zone favorites
- Reduces losses on overvalued fighters
- Complements confidence threshold

### Success Metrics
- Death zone favorites filtered: > 80%
- Overall favorite ROI: > +40%
- False positive rate (filtering good bets): < 20%

### Risk Assessment
- **Medium risk**: Could filter profitable bets if detector is inaccurate
- **Medium effort**: Need labeled data and training
- **Medium testing**: Need to verify detector accuracy

---

## Prioritized Hypothesis Roadmap

### Phase 1: Quick Wins (Week 1)
1. **Hypothesis 2**: Market-aware confidence adjustment
   - No retraining needed
   - Immediate improvement possible
   - Reversible if doesn't work

2. **Hypothesis 3**: Opponent quality feature weighting
   - Simple monotone constraints
   - Clear success metrics
   - Low risk

### Phase 2: Feature Engineering (Week 2)
3. **Hypothesis 1**: Feature interactions
   - striker_volume × opponent_quality
   - tier × recent_form
   - Test and iterate

### Phase 3: Advanced Approaches (Week 3+)
4. **Hypothesis 5**: Paper favorite detector
   - Requires data analysis
   - Specialized training
   - High potential impact

5. **Hypothesis 4**: Separate models (exploratory)
   - Complex but promising
   - Requires significant effort
   - Long-term project

---

## Testing Framework

### A/B Testing Protocol
1. **Control**: Current 70% threshold (+78.5% ROI)
2. **Test A**: Add feature interactions
3. **Test B**: Market-aware adjustment
4. **Test C**: Combined approach

### Success Criteria
- **Must have**: Overall ROI > +70%
- **Must have**: Favorite ROI > +30%
- **Nice to have**: Total bets > 80 (volume)
- **Nice to have**: Underdog ROI > +70%

### Monitoring Metrics
- Per-threshold ROI (50%, 60%, 70%)
- Calibration curves
- Feature importance drift
- Market alignment

---

## Next Steps

**Immediate:**
1. Implement 70% confidence threshold as baseline
2. Add feature interactions (Hypothesis 1)
3. Test opponent quality weighting (Hypothesis 3)

**This Week:**
1. Build and test feature interaction features
2. Evaluate impact on death zone
3. Iterate based on results

**Next Week:**
1. Implement market-aware adjustment
2. Test combined approaches
3. Deploy best performing solution

---

*Last Updated: 2026-03-03*
*Baseline: 70% confidence threshold, +78.5% ROI (84 bets)*
