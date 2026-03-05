# Favorites Fix - Summary of Findings

## Problem
Model has -25% ROI on favorites, while underdogs perform well (+90% ROI).

## Root Cause Analysis

### The Model is Overconfident on "Paper Favorites"
Bad favorites share characteristics:
- Good "on paper" metrics (striking volume, differentials, recent form)
- BUT low opponent quality (fighting weaker competition)
- Mid-tier overall (not championship level)

### Examples of Bad Favorites
- Bogdan Grad (2x losses): Good differentials vs weak opponents
- Marek Bujlo: Good striking metrics but lost
- Zhang Weili vs Shevchenko: Model favored Zhang but Shevchenko won

## Approaches Tested

### 1. Probability Calibration ❌
**Result**: Made it worse
- Calibration uniformly dampened probabilities
- Dropped 18 favorites with +39% ROI (the good ones!)
- Kept 66 favorites with -42% ROI (the bad ones!)
- Favorite ROI went from -25% to -43%

**Lesson**: Can't fix feature-level problems with post-processing calibration

### 2. Fighter Tier Features ⚠️
**Result**: Slight improvement (+4% on favorites)
- Added `fighter_tier_score_diff`, `elite_experience_diff`, `consistency_advantage_diff`
- But features had low importance (rank 23, 103)
- Model still relies heavily on "on paper" stats:
  - `age_difference` (importance 77)
  - `striking_volume_control` (importance 75/72)

**Metrics v1**:
- Favorite ROI: -21.15% (improved from -25.2%)
- Underdog ROI: +83.7% (decreased from +90.4%)
- Overall ROI: +20.35% (decreased from +22.94%)

### 3. Feature Exclusion (striking_volume_control) ❌
**Result**: Made it worse
- Favorite ROI: -24.93% (worse than -21.15%)
- Overall ROI: +11.03% (much worse)
- Accuracy: 64.6% (worse than 65.7%)

**Lesson**: Individual features aren't the problem - it's how they're combined

## What We've Learned

1. **The problem is NOT individual features** - Removing useful features hurts overall performance
2. **The problem is feature interactions** - The model doesn't properly weight opponent quality
3. **Post-processing calibration doesn't work** - It fixes the wrong probabilities
4. **New tier features help but aren't enough** - Need stronger signals

## Next Steps

### Option A: Strengthen Opponent Quality Signals
- Increase weight of opponent quality features
- Add stronger monotone constraints
- Create interaction features: (recent form) × (opponent quality)

### Option B: Targeted Feature Engineering
- Add "elite favorite" flag (tier > 0.6 AND model_prob > 60%)
- Add "paper favorite" penalty (good stats BUT low opponent quality)
- Add market-aware features (market_prob as input)

### Option C: Model Architecture Changes
- Try shallower trees (max_depth=3) for better calibration
- Try higher learning rate with fewer estimators
- Add class weighting to reduce overconfidence

### Option D: Ensemble Approach
- Train separate models for favorites and underdogs
- Blend predictions based on market probability
- Use different feature sets for different probability ranges

## Recommended Next Step

**Option A + Feature Engineering**: Add strong opponent quality interactions

```python
# Add these features:
# 1. Recent form × opponent quality interaction
recent_form_x_opp_quality = (
    recent_win_rate_last_3 *
    opponent_quality_score
) - (
    opponent_win_rate *
    opponent_quality_score
)

# 2. Elite favorite flag
elite_favorite = (
    fighter_tier_score > 0.6 and
    model_prob > 0.60
)

# 3. Paper favorite penalty
paper_favorite_penalty = (
    striking_volume_control > 0.6 and
    opponent_quality_score < 0.2
)
```

This should help the model distinguish:
- Legit favorites: Good stats + high opponent quality
- Paper favorites: Good stats + low opponent quality
