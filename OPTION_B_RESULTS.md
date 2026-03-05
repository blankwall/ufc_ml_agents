# Option B: Clarity-Based Sample Weights - Results

## Summary

**Result**: PARTIAL SUCCESS - Improved favorites and death zone, but hurt overall performance

| Metric | Baseline | Option B | Change |
|--------|----------|----------|--------|
| **Overall ROI** | +22.9% | +17.2% | **-5.7%** |
| **Favorite ROI** | -25.2% | -20.4% | **+4.8%** ✅ |
| **Underdog ROI** | +90.4% | +67.8% | **-22.6%** ❌ |
| **Death Zone Acc** | 0.0% | 9.4% | **+9.4%** ✅ |

## How It Works

Sample weights based on fight "clarity" using feature differentials:
- Very close fights (clarity < 10): weight = 0.3 (learn less)
- Clear mismatches (clarity > 25): weight = 1.3-1.5 (learn more)

Clarity score = striking_diff × 2 + takedown_diff × 5 + age_diff × 0.5

## Confidence Bucket Analysis (Favorites Only)

| Bucket | Baseline n | Baseline ROI | Baseline Acc | Option B n | Option B ROI | Option B Acc |
|--------|-----------|--------------|--------------|-----------|--------------|--------------|
| 50-60% | 30 | -100.0% | 0.0% | 32 | -83.7% | **9.4%** ✅ |
| 60-70% | 30 | -9.6% | 53.3% | 21 | -12.2% | 52.4% |
| 70-80% | 16 | +53.2% | 100.0% | 19 | +46.7% | 94.7% |
| 80-90% | 8 | +39.6% | 100.0% | 10 | +37.6% | 100.0% |

## Why It Partially Works

The approach successfully reduces model confidence in close fights (50-60% bucket), which is exactly where "paper favorites" live. However, it also reduces confidence across ALL bets, including the profitable underdog picks.

## Comprehensive Comparison of All Approaches

| Approach | Favorite ROI | Overall ROI | Death Zone Acc | Status |
|----------|--------------|-------------|----------------|--------|
| Baseline | -25.2% | +22.9% | 0.0% | - |
| 70% Confidence Filter | +48.7% | +78.5% | N/A | Filter only ❌ |
| H1: Feature Interactions | -27.1% | +21.5% | 0.0% | Failed ❌ |
| H3: Opponent Quality Weights | -26.9% | +21.0% | 0.0% | Failed ❌ |
| **Option B: Clarity Weights** | **-20.4%** | **+17.2%** | **9.4%** | Partial ⚠️ |

## Next Steps

1. **Hypothesis 2: Market-Aware Adjustment**
   - Blend model probability with market probability when model is uncertain (50-70% range)
   - Target: Reduce overconfidence specifically in paper favorite situations

2. **Hypothesis 5: Paper Favorite Detector**
   - Binary classifier to identify "paper favorites" (good stats vs weak opposition)
   - Apply specific adjustments only to those fights

3. **Tune Option B Parameters**
   - Adjust clarity score thresholds
   - Try different low_weight values (0.2, 0.5)
   - May find sweet spot that helps favorites without hurting underdogs

## Conclusion

Option B demonstrates that reducing confidence in close fights helps the death zone problem, but the current implementation is too blunt - it affects all low-clarity situations, not just the "paper favorite" cases where the model is truly wrong.

**Recommendation**: Try H2 (Market-aware adjustment) next as it specifically targets the overconfidence issue without reducing overall model performance.
