# Multi-Model Betting Strategy

> **Status:** Proposed — not yet implemented
> **Based on:** 2025 backtest (297 fights) with `mar_4_v2` general model

---

## The Problem the Current Model Has

The current model finds edge on **14% of fights** — roughly 2 bets on a typical 13-fight card.
Most of those edges come from fights where it disagreed with the market. But look at what happens
when we split the 2025 results by market context:

| Context | Fights | Model Accuracy | ROI | Avg Model Edge |
|---|---|---|---|---|
| Heavy favourite (mkt ≥70%) | 109 | 94.5% | +19.8% | **−12.6%** |
| Mild favourite (mkt 55–70%) | 81 | 67.9% | +7.7% | −6.3% |
| Pick'em (mkt 45–55%) | 31 | 61.3% | +23.2% | +5.7% |
| Underdog (mkt <45%) | 76 | 34.2% | −1.0% | +8.6% |

Two problems jump out:

1. **The model undershoots heavy favourites.** When the market says someone is a −300 lock, they
   win 94.5% of the time — but the model only gives them ~58% on average (−12.6% edge). The model
   is *too sceptical* of dominant favourites, leaving value on the table.

2. **The model misses 65.8% of upsets.** When an underdog actually wins, the model predicted it
   correctly only 34.2% of the time. It sided with the market and got burned. A model trained
   specifically to detect upset conditions could catch more of those.

These are two fundamentally different prediction problems that one general model can't fully solve.

---

## The Proposed Architecture

```
                        ┌─────────────────────┐
    Any fight  ──────►  │  ROUTER             │
                        │  (market prob)      │
                        └──────┬──────┬───────┘
                               │      │
               mkt_prob ≥ 0.60 │      │ mkt_prob < 0.60
                               ▼      ▼
                    ┌──────────┐    ┌──────────┐
                    │ MODEL 2  │    │ MODEL 3  │
                    │Favourites│    │Underdogs │
                    └──────────┘    └──────────┘
                         │               │
                         └──────┬────────┘
                                ▼
                    ┌───────────────────────┐
                    │  MODEL 1 (general)    │  ← always runs too
                    │  for cross-validation │
                    └───────────────────────┘
                                │
                    Final signal = weighted ensemble
```

### Model 1 — General (current `mar_4_v2`)
- Trained on all fights, no market price as input
- Used as a sanity check and ensemble component
- Already validated: 68.4% accuracy, +11.5% ROI, 0.968 confidence-accuracy correlation
- **Keep as-is**

### Model 2 — Favourites Specialist
**Question it answers:** "Given this fighter is priced as a favourite, will they actually deliver,
or is there a hidden vulnerability the market is missing?"

**Routing threshold:** market implied probability ≥ 60% (roughly −150 or shorter)

**Training data:**
- Filter to rows where the favourite fighter is `f1` (market_prob_f1 ≥ 0.60)
- Approx. **~10,000 training rows** (favourites are ~60% of fights in training data)
- Add `market_prob_f1` as an explicit input feature — the model needs to know "this person is
  priced at −400" to calibrate its scepticism correctly
- Upsample recent favourite upsets (last 3 years × 3) to help the model learn collapse patterns

**Key features to emphasise (monotone constraints / feature engineering):**
- `f1_recent_finish_losses_last_2` — has the favourite recently been stopped?
- `f1_athleticism_decline` — age-related drop in performance
- `f1_long_layoff_over_1yr` — rust factor
- `f1_avg_beaten_opponent_win_rate` — are their wins actually against good opponents?
- `f1_current_loss_streak` — is this a fading favourite?
- `f2_finish_rate` — can the underdog end it early?
- `style_volatility_mismatch_diff` — does the underdog's style cause problems for the favourite?
- `f1_days_since_last_fight` — inactivity penalty

**Target signal:** binary `target` (favourite wins = 1, upset = 0)

**Hyperparameter changes vs general model:**
- Higher `scale_pos_weight` to penalise false confidence (model should be harder to convince a
  heavy favourite is safe)
- Increase `min_child_weight` — require more evidence before splitting on new patterns
- Reduce `max_depth` to 3 — less overfitting on small upset subsets

**What this adds:**
- Better calibration on heavy-favourite fights (−12.6% current under-shoot corrected)
- Catches "fading favourite" patterns the general model treats as noise
- Finds value in confirmed locks: if Favourites Model *also* says ≥ 70%, that's a stronger
  signal than if only the market says it

---

### Model 3 — Underdogs Specialist
**Question it answers:** "Given this fighter is priced as an underdog, what specific conditions
predict they actually pull the upset?"

**Routing threshold:** market implied probability < 40% (roughly +150 or longer)

**Training data:**
- Filter to rows where the underdog fighter is `f1` (market_prob_f1 < 0.40)
- Approx. **~6,500 training rows** — this is the thin part (see risks below)
- Add `market_prob_f1` as input (know exactly how big the underdog is)
- **Heavy upsample upsets** — the underdog wins ~33% of these fights, use `scale_pos_weight`
  to equalise class balance and make the model focus on detecting wins not just predicting losses

**Key features to emphasise:**
- `f1_finish_rate` — underdog needs to finish (decisions rarely go their way)
- `f2_recent_finish_losses_last_2` — can the underdog exploit the favourite's chin?
- `style_volatility_mismatch_diff` — mismatches cause upsets
- `f1_time_decayed_win_rate` — is the underdog actually on a run the market ignores?
- `f1_opponent_quality_score` — underdog who has beaten quality fighters vs one who hasn't
- `f1_age` vs `f2_age` — young hungry underdog vs aging champion is classic upset territory
- `f1_recent_knockdown_diff_last_3` — hidden KO power the market hasn't priced
- `f2_long_layoff_over_1yr` — rusty favourite is upset-prone

**Hyperparameter changes vs general model:**
- `scale_pos_weight` set to `(n_underdog_losses / n_underdog_wins)` to balance classes
- Lower `learning_rate` (0.03) — smaller steps when learning rare upset patterns
- `subsample` 0.7, `colsample_bytree` 0.6 — more regularisation to prevent overfitting on small
  dataset

**What this adds:**
- Currently we spot only 34.2% of upsets. Even moving to 45% would significantly expand the
  number of high-value bets we can catch
- Specialised upset features (finish vulnerability, style mismatch) have more predictive power
  in this context than the general model can utilise

---

## How the Three Models Work Together

### Prediction pipeline for a given fight

```python
market_prob = compute_market_implied_prob(odds)

# Always run general model
p_general = model_1.predict(features)

if market_prob >= 0.60:
    # Favourite context
    p_specialist = model_2.predict(features_with_market_prob)
    # Ensemble: weighted average, specialist gets more weight
    p_final = 0.40 * p_general + 0.60 * p_specialist
    context = "FAVOURITE"

elif market_prob < 0.40:
    # Underdog context
    p_specialist = model_3.predict(features_with_market_prob)
    p_final = 0.35 * p_general + 0.65 * p_specialist
    context = "UNDERDOG"

else:
    # Pick'em — general model is best here, no specialist needed
    p_final = p_general
    context = "PICKEM"

edge = p_final - market_prob
```

### Betting thresholds stay the same

| Context | Min edge to bet | Min specialist confidence | Notes |
|---|---|---|---|
| Favourite (Model 2) | ≥ 15% | ≥ 65% | Market already discounts, need clear signal |
| Underdog (Model 3) | ≥ 15% | ≥ 55% | Lower confidence threshold OK — odds compensate |
| Pick'em (Model 1) | ≥ 15% | ≥ 60% | Existing rules unchanged |

---

## Why This Is Justified (and Where It Could Fail)

### Why it should work

**Different decision boundaries per context.** A favourite loses for different reasons than an
underdog wins. A general model has to learn one set of rules that covers both. Specialist models
can learn the specific patterns that matter in each context.

**Market price is a signal, not noise.** Currently the model ignores odds entirely — it has no
idea if someone is priced at −150 or −400. Adding `market_prob` as a feature for the specialist
models lets them learn "when the market is *this* confident, here is how often the pattern holds."

**The data supports it.** 94.5% accuracy on heavy favourites with −12.6% average model edge is
strong evidence the general model is systematically underestimating favourites. The favourites
model can correct that bias.

### Where it could fail

| Risk | Mitigation |
|---|---|
| **Less training data per model** — ~6,500 underdog rows is thin | Strict regularisation, walk-forward validation, don't use if accuracy drops |
| **Overfitting to historical upset patterns** | Walk-forward eval on 2023/2024 before deploying on 2025+ |
| **Market price leakage** — adding market_prob as a feature risks the model learning to replicate the market | Keep Model 1 (no market price) as anchor; specialist adds on top |
| **Routing errors** — wrong model applied to a fight | Clear thresholds (≥60% / <40%); mid-range fights always use Model 1 |
| **Small edge improvements not worth the complexity** | Validate both specialist models must beat Model 1 on their segment before production |

---

## Validation Plan

Before deploying any specialist model, it must clear all three gates:

**Gate 1 — Walk-forward accuracy** (2020–2024, same methodology as `walk_forward_eval.py`)
- Model 2 must achieve ≥ 70% accuracy on heavy-favourite fights (vs current ~65%)
- Model 3 must achieve ≥ 40% upset detection rate (vs current 34.2%)

**Gate 2 — ROI on holdout 2025 data**
- Model 2 (favourites context): ROI must beat Model 1's +19.8% on the same fights
- Model 3 (underdog context): ROI must beat Model 1's −1.0% on the same fights — must go
  clearly positive before this model is trusted

**Gate 3 — Confidence-accuracy correlation**
- Specialist models must maintain positive conf-accuracy correlation (> +0.85)
- If it goes negative, the model is overconfident in wrong directions — abort

---

## Implementation Order

```
Phase 1 (2 weeks):  Model 2 — Favourites
  - Filter training data to favourite context
  - Add market_prob feature
  - Tune hyperparameters
  - Walk-forward validate 2020–2024
  - Backtest 2025

Phase 2 (2 weeks):  Model 3 — Underdogs
  - Same process on underdog context
  - Heavier regularisation due to smaller data
  - Walk-forward validate 2020–2024
  - Backtest 2025

Phase 3 (1 week):  Ensemble routing
  - Build router logic
  - Backtest all three together on 2025 full card
  - Compare total ROI vs Model 1 alone
  - Only ship if improvement is statistically significant (p < 0.05 over 50+ bets)
```

---

## Expected Outcome

If specialist models perform as hypothesised:

| Metric | Current (Model 1 only) | Target (3-model ensemble) |
|---|---|---|
| Bets per 13-fight card | ~2 | ~3–4 |
| Overall accuracy on bets | ~65% | ~68% |
| ROI on edge≥15% bets | +64% | +80%+ |
| Upset detection rate | 34.2% | 42%+ |
| Heavy-fav edge correction | −12.6% avg | <−5% avg |

The biggest single win would be the underdog model finding **1 extra high-value underdog per
card** that the general model currently misses. At +200 average odds with 45% accuracy, that adds
roughly **+8% ROI per fight** to the total pool — meaningful compounding over a season.

---

## What This Is Not

- **Not a bet-everything machine.** All three models still funnel through the same edge threshold,
  confidence gate, and bankroll rules from `BETTING_RULES.md`. The rules don't change, the
  information feeding into the decision does.
- **Not dependent on market price during training.** Model 1 stays market-blind. Only the
  specialist models add market_prob as context — and only at inference time routing.
- **Not a replacement.** Model 1 remains the anchor and sanity check. Specialist models are
  additive, not substitutes.
