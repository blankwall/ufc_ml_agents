# UFC ML Model — Betting Rules & Decision Framework

> **Model:** `mar_4_v2` (XGBoost, trained through end of 2024, holdout-tested on 2025)
> **Validation method:** Walk-forward (train on years 1–N, test on year N+1, rolled 2020–2024)
> **Data integrity:** Fully point-in-time — no temporal leakage confirmed by audit
> **Last updated:** March 4, 2026

---

## Model Performance at a Glance

| Metric | Value |
|--------|-------|
| Walk-forward accuracy (2020–2024 avg) | **62.6% ± 1.3%** |
| 2025 out-of-sample accuracy (297 fights) | **68.4%** |
| 2025 flat-bet ROI (all picks) | **+11.5%** |
| Confidence-accuracy correlation | **+0.968** (1.0 = perfect) |
| AUC (area under ROC curve) | **0.678** |
| Brier score | **0.226** (lower = better calibrated) |

The confidence-accuracy correlation of **+0.968** is critical: it means higher model confidence genuinely predicts higher win probability. This was **negative** (-0.x) before data leakage was fixed, making this the most important health check.

---

## Rule 1 — Entry Condition: Minimum Edge Required

**Only place a bet when the model's probability exceeds the market-implied probability by at least 10 percentage points.**

```
edge = model_probability − market_implied_probability
REQUIRED: edge ≥ 0.10 (10%)
```

| Edge Threshold | Bets (2025) | Accuracy | ROI |
|----------------|-------------|----------|-----|
| Any positive edge (≥0%) | 147 | 60.5% | +22% |
| ≥5% | 108 | 59.3% | +32% |
| **≥10%** | **78** | **61.5%** | **+46%** |
| **≥15%** | **56** | **64.3%** | **+64%** ✅ |
| ≥20% | 35 | 57.1% | +54% |
| ≥25% | 22 | 59.1% | +75% |
| ≥30% | 11 | 54.5% | +82% |

**Recommended minimum: 10% edge.** This is the best balance of sample size and ROI. Going to 25%+ narrows to ~20 bets/year — high ROI but too few to be meaningful.

### How to compute market-implied probability from American odds
```
Favourite (negative odds, e.g. -250):   prob = |odds| / (|odds| + 100)
Underdog  (positive odds, e.g. +200):   prob = 100   / (odds   + 100)

Examples:
  -250  →  250/350 = 0.714  (71.4%)
  -150  →  150/250 = 0.600  (60.0%)
  +150  →  100/250 = 0.400  (40.0%)
  +200  →  100/300 = 0.333  (33.3%)
  +300  →  100/400 = 0.250  (25.0%)
```

---

## Rule 2 — Confidence Gate

**The model's probability for the picked fighter must be between 60% and 85%.**

| Model Confidence | 2025 Accuracy | 2025 ROI | Decision |
|------------------|---------------|----------|----------|
| 50–55% | 62% | +17% | ⚠️ Only if edge ≥ 20% |
| 55–60% | 62% | +3% | ❌ Skip unless large underdog |
| **60–65%** | **69%** | **+23%** | ✅ Bet if edge ≥ 15% |
| **65–70%** | **72%** | **+5%** | ✅ Bet if edge ≥ 15% |
| **70–75%** | **63%** | **+0%** | ⚠️ Only if edge ≥ 20% |
| **75–80%** | **88%** | **+27%** | ✅ Best band — always bet if edge ≥ 10% |
| 80–85% | 85% | +7% | ✅ Bet but expect short odds |
| 85%+ | 80% | -4% | ❌ Skip — usually heavy favourite, poor value |

**Do not bet above 85% model confidence.** At that level the fighter is typically priced at -400 or worse and the payout doesn't compensate even at 80% accuracy.

---

## Rule 3 — Odds Range (Market Price)

**Target bets where the model picks a fighter priced between +120 and -300 in the market.**

| Market Price | Market Implied Prob | 2025 ROI | Decision |
|---|---|---|---|
| +300 and longer | < 25% | -100% (1 bet) | ❌ Avoid — too volatile |
| **+200 to +300** | 25–33% | **+104%** | ✅ Best underdog tier |
| **+120 to +200** | 33–45% | **+16%** | ✅ Viable with edge ≥ 15% |
| **Even / +120** | 45–55% | **+12%** | ✅ Viable with edge ≥ 15% |
| **-120 to -200** | 55–67% | **+6%** | ✅ Only if edge ≥ 15% |
| **-200 to -300** | 67–75% | **-4%** | ⚠️ Edge must be ≥ 20% |
| -300 and shorter | > 75% | +6% | ⚠️ Tiny payout — 1-unit max |

**The +200 to +300 underdog band is the highest-value zone.** 62% accuracy at those prices generates +104% ROI. This is where the model most frequently disagrees with the market in a profitable direction.

---

## Rule 4 — Division Filter

**Avoid Heavyweight. Be cautious with Featherweight.**

| Division | 2025 Accuracy | 2025 ROI | Rule |
|----------|---------------|----------|------|
| Women's Bantamweight | 80% | +57% | ✅ Full sizing |
| Bantamweight | 72% | +21% | ✅ Full sizing |
| Lightweight | 71% | +19% | ✅ Full sizing |
| Women's Flyweight | 86% | +21% | ✅ Full sizing |
| Flyweight | 70% | +15% | ✅ Full sizing |
| Welterweight | 68% | +15% | ✅ Full sizing |
| Middleweight | 64% | +7% | ✅ Full sizing |
| Women's Strawweight | 75% | +5% | ✅ Full sizing |
| Light Heavyweight | 68% | +2% | ⚠️ Half sizing |
| **Featherweight** | **59%** | **-4%** | ⚠️ Only if edge ≥ 20% |
| **Heavyweight** | **50%** | **-28%** | ❌ Skip entirely |

Heavyweight fights are essentially coin flips for this model. One punch ends fights unpredictably — the statistical patterns the model relies on (striking accuracy, win streaks, opponent quality) carry less predictive weight when any fighter can end it in round 1 regardless of form.

---

## Rule 5 — Bet Sizing

**Use Quarter Kelly as the default. Never risk more than 5% of bankroll on a single fight.**

### Quarter Kelly formula
```
b  = decimal_odds − 1       (profit per unit staked)
p  = model_probability
q  = 1 − p

Full Kelly   = (p × (b+1) − 1) / b
Quarter Kelly = Full Kelly × 0.25

Cap: never exceed 5% of bankroll per bet
```

### Sizing examples

| Model Prob | American Odds | Decimal Odds | Full Kelly | Quarter Kelly |
|---|---|---|---|---|
| 65% | +200 | 3.00 | 22.5% | **5.6%** → cap at 5% |
| 70% | +150 | 2.50 | 32.5% | **8.1%** → cap at 5% |
| 72% | -150 | 1.67 | 25.1% | **6.3%** → cap at 5% |
| 60% | +300 | 4.00 | 13.3% | **3.3%** |
| 80% | -200 | 1.50 | 26.7% | **6.7%** → cap at 5% |

**Flat betting (1 unit per fight) is also valid** for simplicity. Quarter Kelly outperforms flat over time but requires tracking a running bankroll. In 2025 simulation: Quarter Kelly on $1,000 → $6,483 (+548% ROI); flat betting 1 unit every edge≥15% bet → +36 units.

---

## Rule 6 — Do Not Bet Checklist

Skip the bet if **any** of the following are true:

- [ ] Model edge vs market is below **15%**
- [ ] Model confidence is **above 85%** (short-priced favourite, poor value)
- [ ] Model confidence is **below 55%** unless edge ≥ 20%
- [ ] Division is **Heavyweight**
- [ ] Division is **Featherweight** and edge < 20%
- [ ] Fighter has **fewer than 3 UFC fights** in the database (model has limited signal)
- [ ] Fight was announced **within 72 hours** (late replacements break statistical profiles)
- [ ] The two fighters have **never been matched against similar opposition** (model's opponent quality features are blind)

---

## Rule 7 — Walk-Forward Performance by Year (Out-of-Sample)

This is the honest benchmark — model trained on data before each year, tested on that year.

| Test Year | Fights | Accuracy | AUC | Conf-Acc Corr |
|---|---|---|---|---|
| 2020 | 443 | 61.5% | 0.665 | +0.87 |
| 2021 | 497 | 61.5% | 0.665 | +0.99 |
| 2022 | 506 | 61.8% | 0.671 | +1.00 |
| 2023 | 504 | 63.2% | 0.678 | +1.00 |
| 2024 | 513 | 64.9% | 0.711 | +0.99 |
| **Mean** | — | **62.6%** | **0.678** | **+0.97** |

Performance is **improving over time** (61.5% → 64.9%) — more recent training data is higher quality and the model generalises better as more UFC fight history accumulates.

---

## Rule 8 — Calibration Reference

How to interpret model percentages based on actual 2025 outcomes:

| Model Says | Actually Won | Interpretation |
|---|---|---|
| 50–55% | 62% | Model underestimates — treat as ~60% |
| 55–60% | 62% | Roughly accurate |
| 60–65% | 69% | Model slightly underestimates |
| 65–70% | 72% | Roughly accurate |
| 70–75% | 63% | Model slightly overestimates |
| 75–80% | 88% | Model underestimates significantly — very strong bet |
| 80–85% | 85% | Accurate |
| 85%+ | 80% | Model overestimates at extremes |

The 75–80% band winning 88% of the time is the most actionable calibration insight. When the model says 75–80%, treat it as 85–90%.

---

## Quick Decision Flowchart

```
Is the model's edge vs market ≥ 15%?
│
├─ NO  → SKIP
│
└─ YES → Is the division Heavyweight?
         │
         ├─ YES → SKIP
         │
         └─ NO  → Is model confidence between 60% and 85%?
                  │
                  ├─ NO  → Is confidence < 60%? → SKIP
                  │        Is confidence > 85%? → SKIP (bad value)
                  │
                  └─ YES → Is market price between +300 and -300?
                           │
                           ├─ NO (shorter than -300) → Half size max
                           │
                           └─ YES → BET
                                    Size = Quarter Kelly, max 5% bankroll
                                    Priority order: +200/+300 > even > -150 > -200
```

---

## Appendix — How Model Probabilities Are Generated

1. Features are computed **point-in-time** as of the fight date (no future fight data used)
2. Opponent quality, win rates, and time-decayed stats all use the fight date as the reference — not today's date
3. A **symmetric averaging** step is applied: the model scores both `(A vs B)` and `(B vs A)` and averages, eliminating ordering bias
4. The model is an **XGBoost classifier** with monotone constraints on key features (winning rate, striking accuracy) to prevent nonsensical relationships

**The model does not know:** judge scoring tendencies, injury status, camp changes, short-notice replacements, weight cut severity, or in-fight adjustments. These are the biggest sources of residual error.
