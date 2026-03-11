# DB Deep Dive — Model, Data, and Backtest Architecture

> Written from the FastAPI app build session, March 2026.
> Covers what we learned about mar_4_v2, the underdog blend, the database,
> and how the backtest engine works.

---

## 1. The Model Stack

### mar_4_v2 (General Model)

- **Type:** XGBoost classifier
- **Features:** 251 engineered features (fighter physical attributes, career stats, recent form, opponent quality, style matchup diffs, interaction terms)
- **Training cutoff:** All fights **before 2025** (`--holdout-from-year 2025`)
- **Validation metrics (train/val split):**
  - val accuracy: 62.9%, val AUC: 0.683, val Brier: 0.224
- **2025 holdout (true out-of-sample):**
  - accuracy: ~67-68%, ROI: ~+11% flat-bet on all picks
- **Key calibration finding (BETTING_RULES.md):** When model is 75–80% confident, actual win rate is ~88% — the model is systematically underconfident at the high end.
- **Artifacts:** `models/saved/mar_4_v2.json`, `mar_4_v2_feature_scaler.pkl`, `mar_4_v2_feature_names.pkl`

### underdog_v1 (Specialist Underdog Model)

- **Purpose:** Improve upset detection in fights where the market underdog has implied probability < 40%
- **Extra feature:** `market_prob_f1` is added as feature 252 — the model sees the market's opinion
- **Training:** Filtered to underdog rows only, then **3× upsampling of upset wins** to address class imbalance
- **Metrics:** accuracy 57.5%, AUC 0.572, upset detection rate 45%
- **Artifacts:** `models/saved/underdog_v1.json`, `underdog_v1_feature_scaler.pkl`, `underdog_v1_feature_names.pkl`

### Underdog Blending

When a fight has a market underdog (implied prob < 40%), a blended prediction is used:

```
blended_prob = 0.65 × underdog_v1_prob + 0.35 × mar_4_v2_prob
```

The blend weight (65/35) is configurable. Blending is only applied to the underdog fighter's perspective rows, not to the whole card. The general model runs for all fights; the underdog model only activates when the market flags a fighter as a dog.

### Symmetric Probabilities

The model uses **symmetric averaging** by default: for each fight it scores both `(A vs B)` and `(B vs A)`, then takes:

```
P_sym(A) = (P(A | A is f1) + 1 - P(B | B is f1)) / 2
```

In practice, the features are constructed as signed differentials (e.g. `height_advantage = f1_height - f2_height`), so flipping the fighter order naturally inverts the features and the model produces exactly complementary predictions. Empirically, `model_prob_f1_symmetric == model_prob_f1` in the eval data — the model is **naturally symmetric** without needing the averaging step. Confirmed: all fight pairs sum to exactly 1.0.

---

## 2. Database Structure

SQLite at `data/ufc_database.db`. Eight tables.

### Key Tables for Backtesting

| Table | Rows | Notes |
|---|---|---|
| `events` | 760 | Dates stored as `"November 22, 2025"` (not ISO). `id` is integer PK. |
| `fights` | 8,507 | `id` (int PK), `fight_id` (hash), `event_id` → `events.id`, `fighter_1_id` / `fighter_2_id` / `winner_id` → `fighters.id` |
| `fighters` | 4,451 | `id` (int PK), `name`, stats |
| `betting_odds` | 13,532 | `fight_id` → `fights.id` (integer, NOT the hash). `is_closing_line`, `is_opening_line`. Only ONE fighter's odds per row (either `fighter_1_odds` OR `fighter_2_odds`, rarely both). |
| `fight_stats` | 8,487 | Round-by-round detail, JSON columns |

### Fights with Closing Odds

**6,622 unique fights** across 2007–2026 have closing odds in the DB. By year (approximate):

| Period | ~Fights/yr |
|---|---|
| 2007–2012 | 76–298 |
| 2013–2024 | 400–462 |
| 2025 | 459 |
| 2026 (partial) | 8 |

### Odds Data Quirks

- `betting_odds.fighter_1_odds` corresponds to `fights.fighter_1_id` (DB canonical ordering, which is unrelated to the `training_data.csv` row ordering)
- Only **13 of 6,753 closing-line rows** have both sides' implied probs stored. The rest have only one side.
- The missing side is reliably computed as `1 - known_implied_prob` and then vig-normalised: `market_prob_f1 = f1_impl / (f1_impl + f2_impl)`
- `fighter_1_implied_prob` is the **raw** book probability (e.g. `-282` → `282/382 = 0.738`), not vig-free.

### Join Keys

```
training_data.fight_id   → fights.id          (integer, NOT the hash)
training_data.event_id   → events.id           (integer)
training_data.fighter_1_id → fighters.id       (integer)
betting_odds.fight_id    → fights.id           (integer)
fights.event_id          → events.id           (integer)
```

> **Common mistake:** `fights.fight_id` is a hex hash like `5f5b626e67529056`.
> `training_data.fight_id` is an INTEGER matching `fights.id`, not the hash.
> `betting_odds.fight_id` also uses `fights.id` (integer).

---

## 3. Training Data (`data/processed/training_data.csv`)

- **16,562 rows** = 2 rows per fight (both fighter orderings) × ~8,281 fights
- **258 columns:** 251 model features + `target`, `fight_id`, `event_id`, `fighter_1_id`, `fighter_2_id`, `weight_class`, `is_title_fight`, `method`
- Covers **every year from 1995–2026**, including all 2025 fights (~501 unique fights)
- **All 251 mar_4_v2 features are present** — zero missing features when scoring
- The 2025 fights in this file are the same fights in the DB (confirmed 751/760 events overlap)
- Pre-2025 features were computed using the full current DB (not historical snapshots), meaning there is mild look-ahead bias for older years if using the current model. This is inherent to the training setup.

---

## 4. The Eval CSV vs the DB

The file `reports_mar_4_v2/eval_data_20260304_165004.csv` is a **frozen snapshot** generated March 4, 2026:

| Property | Eval CSV | DB-backed |
|---|---|---|
| Fights | 297 (2025 only) | 6,622 (all years) |
| 2025 fights | 297 | 447–459 |
| Underdog fights (2025) | 88 (after edge filter) | 133+ |
| Date range | Jan–Dec 2025 | 2007–2026 |
| Requires model run | No (pre-scored) | Yes (live scoring) |
| In-sample contamination | 2025 is clean holdout | Pre-2025 is in-sample |

The eval CSV had **fewer 2025 fights** than the DB because it was generated before the full year was scraped. The DB is the authoritative source.

---

## 5. Backtest Design Decisions

### Winner-Perspective Bias (Fixed)

The eval CSV always stores the **actual winner as f1** in the first of the two fight rows. The original backtest engine filtered to `target == 1` (winner as f1), which means:
- The model was always evaluated from the **winner's perspective**
- Market odds for f1 were systematically skewed toward underdog prices (winners aren't always favorites)
- Accuracy appeared inflated

**Fix:** Deduplicate by sorting descending on `model_prob_f1_symmetric` and taking the first row per `fight_id`. This gives one row per fight where **the model picks f1** — completely outcome-independent. It's the same approach used in `analysis/deep_backtest.py`.

### Underdog Focus Row Structure

For the "underdogs" focus, we keep **all rows where `market_prob_f1 < threshold`**, not just one per fight. This gives the full distribution: ~28% upsets (f1/underdog wins), ~72% favorite wins. The model may bet on f1 (underdog) OR f2 (favorite) based on its prediction. Edge is always computed relative to the fighter being bet on.

### Edge Calculation

Edge is computed **relative to the fighter being bet on**, not always from f1's perspective:

```python
direction = +1 if bet_on_f1 else -1
edge = direction × (model_prob_f1 - market_prob_f1)
```

A positive edge means the model thinks the fighter being bet on is undervalued by the market.

### In-Sample Warning

`mar_4_v2` was trained with `--holdout-from-year 2025`. Therefore:
- **2025+:** true out-of-sample, ROI/accuracy numbers are valid
- **Pre-2025:** in-sample — the model has seen these fights during training. ROI numbers will be inflated (2023: ~+40%, 2024: ~+45% vs 2025: ~+11%).
- The backtest UI shows a yellow warning when the date range includes pre-2025 years.

---

## 6. 2025 Holdout Results (True Out-of-Sample)

| Strategy | Fights | Accuracy | ROI | Notes |
|---|---|---|---|---|
| All fights, no edge filter | 447 | 67.1% | +11.3% | Matches BETTING_RULES.md target |
| All fights, edge ≥ 10% | ~90 | ~53% | ~+25% | Fewer but higher value bets |
| Favorites (mkt ≥ 60%) | 151 | 63.6% | +25.1% | Shorter odds, more predictable |
| Underdogs blended, edge ≥ 0% | 133 | 77.4% | +94.4% | Suspicious — needs more scrutiny |
| Underdogs, no blend, edge ≥ 0% | — | lower | ~+35% | Blend meaningfully helps |

> **On the underdog numbers:** 133 fights at 77.4% accuracy is a small sample and the result reflects some of the model's in-period consistency rather than a stable edge. Do not extrapolate directly to future bet sizing.

---

## 7. What's Still Uncertain

1. **Walk-forward validity for pre-2025 years.** The current `training_data.csv` features were computed with the full current DB, including post-fight career updates. A true walk-forward would recompute features using only data available before each fight date. This is the intended next step.

2. **Odds source fidelity.** The DB closing odds come from BestFightOdds scraped retroactively. The opening/closing distinction may not always be reliable for very old fights. The `is_closing_line` flag should be treated as approximate for pre-2020 data.

3. **Underdog blend ROI stability.** The 94%+ ROI on 2025 underdog bets with blending is based on ~133 fights over one year. The model correctly identified several high-profile upsets (Merab, Strickland, etc.) but the sample is too small for confident Kelly sizing.

4. **No Kelly sizing yet.** The backtest engine only supports flat betting. Quarter-Kelly requires a sequential bankroll simulation that accounts for changing stake sizes as the bankroll grows. This is planned.

5. **No opening-line backtest.** The current engine uses closing odds only. Opening-line backtests would measure CLV (closing line value) — whether the model could beat the market before late-money moves the line.
