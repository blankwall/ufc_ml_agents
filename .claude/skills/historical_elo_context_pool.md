# Historical ELO / Context Pool Analysis Skill

## What This Is

This skill performs **read-only, evidence-first historical analysis** over the repo's ELO/context system.

Primary data source:
- `data/enrichment/context_pool.sqlite`

Related source-of-truth files:
- `fastapi_app/services/predict_context_service.py`
- `backtest/deterministic_signal_filter.py`
- `backtest/validate_context_pipeline.py`
- `backtest/validate_combined_evidence.py`
- `tests/test_predict_context_service.py`
- `tests/test_validate_combined_evidence.py`
- `tests/test_elo_review_gate.py`
- `tests/test_golden_elo_service.py`

Use this skill when you need to analyze:
- positive Golden tiers
- negative ELO-against tiers
- favorite caution buckets
- neutral favorite / neutral underdog buckets
- trait/cardio offsets
- tier robustness and sample-size quality

---

## Core Repo Facts

Current generated context pool facts to verify before analysis:
- read `metadata` first
- confirm schema version, source files, graded vs pending rows
- confirm trait evidence coverage before making trait-based claims

Important label conventions:
- `golden_elo_not_expensive` is the app-facing Tier 1 Golden reopen label
- `golden_elo_plus_trait_support` is Tier 2
- `golden_elo_plus_cardio` is Tier 3
- negative-ELO, favorite-fade, neutral, and trait-offset labels may be app/service labels instead of first-class `pattern_stats` rows

Primary trait support currently means:
- `cardio_score_diff >= 10`, or
- `striking_efficiency_score_diff >= 10`, or
- `defensive_exposure_score_diff <= -10`

Cardio support for the highest positive tier means:
- `cardio_score_diff >= 10`

Do **not** assume every app-facing label exists in `pattern_stats`. Some cohorts must be derived ad hoc from `backtest_fight_pool` plus `trait_delta` evidence.

---

## Analysis Workflow

### Step 1 — Sanity-check the pool

Always report:
- schema version
- source results files
- total rows
- graded rows
- pending/ungraded rows
- trait evidence coverage

Never mix pending rows into ROI claims.

### Step 2 — Determine the ask type

1. **Explicit pattern analysis**
   - use `pattern_stats` first
2. **Derived bucket analysis**
   - compute from `backtest_fight_pool`
   - join `evidence_items` for `trait_delta`
3. **Composite-family validation**
   - use `validate_combined_evidence.py`
   - prefer leave-one-out or temporal mode

### Step 3 — Tier families to support

#### Positive Golden tiers

Treat these as the main positive ELO support family:
- `golden_elo_not_expensive`
- `golden_elo_plus_trait_support`
- `golden_elo_plus_cardio`
- related `pattern_stats` families like:
  - `skip_50_65_elo_50_plus`
  - `skip_50_65_elo_50_plus_not_expensive`
  - `skip_50_65_elo_100_plus`

#### Negative ELO-against tiers

Analyze:
- `model_pick_lower_elo`
- `underdog_elo_against`
- `bet_elo_against`
- derived favorite-vs-ELO buckets where `pick_odds < 0` and `pick_elo_diff <= -50`
- derived underdog negative-ELO buckets with and without trait/cardio offsets

#### Favorite caution buckets

Derive ad hoc when needed, such as:
- expensive favorites: `pick_odds <= -300`
- favorite with ELO disagreement: `pick_odds < 0 AND pick_elo_diff <= -50`
- stronger favorite disagreement: `pick_odds < 0 AND pick_elo_diff <= -100`
- favorite tax / over-ELO triangles using `model_market_elo_triangle`

#### Neutral favorite / neutral underdog buckets

Derive:
- neutral favorite: `pick_odds < 0 AND ABS(COALESCE(pick_elo_diff, 0)) < 50`
- neutral underdog: `pick_odds > 0 AND ABS(COALESCE(pick_elo_diff, 0)) < 50`

#### Trait/cardio offsets

Join `evidence_items` where `evidence_type = 'trait_delta'`.

Key offset families:
- trait offset in ELO-against cohort
- cardio-only offset in ELO-against cohort
- non-cardio trait support in Golden cohort
- trait caution inside otherwise positive ELO cohorts

Use current repo thresholds unless explicitly asked otherwise:
- cardio support: `cardio_score_diff >= 10`
- striking efficiency support: `striking_efficiency_score_diff >= 10`
- safer exposure support: `defensive_exposure_score_diff <= -10`
- cardio caution: `cardio_score_diff <= -10`
- exposure caution: `defensive_exposure_score_diff >= 10`

### Step 4 — Output format

Return concise sections:

1. **Question being answered**
2. **Data quality**
   - schema version
   - graded vs pending
   - whether derived or first-class patterns were used
3. **Tier table**
   - tier
   - N
   - W-L
   - WinRate
   - Profit
   - ROI
   - AvgConf
   - AvgEdge
   - AvgELODiff
4. **What is strongest**
5. **What is fragile / cautionary**
6. **Offsets and interactions**
7. **Robustness notes**
   - raw vs deduped if applicable
   - leave-one-out / temporal if applicable
8. **Next investigations**
   - concrete follow-up queries or cohorts

---

## Guardrails

- Never report ROI without sample size.
- Never mix graded and pending rows in the same ROI stat.
- Never present nearest-example rows as proof.
- Distinguish:
  - first-class `pattern_stats` cohorts
  - app aliases
  - ad hoc derived cohorts
- If a bucket is newly derived, say so explicitly.
- If `n < 20`, call it anecdotal.
- If `20 <= n < 50`, call it exploratory.
- If using composite families, prefer leave-one-out or temporal validation before making strong claims.
- If duplicate `main_fight_id` values may matter, show raw and deduped summaries.
- Do not frame outputs as betting advice.
- Do not claim a trait causes ROI. Trait signals are overlays, not proof of mechanism.
- Use percent points clearly for probability deltas and percentages for ROI/win rate.
- If odds/provenance fields are null or mixed, say so.

---

## Suggested Questions This Skill Should Answer

- "Show the full current tier map."
- "Rank Golden tiers by robustness, not just ROI."
- "Compare underdog ELO-support vs underdog ELO-against."
- "Do trait/cardio offsets improve negative ELO-against buckets?"
- "Break down favorite caution buckets and expensive favorite tax."
- "Compare neutral favorites and neutral underdogs."
- "Which current tier families are most likely misleading due to sample size?"
- "Validate combined Golden + trait families with leave-one-out and temporal modes."

---

## Reusable SQL / Query Playbook

### Pool sanity / coverage

```sql
SELECT key, value
FROM metadata
ORDER BY key;

SELECT
  COUNT(*) AS total_rows,
  SUM(CASE WHEN pick_correct IS NOT NULL THEN 1 ELSE 0 END) AS graded_rows,
  SUM(CASE WHEN pick_correct IS NULL THEN 1 ELSE 0 END) AS pending_rows
FROM backtest_fight_pool;
```

### First-class pattern scoreboard

```sql
SELECT
  pattern_name,
  sample_size AS n,
  wins,
  losses,
  ROUND(win_rate * 100, 1) AS win_rate_pct,
  ROUND(profit, 2) AS profit_u,
  ROUND(roi * 100, 1) AS roi_pct,
  ROUND(avg_confidence * 100, 1) AS avg_conf_pct,
  ROUND(avg_edge * 100, 1) AS avg_edge_pct,
  ROUND(avg_elo_diff, 1) AS avg_elo_diff
FROM pattern_stats
ORDER BY sample_size DESC, roi DESC;
```

### Canonical tier family summary from first-class + derived tiers

```sql
WITH base AS (
  SELECT
    p.id,
    p.date,
    p.fighter1,
    p.fighter2,
    p.pick,
    p.pick_prob,
    p.pick_odds,
    p.edge,
    p.bet,
    p.skip_reason,
    p.pick_correct,
    p.actual_pnl,
    p.pick_elo_diff,
    p.model_market_elo_triangle,
    json_extract(e.data_json, '$.deltas.cardio_score_diff') AS cardio_diff,
    json_extract(e.data_json, '$.deltas.striking_efficiency_score_diff') AS strike_eff_diff,
    json_extract(e.data_json, '$.deltas.defensive_exposure_score_diff') AS def_exp_diff
  FROM backtest_fight_pool p
  LEFT JOIN evidence_items e
    ON e.fight_pool_id = p.id
   AND e.evidence_type = 'trait_delta'
  WHERE p.pick_correct IS NOT NULL
),
tiers AS (
  SELECT *,
    CASE
      WHEN bet = 0
       AND pick_prob >= 0.50 AND pick_prob < 0.65
       AND pick_elo_diff >= 50
       AND pick_odds > -300
        THEN 'golden_elo_not_expensive'
      WHEN bet = 0
       AND pick_prob >= 0.50 AND pick_prob < 0.65
       AND pick_elo_diff >= 100
       AND (
         cardio_diff >= 10 OR strike_eff_diff >= 10 OR def_exp_diff <= -10
       )
       AND cardio_diff < 10
        THEN 'golden_elo_plus_trait_support'
      WHEN bet = 0
       AND pick_prob >= 0.50 AND pick_prob < 0.65
       AND pick_elo_diff >= 100
       AND (
         cardio_diff >= 10 OR strike_eff_diff >= 10 OR def_exp_diff <= -10
       )
       AND cardio_diff >= 10
        THEN 'golden_elo_plus_cardio'
      WHEN pick_odds > 0 AND pick_elo_diff > 0
        THEN 'underdog_elo_support'
      WHEN pick_odds > 0 AND pick_elo_diff < 0
        THEN 'underdog_elo_against'
      WHEN pick_odds < 0 AND pick_elo_diff <= -50
        THEN 'favorite_caution_elo_against'
      WHEN pick_odds < 0 AND ABS(COALESCE(pick_elo_diff, 0)) < 50
        THEN 'neutral_favorite'
      WHEN pick_odds > 0 AND ABS(COALESCE(pick_elo_diff, 0)) < 50
        THEN 'neutral_underdog'
    END AS tier
  FROM base
)
SELECT
  tier,
  COUNT(*) AS n,
  SUM(CASE WHEN pick_correct = 1 THEN 1 ELSE 0 END) AS wins,
  SUM(CASE WHEN pick_correct = 0 THEN 1 ELSE 0 END) AS losses,
  ROUND(AVG(CASE WHEN pick_correct IS NOT NULL THEN 1.0 * (pick_correct = 1) END) * 100, 1) AS win_rate_pct,
  ROUND(SUM(actual_pnl), 2) AS profit_u,
  ROUND(AVG(actual_pnl) * 100, 1) AS roi_pct,
  ROUND(AVG(pick_prob) * 100, 1) AS avg_conf_pct,
  ROUND(AVG(edge) * 100, 1) AS avg_edge_pct,
  ROUND(AVG(pick_elo_diff), 1) AS avg_elo_diff
FROM tiers
WHERE tier IS NOT NULL
GROUP BY tier
ORDER BY n DESC, roi_pct DESC;
```

### Trait/cardio offsets inside negative ELO-against buckets

```sql
WITH base AS (
  SELECT
    p.*,
    json_extract(e.data_json, '$.deltas.cardio_score_diff') AS cardio_diff,
    json_extract(e.data_json, '$.deltas.striking_efficiency_score_diff') AS strike_eff_diff,
    json_extract(e.data_json, '$.deltas.defensive_exposure_score_diff') AS def_exp_diff
  FROM backtest_fight_pool p
  LEFT JOIN evidence_items e
    ON e.fight_pool_id = p.id
   AND e.evidence_type = 'trait_delta'
  WHERE p.pick_correct IS NOT NULL
    AND p.pick_elo_diff < 0
)
SELECT
  CASE
    WHEN cardio_diff >= 10 THEN 'cardio_offset_elo_against'
    WHEN strike_eff_diff >= 10 OR def_exp_diff <= -10 THEN 'non_cardio_trait_offset_elo_against'
    ELSE 'plain_elo_against'
  END AS bucket,
  COUNT(*) AS n,
  SUM(CASE WHEN pick_correct = 1 THEN 1 ELSE 0 END) AS wins,
  ROUND(AVG(actual_pnl) * 100, 1) AS roi_pct,
  ROUND(AVG(pick_prob) * 100, 1) AS avg_conf_pct,
  ROUND(AVG(edge) * 100, 1) AS avg_edge_pct,
  ROUND(AVG(pick_elo_diff), 1) AS avg_elo_diff
FROM base
GROUP BY bucket
ORDER BY n DESC;
```

### Favorite caution buckets

```sql
SELECT
  CASE
    WHEN pick_odds <= -300 THEN 'expensive_favorite'
    WHEN pick_odds < 0 AND pick_elo_diff <= -100 THEN 'favorite_strong_elo_against'
    WHEN pick_odds < 0 AND pick_elo_diff <= -50 THEN 'favorite_moderate_elo_against'
    WHEN pick_odds < 0 AND model_market_elo_triangle = 'model_and_market_over_elo' THEN 'favorite_tax_over_elo'
    WHEN pick_odds < 0 AND ABS(COALESCE(pick_elo_diff, 0)) < 50 THEN 'neutral_favorite'
  END AS bucket,
  COUNT(*) AS n,
  SUM(CASE WHEN pick_correct = 1 THEN 1 ELSE 0 END) AS wins,
  ROUND(AVG(actual_pnl) * 100, 1) AS roi_pct,
  ROUND(AVG(edge) * 100, 1) AS avg_edge_pct
FROM backtest_fight_pool
WHERE pick_correct IS NOT NULL
  AND pick_odds < 0
GROUP BY bucket
HAVING bucket IS NOT NULL
ORDER BY n DESC;
```

### Audit rows for any tier

```sql
SELECT
  fight_pool_id,
  date,
  fighter1,
  fighter2,
  pick,
  current_decision,
  pick_prob,
  pick_odds,
  edge,
  pick_elo_diff,
  model_market_elo_triangle,
  pick_correct,
  actual_pnl
FROM v_context_targets
WHERE current_decision = 'skip'
  AND pick_prob >= 0.50
  AND pick_prob < 0.65
  AND pick_elo_diff >= 50
ORDER BY date DESC
LIMIT 25;
```

### Raw vs deduped by `main_fight_id`

```sql
WITH ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY COALESCE(CAST(main_fight_id AS TEXT), CAST(id AS TEXT))
      ORDER BY row_num, id
    ) AS rn
  FROM backtest_fight_pool
  WHERE pick_correct IS NOT NULL
)
SELECT
  COUNT(*) AS deduped_n,
  SUM(CASE WHEN pick_correct = 1 THEN 1 ELSE 0 END) AS wins,
  ROUND(AVG(actual_pnl) * 100, 1) AS roi_pct
FROM ranked
WHERE rn = 1;
```

---

## Validation Commands

Use these when the ask needs robustness checks:

```bash
python backtest/validate_context_pipeline.py --mode leave-one-out
python backtest/validate_context_pipeline.py --mode temporal
python backtest/validate_combined_evidence.py --mode leave-one-out
python backtest/validate_combined_evidence.py --mode temporal
```

Use service/tests as label definitions:
- `fastapi_app/services/predict_context_service.py`
- `backtest/deterministic_signal_filter.py`
- `tests/test_predict_context_service.py`
- `tests/test_validate_combined_evidence.py`
- `tests/test_elo_review_gate.py`
- `tests/test_golden_elo_service.py`

---

## Minimal Rollout

1. Add this file under `.claude/skills/`
2. Benchmark on:
   - full tier map
   - Golden tiers only
   - negative ELO-against only
   - favorite caution + neutral buckets
   - trait/cardio offsets inside ELO-against
3. Require every answer to:
   - report metadata first
   - separate graded from pending
   - label first-class vs derived tiers
   - show N alongside ROI
   - mention robustness mode when used
