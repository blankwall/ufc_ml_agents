# Backtesting Skill

## What This Is

The backtest pipeline evaluates the `mar_4_v2` model against real historical odds to measure ROI, calibration, and strategy performance. It is a **two-step process: generate results CSV → analyze with bucket_analysis**.

---

## Step 1 — Generate a Results CSV

`backtest/backtest_2025.py` is the primary backtest runner. It:
- Loads odds from a canonical CSV (`backtest/odds/ufc_2025_odds.csv` or generated `backtest/odds/ufc_2026_odds.csv`)
- Resolves fighter names to DB IDs (with `_NAME_FIXES` alias dict for BFO→DB mismatches)
- Runs **in-process symmetric predictions**: scores both `(A vs B)` and `(B vs A)`, averages → `P_sym`
- Applies underdog blend (if enabled in config: `UNDERDOG_BLEND=False` by default)
- Runs each fight through `should_bet()` against `backtest_config.json` thresholds
- Writes one row per fight to a results CSV

```bash
# Standard run — 2025 season
python backtest/backtest_2025.py --odds backtest/odds/ufc_2025_odds.csv --model mar_4_v2

# With explicit config
python backtest/backtest_2025.py --odds backtest/odds/ufc_2025_odds.csv --config backtest/backtest_config.json

# 2026 season (same script, point at the 2026 odds CSV with a future cutoff)
python backtest/backtest_2025.py \
  --odds backtest/odds/ufc_2026_odds.csv \
  --cutoff 2027-01-01 \
  --quiet
```

### Results CSV schema

`date, fighter1, fighter2, odds1, odds2, prob1, prob2, pick, pick_odds, pick_prob, ev1, ev2, winner, pick_correct, actual_pnl, bet, skip_reason, error, female`

Key columns:
- `pick_prob` — model's probability for its chosen fighter (always the higher-prob side)
- `pick_odds` — American odds for the picked fighter
- `actual_pnl` — profit/loss per 1 unit if bet=True (e.g. +1.98, -1.0)
- `pick_correct` — True/False outcome
- `bet` — whether the fight passed all config filters
- `skip_reason` — why it was skipped (e.g. `"favorite confidence (58.4% < 65.0%)"`, `"underdog cap (310 >= 300)"`, `"min_fights"`, `"female"`)
- `female` — True for Women's weight classes

### Existing result files

| File | Period | Fights | Notes |
|---|---|---|---|
| `backtest/backtest_2025_results.csv` | 2025 | ~359 | True out-of-sample for `mar_4_v2` |
| `backtest/backtest_2026_results.csv` | 2026 | ~136 | Growing as events are scraped |

---

## Step 2 — Bucket Analysis

`backtest/bucket_analysis.py` is the analytical layer. It reads the results CSV (optionally filtered to a `bets.txt` file of actually-placed bets) and produces **5 analysis sections**.

```bash
# Full analysis — all 5 sections, auto-loads config/betting_config.json for weighted ROI
python backtest/bucket_analysis.py --results backtest/backtest_2025_results.csv
python backtest/bucket_analysis.py --results backtest/backtest_2026_results.csv

# Filter to only fights that were actually bet on
python backtest/bucket_analysis.py \
  --results backtest/backtest_2025_results.csv \
  --bets backtest/bets_2025.txt

python backtest/bucket_analysis.py \
  --results backtest/backtest_2026_results.csv \
  --bets backtest/bets.txt

# Single section
python backtest/bucket_analysis.py --results backtest/backtest_2025_results.csv --section edge
# Sections: buckets | edge | confidence | skip_reasons | weighted
```

### Section 1 — ODDS BUCKET BREAKDOWN (`--section buckets`)

Groups bets by **market odds of the picked fighter**:

| Bucket key | Odds range | Implied prob |
|---|---|---|
| `-400` | < −350 | 78–99% |
| `-300` | −350 to −250 | 71–78% |
| `-200` | −250 to −150 | 60–71% |
| `+200` | −150 to +250 | 29–60% |
| `+300` | +250 to +350 | 22–29% |
| `+400` | > +350 | < 22% |

Shows per bucket: N, W, L, WinRate, Profit (units), ROI%, AvgEdge, AvgConf — with M/F gender split. Best ROI bucket is highlighted. This is the primary diagnostic for which price range the model is profitable in.

### Section 2 — EDGE-TIER BREAKDOWN (`--section edge`)

Groups by **model edge** (pick_prob − market implied prob):
- `0–5%`, `5–10%`, `10–15%`, `15%+`

Same stats as Section 1 + gender split. Answers: does more edge actually mean more ROI?

### Section 3 — CONFIDENCE SCORE BANDS (`--section confidence`)

Uses `backtest/confidence_profile.py` to divide all `pick_prob` values into **10 equal-size decile buckets (score 1–10)**. Built from both `backtest_2025_results.csv` and `backtest_2026_results.csv` combined. Shows `AvgPred` vs actual `WinRate` and the gap (calibration error in pp). Score 10 = highest confidence decile.

This is the canonical calibration report. The `/api/predict` endpoint calls `describe_confidence()` from `confidence_profile.py` to attach a score 1–10 badge to every prediction on the events UI.

### Section 4 — SKIP REASON BREAKDOWN (`--section skip_reasons`)

Only shown when no `--bets` filter is active. Tallies every skip code, collapsed into canonical categories:
- `favorite confidence` — model prob below `confidence_favorite` threshold
- `favorite cap` — odds shorter than `favorite_odds_cap` (e.g. -400 when cap is -300)
- `underdog confidence` — model prob below `confidence_underdog` threshold
- `underdog cap` — odds longer than `underdog_odds_cap` (e.g. +400 when cap is +300)
- `underdog edge` — edge below `edge_underdog` threshold
- `min_fights` — one fighter has fewer than `min_fights` DB appearances before the fight date
- `female` — fight skipped because `female=false` in config

Use this to tune `backtest_config.json` thresholds. If `favorite confidence` is skipping fights that are winning, lower the threshold.

### Section 5 — WEIGHTED ROI (`--section weighted`)

Reads `config/betting_config.json` → `edge_buckets` and applies variable sizing:

| Edge range | Multiplier |
|---|---|
| 0–5% | skip |
| 5–10% | 1.0× |
| 10–20% | 1.5× |
| 20%+ | 2.0× |

WMMA fights: cap multiplier to 1.0× and require ≥10% edge (from `wmma_rules` in config).

Shows per tier: Mult, N/W/L, WinRate, Staked $, Profit $, ROI%, AvgEdge. Totals row shows **WEIGHTED TOTAL** vs **FLAT $100 TOTAL** side-by-side with lift in pp. This is the most realistic P&L projection — what you'd actually make with the sizing rules.

---

## Step 3 — Optimize Config (optional)

`backtest/optimize_config.py` grid-searches parameter combinations to maximize P&L. Pure pandas — no model inference, runs in seconds on existing results CSVs. Writes generated `backtest/optimize_results.csv` locally; that file is ignored and should not be committed.

```bash
python backtest/optimize_config.py --results backtest/backtest_2026_results.csv --top 20 --sort-by roi
```

Parameters tuned: `edge_underdog`, `confidence_favorite`, `confidence_underdog`, `favorite_odds_cap`, `underdog_odds_cap`, `female` (True/False).

---

## Step 4 — ELO Analysis (optional sidecar analysis)

`backtest/elo_analysis.py` post-processes existing backtest result CSVs with Sergey sidecar pre-fight ELO. It does **not** rerun model inference. It joins by `main_fight_id` when present, otherwise falls back to date + alias-normalized fighter pair for legacy CSVs.

```bash
python backtest/elo_analysis.py --results backtest/backtest_2026_results.csv
python backtest/elo_analysis.py --results backtest/backtest_2025_results.csv --section low_confidence

# Optional row-level export for notebooks/manual review
python backtest/elo_analysis.py \
  --results backtest/backtest_2026_results.csv \
  --write-enriched backtest/elo_2026_enriched.csv
```

Sections: `coverage`, `pick_diff`, `agreement`, `low_confidence`, `pure_elo`, `rules`, `all`.

Key questions answered:
- Does the model perform differently when its pick has higher/lower ELO?
- Which 50-65% confidence picks are upgraded by strong ELO support?
- How does pure "bet higher ELO" perform by ELO-gap bucket?
- Are current skip rules rejecting low-confidence but ELO-supported winners?

Requires `data/enrichment/sergey_sidecar.sqlite` with `fight_identity_map` built by `scripts/build_sergey_fight_map.py`.

---

## Step 5 — Context Pool (optional evidence retrieval DB)

`backtest/build_context_pool.py` materializes ELO-enriched 2025/2026 backtest rows into a generated SQLite database for fast historical pattern, similar-fight, and agent evidence queries.

```bash
python backtest/build_context_pool.py
# outputs data/enrichment/context_pool.sqlite

# Print nearest historical rows by model confidence and ELO delta
python backtest/build_context_pool.py --similar-elo 0.61 120 --limit 10

# Print agent-facing schema docs and example SQL
python backtest/context_schema.py
```

Generated tables:

| Table | Purpose |
|---|---|
| `backtest_fight_pool` | One row per backtested fight with model result, market edge, current bet/skip decision, and ELO context |
| `pattern_stats` | Reusable aggregate patterns like `skip_50_65_elo_50_plus`, `model_pick_higher_elo`, and underdog/ELO splits. Scoring uses graded rows only and separately reports pending/ungraded matches. |
| `evidence_items` | Agent-ready target-level evidence rows with an evidence role, summary, source pointer, and JSON payload. Roles include `target`, `context_metric`, `aggregate_pattern`, and `audit_detail`. Evidence types include `trait_delta` when `trait_snapshots.sqlite` is available. |
| `metadata` | Source files, sidecar path, trait sidecar path, schema version, creation timestamp |

Generated views:

| View | Purpose |
|---|---|
| `v_context_targets` | Compact fight-level target rows for finding candidates to inspect |
| `v_pattern_evidence` | Pattern-stat evidence joined to target fight fields |
| `v_recent_fight_evidence` | Recent-fight audit rows joined to target fight fields |
| `v_agent_packet_evidence` | All target-level evidence rows joined to target fight fields for packet-style retrieval |

This is the deterministic evidence pool for future context packets and LLM review. It is generated from existing backtest CSVs plus Sergey sidecar ELO; it should not be edited manually.

When `data/enrichment/trait_snapshots.sqlite` exists, the pool also materializes one `trait_delta` evidence row per joinable backtest pick. Current baseline:

```text
schema_version: 8
trait_delta evidence rows: 411
```

Schema v8 keeps provenance explicit for agent audits:

- `row_num` is the physical CSV line number in `source_results` (header is line 1).
- `source_row_key` is `{source_results}:{row_num}` for stable citations.
- Optional odds-source fields (`odds_source_file`, `odds_source_line`, `odds_source_type`, `odds_source_row`, `bookmaker`, `odds_timestamp`, `odds_is_opening_line`, `odds_is_closing_line`, `source_event_id`, `source_url`, `scraped_at`) are preserved when present in the odds input. Legacy 2025/2026 scalar CSVs mostly leave these null.

Inspect trait evidence:

```sql
SELECT fight_pool_id, date, pick, summary, data_json
FROM v_agent_packet_evidence
WHERE evidence_type = 'trait_delta'
ORDER BY date, fight_pool_id
LIMIT 20;
```

The pool also includes opponent-quality v0 fields derived point-in-time from Sergey fight history:

```text
pick_avg_prior_opponent_elo
opponent_avg_prior_opponent_elo
pick_recent3_prior_opponent_elo
opponent_recent3_prior_opponent_elo
pick_best_win_opponent_elo
opponent_best_win_opponent_elo
pick_opponent_quality_diff
pick_current_vs_peak_decline
opponent_current_vs_peak_decline
pick_decline_diff
pick_recent_fights_json
opponent_recent_fights_json
market_implied_prob
elo_implied_prob
model_minus_elo_prob
market_minus_elo_prob
model_market_elo_triangle
```

For a target fight on date D, these fields use fights before D; same-date fights are not allowed to leak into each other.
The recent-fight JSON fields store the actual last three prior fights with opponent name, opponent ELO, result, method, date, and fight ID; packet output prints these under the opponent-quality section.

---

## Step 6 — Single-Fight Context Packet (optional report)

`backtest/context_packet.py` generates a deterministic JSON evidence packet for one fight from `context_pool.sqlite`. It attaches model/market result, ELO context, matching aggregate patterns, support/risk flags, and nearest historical examples.

```bash
python backtest/context_packet.py \
  --fighter1 "Dominick Reyes" \
  --fighter2 "Johnny Walker" \
  --date 2026-04-11

# JSON-only for future LLM/tool consumption
python backtest/context_packet.py \
  --fighter1 "Dominick Reyes" \
  --fighter2 "Johnny Walker" \
  --date 2026-04-11 \
  --json-only

# Audit the exact graded rows behind the selected pattern_score_v0 source pattern
python backtest/context_packet.py \
  --fighter1 "Dominick Reyes" \
  --fighter2 "Johnny Walker" \
  --date 2026-04-11 \
  --expand-source-pattern

# Include current/future pending rows in that expansion
python backtest/context_packet.py \
  --fighter1 "Dominick Reyes" \
  --fighter2 "Johnny Walker" \
  --date 2026-04-11 \
  --expand-source-pattern \
  --include-pending
```

Run `python backtest/build_context_pool.py` first if `data/enrichment/context_pool.sqlite` is missing or stale.

The packet JSON/human summary includes `trait_deltas_v0` when `trait_delta` evidence exists in `evidence_items`. These deltas are context only; the aggregate `pattern_score_v0` remains ELO-pattern evidence and reports traits separately.

Project subagent for this workflow:

```text
.claude/agents/ufc-fight-context-analyst.md
```

Use it when the task is "tell me about this fight" style analysis. It is instructed to:
- start with structured MCP fight tools (`search_context_targets` -> `get_fight_basics` -> `get_fight_model_market` -> `get_fight_historical_patterns` -> `get_fight_style_flags`)
- treat `get_fight_elo_context` and `get_fighter_elo_history` as first-class ELO tools rather than relying on one monolithic packet
- treat those focused tools as the primary first-pass evidence surface, not a single monolithic packet
- drill into read-only SQL / validation only when the structured tools suggest uncertainty or a promising line of inquiry

`backtest/context_agent_review.py` is the review harness on top of `v_agent_packet_evidence`.

```bash
# Deterministic cited scaffold
python backtest/context_agent_review.py \
  --fighter1 "Dominick Reyes" \
  --fighter2 "Johnny Walker" \
  --date 2026-04-11

# Optional LLM-backed evidence-only cited reasoning
python backtest/context_agent_review.py \
  --fighter1 "Dominick Reyes" \
  --fighter2 "Johnny Walker" \
  --date 2026-04-11 \
  --llm \
  --json-only
```

The default output groups evidence deterministically into support/caution/context/audit buckets. `--llm` uses that deterministic scaffold plus raw evidence rows as prompt input and validates that every generated claim cites known `evidence_id` values from the packet. It is still evidence-only and should not emit a recommendation.

Evidence roles:
- `matching_patterns` are decision support because they come from aggregate `pattern_stats`.
- `nearest_historical_examples` are qualitative sanity checks only; do not treat small nearest-neighbor samples as backtested rules.

Packets also include `pattern_score_v0`, a data-derived score from the strongest specific applicable aggregate pattern. Current thresholds:
- strong: `N>=50`, win rate `>=70%`, ROI `>=15%` → score 8
- moderate: `N>=30`, win rate `>=65%`, ROI `>=5%` → score 7
- mild: `N>=20`, win rate `>=60%`, ROI `>0%` → score 6
- negative/unprofitable: `N>=20`, ROI `<=0%` → score 4

`N` means graded outcomes only. Packet output also shows pending/ungraded matches separately, e.g. `N=64 graded + 4 pending`.
`--expand-source-pattern` is graded-only by default so the printed audit rows match the scored sample.

`pattern_score_v0` is empirical evidence only, not a final betting action. Labels such as `strong_empirical_support` and `negative_empirical_signal` describe evidence strength for a downstream analyst/agent.

---

## Step 7 — Batch Context Evidence Watchlist (optional report)

`backtest/context_candidates.py` scans pending/ungraded skipped picks in `context_pool.sqlite`, scores each with the same `pattern_score_v0` logic, and prints a ranked watchlist of spots where historical aggregate evidence may justify deeper review. It is not a betting recommendation or rule engine.

```bash
# Current/future skipped picks only, min empirical score 7 by default
python backtest/context_candidates.py

# Strong evidence only
python backtest/context_candidates.py --min-score 8

# Include older pending rows such as no-contests or unresolved historical rows
python backtest/context_candidates.py --include-past-pending

# JSON for downstream tooling
python backtest/context_candidates.py --json-only
```

This report does not change betting rules. It is a deterministic audit/watchlist for skipped model picks that deserve context review.

---

## Step 8 — Independent Context Pipeline Validation

`backtest/validate_context_pipeline.py` stress-tests `pattern_score_v0` on historical graded rows without wiring the context layer into the app or betting rules. Default mode is leave-one-out: each target fight is scored using aggregate pattern evidence that excludes that target row.

```bash
# Validate all graded historical rows with leave-one-out aggregates
python backtest/validate_context_pipeline.py

# Validate only rows the current betting rules skipped
python backtest/validate_context_pipeline.py --skips-only

# Focus on stronger scores
python backtest/validate_context_pipeline.py --min-score 7

# More conservative chronological evidence: only prior fights can support a target
python backtest/validate_context_pipeline.py --mode temporal
```

The report groups results by score, source pattern, season, bet/skip status, edge bucket, and odds bucket. Use this as the promotion gate before exposing context scores in `/api/predict`, `/events`, or any LLM review layer.

---

## Step 8b — Combined Evidence Temporal Validation

`backtest/validate_combined_evidence.py` now supports explicit target-date windows and multi-mode comparison so you can isolate later holdout slices without changing the evidence families themselves.

```bash
# Compare leave-one-out / temporal / in-sample on a holdout window
python backtest/validate_combined_evidence.py \
  --compare-modes \
  --dedupe-main-fight \
  --min-date 2026-01-01

# Audit the cardio-supported family on a holdout window
python backtest/validate_combined_evidence.py \
  --dedupe-main-fight \
  --min-date 2026-01-01 \
  --audit-rule golden_elo_not_expensive_plus_cardio
```

Audit output now includes:
- source-row verification status
- odds provenance presence vs legacy-missing
- temporal half split
- temporal quartile split

This is useful for checking whether the cardio-supported ELO family stays intact across later chronological slices instead of only looking good in pooled historical results.

---

## Step 9 — Trait Snapshots (independent Phase 3 sidecar)

`backtest/build_trait_snapshots.py` builds the first point-in-time fighter trait layer from UFCStats fight-total JSON plus Sergey identity mapping. It writes a generated SQLite sidecar and does not mutate the main DB.

```bash
.venv/bin/python backtest/build_trait_snapshots.py
# outputs data/enrichment/trait_snapshots.sqlite

.venv/bin/python backtest/validate_trait_snapshots.py
```

Generated objects:

| Object | Purpose |
|---|---|
| `fighter_trait_snapshots` | One row per fighter before a target fight, using only prior fights |
| `v_trait_pair_deltas` | Fighter-vs-opponent trait deltas for matchup/backtest inspection |

Current v0 trait scores:

```text
experience_score
recent_form_score
cardio_score
durability_risk_score
defensive_exposure_score
offensive_control_score
anti_control_score
scramble_score
striking_pressure_score
striking_efficiency_score
grappling_threat_score
finishing_threat_score
variance_score
trait_confidence
```

Important limitations:
- `round_by_round` is currently empty in `fight_stats`, so `cardio_score` is a late-fight outcome/control proxy, not a true round-to-round pace-retention metric.
- Higher risk scores mean more risk; higher ability scores mean more evidence of that ability.
- Traits are packet evidence only until validated; do not wire them into live betting logic.

Current coverage baseline:

```text
snapshots: 17,310
with prior fight history: 14,563 (84.1%)
with 3+ prior fights: 10,434 (60.3%)
with Sergey identity: 15,224 (87.9%)
with cardio proxy: 9,459 (54.6%)
```

Current Sergey assessment sanity checks:

```text
pace_retention vs cardio_score                  n=77   corr=+0.454
distance_control vs striking_efficiency_score   n=154  corr=+0.244
fight_iq vs recent_form_score                   n=145  corr=+0.167
scramble vs scramble_score                      n=152  corr=+0.186
scramble vs anti_control_score                  n=152  corr=+0.145
hittability vs defensive_exposure_score         n=138  corr=-0.135
hittability vs defensive_responsibility_score   n=138  corr=+0.135
```

Treat these as formula-validation evidence, not betting rules. Cardio/late-fight and striking-efficiency are the strongest first-pass alignments. Anti-control was revised in `trait_v0_1_stats_totals`; defensive exposure remains ambiguous and should be interpreted with its inverse responsibility score in mind.

Combined evidence validation:

```bash
.venv/bin/python backtest/validate_combined_evidence.py
.venv/bin/python backtest/validate_combined_evidence.py --mode temporal
```

Current preservation targets:

```text
golden_elo_plus_trait_support                  N=33  W-L=29-4  WR=87.9%  ROI=+29.7%
golden_elo_not_expensive_plus_trait_support    N=22  W-L=19-3  WR=86.4%  ROI=+39.5%
golden_elo_not_expensive_plus_cardio           N=14  W-L=14-0  WR=100.0% ROI=+61.3%
golden_elo_not_expensive_non_cardio_trait      N=8   W-L=5-3   WR=62.5%  ROI=+1.4%
```

These are empirical evidence families only. The current useful split is cardio/late-fight support, not generic trait support.

Artifact audit:

```bash
.venv/bin/python backtest/validate_combined_evidence.py \
  --audit-rule golden_elo_not_expensive_plus_trait_support \
  --show-rows
```

Current audit notes:
- Unique main fights: `N=21`, `W-L=18-3`, `ROI=+38.2%`; one duplicate line row exists for Alexander Volkanovski vs Diego Lopes.
- By season: 2025 `11-2`, 2026 `8-1`.
- By gender: men `17-3`, WMMA `2-0`.
- Best price band: `-200` to `-101` went `12-0`; `-299` to `-201` was weaker at `7-2`; the lone `>= +200` row lost.
- Bookmaker coverage is incomplete in the main DB for 8/22 rows, so line-source rigor remains a promotion blocker.

---

## Step 10 — Deterministic Agent Evidence Review

`backtest/context_agent_review.py` is the non-LLM harness for the future agent-review flow. It consumes `v_agent_packet_evidence`, groups rows into support/caution/context/audit sections, and cites `evidence_id` values.

```bash
.venv/bin/python backtest/context_agent_review.py \
  --fighter1 "Dominick Reyes" \
  --fighter2 "Johnny Walker" \
  --date 2026-04-11

# Or by context pool row
.venv/bin/python backtest/context_agent_review.py --fight-pool-id 443
```

Rules:
- It does not create probabilities, stakes, or bet/skip recommendations.
- Every claim should cite an `evidence_id` such as `[E4861]`.
- Future LLM review should consume this cited evidence, not raw tables alone.

---

## Golden Context Baseline — `skip_50_65_elo_50_plus`

This is the current strongest independent context signal and should be preserved while adding new signals.

Definition:

```text
current rules skipped the pick
AND 50% <= model pick probability < 65%
AND model pick has at least +50 pre-fight Sergey ELO over opponent
```

Current expected result:

```text
N=64 graded + 4 pending
W-L=51-13
WR=79.7%
ROI=+18.4%
score=8/10
```

Reproduce it:

```bash
.venv/bin/python backtest/build_context_pool.py

.venv/bin/python backtest/context_packet.py \
  --fighter1 "Dominick Reyes" \
  --fighter2 "Johnny Walker" \
  --date 2026-04-11 \
  --expand-source-pattern

.venv/bin/python backtest/validate_context_pipeline.py --skips-only --min-score 7
```

Expected packet line:

```text
skip_50_65_elo_50_plus  N=64 graded + 4 pending  W-L=51-13  WR=79.7%  ROI=+18.4%
```

Expected validation line:

```text
skip_50_65_elo_50_plus | 64 | 51-13 | 79.7% | +11.76 | +18.4%
```

Do not allow broad `model_pick_higher_elo` to upgrade skipped fights by itself; validation showed the skipped-fight subset was not strong enough. For skipped upgrades, prefer specific, audited opportunity patterns.

Opponent-quality v0 currently supports packet context but is not a standalone upgrade rule:

```text
skip_50_65_elo_50_plus_opp_quality_support  N=35  W-L=28-7  WR=80.0%  ROI=+15.5%
skip_50_65_elo_50_plus_opp_quality_against  N=29  W-L=23-6  WR=79.3%  ROI=+21.9%
```

Model/market/ELO triangle fields are also included. ELO-implied probability uses:

```text
elo_implied_prob = 1 / (1 + 10 ** (-pick_elo_diff / 400))
```

Current pricing refinement to preserve/test:

```text
skip_50_65_elo_50_plus_not_expensive
N=44
W-L=35-9
WR=79.5%
ROI=+27.6%
```

Walk-forward checks for the same rule:

```text
after prior N>=10: N=34, W-L=26-8, ROI=+24.5%
after prior N>=20: N=24, W-L=18-6, ROI=+22.2%
after prior N>=30 / prior rule grade>=7: N=12, W-L=10-2, ROI=+39.6%
```

Definition:

```text
current rules skipped the pick
AND 50% <= model pick probability < 65%
AND pick_elo_diff >= +50
AND pick_odds > -300
```

The simple `market_minus_elo_prob <= -10%` split was weaker:

```text
N=14
W-L=8-6
WR=57.1%
ROI=+4.2%
```

So do not use that market-gap split as a standalone upgrade rule yet.

---

## Config Files

### `backtest/backtest_config.json` — per-run backtest parameters
```json
{
  "model": "mar_4_v2",
  "cutoff_date": "2027-01-01",
  "edge_min": 0.05,
  "edge_underdog": 0.1,
  "confidence_favorite": 0.65,
  "confidence_underdog": 0.53,
  "favorite_odds_cap": -300,
  "underdog_odds_cap": 300,
  "min_fights": 2,
  "female": true,
  "underdog_blend": false
}
```

### `config/betting_config.json` — site-facing config (also used by bucket_analysis weighted section)
```json
{
  "filters": {
    "edge_min": 0.04,
    "favorite_confidence_min": 0.65,
    "underdog_confidence_min": 0.53,
    "favorite_odds_cap": -300,
    "underdog_odds_cap": 300,
    "min_fights": 2
  },
  "edge_buckets": [
    { "min_edge": 0.00, "max_edge": 0.05, "action": "skip" },
    { "min_edge": 0.05, "max_edge": 0.10, "multiplier": 1.0 },
    { "min_edge": 0.10, "max_edge": 0.20, "multiplier": 1.5 },
    { "min_edge": 0.20, "max_edge": 1.00, "multiplier": 2.0 }
  ],
  "betting": { "base_unit": 100 },
  "wmma_rules": { "enabled": true, "min_edge": 0.10, "max_multiplier": 1.0 }
}
```

---

## Bets Files

`backtest/bets.txt` (2026) and `backtest/bets_2025.txt` (2025) contain manually recorded real bets placed. Each line:

```
[YYYY-MM-DD] Fighter Name  @  +165  prob=66.1%  ev=+0.75  WON  (+1.64)  vs Opponent Name
```

When `--bets` is passed to `bucket_analysis.py`, only rows where `(date, normalized_fighter_name)` appears in the bets file are analyzed. Use this to see how actually-placed bets performed vs all model picks.

---

## 2026 Backtest Data

2026 uses the same `backtest_2025.py` script — no separate live runner. `backtest/odds/ufc_2026_odds.csv` is a **generated file** — build it first with `rebuild_2026_odds.py`, which merges all 2026 data sources (BFO CSVs, user event JSONs, DB results).

```bash
# Step 0 — build the 2026 odds input file (must run before backtest)
python backtest/rebuild_2026_odds.py
# outputs backtest/odds/ufc_2026_odds.csv

# Step 1 — run the backtest
python backtest/backtest_2025.py \
  --odds backtest/odds/ufc_2026_odds.csv \
  --cutoff 2027-01-01 \
  --quiet
```

| File | Purpose |
|---|---|
| `backtest/rebuild_2026_odds.py` | Builds `backtest/odds/ufc_2026_odds.csv` from all 2026 sources |
| `backtest/odds/ufc_2026_odds.csv` | Generated odds input (not committed) |
| `data/future_fight_odds/ufc*.csv` | Per-event BFO odds CSVs |
| `data/future_fight_odds/outcomes.csv` | UFC Stats results |
| `data/user_events/*.json` | Events added via `POST /api/add-event` |
| `backtest/backtest_2026_results.csv` | Output results CSV |

---

## Cleaned Backtest Layout

Canonical backtesting now lives in a small active surface:

| Path | Status |
|---|---|
| `backtest/backtest_2025.py` | Active formal runner for 2025 and 2026 |
| `backtest/rebuild_2026_odds.py` | Active 2026 odds-input generator |
| `backtest/bucket_analysis.py` | Active analyzer |
| `backtest/confidence_profile.py` | Active confidence scoring helper |
| `backtest/optimize_config.py` | Optional active grid-search helper |
| `backtest/elo_analysis.py` | Optional Sergey sidecar ELO analyzer |
| `backtest/odds/ufc_2025_odds.csv` | Tracked canonical 2025 odds input |
| `backtest/odds/ufc_2026_odds.csv` | Generated/ignored 2026 odds input |
| `backtest/archive/backtest_live.py` | Archived legacy prototype; do not use for formal backtesting |
| `backtest/archive/backtest_underdog.py` | Archived underdog-model research |

Generated scratch artifacts are intentionally removed/ignored:
- `backtest/backtest_results.csv`
- `backtest/optimize_results.csv`

---

## `should_bet()` Logic (in `backtest_2025.py`)

The bet filter applied per fight:

```python
# Favorites (pick_odds < 0)
if pick_odds <= favorite_odds_cap:  skip ("favorite cap")
if pick_prob < confidence_favorite: skip ("favorite confidence")
if prob_edge < edge_min:            skip ("edge")

# Underdogs (pick_odds > 0)
if pick_odds >= underdog_odds_cap:  skip ("underdog cap")
if pick_prob < confidence_underdog: skip ("underdog confidence")
if prob_edge < edge_underdog:       skip ("underdog edge")
```

Where `prob_edge = pick_prob - market_implied_prob(pick_odds)`.

---

## Common Workflows

**Re-run 2025 backtest with new config thresholds:**
```bash
python backtest/backtest_2025.py --odds backtest/odds/ufc_2025_odds.csv --model mar_4_v2
python backtest/bucket_analysis.py --results backtest/backtest_2025_results.csv
```

**Run 2026 backtest (rebuild odds file first):**
```bash
python backtest/rebuild_2026_odds.py
python backtest/backtest_2025.py \
  --odds backtest/odds/ufc_2026_odds.csv \
  --cutoff 2027-01-01 \
  --quiet
python backtest/bucket_analysis.py --results backtest/backtest_2026_results.csv --bets backtest/bets.txt
```

**Find optimal config parameters:**
```bash
python backtest/optimize_config.py --results backtest/backtest_2026_results.csv --top 20
```

**Check single analysis section quickly:**
```bash
python backtest/bucket_analysis.py --results backtest/backtest_2025_results.csv --section skip_reasons
```
