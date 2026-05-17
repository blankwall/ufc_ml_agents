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
