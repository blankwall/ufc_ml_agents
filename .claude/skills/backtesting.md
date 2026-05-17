# Backtesting Skill

## What This Is

The backtest pipeline evaluates the `mar_4_v2` model against real historical odds to measure ROI, calibration, and strategy performance. It is a **two-step process: generate results CSV → analyze with bucket_analysis**.

---

## Step 1 — Generate a Results CSV

`backtest/backtest_2025.py` is the primary backtest runner. It:
- Loads odds from a CSV (exported from DB via `scripts/export_odds_from_db.py`)
- Resolves fighter names to DB IDs (with `_NAME_FIXES` alias dict for BFO→DB mismatches)
- Runs **in-process symmetric predictions**: scores both `(A vs B)` and `(B vs A)`, averages → `P_sym`
- Applies underdog blend (if enabled in config: `UNDERDOG_BLEND=False` by default)
- Runs each fight through `should_bet()` against `backtest_config.json` thresholds
- Writes one row per fight to a results CSV

```bash
# Standard run — 2025 season
python backtest/backtest_2025.py --odds data/odds/db_odds_2025.csv --model mar_4_v2

# With explicit config
python backtest/backtest_2025.py --odds data/odds/db_odds_2025.csv --config backtest/backtest_config.json

# 2026 live events (reads scraped BFO CSVs + outcomes.csv directly, no DB odds needed)
python backtest/backtest_live.py
python backtest/backtest_live.py --event "UFC 327" --edge 0.10 --quiet
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
| `backtest/backtest_results.csv` | 2026 | ~136 | Alias for current year results |

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

`backtest/optimize_config.py` grid-searches parameter combinations to maximize P&L. Pure pandas — no model inference, runs in seconds on existing results CSVs. Saves all combos to `backtest/optimize_results.csv`.

```bash
python backtest/optimize_config.py --results backtest/backtest_results.csv --top 20 --sort-by roi
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

## 2026 Live Backtest Data

| File | Purpose |
|---|---|
| `data/future_fight_odds/all_events.csv` | Scraped BFO odds (~60 rows, 9+ events) |
| `data/future_fight_odds/outcomes.csv` | UFC Stats results (~75 rows) |
| `data/future_fight_odds/predictions.csv` | Output written by `backtest_live.py` |
| `data/user_events/*.json` | Events added via `POST /api/add-event` or `/api/analyze` |

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
python backtest/backtest_2025.py --odds data/odds/db_odds_2025.csv --model mar_4_v2
python backtest/bucket_analysis.py --results backtest/backtest_2025_results.csv
```

**Check what the model actually made on real bets this year:**
```bash
python backtest/bucket_analysis.py --results backtest/backtest_2026_results.csv --bets backtest/bets.txt
```

**Find optimal config parameters:**
```bash
python backtest/optimize_config.py --results backtest/backtest_results.csv --top 20
```

**Check single analysis section quickly:**
```bash
python backtest/bucket_analysis.py --results backtest/backtest_2025_results.csv --section skip_reasons
```
