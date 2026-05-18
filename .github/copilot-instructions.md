# Copilot Instructions — UFC ML Agents

## System Overview

UFC fight prediction platform: XGBoost model (`mar_4_v2`, 251 features) → FastAPI web app + REST API. Predicts outcomes and measures edge vs. market odds — does **not** place bets. Two main surfaces: event dashboard (`/events`) and backtest analysis (`/backtest`).

## Running the App

```bash
# FastAPI dev server (from repo root)
cd fastapi_app && ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Run tests
.venv/bin/pytest -q tests/
```

## Backtest Pipeline

The backtest pipeline is two commands: **generate results CSV** → **analyze with bucket_analysis**.

### Step 1 — Generate results CSV

`backtest/backtest_2025.py` loads odds from a canonical CSV, runs in-process symmetric predictions (no subprocess) for every fight, applies the config-driven `should_bet()` filter, and writes a results CSV.

```bash
# 2025 season (true out-of-sample for mar_4_v2)
python backtest/backtest_2025.py --odds backtest/odds/ufc_2025_odds.csv --model mar_4_v2

# Use a custom config (edge_min, confidence thresholds, odds caps)
python backtest/backtest_2025.py --config backtest/backtest_config.json

# 2026 season (rebuild generated odds input, then run the same backtest runner)
python backtest/rebuild_2026_odds.py
python backtest/backtest_2025.py \
  --odds backtest/odds/ufc_2026_odds.csv \
  --cutoff 2027-01-01 \
  --quiet
```

**Output CSV columns:** `date, fighter1, fighter2, odds1, odds2, prob1, prob2, pick, pick_odds, pick_prob, ev1, ev2, winner, pick_correct, actual_pnl, bet, skip_reason, error, female`

- `bet=True/False` — whether the bet passed the config filter
- `skip_reason` — human-readable reason for skips (e.g. `"favorite confidence (58.4% < 65.0%)"`)
- `actual_pnl` — P&L per unit if the bet was placed (positive=win, -1=loss)
- `female=True` — women's divisions (detected via `Fight.weight_class.startswith("Women's")`)

**Key files:**
- `backtest/backtest_2025_results.csv` — 2025 results (359 fights)
- `backtest/backtest_2026_results.csv` — 2026 results (136 fights, growing)
- `backtest/odds/ufc_2025_odds.csv` — tracked 2025 holdout odds input
- `backtest/odds/ufc_2026_odds.csv` — generated 2026 odds input from `backtest/rebuild_2026_odds.py`
- `backtest/backtest_config.json` — per-run config (model, cutoffs, edge/confidence thresholds, odds caps)
- `config/betting_config.json` — site-facing config (also used by `bucket_analysis --config`)
- `backtest/bets_2025.txt` — manually curated actual bets placed in 2025 (line format: `[YYYY-MM-DD] Fighter @ odds  prob=X%  ev=+Y  WON/LOST (pnl)  vs Opponent`)
- `backtest/bets.txt` — 2026 actual bets

### Step 2 — Bucket analysis

`backtest/bucket_analysis.py` is the post-backtest analytical layer. It slices the results CSV five ways.

```bash
# Full analysis — all 5 sections
python backtest/bucket_analysis.py --results backtest/backtest_2025_results.csv

# Filter to only fights that were actually bet on (bets.txt contains real placed bets)
python backtest/bucket_analysis.py --results backtest/backtest_2025_results.csv --bets backtest/bets_2025.txt

# 2026 with betting config for weighted ROI
python backtest/bucket_analysis.py --results backtest/backtest_2026_results.csv --bets backtest/bets.txt

# Single section (buckets | edge | confidence | skip_reasons | weighted)
python backtest/bucket_analysis.py --results backtest/backtest_2025_results.csv --section edge
```

**The 5 analysis sections:**

1. **ODDS BUCKET BREAKDOWN** (`--section buckets`) — groups bets by market odds: `-400` (78–99%), `-300` (71–78%), `-200` (60–71%), `+200` (37–60%), `+300` (22–37%), `+400` (<22%). Shows N/W/L, WinRate, Profit, ROI, AvgEdge, AvgConf per bucket + M/F gender split.

2. **EDGE-TIER BREAKDOWN** (`--section edge`) — groups by model edge: `0–5%`, `5–10%`, `10–15%`, `15%+`. Same stats. Shows whether higher edge actually translates to higher ROI.

3. **CONFIDENCE SCORE BANDS** (`--section confidence`) — decile-based confidence scoring (1–10). Computed from `confidence_profile.py` which splits all pick_prob values into 10 equal-size buckets ranked by prob. Shows AvgPred vs actual WinRate per band — the calibration gap. This is the canonical way to report how well-calibrated the model is.

4. **SKIP REASON BREAKDOWN** (`--section skip_reasons`) — only shown when no `--bets` filter. Tallies which config filter rejected the most fights: `favorite confidence`, `favorite cap`, `underdog confidence`, `underdog cap`, `underdog edge`, `min_fights`, `female`. Critical for tuning `backtest_config.json`.

5. **WEIGHTED ROI ANALYSIS** (`--section weighted`) — reads `config/betting_config.json` edge_buckets and applies variable bet sizing: 0–5% skip, 5–10% at 1x, 10–20% at 1.5x, 20%+ at 2x. Shows side-by-side flat vs weighted P&L. WMMA (women's) fights capped at 1x and require ≥10% edge. This is the most realistic P&L projection.

### Step 3 — Optimize config (optional)

`backtest/optimize_config.py` grid-searches across edge, confidence, and odds-cap parameters to maximize P&L. Vectorized — no model inference, runs in seconds. Writes generated `backtest/optimize_results.csv` locally; that file is ignored and should not be committed.

```bash
python backtest/optimize_config.py --results backtest/backtest_2026_results.csv --top 20 --sort-by roi
```

### Step 4 — 2026 input rebuild

`backtest/rebuild_2026_odds.py` builds `backtest/odds/ufc_2026_odds.csv` from `data/future_fight_odds/ufc*.csv`, `data/future_fight_odds/the_odds_api*.csv`, `data/future_fight_odds/outcomes.csv`, user-event JSON, and DB outcomes. The Odds API current-value CSV is exported from `data/future_fight_odds/the_odds_api_events.json`, which keeps grouped-date fights plus per-fight odds history. `backtest/archive/backtest_live.py` is legacy/prototype only and is not used for formal backtesting.

### The confidence scoring system

`backtest/confidence_profile.py` provides `build_confidence_bands()` which splits all `pick_prob` values from both `backtest_2025_results.csv` and `backtest_2026_results.csv` into 10 decile bands (score 1–10). The `/api/predict` endpoint uses `describe_confidence()` from this module to attach a confidence score to every prediction — this is what the events UI displays as the confidence score badge.

## Architecture

### Prediction Pipeline
1. `features/matchup_features.py` — `MatchupFeatureExtractor` computes 251 point-in-time features from the DB (strict `as_of_date` to prevent look-ahead leakage)
2. `fastapi_app/services/predict_service.py` — loads model artifacts, runs symmetric scoring (predict A vs B and B vs A, average), caches results in `data/future_fight_odds/predictions_cache.json`
3. Symmetric formula: `P_sym(A) = (P(A|A is f1) + 1 - P(B|B is f1)) / 2`

### Config-Driven Betting
`config/betting_config.json` drives everything: filter thresholds (min confidence, odds caps, min edge), edge-based bet sizing (skip 0-5%, 1x at 5-10%, 1.5x at 10-20%, 2x at 20%+), and WMMA rules. The frontend fetches this via `GET /api/config` on boot.

### FastAPI App Structure
- **Routers** (`fastapi_app/routers/`): 6 routers, all mounted at `/api` prefix
- **Services** (`fastapi_app/services/`): `predict_service.py` (model loading + prediction), `backtest_engine.py` (interactive backtest), `scraper_service.py` (BFO/UFC Stats scraping), `ai_service.py` (Claude integration), `the_odds_api_service.py` (daily new-event sync from The Odds API), `ufcstats_sync_service.py` (conservative completed-event UFCStats DB sync)
- **Templates**: Jinja2 (`fastapi_app/templates/`), dark theme, Google Fonts Inter
- **Static**: Vanilla JS (no frameworks), Plotly 2.32.0 via CDN for charts, CSS custom properties for theming

### Frontend Conventions
- Dark theme tokens in `style.css`: `--bg`, `--surface`, `--border`, `--text`, `--accent`, `--green`, `--red`
- Vanilla JS with `fetch()` API, DOM manipulation via `getElementById`/`classList`
- State via module-level globals
- Plotly charts with transparent backgrounds matching the dark theme

## Critical Gotchas

### Database Joins
`fights` table has two ID columns: `fights.id` (integer PK) and `fights.fight_id` (hex hash). **Always join on `fights.id`**. `betting_odds.fight_id` references `fights.id` (the integer), not the hash.

### Model Training Boundary
`mar_4_v2` was trained on fights **before 2025**. Any 2025+ fight is true out-of-sample. Pre-2025 backtest results are in-sample and artificially inflated (~+40-45% ROI).

### WMMA Detection
`is_wmma` is detected via `Fight.weight_class` starting with `"Women's"`. The field is `True`/`False`/`None` (None = unknown, treated conservatively). WMMA rules cap bet multiplier to 1.0x and require minimum 10% edge.

### Fighter Name Resolution
Fighter names vary between data sources. `FIGHTER_ALIASES` dict in `predict_service.py` maps variants. When adding new fighters, check for name mismatches.

### The Odds API Sync
If `THE_ODDS_API_KEY` is set, `fastapi_app/main.py` starts a background loop that checks hourly and fetches at most once every 24 hours into `data/future_fight_odds/the_odds_api_events.json`, then exports current scalar rows to `data/future_fight_odds/the_odds_api_new_events.csv`. This integration is isolated to the new flow: legacy `ufc*.csv` and user-event JSON remain unchanged and keep higher precedence. The JSON store groups fights by calendar date and preserves `odds_history` arrays per fight as prices move over time. The default ingest horizon is 31 days via `THE_ODDS_API_WINDOW_DAYS`, which acts as the main filter against speculative far-future fights because the feed does not expose a reliable confirmed-bout flag.

### UFCStats Completed Sync
`ufcstats_sync_service.py` is for newly completed events only, not future-card discovery. It should stay conservative: fetch recent completed UFCStats events, run a dry-run ingest first, create one DB backup per sync run, commit only unseen events that pass the dry-run screen, then validate DB winners against UFCStats fight-details pages before marking the event synced in `data/ufcstats_sync_state.json`.

### Underdog Blend
`UNDERDOG_BLEND = False` in `predict_service.py`. The secondary `underdog_v1` model exists but is disabled — the general model performs better alone.

## Test Suite

```bash
.venv/bin/pytest -q tests/   # 33 tests, ~30s
```

| Test file | What it covers |
|---|---|
| `test_skip_codes_exhaustive.py` | Every skip code in `_evaluate_bet()` (predict router). Catches ordering bugs that shadow later checks. Uses synthetic inputs against `config/betting_config.json`. |
| `test_bet_sizing_buckets.py` | Edge bucket boundaries and WMMA multiplier caps mirror JS logic in `events.js`. Pins the spec: 0–5% skip, 5–10% 1×, 10–20% 1.5×, 20%+ 2×. |
| `test_confidence_profile.py` | `build_confidence_bands()` decile logic — verifies 10 bands, monotonic prob ordering, edge cases. |
| `test_fighter_alias_resolution.py` | FIGHTER_ALIASES lookup and name normalization in `predict_service.py`. |
| `test_no_lookahead_leakage.py` | Point-in-time integrity: features at `as_of=D` and `as_of=D-1day` must be identical (fight at D excluded from both). Features at `as_of=D+1day` must differ. Guards against `<` → `<=` regressions in feature extractor. |
| `test_predict_response.py` | `/api/predict` endpoint response shape and field presence. |
| `test_predict_symmetry.py` | Model is symmetric: `P(A beats B) + P(B beats A) ≈ 1.0`. |
| `test_ufc_328_consistency.py` | Event-level regression: UFC 328 predictions are stable across runs. |

**Note:** `test_ufc_328_consistency.py` requires `playwright` (`pip install playwright`). All others have no heavy external dependencies beyond the repo's `.venv`.

## Key Model Artifacts

All in `models/saved/`:
- `mar_4_v2.json` — XGBoost model
- `mar_4_v2_feature_scaler.pkl` — StandardScaler for features
- `mar_4_v2_feature_names.pkl` — ordered feature name list (251 items)
