# Copilot Instructions — UFC ML Agents

## System Overview

UFC fight prediction platform: XGBoost model (`mar_4_v2`, 251 features) → FastAPI web app + REST API. Predicts outcomes and measures edge vs. market odds — does **not** place bets. Two main surfaces: event dashboard (`/events`) and backtest analysis (`/backtest`).

## Running the App

```bash
# FastAPI dev server (from repo root)
cd fastapi_app && ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Bucket analysis (post-backtest)
.venv/bin/python backtest/bucket_analysis.py --results backtest/backtest_2026_results.csv --bets backtest/bets.txt
.venv/bin/python backtest/bucket_analysis.py --results backtest/backtest_2025_results.csv --bets backtest/bets_2025.txt
```

There is no automated test suite. Validate changes by running the app and checking outputs manually.

## Architecture

### Prediction Pipeline
1. `features/matchup_features.py` — `MatchupFeatureExtractor` computes 251 point-in-time features from the DB (strict `as_of_date` to prevent look-ahead leakage)
2. `fastapi_app/services/predict_service.py` — loads model artifacts, runs symmetric scoring (predict A vs B and B vs A, average), caches results in `data/future_fight_odds/predictions_cache.json`
3. Symmetric formula: `P_sym(A) = (P(A|A is f1) + 1 - P(B|B is f1)) / 2`

### Config-Driven Betting
`config/betting_config.json` drives everything: filter thresholds (min confidence, odds caps, min edge), edge-based bet sizing (skip 0-5%, 1x at 5-10%, 1.5x at 10-20%, 2x at 20%+), and WMMA rules. The frontend fetches this via `GET /api/config` on boot.

### FastAPI App Structure
- **Routers** (`fastapi_app/routers/`): 6 routers, all mounted at `/api` prefix
- **Services** (`fastapi_app/services/`): `predict_service.py` (model loading + prediction), `backtest_engine.py` (interactive backtest), `scraper_service.py` (BFO/UFC Stats scraping), `ai_service.py` (Claude integration)
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

### Underdog Blend
`UNDERDOG_BLEND = False` in `predict_service.py`. The secondary `underdog_v1` model exists but is disabled — the general model performs better alone.

## Backtest Data Flow

```
backtest/backtest_2025_results.csv  ─┐
backtest/bets_2025.txt              ─┤─→ bucket_analysis.py ─→ odds buckets, edge tiers, weighted ROI
backtest/backtest_2026_results.csv  ─┤
backtest/bets.txt                   ─┘
```

CSV columns: `date, fighter1, fighter2, odds1, odds2, prob1, prob2, pick, pick_odds, pick_prob, ev1, ev2, winner, pick_correct, actual_pnl, bet, skip_reason, error, female`

## Key Model Artifacts

All in `models/saved/`:
- `mar_4_v2.json` — XGBoost model
- `mar_4_v2_feature_scaler.pkl` — StandardScaler for features
- `mar_4_v2_feature_names.pkl` — ordered feature name list (251 items)
