# FastAPI App Skill

## What This Is

The FastAPI app is the web/API layer for the UFC ML system. It serves the event prediction dashboard, backtest dashboard, UFCStats ingestion UI, fighter profile UI, and JSON APIs used by those pages.

Run it from the repo root:

```bash
cd fastapi_app && ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

The app is not a betting executor. It displays model probabilities, market edge, backtest metrics, fighter profiles, and ingestion results.

---

## App Entry Point

`fastapi_app/main.py` creates the `FastAPI` app, mounts static assets, registers routers under `/api`, and serves HTML pages.

HTML routes:

| Route | Template | Purpose |
|---|---|---|
| `/` | redirect | Redirects to `/events` |
| `/events` | `templates/events.html` | Event predictions dashboard |
| `/backtest` | `templates/backtest.html` | Backtest/bucket-analysis dashboard |
| `/ingest` | `templates/ingest.html` | UFCStats event ingestion UI |
| `/fighter` | `templates/fighter.html` | Fighter search/profile UI |

API routers mounted with `prefix="/api"`:

| Router | Main responsibility |
|---|---|
| `routers/events.py` | Events data, betting config, recent fighter odds |
| `routers/predict.py` | One-off fight prediction endpoint |
| `routers/scraper.py` | BFO/user-event scraping and saved event management |
| `routers/database.py` | UFCStats DB ingestion and fighter profile APIs |
| `routers/analyze.py` | Matchup stats and independent AI analysis |
| `routers/backtest.py` | Interactive DB-backed backtest engine |
| `routers/bucket_analysis.py` | Precomputed 2025/2026 bucket-analysis dashboard data |

---

## Directory Structure

| Path | Role |
|---|---|
| `fastapi_app/main.py` | App construction, router registration, HTML route registration |
| `fastapi_app/routers/` | Thin API endpoint layer; validates requests and maps exceptions to HTTP responses |
| `fastapi_app/services/` | Business logic: prediction, backtesting, scraping, AI analysis |
| `fastapi_app/templates/` | Jinja2 HTML shells for pages |
| `fastapi_app/static/js/` | Vanilla JS page controllers using `fetch()` |
| `fastapi_app/static/css/` | Dark-theme CSS for each page plus shared `style.css` |

Most routers add `ROOT_DIR` to `sys.path` so they can import repo-root modules like `database`, `features`, `models`, `backtest`, and `scrapers`.

---

## Events Dashboard Flow

The main product surface is `/events`.

```text
/events page
  -> static/js/events.js
  -> GET /api/events and GET /api/config
  -> routers/events.py
  -> services/predict_service.get_events_data()
  -> _load_all_odds() + _load_outcomes() + _run_prediction_loop()
```

`predict_service.py`:

- Loads event odds from `data/future_fight_odds/ufc*.csv`
- Loads generated The Odds API current-value rows from `data/future_fight_odds/the_odds_api*.csv`
- Loads user-added event JSON from `data/user_events/*.json`
- Loads outcomes from `data/future_fight_odds/outcomes.csv` and user-event JSON
- Deduplicates fights by alias-normalized fighter pair
- Resolves fighters against `data/ufc_database.db`
- Extracts point-in-time features through `MatchupFeatureExtractor`
- Runs symmetric `mar_4_v2` predictions
- Caches predictions in `data/future_fight_odds/predictions_cache.json`
- Returns event groups with fight-level odds, market probability, model probability, pick, edge, result, P&L, thin-data counts, and WMMA flag

The frontend applies filters client-side using `config/betting_config.json` loaded from `GET /api/config`.

If `THE_ODDS_API_KEY` is set, `main.py` also starts a background sync loop that checks hourly and pulls MMA odds at most once every 24 hours into a JSON history store at `data/future_fight_odds/the_odds_api_events.json`, then exports current scalar rows to `data/future_fight_odds/the_odds_api_new_events.csv`. By default the fetch is limited to the next **31 days** (`THE_ODDS_API_WINDOW_DAYS`) to avoid loading speculative far-future fights.

There is also an optional **completed UFCStats sync** service in `fastapi_app/services/ufcstats_sync_service.py`. When `UFCSTATS_AUTO_SYNC=1`, it polls recent completed UFCStats events, runs a dry-run ingest first, creates a single DB backup for the run, commits only safe unseen events, then validates DB winners against UFCStats fight-details pages before marking the event synced in `data/ufcstats_sync_state.json`.

Important UI behavior:

- Event cards render from the `/api/events` response only; the browser does not run model inference.
- Filters in `events.js` mirror betting config thresholds and bet sizing buckets.
- The "Add Event" modal posts BFO/UFCStats URLs to scraper endpoints and then reloads `/api/events`.
- Missing fighters surface as per-fight `error` values from `predict_service`, not as a whole-page failure.

---

## One-Off Prediction Flow

`POST /api/predict` is separate from the event dashboard. It accepts two fighter names, optional fight date, and optional American odds.

Request model:

```text
fighter1, fighter2, fight_date?, fighter1_odds?, fighter2_odds?
```

Behavior:

- Resolves aliases through `FIGHTER_ALIASES`
- Resolves fighters through `_resolve_fighter()`
- Converts odds to vig-normalized market probability when both sides are supplied
- Uses `fight_date` as `as_of_date` for feature extraction
- Scores with `_score_row()` from `predict_service.py`
- Applies `_evaluate_bet()` against `config/betting_config.json`
- Adds confidence score via `backtest/confidence_profile.py`

Use this endpoint for direct API prediction. Use `/api/events` for card-level predictions from stored odds files.

---

## Event Scraping and User-Added Events

Manual event addition is handled by `routers/scraper.py` and `services/scraper_service.py`.

```text
events.js Add Event modal
  -> POST /api/add-event
  -> scraper_service.scrape_and_save()
  -> scrapers.bestfightodds_scraper.BestFightOddsScraper
  -> optional scrapers.scrape_outcomes.scrape_event()
  -> data/user_events/<slug>.json
  -> picked up by predict_service._load_all_odds()
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /api/add-event` | Scrape BFO odds and optional UFCStats results into `data/user_events` |
| `GET /api/user-events` | List saved user events |
| `DELETE /api/user-events/{slug}` | Delete a saved user event |
| `POST /api/user-events/{slug}/results` | Add/update UFCStats results without re-scraping odds |
| `POST /api/analyze` | One-shot scrape + prediction response for a BFO event |
| `GET /api/analyze/{slug}` | Re-run analysis for a saved event |

User-event JSON is a live-dashboard source, not the canonical historical DB ingestion path.

---

## UFCStats DB Ingestion UI

`/ingest` is the browser UI for canonical UFCStats event ingestion.

```text
/ingest page
  -> static/js/ingest.js
  -> POST /api/db/ingest
  -> routers/database.py
  -> scrapers.event_populator.EventPopulator
  -> data/ufc_database.db
```

`POST /api/db/ingest` runs blocking ingestion in an executor so the FastAPI event loop is not blocked directly.

Current ingestion options in `database.py`:

```text
include_fight_stats=True
force_refresh_fighters=True
bust_cache=True
commit=True
```

For careful production ingestion, the CLI flow in `scrapers/event_populator.py` is still preferred because it supports dry-run and validation flags. The UI is convenient, but it commits immediately.

---

## Fighter and Matchup APIs

The fighter page uses:

```text
/fighter page
  -> static/js/fighter.js
  -> GET /api/db/fighters/search?q=...
  -> GET /api/db/fighter/{name}
```

`routers/database.py` returns:

- Fighter identity and record
- Physical stats
- UFCStats career stat fields
- Fight history sorted by event date
- Opening/closing odds where available from `betting_odds`

Matchup and AI-analysis endpoints live in `routers/analyze.py`:

| Endpoint | Purpose |
|---|---|
| `GET /api/matchup/{fighter1}/{fighter2}` | Side-by-side fighter profiles and recent fights |
| `GET /api/fight-stats/{fighter}/{opponent}` | Detailed fight stats lookup |
| `POST /api/matchup/analyze` | Independent AI statistical analysis |

`services/ai_service.py` deliberately does not receive betting odds or model predictions. It should stay independent from the model signal.

---

## Backtest Surfaces

There are two backtest-related app surfaces.

### `/backtest` dashboard

The current page uses `static/js/backtest.js` and reads:

```text
GET /api/bucket-analysis
```

`routers/bucket_analysis.py` serves precomputed 2025 and 2026 results from:

- `backtest/backtest_2025_results.csv`
- `backtest/backtest_2026_results.csv`
- `backtest/bets_2025.txt`
- `backtest/bets.txt`
- `config/betting_config.json`

It mirrors the CLI `backtest/bucket_analysis.py` concepts and adds DB weight-class lookup for dashboard grouping.

### Interactive backtest API

`routers/backtest.py` exposes:

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | Available date range, weight classes, fight count |
| `POST /api/backtest` | Run DB-backed interactive backtest with supplied params |

This uses `services/backtest_engine.py`, which builds a cached DB/training-data dataset and scores with model artifacts. This is separate from the formal CSV-first backtest pipeline documented in the backtesting skill.

---

## Prediction Service Internals

`services/predict_service.py` is the highest-risk app file because it connects odds files, DB fighter resolution, feature extraction, model scoring, outcomes, and UI response shaping.

Key constants:

| Constant | Meaning |
|---|---|
| `ODDS_DIR` | `data/future_fight_odds` |
| `USER_EVENTS_DIR` | `data/user_events` |
| `OUTCOMES_CSV` | `data/future_fight_odds/outcomes.csv` |
| `CACHE_FILE` | `data/future_fight_odds/predictions_cache.json` |
| `MODEL_DIR` | `models/saved` |
| `UNDERDOG_BLEND` | Disabled by default; general `mar_4_v2` model is used alone |

Important functions:

| Function | Role |
|---|---|
| `_load_all_odds()` | Reads odds CSV/user-event JSON, normalizes columns, deduplicates rows |
| `_load_outcomes()` | Reads outcomes from CSV and user-event JSON |
| `_resolve_fighter()` | Fuzzy/alias fighter lookup |
| `_score_row()` | Symmetric model prediction for one matchup |
| `_run_prediction_loop()` | Core card loop used by `/api/events` and one-shot event analysis |
| `get_events_data()` | Public entry for `GET /api/events` |
| `analyze_event()` | One-shot BFO event scrape/analyze path |

Do not casually clear or rewrite `predictions_cache.json`; it prevents repeated slow feature extraction.

---

## Data and Config Sources

| Data/config | Used by |
|---|---|
| `data/ufc_database.db` | DB-backed APIs, feature extraction, ingestion, fighter pages |
| `data/future_fight_odds/ufc*.csv` | Events dashboard odds source |
| `data/future_fight_odds/outcomes.csv` | Events dashboard outcome source |
| `data/user_events/*.json` | Manual event odds/results source |
| `data/future_fight_odds/predictions_cache.json` | Cached event-dashboard model predictions |
| `config/betting_config.json` | UI filters, `/api/predict` bet evaluation, bucket-analysis weighting |
| `models/saved/mar_4_v2*` | Production model, scaler, feature order |

The real DB path is `data/ufc_database.db`. Ignore root-level `ufc_database.db` if present.

---

## Frontend Conventions

- Vanilla JS only; no frontend framework.
- Each page has one JS controller under `fastapi_app/static/js/`.
- Shared dark-theme CSS lives in `static/css/style.css`; page-specific CSS lives beside it.
- Pages fetch JSON APIs directly with `fetch()`.
- State is mostly module-level globals in each JS file.
- Plotly is loaded by CDN on the backtest page.
- Keep API response shapes stable; the JS is not strongly typed and expects exact field names.

Primary JS/API pairings:

| JS file | APIs |
|---|---|
| `events.js` | `/api/events`, `/api/config`, `/api/add-event`, `/api/user-events`, `/api/user-events/{slug}/results`, matchup endpoints |
| `backtest.js` | `/api/bucket-analysis` |
| `ingest.js` | `/api/db/ingest` |
| `fighter.js` | `/api/db/fighters/search`, `/api/db/fighter/{name}` |

---

## Error Handling Patterns

- Routers generally convert known user/input failures to `HTTPException` with 4xx status.
- Unexpected scraper/analysis failures are typically returned as 500 with a descriptive `detail`.
- The event-dashboard prediction loop captures per-fight model/fighter-resolution errors and includes them in the fight row so the rest of the card still renders.
- Avoid broad silent catches when adding new behavior. Existing broad catches around optional file loading are legacy tolerance for malformed local data; do not use them as a pattern for important API failures.

---

## App Gotchas

1. `/api/events` does not call external odds APIs during requests. It reads local CSV/JSON files and runs cached model prediction.
2. `_load_all_odds()` loads `data/future_fight_odds/ufc*.csv`, `data/future_fight_odds/the_odds_api*.csv`, and user-event JSON, with The Odds API rows intentionally lower-priority than manual/user-added sources.
3. The Odds API replay/update history lives in `data/future_fight_odds/the_odds_api_events.json`; the UI still consumes only the exported scalar CSV rows.
4. Event prediction cache keys include normalized fighter pair and date when available. Fighter DB updates do not automatically invalidate cached predictions.
5. Event-dashboard filters are client-side and mirror `config/betting_config.json`; `/api/events` returns raw prediction data, not pre-filtered bets.
6. `/api/predict` is fresh per request and has its own bet-evaluation logic in `routers/predict.py`.
7. `/api/db/ingest` commits immediately; use `scrapers/event_populator.py --dry-run` from CLI for safer ingestion rehearsals.
8. `betting_odds` is used for fighter recent-fight odds and DB-backed backtest surfaces, but formal backtests are still CSV-first.
9. `fights.id` is the integer primary key for DB joins; do not join app logic on `fights.fight_id` hash.
10. AI matchup analysis is intentionally independent and should not be given odds, model predictions, or betting recommendations.

---

## Safe Change Checklist

When changing FastAPI app behavior:

1. Identify the full path: template -> JS -> router -> service -> data/model source.
2. Preserve existing response field names unless you update every JS consumer.
3. Reuse helpers in `predict_service.py` for fighter normalization and model scoring.
4. Keep external API calls out of page-load routes unless explicitly required.
5. Add fixture-based tests for parsers/loaders; do not depend on live network calls in tests.
6. Run focused tests for affected prediction/betting behavior:

```bash
.venv/bin/pytest -q tests/test_predict_response.py tests/test_predict_symmetry.py tests/test_skip_codes_exhaustive.py tests/test_bet_sizing_buckets.py
```

For broader app confidence, run:

```bash
.venv/bin/pytest -q tests/
```
