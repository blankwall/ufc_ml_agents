# UFC ML Agents — LLM Onboarding Context

> Quick-start guide for a new AI assistant to understand this codebase, the model stack, the FastAPI app, and the validation work done to date.

---

## What This Is

A UFC fight prediction and betting strategy analysis system built on XGBoost models trained on historical UFC fight data. It has two surfaces:

1. **A FastAPI web app** (`fastapi_app/`) — a dark-themed UI for backtesting betting strategies and viewing model predictions on upcoming/past events
2. **A REST API** — all features of the UI are backed by API endpoints, enabling programmatic access for automation, additional tooling, or agent integration

The system does **not** place bets. It predicts outcomes, measures historical edge, and surfaces model confidence vs. market odds.

---

## Model Stack

### mar_4_v2 — General Model (primary)
- **Type:** XGBoost classifier
- **Features:** 251 engineered features — fighter physical attributes, career stats, recent form, opponent quality, style matchup differentials, interaction terms
- **Training cutoff:** All fights **before 2025** (holdout from 2025 onward)
- **Validation (val split):** accuracy 62.9%, AUC 0.683, Brier 0.224
- **2025 holdout (true out-of-sample):** ~67–68% accuracy, ~+11% ROI flat-bet on all picks
- **Artifacts:** `models/saved/mar_4_v2.json`, `mar_4_v2_feature_scaler.pkl`, `mar_4_v2_feature_names.pkl`

### underdog_v1 — Specialist Underdog Model
- **Purpose:** Improve upset detection when market implied prob < 40%
- **Extra feature:** `market_prob_f1` added as feature 252 (model sees market opinion)
- **Training:** Filtered to underdog rows, 3× upsampled upset wins to fix class imbalance
- **Metrics:** accuracy 57.5%, AUC 0.572, upset detection rate 45%
- **Artifacts:** `models/saved/underdog_v1.json`, `underdog_v1_feature_scaler.pkl`, `underdog_v1_feature_names.pkl`

### Underdog Blending
When market implied prob for a fighter < 40%, a blended prediction is used:
```
blended_prob = 0.65 × underdog_v1_prob + 0.35 × mar_4_v2_prob
```
Blending is applied only to fights flagged as underdog scenarios, not the whole card.

### Symmetric Scoring
Every fight is scored twice — `(A vs B)` and `(B vs A)` — and averaged:
```
P_sym(A) = (P(A|A is f1) + 1 - P(B|B is f1)) / 2
```
In practice the model is naturally symmetric (features are signed differentials) so both directions sum to 1.0. This avoids winner-perspective bias in evaluation.

### Feature Extraction
Live feature extraction for new fights uses `MatchupFeatureExtractor` in `features/matchup_features.py`. It reads fighter career stats from the DB as of the fight date (`as_of_date` parameter prevents look-ahead), computes all 251 differentials, and feeds them to the model.

---

## Key Validation Findings

### 2025 Holdout Results (True Out-of-Sample)
| Strategy | Fights | Accuracy | ROI |
|---|---|---|---|
| All fights, no filter | 447 | 67.1% | +11.3% |
| Favorites (mkt ≥ 60%) | 151 | 63.6% | +25.1% |
| Underdogs + blend, edge ≥ 0% | 133 | 77.4% | +94.4%* |

*The underdog number is a small sample (133 fights, one year) and should not be used for aggressive Kelly sizing yet.

### In-Sample Warning (Critical)
`mar_4_v2` was trained on pre-2025 data. Therefore:
- **2025+:** true out-of-sample — ROI numbers are valid
- **Pre-2025:** in-sample — model has seen these fights during training, ROI is artificially inflated (2023: ~+40%, 2024: ~+45%)
- The backtest UI shows a yellow warning when the date range includes pre-2025 data

### Winner-Perspective Bias (Fixed)
The original eval CSV always placed the actual winner as `f1`, which skewed backtest results. The fix: deduplicate fights by taking the row with the highest `model_prob_f1_symmetric` — outcome-independent canonical representation. This is how the backtest engine now works.

### Calibration Finding
When the model is 75–80% confident, the actual win rate is ~88% — the model is systematically **underconfident** at the high end. See `BETTING_RULES.md` for full calibration table.

---

## Database

**SQLite** at `data/ufc_database.db` (~15MB).

### Key Tables
| Table | Rows | Notes |
|---|---|---|
| `events` | 760 | Dates as `"November 22, 2025"` (not ISO) |
| `fights` | 8,507 | `id` (int PK), `fight_id` (hex hash — different!) |
| `fighters` | 4,451 | `id`, `name`, career stats |
| `betting_odds` | 13,532 | `fight_id` → `fights.id` (int). Closing/opening flag. |
| `fight_stats` | 8,487 | Round-by-round detail |

### Critical Join Key Gotcha
- `fights.fight_id` = hex hash like `5f5b626e67529056`
- `fights.id` = plain integer
- `training_data.fight_id` = integer matching `fights.id` (NOT the hash)
- `betting_odds.fight_id` = integer matching `fights.id`

**Always join on `fights.id` (integer), never the hash.**

### Odds Quirks
- Most rows have only ONE fighter's odds stored. The other side is computed as `1 - known_prob`, then vig-normalised.
- `is_closing_line` flag is reliable for 2020+, approximate for older data.

---

## FastAPI App (`fastapi_app/`)

### Entry Point
```
fastapi_app/main.py       — FastAPI app, mounts routers
```

### Routers (`fastapi_app/routers/`)
| File | Prefix | Purpose |
|---|---|---|
| `backtest.py` | `/api` | Run backtests with configurable parameters |
| `events.py` | `/api` | Return all events + model predictions |
| `scraper.py` | `/api` | Scrape new events, one-shot analyze |

### Services (`fastapi_app/services/`)
| File | Purpose |
|---|---|
| `backtest_engine.py` | Core backtesting logic — queries DB, runs model, computes ROI/drawdown/Sharpe |
| `predict_service.py` | Loads odds CSVs + user events, runs model predictions, joins outcomes |
| `scraper_service.py` | Scrapes BFO odds and UFC Stats results, saves to `data/user_events/` |

### Key API Endpoints

```
GET  /                          — Backtest UI (HTML)
GET  /events                    — Events page (HTML)

GET  /api/backtest              — Run backtest
  params: start_date, end_date, focus (all|favorites|underdogs),
          min_confidence, max_confidence, min_edge, ud_blend,
          underdog_cutoff, weight_classes, flat_bet

GET  /api/events                — All events with fight predictions
GET  /api/events/list           — List available events for quickselect

POST /api/analyze               — One-shot: scrape BFO URL + predict all fights
  body: { bfo_url, ufc_stats_url?, force_rescrape? }
  returns: { event_name, event_date, summary, fights: [...] }

GET  /api/analyze/{slug}        — Re-analyze a previously scraped event

POST /api/add-event             — Scrape + save without returning predictions
GET  /api/user-events           — List all saved user-added events
DELETE /api/user-events/{slug}  — Delete a saved event

GET  /docs                      — FastAPI auto-generated API docs
```

### Fight Response Shape
```json
{
  "fighter1": "Justin Gaethje",
  "fighter2": "Paddy Pimblett",
  "f1_odds": 200,
  "f2_odds": -245,
  "market_prob_f1": 33.3,
  "model_prob_f1": 46.8,
  "model_source": "blended",
  "model_pick": "Paddy Pimblett",
  "edge": 13.5,
  "winner": "Justin Gaethje",
  "method": "U-DEC",
  "round": "5",
  "correct": false,
  "pnl": -100.0,
  "error": null,
  "source_type": "csv"
}
```
`source_type` is `"csv"` for static historical events, `"user_added"` for events scraped via the API.

---

## Data Sources

### Static Events (historical, Jan–Mar 2026)
`data/future_fight_odds/*.csv` — one CSV per event, manually populated:
- Columns: `event_name, event_date, event_url, fighter1, fighter2, fighter1_odds, fighter2_odds, fighter1_prob, fighter2_prob`

`data/future_fight_odds/outcomes.csv` — fight results scraped from UFC Stats:
- Populated by running `scrapers/scrape_outcomes.py <ufcstats-url>`

### User-Added Events
`data/user_events/*.json` — events scraped via `POST /api/add-event` or `POST /api/analyze`.
Each JSON contains `fights` (odds) + `outcomes` (results) bundled together. These are picked up automatically by `predict_service.get_events_data()`.

### Predictions Cache
`data/future_fight_odds/predictions_cache.json` — keyed by fight pair, stores model output so feature extraction doesn't run on every request. Automatically invalidated when new fights are added.

---

## Scrapers

| Script | Purpose |
|---|---|
| `scrapers/bestfightodds_scraper.py` | Full BFO scraper — takes event URL, returns odds. Has disk cache in `.cache/bfo/`. |
| `scrapers/scrape_outcomes.py` | Takes UFC Stats event URL, returns fight results. Merges into `outcomes.csv`. |
| `scrapers/event_scraper.py` | Scrapes UFC Stats for event/fighter data into DB |
| `scrapers/fighter_scraper.py` | Scrapes individual fighter pages into DB |

---

## Known Limitations & Open Items

1. **No walk-forward validation for pre-2025.** Features were computed with full current DB, not historical snapshots. True walk-forward would recompute features using only data available before each fight.
2. **No Kelly sizing.** Backtest uses flat-bet only. Quarter-Kelly requires sequential bankroll simulation.
3. **No opening-line backtest.** Current engine uses closing odds only. Opening-line would measure CLV.
4. **Underdog ROI (94%) is small-sample.** 133 fights over one year — directionally positive but not yet stable enough for aggressive sizing.
5. **Fighter name normalization is fragile.** The `FIGHTER_ALIASES` dict in `predict_service.py` and `_resolve_fighter()` handle many variants, but new typos/nicknames in scraped CSVs require manual alias additions.

---

## Running the App

```bash
cd fastapi_app
uv run uvicorn main:app --host 0.0.0.0 --port 8002
```

The app runs from the `fastapi_app/` directory. All paths in services use `ROOT_DIR = Path(__file__).parent.parent.parent` to resolve to the repo root, so the working directory doesn't matter for data access.
