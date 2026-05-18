# UFC Backtest — FastAPI App

Minimal FastAPI + Plotly backtest UI for the `mar_4_v2` model with underdog blending.

## Start

```bash
cd fastapi_app
# fill in runtime.env as needed (THE_ODDS_API_KEY is blank by default)
../run.sh
```

Then open: http://localhost:8001

## Runtime sync config

`fastapi_app/runtime.env` is sourced by `run.sh` before Uvicorn starts.

Use it to control:
- `THE_ODDS_API_KEY`
- `THE_ODDS_API_AUTO_SYNC`
- `THE_ODDS_API_SYNC_INTERVAL_HOURS`
- `THE_ODDS_API_SYNC_CHECK_SECONDS`
- `THE_ODDS_API_WINDOW_DAYS`
- `UFCSTATS_AUTO_SYNC`
- `UFCSTATS_SYNC_INTERVAL_HOURS`
- `UFCSTATS_SYNC_CHECK_SECONDS`
- `UFCSTATS_COMPLETED_LOOKBACK_DAYS`
- `UFCSTATS_COMPLETED_MAX_PAGES`
- `UFCSTATS_COMPLETED_MIN_FIGHTS`
- `UFCSTATS_COMPLETED_MAX_EVENTS_PER_RUN`

## API

All actions are available programmatically:

```bash
# Available params (date range, weight classes)
curl http://localhost:8001/api/meta

# Run a backtest
curl -X POST http://localhost:8001/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "focus": "underdogs",
    "ud_threshold": 0.40,
    "min_confidence": 0.55,
    "max_confidence": 0.85,
    "min_edge": 0.08,
    "min_american_odds": -300,
    "max_american_odds": null,
    "weight_classes": [],
    "use_underdog_blend": true,
    "blend_weight": 0.65,
    "flat_bet": 100
  }'
```

## Backtest Parameters

| Param | Default | Description |
|---|---|---|
| `start_date` | `2025-01-11` | Start of date range |
| `end_date` | `2025-12-06` | End of date range |
| `focus` | `all` | `all`, `favorites`, or `underdogs` |
| `ud_threshold` | `0.40` | Market prob cutoff for underdog label |
| `min_confidence` | `0.50` | Min model confidence to place bet |
| `max_confidence` | `1.00` | Max model confidence (cull heavy favs) |
| `min_edge` | `0.00` | Min model edge vs market to bet |
| `min_american_odds` | `null` | Skip bets shorter than this (e.g. `-300`) |
| `max_american_odds` | `null` | Skip bets longer than this (e.g. `+500`) |
| `weight_classes` | `[]` | Filter by weight class; empty = all |
| `use_underdog_blend` | `true` | Apply 65/35 underdog model blend |
| `blend_weight` | `0.65` | Weight for `underdog_v1` in blend |
| `flat_bet` | `100` | Flat bet size in dollars |

## Data Source

`reports_mar_4_v2/eval_data_20260304_165004.csv` — 297 fights, Jan–Dec 2025 holdout set, pre-scored with `mar_4_v2` general model. Underdog blending is applied live using `models/saved/underdog_v1.json`.
