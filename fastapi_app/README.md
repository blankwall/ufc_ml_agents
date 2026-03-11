# UFC Backtest — FastAPI App

Minimal FastAPI + Plotly backtest UI for the `mar_4_v2` model with underdog blending.

## Start

```bash
cd fastapi_app
uv run uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Then open: http://localhost:8001

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
