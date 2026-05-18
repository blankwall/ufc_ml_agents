# Testing Skill

## What This Is

This skill documents the main validation layers for the UFC ML system: fast unit/integration checks, prediction invariants, and the browser-level consistency test that verifies the live FastAPI events page matches `POST /api/predict`.

Use this skill when changing:

- `fastapi_app/services/predict_service.py`
- `fastapi_app/routers/predict.py`
- `fastapi_app/static/js/events.js`
- odds-loading or The Odds API event shaping
- betting-rule display and edge/market probability rendering

---

## Core Test Commands

Primary maintained suite:

```bash
.venv/bin/pytest -q tests/
```

Focused fast checks:

```bash
.venv/bin/pytest -q tests/test_predict_response.py
.venv/bin/pytest -q tests/test_predict_symmetry.py
.venv/bin/pytest -q tests/test_skip_codes_exhaustive.py
.venv/bin/pytest -q tests/test_bet_sizing_buckets.py
.venv/bin/pytest -q tests/test_the_odds_api_sync.py
```

If you want broad coverage but not the browser/live-site check:

```bash
.venv/bin/pytest -q tests/ -k 'not ufc_328_consistency'
```

---

## Browser-to-API Consistency Test

The repo already includes a real end-to-end validation:

```bash
.venv/bin/python -m pytest tests/test_ufc_328_consistency.py -v -s
```

What it does:

1. Uses Playwright to load `/events`
2. Selects a target event tab
3. Scrapes displayed fight-card values from the page
4. Calls `POST /api/predict` for each displayed fight
5. Verifies that the site and API agree on:
   - vig-removed market probability
   - model probability
   - model pick
   - edge

This is the best regression test when changing event rendering or prediction-response logic because it validates what the user actually sees, not just backend internals.

For a **full parity sweep across every event tab** on the remote site:

```bash
.venv/bin/python -m pytest tests/test_remote_events_parity.py -v -s
```

That test:

1. Fetches `GET /api/events`
2. Opens `/events` in Playwright
3. Switches the page to **raw view**
4. Walks every event tab in UI order
5. Scrapes every visible fight card
6. Calls `POST /api/predict` with the same fighters, odds, and event date
7. Asserts parity for:
   - vig-removed market probability
   - model probability
   - model pick
   - edge
   - visible fight count vs the event payload

This is the best high-confidence upstream regression sweep when you want to know whether frontend/event-loop changes drifted away from the model path.

---

## Environment for the Consistency Test

The test uses environment variables:

```text
SITE_URL
EVENT_NAME
EVENT_DATE
MIN_FIGHTS
```

Defaults in the test:

```text
SITE_URL=http://107.175.94.166:8002
EVENT_NAME=UFC 328
EVENT_DATE=2026-05-10
```

Override them when validating a different deployment or event:

```bash
SITE_URL=http://127.0.0.1:8001 \
EVENT_NAME="UFC 328" \
EVENT_DATE=2026-05-10 \
.venv/bin/python -m pytest tests/test_ufc_328_consistency.py -v -s
```

The full parity sweep also supports:

```text
SITE_URL
EVENT_LIMIT
RAW_VIEW
```

Example:

```bash
SITE_URL=http://107.175.94.166:8002 \
EVENT_LIMIT=0 \
RAW_VIEW=1 \
.venv/bin/python -m pytest tests/test_remote_events_parity.py -v -s
```

---

## Playwright Requirement

`tests/test_ufc_328_consistency.py` requires Playwright.

Install it in the repo venv if needed:

```bash
.venv/bin/pip install playwright
.venv/bin/playwright install chromium
```

Without Playwright, the normal Python test suite still works, but this browser-level validation will not run.

---

## What Each Test Layer Catches

| Test | Main purpose |
|---|---|
| `test_predict_response.py` | Response shape and field presence for `/api/predict` |
| `test_predict_symmetry.py` | `P(A beats B) + P(B beats A) ≈ 1` |
| `test_skip_codes_exhaustive.py` | Betting-rule ordering and skip-code correctness |
| `test_bet_sizing_buckets.py` | Edge bucket and WMMA sizing rules |
| `test_the_odds_api_sync.py` | The Odds API ingest/history/store/export behavior |
| `test_ufc_328_consistency.py` | Live UI values match backend prediction output |
| `test_remote_events_parity.py` | Full all-events UI/API parity sweep on a deployment |

---

## Recommended Testing Workflow

For prediction or events-page changes:

1. Run the focused backend tests relevant to the change
2. Run `tests/test_ufc_328_consistency.py`
3. For deployment-wide confidence, run `tests/test_remote_events_parity.py`
4. If changing odds ingestion or event shaping, also run `tests/test_the_odds_api_sync.py`
5. If changing bet rules or displayed bet filtering, also run:

```bash
.venv/bin/pytest -q tests/test_skip_codes_exhaustive.py tests/test_bet_sizing_buckets.py
```

---

## Important Notes

- `/events` renders from `GET /api/events`; the browser does not run the model itself.
- The consistency test is valuable because it compares the rendered site against `POST /api/predict`, catching UI/API drift.
- `test_ufc_328_consistency.py` is event-specific by default, but reusable by env vars.
- `test_remote_events_parity.py` is the deployment-wide version for every current event tab.
