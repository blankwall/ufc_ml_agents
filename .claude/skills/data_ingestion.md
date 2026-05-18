# Data Ingestion Skill

## What This Is

This skill documents the focused data-ingestion flow for the UFC ML system: UFCStats event ingestion into SQLite, odds-file handling for backtests and live-year data, and the feature/schema handoff into model training.

The safe canonical workflow is:

```text
ingest UFCStats event -> validate DB outcomes -> rebuild feature dataset -> use CSV odds for backtests -> export schema only through model training
```

Do not revive archived one-off ingestion scripts unless you are explicitly doing recovery or migration work.

---

## Canonical Data Sources

| Data | Canonical source | Active path |
|---|---|---|
| Fighters, events, fights, fight results | UFCStats | `scrapers/event_populator.py` |
| Fight detail stats | UFCStats fight-details pages | `scrapers/event_populator.py --include-fight-stats` |
| 2025 holdout odds | Tracked CSV | `backtest/odds/ufc_2025_odds.csv` |
| 2026/live-year odds input | Generated CSV | `backtest/rebuild_2026_odds.py` -> `backtest/odds/ufc_2026_odds.csv` |
| Historical BFO odds research CSV | BestFightOdds scraper | `scrapers/bestfightodds_scraper.py` -> `data/odds/historical_odds.csv` |
| Feature contract | Model export | `schema/feature_schema.json` |

The configured database is `data/ufc_database.db` through `config/config.yaml`. Ignore the root-level `ufc_database.db` if present.

---

## Step 1 - Ingest a UFCStats Event

Use `scrapers/event_populator.py` for all UFCStats DB ingestion. It upserts:

| Table | Key fields |
|---|---|
| `events` | `event_id`, `name`, `date`, `location`, `venue`, `url` |
| `fighters` | UFCStats `fighter_id`, name, physical stats, career stat snapshots |
| `fights` | `fight_id`, integer `event_id`, integer fighter IDs, result, method, weight class |
| `fight_stats` | Optional totals and significant-strike JSON from fight-details pages |

Dry-run first:

```bash
python scrapers/event_populator.py \
  --event-id "<UFCSTATS_EVENT_ID>" \
  --dry-run
```

Commit with fight-detail stats and validate persisted outcomes:

```bash
python scrapers/event_populator.py \
  --event-id "<UFCSTATS_EVENT_ID>" \
  --include-fight-stats \
  --validate \
  --validate-details
```

Useful variants:

```bash
# Use a full UFCStats event URL instead of an event_id
python scrapers/event_populator.py --event-url "http://ufcstats.com/event-details/<id>" --include-fight-stats --validate --validate-details

# Avoid refreshing fighter pages; creates/updates minimal fighter records only
python scrapers/event_populator.py --event-id "<id>" --no-fighter-scrape --validate --validate-details

# Force a fresh fighter scrape when cached/stale fighter data is suspected
python scrapers/event_populator.py --event-id "<id>" --force-refresh-fighters --validate --validate-details
```

Implementation notes:
- Event/fighter/fight writes are idempotent upserts.
- SQLite commits create a backup before persisting changes.
- Fight-details pages are preferred for result/method/round/time by default because they are more reliable than event-page winner heuristics.
- `--validate-details` is the strongest winner check and should be used for completed events.

---

## Step 2 - Database Schema Rules

Critical join rule:

```text
betting_odds.fight_id -> fights.id
fight_stats.fight_id -> fights.id
predictions.fight_id -> fights.id
```

`fights` has two identifiers:

| Column | Meaning |
|---|---|
| `fights.id` | Integer primary key; use this for joins |
| `fights.fight_id` | UFCStats fight-detail hash; do not join DB tables on this |

`Fight.result` must be one of:

```text
fighter_1, fighter_2, draw, no_contest, NULL
```

Training only uses completed non-draw/non-NC fights where `result` is `fighter_1` or `fighter_2`.

---

## Step 3 - Feature Dataset Handoff

After DB ingestion changes, rebuild the training dataset:

```bash
python -m features.feature_pipeline --create
```

Useful variants:

```bash
python -m features.feature_pipeline --create --no-cache
python -m features.feature_pipeline --create --feature-set advanced
```

Feature creation rules:
- `features.matchup_features.create_training_dataset()` reads completed DB fights.
- Each fight becomes two rows: winner as `fighter_1` with `target=1`, loser as `fighter_1` with `target=0`.
- Features are extracted using `as_of_date = fight.event.date`.
- Fight history uses strict point-in-time filtering before the event date; do not change this to include same-day fights.
- Metadata columns include `fight_id`, `event_id`, `fighter_1_id`, `fighter_2_id`, `weight_class`, `method`, and `target`.
- Model feature columns are the remaining columns, sorted deterministically in `FeaturePipeline.prepare_features()`.

Important ingestion requirement: event dates must parse cleanly. Bad or missing dates can break point-in-time feature history.

---

## Step 4 - Feature Schema Handoff

`schema/feature_schema.json` is the contract between training, prediction, API usage, and exports.

Do not edit it by hand. Regenerate it through model training/export:

```bash
python -m models.xgboost_model \
  --train \
  --evaluate \
  --check-calibration \
  --save-plots \
  --export-schema \
  --data-path data/processed/training_data.csv \
  --holdout-from-year 2025 \
  --model-name "$MODEL_NAME"
```

Schema guardrails:
- Review schema diffs before committing.
- Keep model-specific scaler and feature-name files with the model artifact.
- If feature columns change, retrain before using the schema for predictions.
- Run feature/schema validation when touching ingestion or feature code:

```bash
python validate_changes.py
```

---

## Step 5 - Odds Data Policy

Use CSV odds for formal backtests. Do not rely on `betting_odds` as the canonical backtest odds source until the DB odds importer is repaired and normalized.

Canonical formal odds files:

| File | Purpose |
|---|---|
| `backtest/odds/ufc_2025_odds.csv` | Frozen 2025 holdout odds input |
| `backtest/odds/ufc_2026_odds.csv` | Generated 2026/live-year odds input |
| `data/odds/historical_odds.csv` | BFO historical/research source |
| `data/future_fight_odds/ufc*.csv` | Per-event live-year source files |
| `data/future_fight_odds/the_odds_api_new_events.csv` | Generated append-only The Odds API source for newly discovered MMA matchups |
| `data/future_fight_odds/the_odds_api_events.json` | Generated grouped-date store for The Odds API fights plus per-fight odds history |
| `data/future_fight_odds/outcomes.csv` | Outcome source for live-year rebuilds |
| `data/user_events/*.json` | User-added event odds/outcome source |

Build the 2026 backtest odds input:

```bash
python backtest/rebuild_2026_odds.py
```

Fetch new events from The Odds API into the live-year odds pool:

```bash
THE_ODDS_API_KEY=... .venv/bin/python scripts/fetch_the_odds_api.py
```

Behavior notes:
- This source is isolated to the new The Odds API flow; legacy `ufc*.csv` and `data/user_events/*.json` behavior stays unchanged.
- The JSON store groups The Odds API fights by calendar date and keeps an `odds_history` array per fight as prices move over time.
- The site still reads a single scalar odds pair from `the_odds_api_new_events.csv`; that CSV is exported from the JSON store using the **current/latest** odds, while the history array remains for replay/audit only.
- The default ingest horizon is **31 days** via `THE_ODDS_API_WINDOW_DAYS`; this is the practical filter for speculative far-future fights because the feed does not expose a trustworthy “confirmed bout” flag.
- Existing tracked fights from legacy sources are still left untouched by this flow.
- Fights missing from a later API sync are marked inactive in the JSON store and dropped from the exported The Odds API CSV.
- Generated event labels are synthetic date groups because the odds feed does not expose clean UFC card names.

Then run the formal backtest:

```bash
python backtest/backtest_2025.py \
  --odds backtest/odds/ufc_2026_odds.csv \
  --cutoff 2027-01-01 \
  --quiet
```

For historical BFO research scraping:

```bash
python scrapers/bestfightodds_scraper.py \
  --events "UFC 300" \
  --output data/odds/historical_odds.csv
```

---

## Archived / Non-Canonical Paths

The following paths are archived and should not be used for normal ingestion:

| Archived path | Reason |
|---|---|
| `scrapers/archive/scrape_fighter_to_db.py` | Duplicate fighter-history DB ingester; event-populator is safer |
| `add_odds_db/archive/*.py` | Experimental DB odds importer/preview flow; writes partial per-fighter odds rows and has duplicate/update risks |
| `backtest/archive/*` | Legacy/prototype backtest scripts |

If a task asks for odds ingestion into `betting_odds`, treat it as a repair/refactor task first. The current formal betting and backtest pipeline is CSV-first.

---

## Gotchas

- Always join DB tables through `fights.id`, not `fights.fight_id`.
- `EventPopulator` can fall back to minimal fighter records if fighter scraping fails; inspect summaries before trusting a new event.
- Missing `fight_stats` weakens recent striking/grappling feature groups.
- Duplicate DB `Fight` rows will duplicate training samples because the feature builder iterates all completed fights.
- `mar_4_v2` was trained before 2025; any 2025+ fight remains true out-of-sample only if you do not train on 2025+ data.
- WMMA detection depends on `Fight.weight_class` starting with `"Women's"`.
- Fighter names vary by data source; update aliases in prediction/backtest code when odds or event sources use different names.

---

## Quick Safe Flow

```bash
# 1) Ingest completed UFCStats event safely
python scrapers/event_populator.py --event-id "<id>" --dry-run
python scrapers/event_populator.py --event-id "<id>" --include-fight-stats --validate --validate-details

# 2) Rebuild feature dataset
python -m features.feature_pipeline --create

# 3) Rebuild live-year odds input if needed
python backtest/rebuild_2026_odds.py

# 4) Run maintained tests
.venv/bin/pytest -q tests/
```
