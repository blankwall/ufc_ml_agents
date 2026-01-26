## Scrapers

This folder contains utilities for scraping UFCStats and populating the local database.

### Event → Database Populator

`event_populator.py` ingests a single UFCStats event and updates the database:
- **Event metadata** comes from the event page
- **Fighter stats** come from **fighter pages** (source of truth)
- **Fights** are created/updated from the event’s fight list (and store `fight_detail_url`)

#### Usage

- Populate by event URL:

```bash
python3 scrapers/event_populator.py --event-url "https://ufcstats.com/event-details/<EVENT_ID>"
```

- Populate by event id:

```bash
python3 scrapers/event_populator.py --event-id "<EVENT_ID>"
```

#### Options

- `--no-fighter-scrape`: don’t scrape fighter pages (creates minimal fighter records only)
- `--force-refresh-fighters`: re-scrape fighters even if they already exist in DB
- `--include-fight-stats`: also scrape fight detail pages and upsert `fight_stats`
- `--dry-run`: run everything, then rollback (no DB changes persisted)
- `--validate`: after running, compare each fight’s event-page outcome vs what’s in the DB (prints mismatches)
- `--validate-details`: when used with `--validate`, validates DB winners against UFCStats fight-details pages (strongest check)

#### Safety backup (SQLite)

When using SQLite and **not** `--dry-run`, the populator creates a timestamped backup of your DB
before committing changes:

- `data/backups/ufc_database_<timestamp>.db`


