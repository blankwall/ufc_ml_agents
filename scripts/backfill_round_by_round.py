"""Backfill fight_stats.round_by_round from cached UFCStats fight-detail pages.

Dry-run first by design: no database rows are written unless --apply (alias --all)
is passed. Cached HTML under data/raw/events/fight_*.html is preferred; live
scraping only happens when --fetch-missing is explicitly requested.

Examples:
    .venv/bin/python scripts/backfill_round_by_round.py --dry-run
    .venv/bin/python scripts/backfill_round_by_round.py --limit 500
    .venv/bin/python scripts/backfill_round_by_round.py --all
    .venv/bin/python scripts/backfill_round_by_round.py --all --force
    .venv/bin/python scripts/backfill_round_by_round.py --fetch-missing --rate-limit 1.0 --all
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loguru import logger

from database.db_manager import DatabaseManager
from database.schema import Event, Fight, FightStats
from scrapers.event_scraper import EventScraper


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backfill fight_stats.round_by_round from cached UFCStats fight-detail pages."
    )
    p.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    p.add_argument("--fight-id", help="Only process a single UFCStats fight_id")
    p.add_argument("--limit", type=int, default=None, help="Process at most N fights")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report candidates without writing (this is also the default).",
    )
    p.add_argument(
        "--apply",
        "--all",
        dest="apply",
        action="store_true",
        help="Persist round_by_round updates. Default is dry-run.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite already-populated round_by_round rows (default: skip them).",
    )
    p.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Live-scrape fight-detail pages that are not cached (conservative, opt-in).",
    )
    p.add_argument(
        "--rate-limit",
        type=float,
        default=None,
        help="Override scraper rate limit (seconds) when --fetch-missing is used.",
    )
    return p


def _is_empty(value) -> bool:
    return value is None or value == [] or value == {}


def _backup_sqlite_db(db: DatabaseManager) -> Path | None:
    """Copy the SQLite DB file to data/backups/ before any mutation. Returns path."""
    db_config = db.config.get("database", {})
    if db_config.get("type") != "sqlite":
        logger.warning("Non-sqlite database configured; skipping file backup.")
        return None

    src = Path(db_config["sqlite_path"])
    if not src.is_absolute():
        src = ROOT_DIR / src
    if not src.exists():
        logger.warning(f"SQLite DB not found at {src}; skipping backup.")
        return None

    backup_dir = src.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"ufc_database_before_round_by_round_{stamp}.db"
    shutil.copy2(src, dest)
    return dest


def main(argv: list[str] | None = None) -> dict:
    args = _build_cli().parse_args(argv)

    db = DatabaseManager(config_path=args.config)
    scraper = EventScraper(config_path=args.config)
    if args.rate_limit is not None:
        scraper.rate_limit = args.rate_limit

    session = db.get_session()

    fights_query = (
        session.query(Fight)
        .join(Event, Event.id == Fight.event_id)
        .filter(Fight.fight_detail_url.isnot(None))
        .filter(Fight.fight_detail_url != "")
    )
    if args.fight_id:
        fights_query = fights_query.filter(Fight.fight_id == args.fight_id)

    fights = fights_query.all()
    if args.limit:
        fights = fights[: args.limit]

    summary = {
        "checked": 0,
        "cache_missing": 0,
        "detail_missing": 0,
        "parse_failures": 0,
        "no_db_stats_row": 0,
        "one_round_fights": 0,
        "multi_round_fights": 0,
        "already_populated_skipped": 0,
        "would_update": 0,
        "updated": 0,
    }
    sample_updates: list[str] = []

    backup_path = None
    if args.apply:
        backup_path = _backup_sqlite_db(db)
        if backup_path:
            logger.warning(f"SQLite backup created before backfill: {backup_path}")

    try:
        for fight in fights:
            summary["checked"] += 1
            cache_file = scraper.cache_dir / f"fight_{fight.fight_id}.html"

            if not cache_file.exists() and not args.fetch_missing:
                summary["cache_missing"] += 1
                continue

            try:
                details = scraper.scrape_fight_details(fight.fight_detail_url)
            except Exception as exc:  # pragma: no cover - network/parse edge
                summary["parse_failures"] += 1
                logger.error(f"Failed scraping fight details for {fight.fight_id}: {exc}")
                if args.apply:
                    session.rollback()
                continue

            if not details:
                summary["detail_missing"] += 1
                continue

            _f1, _f2, _sig, round_by_round = db.remap_fight_details_to_db_slots(details, fight)
            if _is_empty(round_by_round):
                summary["parse_failures"] += 1
                continue

            if len(round_by_round) <= 1:
                summary["one_round_fights"] += 1
            else:
                summary["multi_round_fights"] += 1

            stats = fight.fight_stats
            if not stats:
                summary["no_db_stats_row"] += 1
                continue

            if not _is_empty(stats.round_by_round) and not args.force:
                summary["already_populated_skipped"] += 1
                continue

            summary["would_update"] += 1
            if len(sample_updates) < 20:
                sample_updates.append(fight.fight_id)

            if args.apply:
                stats.round_by_round = round_by_round
                summary["updated"] += 1
                if summary["updated"] % 100 == 0:
                    session.commit()
                    logger.info("Committed {} round_by_round updates so far", summary["updated"])

        if args.apply:
            session.commit()
        else:
            session.rollback()
            logger.info("Dry-run (no --apply): no database rows were written.")

        logger.info("round_by_round backfill summary: {}", summary)
        if backup_path:
            logger.info("Backup path: {}", backup_path)
        if sample_updates:
            logger.info("Sample candidate fights: {}", sample_updates)
        return summary
    finally:
        session.close()


if __name__ == "__main__":
    main()
    raise SystemExit(0)
