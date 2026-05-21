from __future__ import annotations

import argparse
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
        description="Audit and backfill UFCStats fight_stats slot alignment using fighter-ID remapping."
    )
    p.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    p.add_argument("--fight-id", help="Only process a single UFCStats fight_id")
    p.add_argument("--limit", type=int, default=None, help="Process at most N fights")
    p.add_argument(
        "--since-date",
        help="Only process fights on or after YYYY-MM-DD based on Event.date",
    )
    p.add_argument(
        "--recent-first",
        action="store_true",
        help="Process newer fights first after in-Python date parsing.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Persist rewritten fight_stats rows. Default is dry-run audit only.",
    )
    p.add_argument(
        "--cache-only",
        action="store_true",
        help="Only use cached fight-details HTML; skip rows whose cache is missing.",
    )
    p.add_argument(
        "--bust-cache",
        action="store_true",
        help="Refresh cached fight-details HTML before parsing.",
    )
    return p


def _payload_changed(current, new) -> bool:
    return (current or None) != (new or None)


def _parse_event_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def main() -> int:
    args = _build_cli().parse_args()

    db = DatabaseManager(config_path=args.config)
    scraper = EventScraper(config_path=args.config)
    session = db.get_session()
    since_dt = datetime.strptime(args.since_date, "%Y-%m-%d") if args.since_date else None

    fights_query = (
        session.query(Fight, Event.date)
        .join(FightStats, FightStats.fight_id == Fight.id)
        .join(Event, Event.id == Fight.event_id)
        .filter(Fight.fight_detail_url.isnot(None))
        .filter(Fight.fight_detail_url != "")
    )
    if args.fight_id:
        fights_query = fights_query.filter(Fight.fight_id == args.fight_id)

    fights_with_dates = []
    for fight, event_date in fights_query.all():
        parsed_event_dt = _parse_event_date(event_date)
        if since_dt and (parsed_event_dt is None or parsed_event_dt < since_dt):
            continue
        fights_with_dates.append((fight, parsed_event_dt))

    fights_with_dates.sort(
        key=lambda item: (item[1] or datetime.min, item[0].id),
        reverse=args.recent_first,
    )
    if args.limit:
        fights_with_dates = fights_with_dates[: args.limit]
    fights = [fight for fight, _ in fights_with_dates]
    summary = {
        "checked": 0,
        "cache_missing": 0,
        "detail_missing": 0,
        "unchanged": 0,
        "would_rewrite": 0,
        "rewritten": 0,
        "errors": 0,
    }
    rewritten_fights: list[str] = []

    backup_path = None
    if args.apply:
        backup_fn = getattr(db, "_backup_db_if_sqlite", None)
        if callable(backup_fn):
            backup_path = backup_fn()
        if backup_path:
            logger.warning(f"SQLite backup created before backfill: {backup_path}")

    try:
        for fight in fights:
            summary["checked"] += 1
            cache_file = scraper.cache_dir / f"fight_{fight.fight_id}.html"
            if args.cache_only and not cache_file.exists():
                summary["cache_missing"] += 1
                continue

            try:
                details = scraper.scrape_fight_details(
                    fight.fight_detail_url,
                    bust_cache=args.bust_cache,
                )
            except Exception as exc:
                summary["errors"] += 1
                logger.error(f"Failed scraping fight details for {fight.fight_id}: {exc}")
                if args.apply:
                    session.rollback()
                continue

            if not details:
                summary["detail_missing"] += 1
                continue

            stats = fight.fight_stats
            if not stats:
                summary["detail_missing"] += 1
                continue

            remapped_f1, remapped_f2, remapped_sig = db.remap_fight_details_to_db_slots(details, fight)
            changed = (
                _payload_changed(stats.fighter_1_totals, remapped_f1)
                or _payload_changed(stats.fighter_2_totals, remapped_f2)
                or _payload_changed(stats.significant_strikes, remapped_sig)
            )

            if not changed:
                summary["unchanged"] += 1
                continue

            summary["would_rewrite"] += 1
            logger.info(
                "Needs rewrite: {} :: {} vs {}",
                fight.fight_id,
                fight.fighter_1.name if fight.fighter_1 else "?",
                fight.fighter_2.name if fight.fighter_2 else "?",
            )

            if not args.apply:
                continue

            stats.fighter_1_totals = remapped_f1
            stats.fighter_2_totals = remapped_f2
            stats.significant_strikes = remapped_sig
            rewritten_fights.append(fight.fight_id)
            summary["rewritten"] += 1

            if summary["rewritten"] % 100 == 0:
                session.commit()
                logger.info("Committed {} rewritten fight_stats rows so far", summary["rewritten"])

        if args.apply:
            session.commit()
        else:
            session.rollback()

        logger.info("Fight-stats backfill summary: {}", summary)
        if backup_path:
            logger.info("Backup path: {}", backup_path)
        if rewritten_fights:
            logger.info("Sample rewritten fights: {}", rewritten_fights[:20])
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
