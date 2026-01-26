"""
Event Populator - Ingest a single UFCStats event into the database.

Goal:
- Given a UFCStats event URL (or event_id), scrape the event page to get fights.
- Scrape each unique fighter page to get fighter stats (fighter pages are source of truth).
- Upsert Event, Fighters, and Fights into the database (idempotent).

Notes:
- Fighter lookups are by Fighter.fighter_id (UFCStats string id).
- Fight relationships store Fighter.id (DB integer PK).
"""

from __future__ import annotations

import sys
import argparse
import sqlite3
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set

from loguru import logger

# Ensure project root is on sys.path when running as a script:
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db_manager import DatabaseManager
from database.schema import Fighter, FightStats, Fight  # noqa: E402
from scrapers.event_scraper import EventScraper  # noqa: E402
from scrapers.fighter_scraper import FighterScraper  # noqa: E402


@dataclass(frozen=True)
class PopulatorOptions:
    scrape_fighters: bool = True
    force_refresh_fighters: bool = False
    include_fight_stats: bool = False
    prefer_fight_detail_results: bool = True
    commit: bool = True


class EventPopulator:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.event_scraper = EventScraper(config_path=config_path)
        self.fighter_scraper = FighterScraper(config_path=config_path)
        self.db = DatabaseManager(config_path=config_path)

    @staticmethod
    def normalize_ufcstats_url(url: str) -> str:
        """
        UFCStats is commonly served over HTTP. If the user passes an HTTPS UFCStats URL,
        normalize it to HTTP to avoid connection/SSL issues in some environments.
        """
        u = (url or "").strip()
        if u.startswith("https://ufcstats.com/"):
            return "http://ufcstats.com/" + u[len("https://ufcstats.com/") :]
        return u

    @staticmethod
    def extract_event_id(event_url: str) -> str:
        """
        Extract UFCStats event_id from an event URL.
        Accepts:
        - .../event-details/<event_id>
        - .../event-details/<event_id>/
        """
        event_url = (event_url or "").strip()
        if not event_url:
            raise ValueError("event_url is empty")
        # best-effort: last path segment
        return event_url.rstrip("/").split("/")[-1]

    def populate_event_from_url(
        self,
        event_url: str,
        *,
        options: PopulatorOptions = PopulatorOptions(),
    ) -> Dict[str, object]:
        event_url = self.normalize_ufcstats_url(event_url)
        event_id = self.extract_event_id(event_url)
        return self.populate_event(event_url=event_url, event_id=event_id, options=options)

    def populate_event(
        self,
        *,
        event_url: str,
        event_id: str,
        options: PopulatorOptions = PopulatorOptions(),
    ) -> Dict[str, object]:
        """
        Populate DB with a single event.

        Returns a summary dict.
        """
        summary = {
            "event_id": event_id,
            "event_url": event_url,
            "event_name": None,
            "fighters_total": 0,
            "fighters_scraped": 0,
            "fighters_skipped_existing": 0,
            "fighters_failed": 0,
            "fights_total": 0,
            "fights_upserted": 0,
            "fight_stats_upserted": 0,
            "committed": False,
        }

        event_url = self.normalize_ufcstats_url(event_url)
        logger.info(f"Scraping event {event_id} …")
        event_data = self.event_scraper.scrape_event(event_url, event_id)
        if not event_data:
            raise RuntimeError(f"Failed to scrape event: {event_url}")

        summary["event_name"] = event_data.get("name")

        fights = list(event_data.get("fights", []) or [])
        summary["fights_total"] = len(fights)

        fighters_to_process = self._collect_unique_fighters(fights)
        summary["fighters_total"] = len(fighters_to_process)

        session = self.db.get_session()
        try:
            # Safety: backup sqlite DB before we persist anything.
            # This creates a restore point of the *current* DB state.
            if options.commit:
                self._backup_db_if_sqlite()

            # Upsert event first (need DB event.id for fights)
            event = self.db.add_event(session, event_data)
            session.flush()

            # Upsert fighters
            fighter_objs_by_ufc_id: Dict[str, Fighter] = {}
            for fighter_id, (fighter_name, fighter_url) in fighters_to_process.items():
                fighter_obj, did_scrape, failed = self._ensure_fighter(
                    session,
                    fighter_id=fighter_id,
                    fighter_name=fighter_name,
                    fighter_url=fighter_url,
                    options=options,
                )
                fighter_objs_by_ufc_id[fighter_id] = fighter_obj

                if failed:
                    summary["fighters_failed"] += 1
                elif did_scrape:
                    summary["fighters_scraped"] += 1
                else:
                    summary["fighters_skipped_existing"] += 1

            # Upsert fights (event page provides fight detail url/id)
            fights_upserted = 0
            fight_stats_upserted = 0
            fight_details_cache: Dict[str, Dict] = {}

            for fight_data in fights:
                f1_ufc_id = (fight_data.get("fighter_1_id") or "").strip()
                f2_ufc_id = (fight_data.get("fighter_2_id") or "").strip()
                if not f1_ufc_id or not f2_ufc_id:
                    continue

                f1_obj = fighter_objs_by_ufc_id.get(f1_ufc_id)
                f2_obj = fighter_objs_by_ufc_id.get(f2_ufc_id)
                if not f1_obj or not f2_obj:
                    continue

                # Prefer fight-details page for result/method/round/time, if available.
                if options.prefer_fight_detail_results:
                    self._enrich_fight_data_from_details(fight_data, fight_details_cache)

                fight_obj = self.db.add_fight(session, fight_data, event, f1_obj, f2_obj)
                fights_upserted += 1

                # Debug log: print who won (W/L/D/NC tags)
                logger.debug(
                    self._format_fight_result_line(
                        fight_number=fight_data.get("fight_number"),
                        f1_name=f1_obj.name or fight_data.get("fighter_1_name", "Unknown"),
                        f2_name=f2_obj.name or fight_data.get("fighter_2_name", "Unknown"),
                        result=fight_obj.result,
                    )
                )

                if options.include_fight_stats:
                    # Ensure fight_obj.id is available for FightStats FK
                    if fight_obj.id is None:
                        session.flush()
                    if self._upsert_fight_stats_if_available(session, fight_obj, fight_data, fight_details_cache):
                        fight_stats_upserted += 1

            summary["fights_upserted"] = fights_upserted
            summary["fight_stats_upserted"] = fight_stats_upserted

            if options.commit:
                session.commit()
                summary["committed"] = True
                logger.success(
                    f"Populated event '{summary['event_name']}' ({event_id}): "
                    f"{summary['fighters_total']} fighters, {summary['fights_upserted']} fights"
                )
            else:
                session.rollback()
                logger.warning("Dry-run mode: rolled back DB transaction (no changes persisted).")

            return summary
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _backup_db_if_sqlite(self) -> Optional[Path]:
        """
        If we're using SQLite, create a timestamped backup before committing.

        Uses sqlite3 backup API for a consistent snapshot.
        Returns the backup path if created.
        """
        try:
            db_cfg = (self.db.config or {}).get("database", {})
            if db_cfg.get("type") != "sqlite":
                return None

            db_path = Path(db_cfg.get("sqlite_path", "data/ufc_database.db"))
            if not db_path.exists():
                # Nothing to back up yet
                return None

            backups_dir = db_path.parent / "backups"
            backups_dir.mkdir(parents=True, exist_ok=True)

            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backups_dir / f"{db_path.stem}_{ts}{db_path.suffix}"

            # Consistent snapshot using SQLite backup
            src = sqlite3.connect(str(db_path))
            try:
                dst = sqlite3.connect(str(backup_path))
                try:
                    src.backup(dst)
                    dst.commit()
                finally:
                    dst.close()
            finally:
                src.close()

            logger.warning(f"SQLite backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"Failed to create SQLite backup: {e}")
            return None

    def validate_event_against_db(self, event_url: str, *, use_fight_details: bool = False) -> Dict[str, object]:
        """
        Validate that DB fight outcomes match the event page outcomes.

        - Scrapes event (cache respected)
        - For each fight, looks it up in DB by fight_detail_id (stored as Fight.fight_id)
        - Prints event winner/loser vs DB winner/loser
        - If use_fight_details=True, also compares DB winner to fight-details winner (strongest check)
        """
        event_url = self.normalize_ufcstats_url(event_url)
        event_id = self.extract_event_id(event_url)

        logger.info(f"Validating event {event_id} …")
        event_data = self.event_scraper.scrape_event(event_url, event_id)
        if not event_data:
            raise RuntimeError(f"Failed to scrape event for validation: {event_url}")

        fights = list(event_data.get("fights", []) or [])
        total = 0
        missing = 0
        mismatches = 0

        session = self.db.get_session()
        try:
            for fight_data in fights:
                fight_detail_id = (fight_data.get("fight_detail_id") or "").strip()
                if not fight_detail_id:
                    continue

                total += 1

                db_fight = session.query(Fight).filter_by(fight_id=fight_detail_id).first()
                if not db_fight:
                    missing += 1
                    logger.warning(
                        f"[MISSING] Fight #{fight_data.get('fight_number')}: "
                        f"{fight_data.get('fighter_1_name')} vs {fight_data.get('fighter_2_name')} "
                        f"(fight_detail_id={fight_detail_id}) not found in DB"
                    )
                    continue

                # Event page perspective (after scrape_event parsing)
                ev_result = fight_data.get("result")
                ev_line = self._format_fight_result_line(
                    fight_number=fight_data.get("fight_number"),
                    f1_name=fight_data.get("fighter_1_name", "Unknown"),
                    f2_name=fight_data.get("fighter_2_name", "Unknown"),
                    result=ev_result,
                )

                # DB perspective
                db_line = self._format_fight_result_line(
                    fight_number=fight_data.get("fight_number"),
                    f1_name=db_fight.fighter_1.name if db_fight.fighter_1 else "Unknown",
                    f2_name=db_fight.fighter_2.name if db_fight.fighter_2 else "Unknown",
                    result=db_fight.result,
                )

                # Compare DB winner against event ordering (weak check)
                same_event = (ev_result == db_fight.result) or (
                    ev_result in ("fighter_1", "fighter_2") and db_fight.result in ("fighter_1", "fighter_2")
                    and self._winner_matches_event_order(fight_data, db_fight)
                )

                # Strong check: compare DB winner against fight-details page (source of truth)
                same_details = True
                details_line = None
                if use_fight_details:
                    fight_detail_url = (fight_data.get("fight_detail_url") or "").strip()
                    fight_detail_url = self.normalize_ufcstats_url(fight_detail_url)
                    details = self.event_scraper.scrape_fight_details(fight_detail_url) if fight_detail_url else None
                    if details:
                        # Winner UFCStats id from details
                        winner = details.get("winner")
                        if winner in ("draw", "no_contest"):
                            details_winner_id = winner
                        elif winner == "fighter_1":
                            details_winner_id = details.get("fighter_1_id")
                        elif winner == "fighter_2":
                            details_winner_id = details.get("fighter_2_id")
                        else:
                            details_winner_id = None

                        # Winner UFCStats id from DB
                        if db_fight.result == "draw":
                            db_winner_id = "draw"
                        elif db_fight.result == "no_contest":
                            db_winner_id = "no_contest"
                        elif db_fight.result == "fighter_1":
                            db_winner_id = db_fight.fighter_1.fighter_id if db_fight.fighter_1 else None
                        elif db_fight.result == "fighter_2":
                            db_winner_id = db_fight.fighter_2.fighter_id if db_fight.fighter_2 else None
                        else:
                            db_winner_id = None

                        same_details = (details_winner_id is not None) and (details_winner_id == db_winner_id)
                        details_line = f"Details winner_id={details_winner_id} vs DB winner_id={db_winner_id}"
                    else:
                        same_details = False
                        details_line = "Details: unable to load fight-details page"

                ok = same_details if use_fight_details else same_event
                if not ok:
                    mismatches += 1
                    logger.error(f"[MISMATCH] EVENT: {ev_line}")
                    logger.error(f"[MISMATCH]   DB: {db_line}")
                    if details_line:
                        logger.error(f"[MISMATCH] {details_line}")
                else:
                    logger.info(f"[OK] EVENT: {ev_line}")
                    logger.info(f"[OK]   DB: {db_line}")
                    if details_line:
                        logger.info(f"[OK]   {details_line}")

            logger.info(f"Validation summary: fights_checked={total}, missing_in_db={missing}, mismatches={mismatches}")
            return {"fights_checked": total, "missing_in_db": missing, "mismatches": mismatches}
        finally:
            session.close()

    @staticmethod
    def _winner_matches_event_order(event_fight_data: Dict, db_fight: Fight) -> bool:
        """
        Compare winner identity, independent of fighter_1/fighter_2 ordering differences.
        """
        ev_result = event_fight_data.get("result")
        if ev_result in ("draw", "no_contest"):
            return db_fight.result == ev_result

        ev_f1_id = event_fight_data.get("fighter_1_id")
        ev_f2_id = event_fight_data.get("fighter_2_id")
        if not ev_f1_id or not ev_f2_id:
            return False

        # Determine winner UFCStats id from event perspective
        if ev_result == "fighter_1":
            ev_winner = ev_f1_id
        elif ev_result == "fighter_2":
            ev_winner = ev_f2_id
        else:
            return False

        # Determine winner UFCStats id from DB perspective
        if db_fight.result == "fighter_1":
            db_winner = db_fight.fighter_1.fighter_id if db_fight.fighter_1 else None
        elif db_fight.result == "fighter_2":
            db_winner = db_fight.fighter_2.fighter_id if db_fight.fighter_2 else None
        else:
            db_winner = None

        return bool(db_winner) and (ev_winner == db_winner)

    def _collect_unique_fighters(self, fights: List[Dict]) -> Dict[str, Tuple[str, str]]:
        """
        Returns mapping: fighter_id (UFCStats) -> (name, url)
        """
        fighters: Dict[str, Tuple[str, str]] = {}
        for fight in fights:
            for side in ("fighter_1", "fighter_2"):
                fid = (fight.get(f"{side}_id") or "").strip()
                name = (fight.get(f"{side}_name") or "").strip()
                url = (fight.get(f"{side}_url") or "").strip()
                if not fid:
                    continue
                # Keep first seen; URLs are stable for a given id.
                if fid not in fighters:
                    fighters[fid] = (name or "Unknown", url)
        return fighters

    @staticmethod
    def _format_fight_result_line(
        *,
        fight_number: Optional[int],
        f1_name: str,
        f2_name: str,
        result: Optional[str],
    ) -> str:
        """
        Human-readable fight result line for debug logs.

        result values follow DB convention:
        - 'fighter_1', 'fighter_2', 'draw', 'no_contest', or None
        """
        f1_tag = ""
        f2_tag = ""
        winner_text = "UNKNOWN"

        if result == "fighter_1":
            f1_tag, f2_tag = "W", "L"
            winner_text = f1_name
        elif result == "fighter_2":
            f1_tag, f2_tag = "L", "W"
            winner_text = f2_name
        elif result == "draw":
            f1_tag = f2_tag = "D"
            winner_text = "DRAW"
        elif result == "no_contest":
            f1_tag = f2_tag = "NC"
            winner_text = "NO CONTEST"

        prefix = f"Fight #{fight_number}: " if fight_number else "Fight: "
        return f"{prefix}F1: {f1_name} [{f1_tag}] vs F2: {f2_name} [{f2_tag}] — Winner: {winner_text}"

    def _ensure_fighter(
        self,
        session,
        *,
        fighter_id: str,
        fighter_name: str,
        fighter_url: str,
        options: PopulatorOptions,
    ) -> Tuple[Fighter, bool, bool]:
        """
        Ensure fighter exists in DB, and optionally scrape fighter page to upsert full stats.

        Returns: (fighter_obj, did_scrape, failed)
        """
        existing = session.query(Fighter).filter_by(fighter_id=fighter_id).first()
        if existing and (not options.force_refresh_fighters):
            return existing, False, False

        # If scraping disabled, create/minimally update record.
        if not options.scrape_fighters:
            fighter_data_min = {"fighter_id": fighter_id, "name": fighter_name, "url": fighter_url}
            fighter_obj = self.db.add_fighter(session, fighter_data_min)
            session.flush()
            return fighter_obj, False, False

        # Try scrape fighter page
        fighter_data = None
        try:
            url = fighter_url or f"{self.fighter_scraper.base_url}/fighter-details/{fighter_id}"
            url = self.normalize_ufcstats_url(url)
            fighter_data = self.fighter_scraper.scrape_fighter(url, fighter_id)
        except Exception as e:
            logger.warning(f"Failed scraping fighter {fighter_id} ({fighter_name}): {e}")

        if fighter_data:
            fighter_obj = self.db.add_fighter(session, fighter_data)
            session.flush()
            return fighter_obj, True, False

        # Fallback: minimal record (matches existing DB population pattern)
        fighter_data_min = {"fighter_id": fighter_id, "name": fighter_name, "url": fighter_url}
        fighter_obj = self.db.add_fighter(session, fighter_data_min)
        session.flush()
        return fighter_obj, False, True

    def _upsert_fight_stats_if_available(
        self,
        session,
        fight_obj: Fight,
        fight_data: Dict,
        fight_details_cache: Optional[Dict[str, Dict]] = None,
    ) -> bool:
        """
        Optionally scrape and upsert FightStats for a fight if we have a fight detail URL.
        Returns True if stats were upserted.
        """
        fight_detail_url = (fight_data.get("fight_detail_url") or "").strip()
        if not fight_detail_url:
            return False

        fight_detail_url = self.normalize_ufcstats_url(fight_detail_url)

        details = None
        if fight_details_cache is not None:
            details = fight_details_cache.get(fight_detail_url)

        if details is None:
            try:
                details = self.event_scraper.scrape_fight_details(fight_detail_url)
            except Exception as e:
                logger.warning(f"Failed scraping fight details for {fight_detail_url}: {e}")
                return False
            if fight_details_cache is not None and details:
                fight_details_cache[fight_detail_url] = details

        if not details:
            return False

        existing_stats = session.query(FightStats).filter_by(fight_id=fight_obj.id).first()
        f1_totals = (details.get("totals") or {}).get("fighter_1")
        f2_totals = (details.get("totals") or {}).get("fighter_2")
        sig = details.get("significant_strikes")

        if existing_stats:
            existing_stats.fighter_1_totals = f1_totals
            existing_stats.fighter_2_totals = f2_totals
            existing_stats.significant_strikes = sig
        else:
            session.add(
                FightStats(
                    fight_id=fight_obj.id,
                    fighter_1_totals=f1_totals,
                    fighter_2_totals=f2_totals,
                    significant_strikes=sig,
                )
            )
        return True

    def _enrich_fight_data_from_details(self, fight_data: Dict, fight_details_cache: Dict[str, Dict]) -> None:
        """
        Use fight-details page to set accurate result + method/round/time.
        This prevents bad event-page heuristics from writing wrong winners.
        """
        fight_detail_url = (fight_data.get("fight_detail_url") or "").strip()
        if not fight_detail_url:
            return

        fight_detail_url = self.normalize_ufcstats_url(fight_detail_url)
        details = fight_details_cache.get(fight_detail_url)
        if details is None:
            try:
                details = self.event_scraper.scrape_fight_details(fight_detail_url)
            except Exception as e:
                logger.debug(f"Fight-details fetch failed for {fight_detail_url}: {e}")
                return
            if details:
                fight_details_cache[fight_detail_url] = details
            else:
                return

        # Update method/round/time if present
        if details.get("method"):
            fight_data["method"] = details.get("method")
        if details.get("round") is not None:
            fight_data["round"] = details.get("round")
        if details.get("time"):
            fight_data["time"] = details.get("time")
        if details.get("method_details"):
            fight_data["method_detail"] = details.get("method_details")

        # Map winner to fight_data result using fighter IDs
        winner = details.get("winner")
        if winner in ("draw", "no_contest"):
            fight_data["result"] = winner
            return

        if winner == "fighter_1":
            winner_ufc_id = details.get("fighter_1_id")
        elif winner == "fighter_2":
            winner_ufc_id = details.get("fighter_2_id")
        else:
            winner_ufc_id = None

        if not winner_ufc_id:
            return

        f1_ufc_id = fight_data.get("fighter_1_id")
        f2_ufc_id = fight_data.get("fighter_2_id")
        if winner_ufc_id == f1_ufc_id:
            fight_data["result"] = "fighter_1"
        elif winner_ufc_id == f2_ufc_id:
            fight_data["result"] = "fighter_2"
        else:
            # Fight-details ordering didn't match event ordering; keep existing result but warn.
            logger.warning(
                f"Winner ID mismatch for fight {fight_data.get('fight_detail_id')}: "
                f"winner_ufc_id={winner_ufc_id} not in ({f1_ufc_id}, {f2_ufc_id})"
            )


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Populate the database from a UFCStats event link.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--event-url", type=str, help="UFCStats event URL, e.g. https://ufcstats.com/event-details/<id>")
    group.add_argument("--event-id", type=str, help="UFCStats event_id, e.g. <id>")

    p.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    p.add_argument("--no-fighter-scrape", action="store_true", help="Do not scrape fighter pages (minimal fighter records only)")
    p.add_argument("--force-refresh-fighters", action="store_true", help="Re-scrape fighters even if they already exist in DB")
    p.add_argument("--include-fight-stats", action="store_true", help="Also scrape fight detail pages and upsert FightStats")
    p.add_argument(
        "--no-fight-detail-results",
        action="store_true",
        help="Do not override result/method/round/time from fight detail pages (use event page only)",
    )
    p.add_argument("--dry-run", action="store_true", help="Do everything but rollback at the end (no DB changes persisted)")
    p.add_argument(
        "--validate",
        action="store_true",
        help="After running (or against existing DB if --dry-run), validate each fight outcome vs the event page",
    )
    p.add_argument(
        "--validate-details",
        action="store_true",
        help="Validate DB winners against UFCStats fight-details pages (strongest check)",
    )
    return p


def main() -> int:
    args = _build_cli().parse_args()

    options = PopulatorOptions(
        scrape_fighters=not bool(args.no_fighter_scrape),
        force_refresh_fighters=bool(args.force_refresh_fighters),
        include_fight_stats=bool(args.include_fight_stats),
        prefer_fight_detail_results=not bool(args.no_fight_detail_results),
        commit=not bool(args.dry_run),
    )

    pop = EventPopulator(config_path=args.config)

    if args.event_url:
        summary = pop.populate_event_from_url(args.event_url, options=options)
        event_url_for_validation = args.event_url
    else:
        event_id = args.event_id.strip()
        event_url = f"{pop.event_scraper.base_url}/event-details/{event_id}"
        summary = pop.populate_event(event_url=event_url, event_id=event_id, options=options)
        event_url_for_validation = event_url

    # Print a small summary for CLI users
    logger.info(
        "Summary: "
        f"event='{summary.get('event_name')}', "
        f"fighters_total={summary.get('fighters_total')}, "
        f"fighters_scraped={summary.get('fighters_scraped')}, "
        f"fights_upserted={summary.get('fights_upserted')}, "
        f"committed={summary.get('committed')}"
    )

    if args.validate:
        if not summary.get("committed") and not args.dry_run:
            logger.warning("Validation requested but ingestion did not commit; validating against existing DB state.")
        pop.validate_event_against_db(event_url_for_validation, use_fight_details=bool(args.validate_details))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


