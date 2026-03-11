#!/usr/bin/env python3
"""
Scrape a single fighter from UFCStats and insert/update in the database.

Also scrapes and inserts all fights from the fighter's history, including
opponents and events.

Usage:
  python scrapers/scrape_fighter_to_db.py --fighter-id 294aa73dbf37d281
  python scrapers/scrape_fighter_to_db.py --fighter-url http://ufcstats.com/fighter-details/294aa73dbf37d281
  python scrapers/scrape_fighter_to_db.py --fighter-id 294aa73dbf37d281 --dry-run
  python scrapers/scrape_fighter_to_db.py --fighter-id 294aa73dbf37d281 --skip-fights
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from database.db_manager import DatabaseManager
from database.schema import Fighter, Event, Fight
from scrapers.fighter_scraper import FighterScraper
from scrapers.event_scraper import EventScraper


def normalize_url(url: str) -> str:
    """Normalize UFCStats URL to HTTP."""
    if url.startswith("https://ufcstats.com/"):
        return "http://ufcstats.com/" + url[len("https://ufcstats.com/"):]
    return url


def extract_id_from_url(url: str) -> str:
    """Extract ID from UFCStats URL."""
    return url.rstrip("/").split("/")[-1]


def parse_fight_result(result_str: str) -> Optional[str]:
    """Parse result string (win/loss/draw/nc) to DB format."""
    if not result_str:
        return None
    result_lower = result_str.strip().lower()
    if result_lower == "win":
        return "fighter_1"  # Will be adjusted based on who is fighter_1
    elif result_lower == "loss":
        return "fighter_2"
    elif result_lower == "draw":
        return "draw"
    elif result_lower in ("nc", "no contest"):
        return "no_contest"
    return None


def parse_method(method_str: str) -> tuple[Optional[str], Optional[str]]:
    """Parse method string into method and method_detail."""
    if not method_str:
        return None, None

    method_str = method_str.strip()
    method = None
    method_detail = None

    # Common patterns
    if "KO/TKO" in method_str:
        method = "KO/TKO"
        method_detail = method_str.replace("KO/TKO", "").strip()
    elif "SUB" in method_str:
        method = "SUB"
        method_detail = method_str.replace("SUB", "").strip()
    elif "Decision" in method_str:
        method = "Decision"
        method_detail = method_str.replace("Decision", "").strip()
    elif "DQ" in method_str:
        method = "DQ"
        method_detail = method_str.replace("DQ", "").strip()
    else:
        method = method_str

    return method, method_detail


def scrape_and_insert_fighter(
    fighter_id: str,
    fighter_url: str | None = None,
    dry_run: bool = False,
    skip_fights: bool = False,
    config_path: str = "config/config.yaml",
) -> dict:
    """
    Scrape fighter from UFCStats and insert/update in database.
    Also scrapes and inserts all fights, opponents, and events.

    Returns summary dict.
    """
    summary = {
        "fighter_id": fighter_id,
        "fighter_name": None,
        "scraped": False,
        "inserted": False,
        "fights_processed": 0,
        "fights_inserted": 0,
        "opponents_scraped": 0,
        "events_created": 0,
        "dry_run": dry_run,
        "errors": [],
    }

    fighter_scraper = FighterScraper(config_path=config_path)
    event_scraper = EventScraper(config_path=config_path)
    db = DatabaseManager(config_path=config_path)

    # Build URL if not provided
    if not fighter_url:
        fighter_url = f"{fighter_scraper.base_url}/fighter-details/{fighter_id}"
    fighter_url = normalize_url(fighter_url)

    # Scrape main fighter
    logger.info(f"Scraping fighter: {fighter_url}")
    fighter_data = fighter_scraper.scrape_fighter(fighter_url, fighter_id)

    if not fighter_data:
        logger.error(f"Failed to scrape fighter {fighter_id}")
        summary["errors"].append(f"Failed to scrape fighter {fighter_id}")
        return summary

    summary["scraped"] = True
    summary["fighter_name"] = fighter_data.get("name")

    logger.info(f"Scraped fighter: {fighter_data.get('name')}")
    logger.info(f"Record: {fighter_data.get('wins', 0)}-{fighter_data.get('losses', 0)}-{fighter_data.get('draws', 0)}")

    fight_history = fighter_data.get("fight_history", [])
    logger.info(f"Fights in history: {len(fight_history)}")

    if dry_run:
        logger.warning("Dry-run mode: not inserting into database")
        return summary

    session = db.get_session()

    try:
        # Insert/update main fighter
        main_fighter = db.add_fighter(session, fighter_data)
        session.flush()
        logger.success(f"Inserted/updated fighter: {main_fighter.name} (ID: {main_fighter.id})")
        summary["inserted"] = True

        if skip_fights or not fight_history:
            session.commit()
            return summary

        # Process each fight
        for fight_idx, fight_record in enumerate(reversed(fight_history), 1):
            # Process fights oldest to newest for proper ordering
            summary["fights_processed"] += 1

            try:
                opponent_url = fight_record.get("opponent_url")
                event_url = fight_record.get("event_url")

                if not opponent_url:
                    logger.warning(f"Fight {fight_idx}: No opponent URL, skipping")
                    continue

                # Get or create opponent fighter
                opponent_url = normalize_url(opponent_url)
                opponent_id = extract_id_from_url(opponent_url)

                opponent = session.query(Fighter).filter_by(fighter_id=opponent_id).first()
                if not opponent:
                    logger.info(f"Scraping opponent: {fight_record.get('opponent')}")
                    opponent_data = fighter_scraper.scrape_fighter(opponent_url, opponent_id)
                    if opponent_data:
                        opponent = db.add_fighter(session, opponent_data)
                        session.flush()
                        summary["opponents_scraped"] += 1
                        logger.debug(f"Added opponent: {opponent.name}")
                    else:
                        # Create minimal fighter record
                        minimal_data = {
                            "fighter_id": opponent_id,
                            "name": fight_record.get("opponent", "Unknown"),
                            "url": opponent_url,
                        }
                        opponent = db.add_fighter(session, minimal_data)
                        session.flush()
                        logger.warning(f"Created minimal opponent record: {opponent.name}")

                # Get or create event
                event = None
                if event_url:
                    event_url = normalize_url(event_url)
                    event_id = extract_id_from_url(event_url)
                    event = session.query(Event).filter_by(id=event_id).first()

                    if not event:
                        # Try to get event data from the event scraper
                        logger.info(f"Scraping event: {event_url}")
                        try:
                            event_data = event_scraper.scrape_event(event_url, event_id)
                            if event_data:
                                event = db.add_event(session, event_data)
                                session.flush()
                                summary["events_created"] += 1
                                logger.debug(f"Added event: {event.name}")
                        except Exception as e:
                            logger.warning(f"Could not scrape event {event_url}: {e}")

                if not event:
                    # Create a minimal event if we couldn't scrape it
                    event_name = fight_record.get("event") or "Unknown Event"
                    event_date = fight_record.get("date") or ""

                    # Try to find existing event by name first
                    event = session.query(Event).filter_by(name=event_name).first()

                    if not event:
                        # Generate a unique ID if we don't have one
                        if not event_url:
                            # Use a hash of the event name as ID
                            import hashlib
                            event_id = hashlib.md5(event_name.encode()).hexdigest()[:16]

                        minimal_event_data = {
                            "id": event_id,
                            "name": event_name,
                            "date": event_date,
                        }
                        event = db.add_event(session, minimal_event_data)
                        session.flush()
                        logger.warning(f"Created minimal event: {event_name}")

                # Determine result
                result_str = fight_record.get("result", "")
                # The main fighter is always fighter_1 in our representation
                result = parse_fight_result(result_str)

                # Parse method
                method, method_detail = parse_method(fight_record.get("method", ""))

                # Build fight data
                fight_data = {
                    "fight_id": fight_record.get("fight_detail_url", "").split("/")[-1] if fight_record.get("fight_detail_url") else None,
                    "result": result,
                    "method": method,
                    "method_detail": method_detail,
                    "round": fight_record.get("round"),
                    "time": fight_record.get("time"),
                    "fight_number": fight_idx,
                }

                # Create the fight
                # main_fighter is fighter_1, opponent is fighter_2
                fight = db.add_fight(session, fight_data, event, main_fighter, opponent)
                session.flush()
                summary["fights_inserted"] += 1
                logger.info(f"Added fight: {main_fighter.name} vs {opponent.name} at {event.name}")

            except Exception as e:
                logger.error(f"Error processing fight {fight_idx}: {e}")
                summary["errors"].append(f"Fight {fight_idx}: {e}")
                continue

        session.commit()
        logger.success(f"Committed all changes for {main_fighter.name}")

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to insert fighter/fights: {e}")
        summary["errors"].append(str(e))
        raise
    finally:
        session.close()

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape a single fighter from UFCStats and insert into database with all fights."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--fighter-id",
        type=str,
        help="UFCStats fighter ID (e.g., 294aa73dbf37d281)",
    )
    group.add_argument(
        "--fighter-url",
        type=str,
        help="Full UFCStats fighter URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape but don't insert into database",
    )
    parser.add_argument(
        "--skip-fights",
        action="store_true",
        help="Only insert fighter, skip fight history",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output scraped data as JSON (implies --dry-run)",
    )
    args = parser.parse_args()

    # Determine fighter_id and URL
    if args.fighter_url:
        fighter_url = args.fighter_url
        fighter_id = extract_id_from_url(fighter_url)
    else:
        fighter_id = args.fighter_id
        fighter_url = None

    # If --json, output raw data and exit
    if args.json:
        scraper = FighterScraper(config_path=args.config)
        if not fighter_url:
            fighter_url = f"{scraper.base_url}/fighter-details/{fighter_id}"
        fighter_url = normalize_url(fighter_url)
        fighter_data = scraper.scrape_fighter(fighter_url, fighter_id)
        if fighter_data:
            print(json.dumps(fighter_data, indent=2, default=str))
            return 0
        else:
            logger.error("Failed to scrape fighter")
            return 1

    # Normal mode
    summary = scrape_and_insert_fighter(
        fighter_id=fighter_id,
        fighter_url=fighter_url,
        dry_run=args.dry_run,
        skip_fights=args.skip_fights,
        config_path=args.config,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Fighter: {summary.get('fighter_name', 'Unknown')}")
    print(f"Scraped: {summary.get('scraped', False)}")
    print(f"Inserted: {summary.get('inserted', False)}")
    print(f"Fights processed: {summary.get('fights_processed', 0)}")
    print(f"Fights inserted: {summary.get('fights_inserted', 0)}")
    print(f"Opponents scraped: {summary.get('opponents_scraped', 0)}")
    print(f"Events created: {summary.get('events_created', 0)}")
    if summary.get("errors"):
        print(f"Errors: {len(summary['errors'])}")
        for err in summary["errors"][:5]:
            print(f"  - {err}")
    print("=" * 60)

    if summary["scraped"] and (args.dry_run or summary["inserted"]):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
