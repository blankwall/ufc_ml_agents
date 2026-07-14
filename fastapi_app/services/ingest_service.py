"""
Ingest Service
==============
Backs the /ingest UI. Lets an operator preview a fighter from a Sherdog URL and
then commit them to the DB, optionally writing/updating a name alias — replacing
the manual `curl /api/recover-fighter` + hand-editing FIGHTER_ALIASES workflow.

v1 scope: fighter identity + alias store. Re-pulling a fighter's per-fight stats
from UFC Stats (what the model actually consumes) is a later phase.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from sqlalchemy import or_

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from database.db_manager import DatabaseManager
from database.schema import Fighter, Fight
from fastapi_app.services import fighter_alias_service
from fastapi_app.services.predict_service import _resolve_fighter
from fastapi_app.services.sherdog_recovery_service import CONFIG_PATH, recover_fighter_from_url


def _fight_count(session, fighter: Fighter) -> int:
    return (
        session.query(Fight)
        .filter(or_(Fight.fighter_1_id == fighter.id, Fight.fighter_2_id == fighter.id))
        .count()
    )


def preview_fighter(sherdog_url: str, requested_name: Optional[str] = None) -> dict:
    """
    Dry-run scrape of a Sherdog fighter page. Writes nothing to the DB. Returns
    the scraped identity plus context to help the operator decide what to save:
    whether the fighter already exists, whether the requested name resolves, and
    whether an alias is needed.
    """
    requested_name = (requested_name or "").strip() or None

    scraped = recover_fighter_from_url(
        fighter_url=sherdog_url,
        requested_name=requested_name,
        dry_run=True,
        trigger="ingest_preview",
    )
    scraped_name = scraped.get("scraped_name") or scraped.get("requested_name")

    db = DatabaseManager(config_path=str(CONFIG_PATH))
    session = db.get_session()
    try:
        existing = (
            session.query(Fighter).filter_by(fighter_id=scraped.get("fighter_id")).first()
        )
        requested_resolves = _resolve_fighter(session, requested_name) if requested_name else None
        canonical_match = _resolve_fighter(session, scraped_name) if scraped_name else None
        has_history = (
            _fight_count(session, canonical_match) > 0 if canonical_match is not None else None
        )

        existing_alias = (
            fighter_alias_service.get_aliases().get(requested_name) if requested_name else None
        )
        suggested_alias = None
        alias_needed = False
        if requested_name and scraped_name and requested_name != scraped_name:
            suggested_alias = {"alias": requested_name, "canonical": scraped_name}
            alias_needed = existing_alias != scraped_name

        return {
            "status": "ok",
            "sherdog_url": scraped.get("fighter_url") or sherdog_url,
            "requested_name": requested_name,
            "scraped_name": scraped_name,
            "fighter_id": scraped.get("fighter_id"),
            "already_in_db": existing is not None,
            "db_name": existing.name if existing else (canonical_match.name if canonical_match else None),
            "requested_name_resolves_to": requested_resolves.name if requested_resolves else None,
            "has_fight_history": has_history,
            "existing_alias": existing_alias,
            "suggested_alias": suggested_alias,
            "alias_needed": alias_needed,
        }
    finally:
        session.close()


def save_fighter(
    sherdog_url: str,
    requested_name: Optional[str] = None,
    write_alias: bool = False,
    alias_from: Optional[str] = None,
    alias_to: Optional[str] = None,
    bust_cache: bool = False,
) -> dict:
    """
    Commit the fighter to the DB (via the shared recovery path) and, if
    requested, upsert an alias into config/fighter_aliases.json.
    """
    requested_name = (requested_name or "").strip() or None

    fighter = recover_fighter_from_url(
        fighter_url=sherdog_url,
        requested_name=requested_name,
        dry_run=False,
        bust_cache=bust_cache,
        trigger="ingest_save",
    )

    alias_result = None
    if write_alias:
        a_from = (alias_from or requested_name or "").strip()
        a_to = (alias_to or fighter.get("scraped_name") or fighter.get("db_name") or "").strip()
        if a_from and a_to and a_from != a_to:
            alias_result = fighter_alias_service.upsert_alias(a_from, a_to)

    return {"status": "saved", "fighter": fighter, "alias": alias_result}
