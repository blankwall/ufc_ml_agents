"""
Database ingestion + fighter profile endpoints.

POST /api/db/ingest                  — scrape & upsert an event from UFCStats URL
GET  /api/db/fighters/search?q=ilia  — autocomplete search
GET  /api/db/fighter/{name}          — full profile + fight history
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.schema import BettingOdds, Event, Fight, Fighter
from scrapers.event_populator import EventPopulator, PopulatorOptions

router = APIRouter()

_DB_PATH = ROOT_DIR / "data" / "ufc_database.db"
_CONFIG_PATH = str(ROOT_DIR / "config" / "config.yaml")
_engine = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_date(d: str) -> Optional[datetime]:
    if not d:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    return None


def _fmt_odds(val: Optional[int]) -> Optional[str]:
    if val is None:
        return None
    return f"+{val}" if val > 0 else str(val)


def _display_pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value * 100, 1) if value <= 1 else round(value, 1)


def _fight_result_for_fighter(fight: Fight, fighter_id: int) -> str:
    if not fight.result:
        return "N/A"
    is_f1 = fight.fighter_1_id == fighter_id
    pos = 1 if is_f1 else 2
    if fight.result == f"fighter_{pos}":
        return "W"
    if fight.result == "draw":
        return "D"
    if fight.result == "no_contest":
        return "NC"
    return "L"


# ── POST /api/db/ingest ──────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    ufc_stats_url: str


def _run_ingest(url: str) -> dict:
    """Blocking ingestion — runs in a thread pool."""
    options = PopulatorOptions(
        include_fight_stats=True,
        force_refresh_fighters=True,
        bust_cache=True,
        commit=True,
    )
    populator = EventPopulator(config_path=_CONFIG_PATH)
    return populator.populate_event_from_url(url, options=options)


@router.post("/db/ingest")
async def ingest_event(req: IngestRequest):
    """Scrape a UFCStats event URL and upsert into DB, then return the results."""
    url = req.ufc_stats_url.strip()
    if not url:
        raise HTTPException(400, detail="ufc_stats_url is required")

    try:
        summary = await asyncio.get_event_loop().run_in_executor(
            None, partial(_run_ingest, url)
        )
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))

    event_name = summary.get("event_name")
    if not event_name:
        return {"summary": summary, "fights": []}

    # Pull back the just-ingested event from DB for display
    from database.db_manager import DatabaseManager
    db = DatabaseManager(config_path=_CONFIG_PATH)
    fights = db.get_event_results(event_name)

    # Also grab event metadata
    session = _Session()
    try:
        event = session.query(Event).filter(Event.name.ilike(f"%{event_name}%")).first()
        event_meta = None
        if event:
            event_meta = {
                "name": event.name,
                "date": event.date,
                "location": event.location,
            }
    finally:
        session.close()

    return {
        "summary": summary,
        "event": event_meta,
        "fights": fights,
    }


# ── GET /api/db/fighters/search ──────────────────────────────────────────────

@router.get("/db/fighters/search")
async def search_fighters(q: str = Query("", min_length=1)):
    """Search fighters by name (case-insensitive substring match)."""
    session = _Session()
    try:
        results = (
            session.query(Fighter)
            .filter(Fighter.name.ilike(f"%{q}%"))
            .order_by(Fighter.name)
            .limit(20)
            .all()
        )
        return [
            {
                "name": f.name,
                "fighter_id": f.fighter_id,
                "record": f"{f.wins}-{f.losses}-{f.draws}",
            }
            for f in results
        ]
    finally:
        session.close()


# ── GET /api/db/fighter/{name} ───────────────────────────────────────────────

@router.get("/db/fighter/{name}")
async def get_fighter_profile(name: str):
    """Full fighter profile + complete fight history from DB."""
    session = _Session()
    try:
        # Case-insensitive lookup: exact match first, then substring
        fighter = session.query(Fighter).filter(Fighter.name.ilike(name)).first()
        if not fighter:
            fighter = session.query(Fighter).filter(Fighter.name.ilike(f"%{name}%")).first()
        if not fighter:
            raise HTTPException(404, detail=f"Fighter not found: {name}")

        # All fights
        fights = session.query(Fight).filter(
            (Fight.fighter_1_id == fighter.id) | (Fight.fighter_2_id == fighter.id)
        ).all()

        fight_dates = []
        for fight in fights:
            event = session.query(Event).filter_by(id=fight.event_id).first()
            dt = _parse_date(event.date) if event else None
            fight_dates.append((fight, event, dt))
        fight_dates.sort(key=lambda x: x[2] if x[2] else datetime.min, reverse=True)

        # Build full fight history
        fight_history = []
        ufc_wins = 0
        ufc_losses = 0
        ufc_draws = 0
        ufc_no_contests = 0
        for fight, event, dt in fight_dates:
            is_f1 = fight.fighter_1_id == fighter.id
            opp = fight.fighter_2 if is_f1 else fight.fighter_1
            result = _fight_result_for_fighter(fight, fighter.id)
            if result == "W":
                ufc_wins += 1
            elif result == "L":
                ufc_losses += 1
            elif result == "D":
                ufc_draws += 1
            elif result == "NC":
                ufc_no_contests += 1

            # Closing odds
            attr = "fighter_1_odds" if is_f1 else "fighter_2_odds"
            close_row = session.query(BettingOdds).filter_by(
                fight_id=fight.id, is_closing_line=True
            ).filter(getattr(BettingOdds, attr).isnot(None)).first()

            fight_history.append({
                "date": event.date if event else None,
                "event": event.name if event else "Unknown",
                "opponent": opp.name if opp else "Unknown",
                "result": result,
                "method": fight.method,
                "round": fight.round_finished,
                "time": fight.time,
                "weight_class": fight.weight_class,
                "closing_odds": _fmt_odds(getattr(close_row, attr, None) if close_row else None),
            })

        # Nickname
        nickname = fighter.nickname
        decided_bouts = ufc_wins + ufc_losses + ufc_draws
        total_bouts = len(fights)
        win_rate = round((fighter.wins / max(fighter.wins + fighter.losses + fighter.draws, 1)) * 100, 1)
        ufc_win_rate = round((ufc_wins / max(decided_bouts, 1)) * 100, 1) if decided_bouts else None

        return {
            "name": fighter.name,
            "nickname": nickname,
            "record": f"{fighter.wins}-{fighter.losses}-{fighter.draws}",
            "overall_record": {
                "wins": fighter.wins,
                "losses": fighter.losses,
                "draws": fighter.draws,
                "no_contests": fighter.no_contests or 0,
            },
            "ufc_record": {
                "wins": ufc_wins,
                "losses": ufc_losses,
                "draws": ufc_draws,
                "no_contests": ufc_no_contests,
            },
            "age": fighter.age,
            "stance": fighter.stance,
            "height_cm": fighter.height_cm,
            "weight_lbs": fighter.weight_lbs,
            "reach_inches": fighter.reach_inches,
            "sig_strikes_landed_per_min": fighter.sig_strikes_landed_per_min,
            "striking_accuracy": fighter.striking_accuracy,
            "striking_accuracy_pct": _display_pct(fighter.striking_accuracy),
            "sig_strikes_absorbed_per_min": fighter.sig_strikes_absorbed_per_min,
            "striking_defense": fighter.striking_defense,
            "striking_defense_pct": _display_pct(fighter.striking_defense),
            "takedown_avg_per_15min": fighter.takedown_avg_per_15min,
            "takedown_accuracy": fighter.takedown_accuracy,
            "takedown_accuracy_pct": _display_pct(fighter.takedown_accuracy),
            "takedown_defense": fighter.takedown_defense,
            "takedown_defense_pct": _display_pct(fighter.takedown_defense),
            "submission_avg_per_15min": fighter.submission_avg_per_15min,
            "fight_count": total_bouts,
            "ufc_bout_count": total_bouts,
            "win_rate_pct": win_rate,
            "ufc_win_rate_pct": ufc_win_rate,
            "recent_form": [fight["result"] for fight in fight_history[:5]],
            "fight_history": fight_history,
        }
    finally:
        session.close()
