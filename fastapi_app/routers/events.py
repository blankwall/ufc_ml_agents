from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.schema import BettingOdds, Event, Fight, Fighter
from services.predict_service import FIGHTER_ALIASES, get_events_data

CONFIG_PATH = ROOT_DIR / "config" / "betting_config.json"

router = APIRouter()

# ── shared DB session factory ─────────────────────────────────────────────────
_DB_PATH = ROOT_DIR / "data" / "ufc_database.db"
_engine  = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine)


def _get_session():
    return _Session()


# ── helpers (mirrored from analysis/fighter_recent_fights.py) ─────────────────

def _parse_event_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _normalize(name: str) -> str:
    """Lowercase, collapse whitespace, remove apostrophes, replace hyphens with space."""
    s = name.lower().strip()
    s = re.sub(r"'", "", s)       # drop apostrophes (O'Malley → omalley)
    s = re.sub(r"-", " ", s)      # hyphen → space (Cortes-Acosta → cortes acosta)
    s = re.sub(r"\s+", " ", s)
    return s


def _most_recent_fight_date(session, fighter_id: int) -> datetime:
    """Return the most recent fight date for a fighter (datetime.min if none)."""
    fights = session.query(Fight).filter(
        (Fight.fighter_1_id == fighter_id) | (Fight.fighter_2_id == fighter_id)
    ).all()
    best = datetime.min
    for fight in fights:
        event = session.query(Event).filter_by(id=fight.event_id).first()
        d = _parse_event_date(event.date) if event else None
        if d and d > best:
            best = d
    return best


def _fmt_odds(val: Optional[int]) -> Optional[str]:
    if val is None:
        return None
    return f"+{val}" if val > 0 else str(val)


# ── endpoint ──────────────────────────────────────────────────────────────────

@router.get("/events")
async def api_events():
    """Return all events with fight predictions and outcome results."""
    return get_events_data()


@router.get("/config")
async def api_config():
    """Return the betting configuration."""
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="Config file not found")
    return json.loads(CONFIG_PATH.read_text())


@router.get("/fighter/{fighter_name}/recent")
async def fighter_recent(fighter_name: str, limit: int = 3):
    """Return a fighter's most recent fights with opening/closing odds."""
    session = _get_session()
    try:
        # Resolve known aliases (e.g. "Bobby Green" → "King Green")
        fighter_name = FIGHTER_ALIASES.get(fighter_name, fighter_name)
        target = _normalize(fighter_name)

        # Collect all candidates whose normalized name overlaps with the query
        candidates: list[Fighter] = []
        all_fighters = session.query(Fighter).all()
        for f in all_fighters:
            fn = _normalize(f.name)
            if fn == target or target in fn or fn in target:
                candidates.append(f)

        if not candidates:
            raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter_name}")

        # When multiple candidates (duplicate names), pick the one with the most recent fight
        if len(candidates) == 1:
            fighter = candidates[0]
        else:
            fighter = max(candidates, key=lambda f: _most_recent_fight_date(session, f.id))

        fights = session.query(Fight).filter(
            (Fight.fighter_1_id == fighter.id) | (Fight.fighter_2_id == fighter.id)
        ).all()

        fight_dates = []
        for fight in fights:
            event = session.query(Event).filter_by(id=fight.event_id).first()
            event_date = _parse_event_date(event.date) if event else None
            fight_dates.append((fight, event, event_date))

        fight_dates.sort(key=lambda x: x[2] if x[2] else datetime.min, reverse=True)
        recent = fight_dates[:limit]

        rows = []
        for fight, event, event_date in recent:
            is_f1 = fight.fighter_1_id == fighter.id
            opponent = fight.fighter_2 if is_f1 else fight.fighter_1
            opponent_name = opponent.name if opponent else "Unknown"

            result = "N/A"
            if fight.result:
                pos = 1 if is_f1 else 2
                if fight.result == f"fighter_{pos}":
                    result = "W"
                elif fight.result == "draw":
                    result = "D"
                elif fight.result == "no_contest":
                    result = "NC"
                else:
                    result = "L"

            odds_attr_open = "fighter_1_odds" if is_f1 else "fighter_2_odds"

            open_row = session.query(BettingOdds).filter_by(
                fight_id=fight.id, is_opening_line=True
            ).filter(getattr(BettingOdds, odds_attr_open).isnot(None)).first()

            close_row = session.query(BettingOdds).filter_by(
                fight_id=fight.id, is_closing_line=True
            ).filter(getattr(BettingOdds, odds_attr_open).isnot(None)).first()

            open_val  = getattr(open_row,  odds_attr_open, None) if open_row  else None
            close_val = getattr(close_row, odds_attr_open, None) if close_row else None

            rows.append({
                "result":        result,
                "opponent":      opponent_name,
                "event":         event.name if event else "Unknown",
                "event_date":    event.date if event else None,
                "open_odds":     _fmt_odds(open_val),
                "close_odds":    _fmt_odds(close_val),
            })

        return {
            "name":   fighter.name,
            "record": f"{fighter.wins}-{fighter.losses}-{fighter.draws}",
            "fights": rows,
        }
    finally:
        session.close()

