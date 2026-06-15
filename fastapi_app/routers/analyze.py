"""
GET  /api/matchup/{fighter1}/{fighter2}  — side-by-side stats + recent fights
POST /api/matchup/analyze               — AI analysis (independent of model/odds)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.schema import BettingOdds, Event, Fight, Fighter, FightStats
from services.fighter_identity import FIGHTER_ALIASES, resolve_fighter as _resolve_fighter
from services.ai_service import analyze_matchup

router = APIRouter()

_DB_PATH = ROOT_DIR / "data" / "ufc_database.db"
_engine  = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_date(d: str) -> Optional[datetime]:
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


def _fighter_profile(session, fighter: Fighter) -> dict:
    """Full profile: attributes + recent fights."""
    fights = session.query(Fight).filter(
        (Fight.fighter_1_id == fighter.id) | (Fight.fighter_2_id == fighter.id)
    ).all()

    fight_dates = []
    for fight in fights:
        event = session.query(Event).filter_by(id=fight.event_id).first()
        fight_dates.append((fight, event, _parse_date(event.date) if event else None))
    fight_dates.sort(key=lambda x: x[2] if x[2] else datetime.min, reverse=True)

    recent = []
    for fight, event, _ in fight_dates[:3]:
        is_f1 = fight.fighter_1_id == fighter.id
        opp   = fight.fighter_2 if is_f1 else fight.fighter_1
        pos   = 1 if is_f1 else 2
        result = "N/A"
        if fight.result:
            if fight.result == f"fighter_{pos}":   result = "W"
            elif fight.result == "draw":            result = "D"
            elif fight.result == "no_contest":      result = "NC"
            else:                                   result = "L"

        attr = "fighter_1_odds" if is_f1 else "fighter_2_odds"
        close_row = session.query(BettingOdds).filter_by(
            fight_id=fight.id, is_closing_line=True
        ).filter(getattr(BettingOdds, attr).isnot(None)).first()

        recent.append({
            "result":     result,
            "opponent":   opp.name if opp else "Unknown",
            "event":      event.name if event else "Unknown",
            "close_odds": _fmt_odds(getattr(close_row, attr, None) if close_row else None),
        })

    return {
        "name":         fighter.name,
        "record":       f"{fighter.wins}-{fighter.losses}-{fighter.draws}",
        "age":          fighter.age,
        "height_cm":    fighter.height_cm,
        "reach_inches": fighter.reach_inches,
        "stance":       fighter.stance,
        "weight_lbs":   fighter.weight_lbs,
        "sig_strikes_landed_per_min":    fighter.sig_strikes_landed_per_min,
        "striking_accuracy":             fighter.striking_accuracy,
        "sig_strikes_absorbed_per_min":  fighter.sig_strikes_absorbed_per_min,
        "striking_defense":              fighter.striking_defense,
        "takedown_avg_per_15min":        fighter.takedown_avg_per_15min,
        "takedown_accuracy":             fighter.takedown_accuracy,
        "takedown_defense":              fighter.takedown_defense,
        "submission_avg_per_15min":      fighter.submission_avg_per_15min,
        "fight_count":  len(fights),
        "recent_fights": recent,
    }


def _resolve(session, name: str) -> tuple[Optional[Fighter], str]:
    canonical = FIGHTER_ALIASES.get(name, name)
    fighter   = _resolve_fighter(session, canonical)
    return fighter, canonical


# ── GET /api/matchup/{fighter1}/{fighter2} ────────────────────────────────────

@router.get("/matchup/{fighter1}/{fighter2}")
async def get_matchup(fighter1: str, fighter2: str):
    """Return side-by-side fighter profiles (stats + recent fights)."""
    session = _Session()
    try:
        f1, f1_canon = _resolve(session, fighter1)
        f2, f2_canon = _resolve(session, fighter2)

        missing = [n for n, f in [(fighter1, f1), (fighter2, f2)] if not f]
        if missing:
            raise HTTPException(404, detail=f"Fighter(s) not found: {', '.join(missing)}")

        return {
            "fighter1": _fighter_profile(session, f1),
            "fighter2": _fighter_profile(session, f2),
        }
    finally:
        session.close()


# ── POST /api/matchup/analyze ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    fighter1: str
    fighter2: str


# ── GET /api/fight-stats/{fighter}/{opponent} ─────────────────────────────────

@router.get("/fight-stats/{fighter}/{opponent}")
async def get_fight_stats(fighter: str, opponent: str):
    """
    Return high-level fight stats for the most recent bout between two fighters.
    Returns totals (sig strikes, takedowns, KDs, control) + sig-strike breakdown.
    """
    session = _Session()
    try:
        f, _  = _resolve(session, fighter)
        opp, _ = _resolve(session, opponent)

        missing = [n for n, x in [(fighter, f), (opponent, opp)] if not x]
        if missing:
            raise HTTPException(404, detail=f"Fighter(s) not found: {', '.join(missing)}")

        fights = session.query(Fight).filter(
            ((Fight.fighter_1_id == f.id) & (Fight.fighter_2_id == opp.id)) |
            ((Fight.fighter_1_id == opp.id) & (Fight.fighter_2_id == f.id))
        ).all()

        if not fights:
            raise HTTPException(404, detail=f"No fight found between {f.name} and {opp.name}")

        # Sort by event date, pick most recent
        fight_dates = []
        for fi in fights:
            ev = session.query(Event).filter_by(id=fi.event_id).first()
            dt = _parse_date(ev.date) if ev else None
            fight_dates.append((fi, ev, dt))
        fight_dates.sort(key=lambda x: x[2] if x[2] else datetime.min, reverse=True)

        fi, ev, _ = fight_dates[0]

        is_f1 = fi.fighter_1_id == f.id
        pos = 1 if is_f1 else 2

        result = "N/A"
        if fi.result:
            if fi.result == f"fighter_{pos}": result = "W"
            elif fi.result == "draw":          result = "D"
            elif fi.result == "no_contest":    result = "NC"
            else:                              result = "L"

        stats = session.query(FightStats).filter_by(fight_id=fi.id).first()

        f_totals   = (stats.fighter_1_totals   if is_f1 else stats.fighter_2_totals)   if stats else None
        opp_totals = (stats.fighter_2_totals   if is_f1 else stats.fighter_1_totals)   if stats else None
        sig_key    = "fighter_1" if is_f1 else "fighter_2"
        opp_key    = "fighter_2" if is_f1 else "fighter_1"
        f_sig   = (stats.significant_strikes.get(sig_key)  if stats and stats.significant_strikes else None)
        opp_sig = (stats.significant_strikes.get(opp_key)  if stats and stats.significant_strikes else None)

        return {
            "event":         ev.name if ev else None,
            "event_date":    ev.date if ev else None,
            "result":        result,
            "method":        fi.method,
            "method_detail": fi.method_detail,
            "round":         fi.round_finished,
            "time":          fi.time,
            "fighter": {
                "name":        f.name,
                "totals":      f_totals,
                "sig_strikes": f_sig,
            },
            "opponent": {
                "name":        opp.name,
                "totals":      opp_totals,
                "sig_strikes": opp_sig,
            },
        }
    finally:
        session.close()


@router.post("/matchup/analyze")
async def ai_analyze(req: AnalyzeRequest):
    """
    Independent AI analysis using only fighter stats + history.
    Does NOT receive model probabilities or betting odds.
    Returns: winner, winner_pct, loser_pct, reasons[3].
    """
    session = _Session()
    try:
        f1, _ = _resolve(session, req.fighter1)
        f2, _ = _resolve(session, req.fighter2)

        missing = [n for n, f in [(req.fighter1, f1), (req.fighter2, f2)] if not f]
        if missing:
            raise HTTPException(404, detail=f"Fighter(s) not found: {', '.join(missing)}")

        f1_data = _fighter_profile(session, f1)
        f2_data = _fighter_profile(session, f2)

        result = analyze_matchup(req.fighter1, f1_data, req.fighter2, f2_data)

        if result.get("error") and not result.get("winner"):
            raise HTTPException(502, detail=result["error"])

        return result
    finally:
        session.close()
