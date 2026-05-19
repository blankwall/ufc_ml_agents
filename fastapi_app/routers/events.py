from __future__ import annotations

import csv
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
from services.predict_service import FIGHTER_ALIASES, _resolve_fighter, get_events_data
from services.the_odds_api_service import get_sampled_odds_history, toggle_bet_placed

CONFIG_PATH = ROOT_DIR / "config" / "betting_config.json"
EXTERNAL_BETS_PATH = Path("/tmp/odds.csv")

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
    cleaned = re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", str(date_str).strip(), flags=re.IGNORECASE)
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(cleaned, fmt)
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


def _normalize_label(value: str) -> str:
    normalized = _normalize(value)
    normalized = re.sub(r"[^a-z0-9 ]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _tokenize_label(value: str) -> set[str]:
    stopwords = {"ufc", "fight", "night", "vs", "the", "and"}
    return {token for token in _normalize_label(value).split() if token and token not in stopwords}


def _event_match_score(target_event: str, candidate_event: str) -> float:
    target_norm = _normalize_label(target_event)
    candidate_norm = _normalize_label(candidate_event)
    if not target_norm or not candidate_norm:
        return 0.0
    if target_norm == candidate_norm:
        return 10.0
    if target_norm in candidate_norm or candidate_norm in target_norm:
        return 8.0

    target_tokens = _tokenize_label(target_event)
    candidate_tokens = _tokenize_label(candidate_event)
    if not target_tokens or not candidate_tokens:
        return 0.0

    overlap = len(target_tokens & candidate_tokens)
    if not overlap:
        return 0.0

    score = overlap / max(len(target_tokens), len(candidate_tokens))
    ufc_number_target = re.search(r"\bufc\s+(\d+)\b", target_norm)
    ufc_number_candidate = re.search(r"\bufc\s+(\d+)\b", candidate_norm)
    if ufc_number_target and ufc_number_candidate and ufc_number_target.group(1) == ufc_number_candidate.group(1):
        score += 1.0
    return score


def _parse_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text or text.upper() == "N/A":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value) -> Optional[int]:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _did_bet_win(bet_fighter: str, winner: Optional[str]) -> Optional[bool]:
    if not winner:
        return None
    bet_name = FIGHTER_ALIASES.get(str(bet_fighter), str(bet_fighter))
    winner_name = FIGHTER_ALIASES.get(str(winner), str(winner))
    return _normalize(bet_name) == _normalize(winner_name)


def _canonical_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return str(FIGHTER_ALIASES.get(str(name), str(name)))


def _build_bet_summary(
    *,
    fighter: str,
    opponent: Optional[str],
    stake: float,
    odds: int,
    listed_odds: Optional[int] = None,
    opponent_listed_odds: Optional[int] = None,
    placed_at: Optional[str] = None,
    winner: Optional[str] = None,
    pnl_override: Optional[float] = None,
) -> Optional[dict]:
    try:
        stake = float(stake)
        odds = int(odds)
    except (TypeError, ValueError):
        return None

    if stake <= 0 or odds == 0:
        return None

    won = _did_bet_win(fighter, winner)
    pnl = round(pnl_override, 2) if pnl_override is not None else None
    if pnl is None:
        if won is True:
            pnl = round(stake * (odds / 100 if odds > 0 else 100 / abs(odds)), 2)
        elif won is False:
            pnl = round(-stake, 2)

    return {
        "fighter": fighter,
        "opponent": opponent,
        "stake": stake,
        "odds": odds,
        "listed_odds": listed_odds,
        "opponent_listed_odds": opponent_listed_odds,
        "placed_at": placed_at,
        "settled": won is not None,
        "won": won,
        "pnl": pnl,
        "risk": stake,
    }


def _tracked_bet_summary(fight: dict) -> Optional[dict]:
    bet = fight.get("bet_placed")
    if not bet:
        return None
    return _build_bet_summary(
        fighter=str(bet.get("fighter", "")),
        opponent=bet.get("opponent"),
        stake=bet.get("stake"),
        odds=bet.get("bet_odds"),
        listed_odds=bet.get("listed_odds"),
        opponent_listed_odds=bet.get("opponent_listed_odds"),
        placed_at=bet.get("placed_at"),
        winner=fight.get("winner"),
    )


def _build_card_entry(
    *,
    event_name: str,
    event_date: Optional[str],
    fighter1: Optional[str],
    fighter2: Optional[str],
    bet: dict,
    edge: Optional[float],
    winner: Optional[str],
    method: Optional[str],
    round_number,
    source_type: Optional[str],
    source_label: Optional[str] = None,
    bet_type: Optional[str] = None,
    model_prob: Optional[float] = None,
    manual_confidence: Optional[float] = None,
    notes: Optional[str] = None,
) -> dict:
    return {
        "event_name": event_name,
        "event_date": event_date,
        "fighter1": fighter1,
        "fighter2": fighter2,
        "matchup": f"{fighter1} vs {fighter2}" if fighter1 and fighter2 else bet.get("fighter", "Unknown"),
        "bet": bet,
        "model_pick": bet.get("fighter"),
        "edge": edge,
        "winner": winner,
        "method": method,
        "round": round_number,
        "source_type": source_type,
        "source_label": source_label,
        "bet_type": bet_type,
        "model_prob": model_prob,
        "manual_confidence": manual_confidence,
        "notes": notes,
    }


def _winner_name_for_fight(fight: Fight) -> Optional[str]:
    if fight.result == "draw":
        return "Draw"
    if fight.result == "no_contest":
        return "No Contest"
    if fight.winner_id == fight.fighter_1_id and fight.fighter_1:
        return fight.fighter_1.name
    if fight.winner_id == fight.fighter_2_id and fight.fighter_2:
        return fight.fighter_2.name
    return None


def _db_fight_payload(fight: Fight) -> dict:
    return {
        "fighter1": fight.fighter_1.name if fight.fighter_1 else None,
        "fighter2": fight.fighter_2.name if fight.fighter_2 else None,
        "winner": _winner_name_for_fight(fight),
        "method": fight.method,
        "round": fight.round_finished,
        "source_type": "database",
    }


def _tracked_bet_entries(events: list[dict]) -> list[dict]:
    entries: list[dict] = []
    for event in events:
        if event.get("source_type") != "the_odds_api":
            continue
        event_name = str(event.get("event_name", "")).strip()
        if not event_name.startswith("MMA Card"):
            continue

        for fight in event.get("fights", []):
            tracked_bet = _tracked_bet_summary(fight)
            if not tracked_bet:
                continue
            entries.append(
                _build_card_entry(
                    event_name=event_name,
                    event_date=event.get("event_date"),
                    fighter1=fight.get("fighter1"),
                    fighter2=fight.get("fighter2"),
                    bet=tracked_bet,
                    edge=fight.get("edge"),
                    winner=fight.get("winner"),
                    method=fight.get("method"),
                    round_number=fight.get("round"),
                    source_type=event.get("source_type"),
                    source_label="Odds API tracked",
                )
            )
    return entries


def _read_external_bets_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            reader = csv.DictReader(handle, dialect=dialect)
        except csv.Error:
            reader = csv.DictReader(handle, delimiter="\t")
        return [row for row in reader if row]


def _find_matching_fight_in_db(session, event_name: str, event_date: Optional[str], bet_on: str) -> Optional[dict]:
    target_date = _parse_event_date(event_date) if event_date else None
    if not target_date or not bet_on.strip():
        return None

    candidates: list[tuple[float, int, Fight]] = []
    fighter = _resolve_fighter(session, bet_on)
    if fighter is None:
        return None
    fights = (
        session.query(Fight)
        .join(Event, Event.id == Fight.event_id)
        .filter((Fight.fighter_1_id == fighter.id) | (Fight.fighter_2_id == fighter.id))
        .all()
    )

    for fight in fights:
        if not fight.event:
            continue
        current_date = _parse_event_date(fight.event.date)
        if not current_date:
            continue
        date_distance = abs((current_date.date() - target_date.date()).days)
        if date_distance > 1:
            continue
        score = _event_match_score(event_name, fight.event.name)
        if date_distance == 0:
            score += 2.0
        elif date_distance == 1:
            score += 1.0
        candidates.append((score, -date_distance, fight))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2].id), reverse=True)
    best_score, _, fight = candidates[0]
    if best_score <= 0:
        return None
    return {"event": {"event_name": fight.event.name, "event_date": fight.event.date, "source_type": "database"}, "fight": _db_fight_payload(fight)}


def _find_matching_fight(events: list[dict], event_name: str, event_date: Optional[str], bet_on: str) -> Optional[dict]:
    bet_name = _normalize(bet_on)
    target_date = _parse_event_date(event_date) if event_date else None
    target_event = _normalize_label(event_name)
    matches: list[tuple[int, dict, dict]] = []

    for event in events:
        current_date = _parse_event_date(str(event.get("event_date", "")))
        if target_date and current_date and current_date.date() != target_date.date():
            continue

        event_score = 0
        current_name = _normalize_label(str(event.get("event_name", "")))
        if target_event and current_name == target_event:
            event_score = 3
        elif target_event and target_event in current_name:
            event_score = 2
        elif target_event and current_name in target_event:
            event_score = 1

        for fight in event.get("fights", []):
            fighter1 = _normalize(str(fight.get("fighter1", "")))
            fighter2 = _normalize(str(fight.get("fighter2", "")))
            if bet_name not in {fighter1, fighter2}:
                continue
            matches.append((event_score, event, fight))

    if not matches:
        return None

    matches.sort(
        key=lambda item: (
            item[0],
            _parse_event_date(str(item[1].get("event_date", ""))) or datetime.min,
        ),
        reverse=True,
    )
    _, event, fight = matches[0]
    return {"event": event, "fight": fight}


def _external_csv_bet_entries(events: list[dict]) -> list[dict]:
    entries: list[dict] = []
    tracked_keys = {
        (
            str(entry.get("event_date", "")),
            _normalize(str(entry.get("fighter1", ""))),
            _normalize(str(entry.get("fighter2", ""))),
            _normalize(str(entry["bet"].get("fighter", ""))),
        )
        for entry in _tracked_bet_entries(events)
    }
    session = _get_session()

    try:
        for row in _read_external_bets_rows(EXTERNAL_BETS_PATH):
            event_name = str(row.get("event", "")).strip()
            bet_on = str(row.get("bet_on", "")).strip()
            event_date = _parse_event_date(str(row.get("date", "")).strip())
            odds = _parse_int(row.get("market_odds"))
            stake = _parse_float(row.get("stake"))
            if not event_name or not bet_on or event_date is None or odds is None or stake is None:
                continue

            normalized_date = event_date.strftime("%Y-%m-%d")
            matched = _find_matching_fight_in_db(session, event_name=event_name, event_date=normalized_date, bet_on=bet_on)
            if not matched:
                matched = _find_matching_fight(events, event_name=event_name, event_date=normalized_date, bet_on=bet_on)
            fight = matched["fight"] if matched else {}
            matched_event = matched["event"] if matched else {}

            fighter1 = fight.get("fighter1")
            fighter2 = fight.get("fighter2")
            opponent = None
            if fighter1 and fighter2:
                canonical_bet_on = _normalize(_canonical_name(bet_on))
                canonical_f1 = _normalize(_canonical_name(str(fighter1)))
                opponent = fighter2 if canonical_f1 == canonical_bet_on else fighter1

            dedupe_key = (
                normalized_date,
                _normalize(str(fighter1 or "")),
                _normalize(str(fighter2 or "")),
                _normalize(bet_on),
            )
            if fighter1 and fighter2 and dedupe_key in tracked_keys:
                continue

            display_event_name = str(matched_event.get("event_name") or event_name)
            display_event_date = _parse_event_date(str(matched_event.get("event_date", "")))
            event_date_value = (display_event_date or event_date).strftime("%Y-%m-%d")

            bet = _build_bet_summary(
                fighter=bet_on,
                opponent=opponent,
                stake=stake,
                odds=odds,
                listed_odds=odds,
                opponent_listed_odds=_parse_int(row.get("oppenent_odds")),
                placed_at=None,
                winner=fight.get("winner"),
                pnl_override=_parse_float(row.get("PNL")),
            )
            if not bet:
                continue

            entries.append(
                _build_card_entry(
                    event_name=display_event_name,
                    event_date=event_date_value,
                    fighter1=fighter1,
                    fighter2=fighter2,
                    bet=bet,
                    edge=_parse_float(row.get("edge")) if row.get("edge") else fight.get("edge"),
                    winner=fight.get("winner"),
                    method=fight.get("method"),
                    round_number=fight.get("round"),
                    source_type=str(matched_event.get("source_type", "external_csv")),
                    source_label="Imported from /tmp/odds.csv",
                    bet_type=str(row.get("type", "")).strip() or None,
                    model_prob=_parse_float(row.get("model_prob")),
                    manual_confidence=_parse_float(row.get("manual_confidence")),
                    notes=str(row.get("Notes", "")).strip() or None,
                )
            )
    finally:
        session.close()

    return entries


def _group_bet_entries(entries: list[dict]) -> list[dict]:
    grouped_map: dict[tuple[str, str], dict] = {}

    for entry in entries:
        event_name = str(entry.get("event_name", "")).strip()
        event_date = str(entry.get("event_date", "")).strip()
        key = (event_date, event_name)
        card = grouped_map.get(key)
        if not card:
            card = {
                "event_name": event_name,
                "event_date": event_date,
                "bets": [],
            }
            grouped_map[key] = card
        card["bets"].append(entry)

    grouped = []
    for card in grouped_map.values():
        settled = [bet for bet in card["bets"] if bet["bet"]["settled"] and bet["bet"]["pnl"] is not None]
        wins = sum(1 for bet in settled if bet["bet"]["won"] is True)
        losses = sum(1 for bet in settled if bet["bet"]["won"] is False)
        total_risk = round(sum(bet["bet"]["risk"] for bet in settled), 2)
        total_pnl = round(sum(bet["bet"]["pnl"] for bet in settled), 2)
        grouped.append(
            {
                "event_name": card["event_name"],
                "event_date": card["event_date"],
                "bet_count": len(card["bets"]),
                "settled_count": len(settled),
                "wins": wins,
                "losses": losses,
                "pending_count": len(card["bets"]) - len(settled),
                "accuracy": round((wins / len(settled)) * 100, 1) if settled else None,
                "roi": round((total_pnl / total_risk) * 100, 1) if total_risk else None,
                "total_risk": total_risk,
                "total_pnl": total_pnl,
                "bets": card["bets"],
            }
        )

    grouped.sort(key=lambda item: _parse_event_date(str(item.get("event_date", ""))) or datetime.min, reverse=True)
    return grouped


def _build_bets_payload(events: list[dict]) -> list[dict]:
    entries = _tracked_bet_entries(events)
    entries.extend(_external_csv_bet_entries(events))
    return _group_bet_entries(entries)


def _extract_unresolved_fighters(error: Optional[str]) -> list[str]:
    if not error:
        return []
    _, _, raw_names = str(error).partition(":")
    if not raw_names:
        return []
    return [name.strip() for name in raw_names.split(",") if name.strip()]


def _build_unresolved_fighter_payload(events: list[dict]) -> list[dict]:
    unresolved = []
    for event in events:
        for fight in event.get("fights", []):
            if fight.get("model_source") != "not_found":
                continue
            unresolved.append({
                "event_name": event.get("event_name"),
                "event_date": event.get("event_date"),
                "source_type": event.get("source_type"),
                "fighter1": fight.get("fighter1"),
                "fighter2": fight.get("fighter2"),
                "matchup": f"{fight.get('fighter1')} vs {fight.get('fighter2')}",
                "error": fight.get("error"),
                "unresolved_fighters": _extract_unresolved_fighters(fight.get("error")),
            })
    return unresolved


# ── endpoint ──────────────────────────────────────────────────────────────────

@router.get("/events")
async def api_events():
    """Return all events with fight predictions and outcome results."""
    return get_events_data()


@router.get("/events/unresolved-fighters")
async def api_unresolved_fighters():
    """Return only event fights blocked by unresolved fighter names."""
    return _build_unresolved_fighter_payload(get_events_data())


@router.get("/config")
async def api_config():
    """Return the betting configuration."""
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="Config file not found")
    return json.loads(CONFIG_PATH.read_text())


@router.get("/bets")
async def api_bets():
    """Return grouped tracked bets for The Odds API MMA Card events."""
    return _build_bets_payload(get_events_data())


@router.get("/odds-history")
async def api_odds_history(fighter1: str, fighter2: str, event_date: str):
    """Return first/middle/latest sampled odds history for the new The Odds API flow."""
    data = get_sampled_odds_history(event_date=event_date, fighter1=fighter1, fighter2=fighter2)
    if not data:
        raise HTTPException(status_code=404, detail="Odds history not found for that matchup")
    return data


@router.get("/odds-history/bet-toggle")
async def api_toggle_odds_history_bet(
    fighter1: str,
    fighter2: str,
    event_date: str,
    bet_fighter: str,
    stake: float | None = None,
    custom_odds: int | None = None,
):
    data = toggle_bet_placed(
        event_date=event_date,
        fighter1=fighter1,
        fighter2=fighter2,
        bet_fighter=bet_fighter,
        stake=stake,
        custom_odds=custom_odds,
    )
    if not data:
        raise HTTPException(status_code=404, detail="Fight not found for bet toggle")
    return data


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
