"""
UFC Schedule Allowlist Service
==============================
The Odds API ``mma_mixed_martial_arts`` feed returns *every* MMA promotion's
bouts (UFC, PFL, etc.) with no promotion label, and predict_service groups them
by date into a single "MMA Card". That lets non-UFC fights (e.g. a PFL "Tyson
Pedro") leak in and get scored/bet by the UFC-only model — sometimes resolving
to a same-named UFC fighter and emitting a confident bogus pick.

This service builds a robust **UFC allowlist** from the authoritative upcoming
schedule on UFC Stats (``/statistics/events/upcoming`` + per-event bout lists),
caches it to ``data/ufc_schedule/upcoming_allowlist.json``, and exposes
``is_ufc_fight()`` so predict_service can drop odds fights that are not on a real
upcoming UFC card.

Name matching reuses the existing alias DB via ``resolve_fighter`` (which chains
``config/fighter_aliases.json`` + DB fuzzy resolution) so variants like
"Steve Erceg"/"Stephen Erceg" collapse to the same DB fighter id. For debut
fighters not yet in the DB (``resolve_fighter`` -> None), a thin normalized
last-name/first-name fuzzy fallback is used.

Refreshing is decoupled from serving: ``refresh_allowlist()`` does the scraping
(call it from cron / a script / a manual endpoint); ``is_ufc_fight()`` only reads
the cached file and **fails open** (does not filter) when the allowlist is
missing or stale, so a scrape outage can never hide a real UFC card.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi_app.services.fighter_identity import (
    FIGHTER_ALIASES,
    resolve_fighter as _resolve_fighter,
)

SCHEDULE_DIR = ROOT_DIR / "data" / "ufc_schedule"
ALLOWLIST_PATH = SCHEDULE_DIR / "upcoming_allowlist.json"
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

# Allowlist is only trusted for strict filtering while this fresh; older than
# this we fail open (don't filter) so a stuck cron never hides a real UFC card.
FRESH_MAX_DAYS = 4
# How far ahead to capture upcoming events (odds sync window is ~31 days).
DEFAULT_MAX_DAYS_AHEAD = 40
# Date tolerance when matching an odds card date to a UFC Stats event date
# (US events can cross UTC midnight, shifting the odds commence date by a day).
DATE_TOLERANCE_DAYS = 1
# Throttle lazy bootstrap scrapes (only used when the file is entirely missing).
_BOOTSTRAP_THROTTLE_SEC = 1800

_lock = threading.RLock()
_UNSET = object()
_MEMO: dict[str, Any] = {"mtime": None, "data": None, "slots": None}
_ID_CACHE: dict[str, Optional[int]] = {}
_last_bootstrap_at = 0.0


# ── time / date helpers ───────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_ufcstats_date(value: str | None) -> Optional[date]:
    """UFC Stats renders event dates as 'July 25, 2026'."""
    if not value:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_card_date(value: str | None) -> Optional[date]:
    """Odds card dates are ISO 'YYYY-MM-DD'; tolerate a few other shapes."""
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    iso = _parse_iso(text)
    return iso.date() if iso else None


# ── name normalization / fuzzy fallback ───────────────────────────────────────

def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _name_match(a: str, b: str) -> bool:
    """Thin fuzzy fallback for fighters not resolvable via the alias DB/DB.

    Matches on last-name (exact or one a prefix of the other, >=4 chars) plus a
    lenient first-name check (exact, shared 3-char prefix, or prefix subset).
    Handles 'Muhammad Saidov' vs 'Muhammad Said', 'Steve' vs 'Stephen'.
    """
    ta, tb = _norm(a).split(), _norm(b).split()
    if not ta or not tb:
        return False
    la, lb = ta[-1], tb[-1]
    last_ok = la == lb or (len(la) >= 4 and len(lb) >= 4 and (la.startswith(lb) or lb.startswith(la)))
    if not last_ok:
        return False
    fa, fb = ta[0], tb[0]
    return fa == fb or fa[:3] == fb[:3] or fa.startswith(fb) or fb.startswith(fa)


def _canonical_name(name: str) -> str:
    raw = str(name).strip()
    return str(FIGHTER_ALIASES.get(raw, raw)).strip()


# ── identity resolution (reuses the alias DB via resolve_fighter) ─────────────

def _resolve_id(session, name: str) -> Optional[int]:
    key = name.strip()
    cached = _ID_CACHE.get(key, _UNSET)
    if cached is not _UNSET:
        return cached
    fid: Optional[int] = None
    try:
        f = _resolve_fighter(session, key)
        fid = f.id if f else None
    except Exception:
        fid = None
    _ID_CACHE[key] = fid
    return fid


def _fighter_matches_slot(session, name: str, slot: dict) -> bool:
    """A carded 'slot' is {'id': int|None, 'name': raw}."""
    nid = _resolve_id(session, name)
    if slot.get("id") is not None and nid is not None and slot["id"] == nid:
        return True
    return _name_match(name, slot.get("name", ""))


def _find_slot(session, name: str, slots: list[dict], skip: Optional[int]) -> Optional[int]:
    for idx, slot in enumerate(slots):
        if idx == skip:
            continue
        if _fighter_matches_slot(session, name, slot):
            return idx
    return None


def _event_slots(session, ev: dict) -> list[dict]:
    """Flatten an event's bouts into distinct carded-fighter slots with ids."""
    seen: set[str] = set()
    slots: list[dict] = []
    for bout in ev.get("bouts", []) or []:
        for raw in _bout_names(bout):
            key = _norm(raw)
            if not key or key in seen:
                continue
            seen.add(key)
            slots.append({"id": _resolve_id(session, raw), "name": raw})
    return slots


def _bout_names(bout: Any) -> tuple[str, str]:
    """Read fighter names from both legacy list bouts and metadata-rich bouts."""
    if isinstance(bout, dict):
        return (
            str(bout.get("fighter1", "")).strip(),
            str(bout.get("fighter2", "")).strip(),
        )
    if isinstance(bout, (list, tuple)) and len(bout) >= 2:
        return str(bout[0]).strip(), str(bout[1]).strip()
    return "", ""


def find_upcoming_bout(
    fighter1: str,
    fighter2: str,
    event_date: str,
    *,
    tolerance_days: int = DATE_TOLERANCE_DAYS,
) -> Optional[dict[str, Any]]:
    """Return authoritative UFCStats metadata for an upcoming matchup.

    New allowlist files retain weight class and card order. Legacy list-only
    files still match the bout but return ``None`` for those metadata fields.
    """
    al = load_allowlist()
    target = _parse_card_date(event_date)
    if not al or target is None:
        return None

    canonical_fighter1 = _canonical_name(fighter1)
    canonical_fighter2 = _canonical_name(fighter2)

    for event in al.get("events", []):
        current_date = _parse_card_date(event.get("date"))
        if current_date is None or abs((current_date - target).days) > tolerance_days:
            continue
        for bout in event.get("bouts", []) or []:
            left, right = _bout_names(bout)
            if not left or not right:
                continue
            direct = (
                _name_match(canonical_fighter1, _canonical_name(left))
                and _name_match(canonical_fighter2, _canonical_name(right))
            )
            reverse = (
                _name_match(canonical_fighter1, _canonical_name(right))
                and _name_match(canonical_fighter2, _canonical_name(left))
            )
            if not direct and not reverse:
                continue
            metadata = bout if isinstance(bout, dict) else {}
            return {
                "event_name": event.get("name"),
                "event_date": event.get("date"),
                "event_url": event.get("url"),
                "fighter1": left,
                "fighter2": right,
                "weight_class": metadata.get("weight_class"),
                "fight_number": metadata.get("fight_number"),
            }
    return None


# ── allowlist load / cache ────────────────────────────────────────────────────

def load_allowlist() -> Optional[dict]:
    """Return the cached allowlist dict (memoized by file mtime), or None."""
    with _lock:
        if not ALLOWLIST_PATH.exists():
            _maybe_bootstrap()
            if not ALLOWLIST_PATH.exists():
                return None
        try:
            mtime = ALLOWLIST_PATH.stat().st_mtime
        except OSError:
            return None
        if _MEMO["mtime"] != mtime:
            try:
                _MEMO["data"] = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                _MEMO["data"] = None
            _MEMO["mtime"] = mtime
            _MEMO["slots"] = None
            _ID_CACHE.clear()
        return _MEMO["data"]


def _maybe_bootstrap() -> None:
    """One throttled best-effort scrape when the file is entirely missing."""
    global _last_bootstrap_at
    now = time.monotonic()
    if now - _last_bootstrap_at < _BOOTSTRAP_THROTTLE_SEC:
        return
    _last_bootstrap_at = now
    try:
        refresh_allowlist()
    except Exception as exc:  # never let a scrape failure break serving
        logger.warning("UFC allowlist lazy bootstrap failed: {}", exc)


# ── public query API ──────────────────────────────────────────────────────────

def is_ufc_fight(session, fighter1: str, fighter2: str, event_date: str,
                 *, tolerance_days: int = DATE_TOLERANCE_DAYS) -> tuple[bool, str]:
    """Return (allowed, reason).

    Fails **open** (allowed=True) whenever the allowlist is missing, unparsable,
    or stale, so an outage can never hide a real UFC card. When a fresh allowlist
    is present, an odds fight is allowed only if BOTH fighters appear as two
    distinct fighters on the SAME upcoming UFC event within the date tolerance.
    """
    al = load_allowlist()
    if not al or not al.get("events"):
        return True, "allowlist_unavailable"

    generated = _parse_iso(al.get("generated_at"))
    if generated is None or (_utcnow() - generated).days > FRESH_MAX_DAYS:
        return True, "allowlist_stale"

    target = _parse_card_date(event_date)
    if target is None:
        return True, "unparsable_card_date"

    # The upcoming-schedule allowlist only has authority over the date range it
    # covers. Fail open for anything outside it — past/completed events (which
    # still carry odds rows for results tracking) and far-future dates beyond
    # what we scraped must never be dropped.
    ev_dates = [d for d in (_parse_card_date(e.get("date")) for e in al["events"]) if d]
    if not ev_dates:
        return True, "allowlist_unavailable"
    if target < min(ev_dates) - timedelta(days=tolerance_days) or \
       target > max(ev_dates) + timedelta(days=tolerance_days):
        return True, "outside_allowlist_window"

    for ev in al["events"]:
        ev_date = _parse_card_date(ev.get("date"))
        if ev_date is None or abs((ev_date - target).days) > tolerance_days:
            continue
        slots = _event_slots(session, ev)
        i1 = _find_slot(session, fighter1, slots, skip=None)
        if i1 is None:
            continue
        i2 = _find_slot(session, fighter2, slots, skip=i1)
        if i2 is None:
            continue
        return True, f"matched:{ev.get('name', '')}"

    return False, "not_on_ufc_card"


def get_status() -> dict[str, Any]:
    """Lightweight health/debug summary of the cached allowlist."""
    al = load_allowlist()
    if not al:
        return {"available": False, "events": 0, "bouts": 0, "generated_at": None,
                "age_hours": None, "fresh": False}
    generated = _parse_iso(al.get("generated_at"))
    age_h = None
    fresh = False
    if generated is not None:
        age_h = round((_utcnow() - generated).total_seconds() / 3600, 1)
        fresh = (_utcnow() - generated).days <= FRESH_MAX_DAYS
    return {
        "available": True,
        "events": len(al.get("events", [])),
        "bouts": sum(len(e.get("bouts", [])) for e in al.get("events", [])),
        "generated_at": al.get("generated_at"),
        "age_hours": age_h,
        "fresh": fresh,
        "dates": [e.get("date") for e in al.get("events", [])],
    }


# ── refresh (scrape UFC Stats upcoming) ───────────────────────────────────────

def refresh_allowlist(*, max_days_ahead: int = DEFAULT_MAX_DAYS_AHEAD,
                      max_pages: int = 1) -> dict[str, Any]:
    """Scrape the upcoming UFC schedule + bout lists and write the allowlist.

    DB-free (fighter ids are resolved lazily at query time). Returns a summary.
    """
    from scrapers.event_scraper import EventScraper

    es = EventScraper(config_path=str(CONFIG_PATH))
    upcoming = es.get_all_event_links(completed_only=False, max_pages=max_pages)

    today = date.today()
    lower = today - timedelta(days=2)
    upper = today + timedelta(days=max_days_ahead)

    events: list[dict] = []
    for entry in upcoming:
        ev_date = _parse_ufcstats_date(entry.get("date"))
        if ev_date is None or ev_date < lower or ev_date > upper:
            continue
        url = str(entry.get("url", "")).strip()
        if not url:
            continue
        event_id = entry.get("event_id") or url.rstrip("/").split("/")[-1]
        try:
            detail = es.scrape_event(url, event_id)
        except Exception as exc:
            logger.warning("UFC allowlist: failed to scrape event {}: {}", url, exc)
            continue
        bouts: list[dict[str, Any]] = []
        for fight in (detail.get("fights") if detail else []) or []:
            n1 = str(fight.get("fighter_1_name", "")).strip()
            n2 = str(fight.get("fighter_2_name", "")).strip()
            if n1 and n2:
                bouts.append({
                    "fighter1": n1,
                    "fighter2": n2,
                    "weight_class": str(fight.get("weight_class", "")).strip() or None,
                    "fight_number": fight.get("fight_number"),
                })
        if not bouts:
            continue
        events.append({
            "date": ev_date.isoformat(),
            "name": str(entry.get("name", "")).strip(),
            "url": url,
            "bouts": bouts,
        })

    payload = {
        "generated_at": _iso_z(_utcnow()),
        "source": "ufcstats_upcoming",
        "events": events,
    }

    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ALLOWLIST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(ALLOWLIST_PATH)

    with _lock:
        _MEMO["mtime"] = None  # force reload on next access

    summary = {
        "events": len(events),
        "bouts": sum(len(e["bouts"]) for e in events),
        "dates": [e["date"] for e in events],
        "path": str(ALLOWLIST_PATH),
    }
    logger.info("UFC allowlist refreshed: {} events, {} bouts", summary["events"], summary["bouts"])
    return summary
