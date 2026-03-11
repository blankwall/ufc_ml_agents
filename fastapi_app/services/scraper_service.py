"""
Scraper Service
===============
Fetches odds from a BestFightOdds event URL and (optionally) results from
a UFCStats event URL, saves the combined data to data/user_events/<slug>.json,
and returns a summary.

These user-added events are picked up by predict_service.get_events_data()
alongside the static CSVs, and are visually distinguished on the Events page.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

USER_EVENTS_DIR = ROOT_DIR / "data" / "user_events"

# Patch module-level CACHE_DIR before importing so the scraper writes to
# the repo root's .cache/bfo regardless of the CWD of the uvicorn process.
import scrapers.bestfightodds_scraper as _bfo_mod
_bfo_mod.CACHE_DIR = ROOT_DIR / ".cache" / "bfo"
_bfo_mod.CACHE_DIR.mkdir(parents=True, exist_ok=True)

from scrapers.bestfightodds_scraper import BestFightOddsScraper
from scrapers.scrape_outcomes import scrape_event as _scrape_outcomes


def _slug(url: str) -> str:
    """Stable filesystem-safe slug from a URL (last 60 chars of sanitised form)."""
    s = re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_")
    return s[-60:]


def scrape_and_save(
    bfo_url: str,
    ufc_stats_url: Optional[str] = None,
) -> dict:
    """
    1. Scrape odds from the BFO event page.
    2. Optionally scrape outcomes from the UFC Stats event page.
    3. Save combined payload to data/user_events/<slug>.json.
    4. Return a summary dict.

    Raises ValueError if the BFO URL yields no fights.
    """
    USER_EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Scrape odds ───────────────────────────────────────────────────────────
    scraper = BestFightOddsScraper()
    fights  = scraper.scrape_event(bfo_url)

    if not fights:
        raise ValueError(f"No fights found at BFO URL: {bfo_url}")

    event_name = fights[0].get("event_name", "")
    event_date = fights[0].get("event_date", "")

    # Normalise the prob columns (BFO scraper returns fighter1_prob as 0–1;
    # predict_service expects the same 0–1 float).
    for f in fights:
        f["source_type"] = "user_added"

    # ── Scrape outcomes (optional) ────────────────────────────────────────────
    outcomes: list[dict] = []
    if ufc_stats_url and ufc_stats_url.strip():
        outcomes = _scrape_outcomes(ufc_stats_url.strip())
        # Stamp with event name if missing
        for o in outcomes:
            if not o.get("event_name"):
                o["event_name"] = event_name

    # ── Persist ──────────────────────────────────────────────────────────────
    slug    = _slug(bfo_url)
    payload = {
        "bfo_url":        bfo_url,
        "ufc_stats_url":  ufc_stats_url or "",
        "scraped_at":     datetime.now(timezone.utc).isoformat(),
        "event_name":     event_name,
        "event_date":     event_date,
        "fights":         fights,
        "outcomes":       outcomes,
    }

    out_path = USER_EVENTS_DIR / f"{slug}.json"
    out_path.write_text(json.dumps(payload, indent=2))

    return {
        "status":           "ok",
        "slug":             slug,
        "event_name":       event_name,
        "event_date":       event_date,
        "fights_scraped":   len(fights),
        "outcomes_scraped": len(outcomes),
        "saved_to":         str(out_path.relative_to(ROOT_DIR)),
    }


def list_user_events() -> list[dict]:
    """Return a list of summary dicts for all saved user events."""
    if not USER_EVENTS_DIR.exists():
        return []
    events = []
    for path in sorted(USER_EVENTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            events.append({
                "slug":             path.stem,
                "event_name":       data.get("event_name", ""),
                "event_date":       data.get("event_date", ""),
                "bfo_url":          data.get("bfo_url", ""),
                "ufc_stats_url":    data.get("ufc_stats_url", ""),
                "scraped_at":       data.get("scraped_at", ""),
                "fights_count":     len(data.get("fights", [])),
                "outcomes_count":   len(data.get("outcomes", [])),
            })
        except Exception:
            pass
    return events


def delete_user_event(slug: str) -> bool:
    """Delete a saved user event JSON. Returns True if deleted."""
    path = USER_EVENTS_DIR / f"{slug}.json"
    if path.exists():
        path.unlink()
        return True
    return False
