"""
Full browser-to-API parity sweep across every event tab on /events.

This uses the remote deployment by default and compares the rendered browser UI
against POST /api/predict for every visible fight card after switching the page
to raw view.

Run from the repo root with:

    .venv/bin/python -m pytest tests/test_remote_events_parity.py -v -s

Optional environment variables:

    SITE_URL=http://107.175.94.166:8002
    EVENT_LIMIT=0        # 0 = all events, otherwise first N events
    RAW_VIEW=1           # click "Show Raw" before scraping
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pytest
import requests
from playwright.sync_api import sync_playwright


SITE_URL = os.environ.get("SITE_URL", "http://107.175.94.166:8002")
EVENT_LIMIT = int(os.environ.get("EVENT_LIMIT", "0"))
RAW_VIEW = os.environ.get("RAW_VIEW", "1").strip().lower() not in {"0", "false", "no", "off"}

PROB_TOL = 0.5
EDGE_TOL = 0.5

_ODDS_RE = re.compile(r"([+-]?\d+)\s*·\s*mkt\s*([\d.]+)\s*%", re.IGNORECASE)


@dataclass
class FightCard:
    fighter1: str
    fighter2: str
    f1_odds: Optional[int]
    f2_odds: Optional[int]
    f1_market_prob: Optional[float]
    f2_market_prob: Optional[float]
    f1_model_prob: Optional[float]
    f2_model_prob: Optional[float]
    edge: Optional[float]
    model_pick: Optional[str]


def _parse_fighter_odds_block(text: str) -> tuple[Optional[int], Optional[float]]:
    cleaned = text.replace("\u2212", "-").replace("\u00a0", " ")
    m = _ODDS_RE.search(cleaned)
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def _parse_edge(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"([+-]?\d+\.?\d*)\s*%", text.replace("\u2212", "-"))
    return float(m.group(1)) if m else None


def _to_iso_date(value: str) -> Optional[str]:
    if not value:
        return None
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", str(value).strip())
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%B %d, %Y", "%b %d, %Y", "%B %d", "%b %d"):
        try:
            dt = datetime.strptime(cleaned, fmt)
            if "%Y" not in fmt:
                dt = dt.replace(year=datetime.now().year)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _api_predict(card: FightCard, *, event_date_iso: str | None) -> tuple[Optional[dict], Optional[str]]:
    payload = {
        "fighter1": card.fighter1,
        "fighter2": card.fighter2,
        "fighter1_odds": card.f1_odds,
        "fighter2_odds": card.f2_odds,
    }
    if event_date_iso:
        payload["fight_date"] = event_date_iso
    response = requests.post(f"{SITE_URL}/api/predict", json=payload, timeout=30)
    # A 404 means one of the fighters isn't in the DB (common for obscure
    # non-UFC regional cards). That's a data-coverage gap, not a parity bug, so
    # surface the detail and let the caller skip it instead of aborting the run.
    if response.status_code == 404:
        try:
            detail = response.json().get("detail", "not found")
        except ValueError:
            detail = "not found"
        return None, detail
    response.raise_for_status()
    return response.json(), None


def _scrape_active_event(page) -> list[FightCard]:
    page.wait_for_selector(".fight-card", timeout=10_000)
    time.sleep(0.3)

    cards: list[FightCard] = []
    for el in page.query_selector_all(".fight-card"):
        f1_block = el.query_selector(".fighter-block.f1")
        f2_block = el.query_selector(".fighter-block.f2")
        if not f1_block or not f2_block:
            continue

        f1_name_el = f1_block.query_selector("[data-fighter]")
        f2_name_el = f2_block.query_selector("[data-fighter]")
        f1_name = f1_name_el.get_attribute("data-fighter") if f1_name_el else ""
        f2_name = f2_name_el.get_attribute("data-fighter") if f2_name_el else ""

        f1_odds_el = f1_block.query_selector(".fighter-odds")
        f2_odds_el = f2_block.query_selector(".fighter-odds")
        f1_odds, f1_mkt = _parse_fighter_odds_block(f1_odds_el.inner_text() if f1_odds_el else "")
        f2_odds, f2_mkt = _parse_fighter_odds_block(f2_odds_el.inner_text() if f2_odds_el else "")

        prob_labels = el.query_selector_all(".prob-label")
        f1_mp = f2_mp = None
        if len(prob_labels) >= 2:
            try:
                f1_mp = float(prob_labels[0].inner_text().replace("%", "").strip())
                f2_mp = float(prob_labels[1].inner_text().replace("%", "").strip())
            except ValueError:
                pass

        edge_el = el.query_selector(".meta-edge")
        edge = _parse_edge(edge_el.inner_text()) if edge_el else None

        model_pick = None
        if f1_mp is not None and f2_mp is not None:
            model_pick = f1_name if f1_mp >= f2_mp else f2_name

        cards.append(
            FightCard(
                fighter1=f1_name,
                fighter2=f2_name,
                f1_odds=f1_odds,
                f2_odds=f2_odds,
                f1_market_prob=f1_mkt,
                f2_market_prob=f2_mkt,
                f1_model_prob=f1_mp,
                f2_model_prob=f2_mp,
                edge=edge,
                model_pick=model_pick,
            )
        )
    return cards


def _load_api_events() -> list[dict]:
    response = requests.get(f"{SITE_URL}/api/events", timeout=30)
    response.raise_for_status()
    return response.json()


def test_all_remote_events_match_api():
    api_events = _load_api_events()
    assert api_events, "Remote /api/events returned no events"

    mismatches: list[str] = []
    count_mismatches: list[str] = []
    unresolved: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1800})
            page.goto(f"{SITE_URL}/events", wait_until="networkidle")
            page.wait_for_selector(".event-tab", timeout=15_000)

            if RAW_VIEW:
                raw_btn = page.query_selector("#rawView")
                if raw_btn:
                    raw_btn.click()
                    time.sleep(0.5)

            tabs = page.query_selector_all(".event-tab")
            assert tabs, "No event tabs rendered on remote /events page"

            limit = min(len(api_events), len(tabs))
            if EVENT_LIMIT > 0:
                limit = min(limit, EVENT_LIMIT)

            print(f"\nChecking {limit} event(s) against {SITE_URL}")

            for idx in range(limit):
                tab = page.locator(f"#etab-{idx}")
                tab.click()
                time.sleep(0.4)

                cards = _scrape_active_event(page)
                api_event = api_events[idx]
                event_name = api_event.get("event_name", f"event[{idx}]")
                event_date_iso = _to_iso_date(str(api_event.get("event_date", "")))
                expected_fights = len(api_event.get("fights", []))
                print(f"  [{idx + 1}/{limit}] {event_name} ({event_date_iso or api_event.get('event_date', '')}) -> {len(cards)} UI fights")

                if len(cards) != expected_fights:
                    count_mismatches.append(
                        f"{event_name}: ui_fights={len(cards)} api_event_fights={expected_fights}"
                    )

                for card in cards:
                    if card.f1_model_prob is None:
                        continue
                    api, skip_reason = _api_predict(card, event_date_iso=event_date_iso)
                    if api is None:
                        # Fighter(s) not in the DB (typically obscure non-UFC
                        # regional cards) — record and skip rather than fail.
                        unresolved.append(
                            f"{event_name} :: {card.fighter1} vs {card.fighter2}: {skip_reason}"
                        )
                        continue

                    if card.f1_market_prob is not None and abs(api["market_prob_f1"] - card.f1_market_prob) > PROB_TOL:
                        mismatches.append(
                            f"{event_name} :: {card.fighter1} vs {card.fighter2}: "
                            f"market_prob_f1 site={card.f1_market_prob} api={api['market_prob_f1']}"
                        )

                    if abs(api["model_prob_f1"] - card.f1_model_prob) > PROB_TOL:
                        mismatches.append(
                            f"{event_name} :: {card.fighter1} vs {card.fighter2}: "
                            f"model_prob_f1 site={card.f1_model_prob} api={api['model_prob_f1']}"
                        )

                    if api["model_pick"] != card.model_pick:
                        mismatches.append(
                            f"{event_name} :: {card.fighter1} vs {card.fighter2}: "
                            f"model_pick site={card.model_pick} api={api['model_pick']}"
                        )

                    if card.edge is not None and abs(api["edge"] - card.edge) > EDGE_TOL:
                        mismatches.append(
                            f"{event_name} :: {card.fighter1} vs {card.fighter2}: "
                            f"edge site={card.edge} api={api['edge']}"
                        )
        finally:
            browser.close()

    if unresolved:
        print(
            f"\nSkipped {len(unresolved)} unresolvable matchup(s) "
            f"(fighter not in DB — expected for non-UFC regional cards):"
        )
        for entry in unresolved:
            print(f"  - {entry}")

    assert not count_mismatches, "UI/API event fight-count mismatches:\n  " + "\n  ".join(count_mismatches)
    assert not mismatches, "Remote UI/API mismatches:\n  " + "\n  ".join(mismatches)
