"""
End-to-end consistency check for UFC 328:

1. Use Playwright to load the live /events page, navigate to UFC 328, and
   scrape every fight card's displayed values (odds, market %, model %,
   edge, model pick).
2. For each fight, POST to /api/predict with the same odds and assert the
   API agrees with what the user sees in the browser.

This catches:
- Vig-removal regressions (raw vs vig-removed market_prob mismatches)
- Pick-side bugs (events page picking different fighter than /api/predict)
- Edge calculation drift between the events loop and the predict endpoint

Run from the repo root with the project venv:

    .venv/bin/python -m pytest tests/test_ufc_328_consistency.py -v -s

Set SITE_URL env var to point at a different deployment (default is the
production server).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import pytest
import requests
from playwright.sync_api import sync_playwright


SITE_URL  = os.environ.get("SITE_URL", "http://107.175.94.166:8002")
EVENT     = os.environ.get("EVENT_NAME", "UFC 328")
EVENT_DATE = os.environ.get("EVENT_DATE", "2026-05-10")

# Tolerances. The events loop and /api/predict run the same model on the
# same DB so probabilities should match almost exactly. Allow tiny float
# drift from intermediate rounding.
PROB_TOL   = 0.5    # percentage points
EDGE_TOL   = 0.5    # percentage points


@dataclass
class FightCard:
    fighter1:        str
    fighter2:        str
    f1_odds:         int
    f2_odds:         int
    f1_market_prob:  float   # percent (vig-removed expected post-fix)
    f2_market_prob:  float   # percent
    f1_model_prob:   Optional[float]   # percent
    f2_model_prob:   Optional[float]   # percent
    edge:            Optional[float]   # signed percent
    model_pick:      Optional[str]


_ODDS_RE = re.compile(r"([+-]?\d+)\s*·\s*mkt\s*([\d.]+)\s*%", re.IGNORECASE)


def _parse_fighter_odds_block(text: str):
    """Parse '−165 · mkt 59.3%' (en-dash or hyphen) → (odds, market_prob)."""
    cleaned = text.replace("\u2212", "-").replace("\u00a0", " ")
    m = _ODDS_RE.search(cleaned)
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def _parse_edge(text: str) -> Optional[float]:
    """Parse '+6.6%' or '-3.3%' from the edge meta value."""
    if not text:
        return None
    m = re.search(r"([+-]?\d+\.?\d*)\s*%", text.replace("\u2212", "-"))
    return float(m.group(1)) if m else None


def _scrape_event_via_browser(page, event_name: str) -> List[FightCard]:
    page.goto(f"{SITE_URL}/events", wait_until="networkidle")

    # Wait for the event tabs to render, then click the matching tab.
    page.wait_for_selector(".event-tab", timeout=15_000)
    target = None
    for tab in page.query_selector_all(".event-tab"):
        if event_name.lower() in (tab.inner_text() or "").lower():
            target = tab
            break
    if target is None:
        raise AssertionError(
            f"Event '{event_name}' not found in tabs. Available: "
            + ", ".join(t.inner_text() for t in page.query_selector_all(".event-tab"))
        )
    target.click()
    page.wait_for_selector(".fight-card", timeout=10_000)
    # Brief settle for any client-side filtering pass.
    time.sleep(0.5)

    cards: List[FightCard] = []
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

        # Probability bar shows model probs as labels at each end. Empty
        # for fights without a prediction.
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

        cards.append(FightCard(
            fighter1=f1_name, fighter2=f2_name,
            f1_odds=f1_odds, f2_odds=f2_odds,
            f1_market_prob=f1_mkt, f2_market_prob=f2_mkt,
            f1_model_prob=f1_mp, f2_model_prob=f2_mp,
            edge=edge,
            model_pick=None,    # derived below from model_prob
        ))

    # Derive the displayed model_pick from the prob labels.
    for c in cards:
        if c.f1_model_prob is not None and c.f2_model_prob is not None:
            c.model_pick = c.fighter1 if c.f1_model_prob >= c.f2_model_prob else c.fighter2

    return cards


def _api_predict(card: FightCard) -> dict:
    r = requests.post(
        f"{SITE_URL}/api/predict",
        json={
            "fighter1":      card.fighter1,
            "fighter2":      card.fighter2,
            "fighter1_odds": card.f1_odds,
            "fighter2_odds": card.f2_odds,
            "fight_date":    EVENT_DATE,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def scraped_cards() -> List[FightCard]:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1400, "height": 1800})
            cards = _scrape_event_via_browser(page, EVENT)
        finally:
            browser.close()
    assert cards, f"No fight cards scraped for {EVENT}"
    print(f"\nScraped {len(cards)} fight cards for {EVENT}")
    for c in cards:
        print(f"  {c.fighter1} ({c.f1_odds}) vs {c.fighter2} ({c.f2_odds})  "
              f"mkt {c.f1_market_prob}/{c.f2_market_prob}  "
              f"model {c.f1_model_prob}/{c.f2_model_prob}  edge {c.edge}")
    return cards


# ─── tests ───────────────────────────────────────────────────────────────────

def test_event_has_fights(scraped_cards):
    """Smoke check — playwright actually scraped the event."""
    assert len(scraped_cards) >= 5, f"Expected ≥5 fights for {EVENT}, got {len(scraped_cards)}"


def test_market_probs_sum_to_100(scraped_cards):
    """After vig removal, mkt% for f1 + f2 must sum to ~100."""
    for c in scraped_cards:
        if c.f1_market_prob is None or c.f2_market_prob is None:
            continue
        total = c.f1_market_prob + c.f2_market_prob
        assert abs(total - 100.0) < 0.5, (
            f"{c.fighter1} vs {c.fighter2}: mkt sum = {total:.2f}% (vig not removed?)"
        )


def test_each_fight_matches_api(scraped_cards):
    """For each scraped fight, the API must return the same model prob,
    market prob, model_pick, and edge."""
    mismatches = []
    for c in scraped_cards:
        if c.f1_model_prob is None:
            # No prediction rendered — nothing to compare.
            continue
        api = _api_predict(c)

        # market_prob_f1 (vig-removed)
        if abs(api["market_prob_f1"] - c.f1_market_prob) > PROB_TOL:
            mismatches.append(
                f"{c.fighter1} vs {c.fighter2}: market_prob_f1 site={c.f1_market_prob} "
                f"api={api['market_prob_f1']}")

        # model_prob_f1
        if abs(api["model_prob_f1"] - c.f1_model_prob) > PROB_TOL:
            mismatches.append(
                f"{c.fighter1} vs {c.fighter2}: model_prob_f1 site={c.f1_model_prob} "
                f"api={api['model_prob_f1']}")

        # model_pick
        if api["model_pick"] != c.model_pick:
            mismatches.append(
                f"{c.fighter1} vs {c.fighter2}: model_pick site={c.model_pick} "
                f"api={api['model_pick']}")

        # edge — both /api/predict and the events loop now use pick-side
        # edge (signed; negative = market more confident than model on the
        # pick).
        if c.edge is not None and abs(api["edge"] - c.edge) > EDGE_TOL:
            mismatches.append(
                f"{c.fighter1} vs {c.fighter2}: edge "
                f"site={c.edge} api={api['edge']}")

    assert not mismatches, "API/site mismatches:\n  " + "\n  ".join(mismatches)
