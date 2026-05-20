from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime

import pytest
import requests

SITE_URL = os.environ.get("SITE_URL", "http://localhost:8009")
TIMEOUT = 60


@dataclass
class GoldenFight:
    event_index: int
    event_name: str
    event_date: str
    fighter1: str
    fighter2: str
    f1_odds: int | None
    f2_odds: int | None
    decision_source: str
    review_bucket: str | None
    review_tier: int | str
    review_label: str
    pick_elo_diff: float | None
    bet: bool


def _get_events_payload() -> list[dict]:
    try:
        response = requests.get(f"{SITE_URL}/api/events", timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"Local site unavailable at {SITE_URL}: {exc}")
    payload = response.json()
    if not isinstance(payload, list):
        pytest.fail(f"Expected /api/events to return a list, got {type(payload).__name__}")
    return payload


def _collect_golden_fights(events_payload: list[dict]) -> list[GoldenFight]:
    golden_fights: list[GoldenFight] = []
    for event_index, event in enumerate(events_payload):
        event_name = event.get("event_name", "")
        event_date = event.get("event_date", "")
        for fight in event.get("fights", []):
            if fight.get("decision_source") != "golden_elo_reopen":
                continue
            golden_fights.append(
                GoldenFight(
                    event_index=event_index,
                    event_name=event_name,
                    event_date=event_date,
                    fighter1=fight.get("fighter1", ""),
                    fighter2=fight.get("fighter2", ""),
                    f1_odds=fight.get("f1_odds"),
                    f2_odds=fight.get("f2_odds"),
                    decision_source=fight.get("decision_source"),
                    review_bucket=fight.get("review_bucket"),
                    review_tier=fight.get("review_tier"),
                    review_label=fight.get("review_label", ""),
                    pick_elo_diff=fight.get("pick_elo_diff"),
                    bet=bool(fight.get("bet")),
                )
            )
    return golden_fights


def _expected_bucket_for_tier(tier: int | str) -> str:
    if tier == 3:
        return "golden_elo_plus_cardio"
    if tier == 2:
        return "golden_elo_plus_trait_support"
    if tier == "1A":
        return "golden_elo_tier_1a"
    return "golden_elo_not_expensive"


def _normalize_predict_date(event_date: str) -> str | None:
    if re.match(r"^\d{4}-\d{2}-\d{2}$", event_date):
        return event_date

    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", event_date or "").strip()
    if not cleaned:
        return None

    for fmt in ("%B %d", "%b %d"):
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.replace(year=datetime.now().year).date().isoformat()
        except ValueError:
            continue
    return None


def _select_event_tab(page, event_index: int, event_name: str, fighter1: str, fighter2: str) -> None:
    page.goto(f"{SITE_URL}/events", wait_until="networkidle")
    page.wait_for_selector(".event-tab", timeout=15_000)
    page.evaluate("(idx) => window.selectEvent(idx)", event_index)
    page.wait_for_function(
        """([eventName, f1, f2]) => {
            const title = document.querySelector('.event-panel-title')?.textContent || '';
            const panel = document.querySelector('#eventPanel')?.textContent || '';
            return title.includes(eventName) && panel.includes(f1) && panel.includes(f2);
        }""",
        arg=[event_name, fighter1, fighter2],
        timeout=15_000,
    )


def _find_fight_card(page, fighter1: str, fighter2: str):
    for card in page.query_selector_all(".fight-card"):
        text = card.inner_text()
        if fighter1 in text and fighter2 in text:
            return card
    return None


@pytest.fixture(scope="module")
def golden_fights() -> list[GoldenFight]:
    fights = _collect_golden_fights(_get_events_payload())
    if not fights:
        pytest.skip(f"No Golden ELO reopen fights returned by {SITE_URL}/api/events")
    return fights


@pytest.fixture(scope="module")
def sample_golden_fight(golden_fights: list[GoldenFight]) -> GoldenFight:
    return golden_fights[0]


def test_local_events_api_exposes_golden_elo_fields(golden_fights: list[GoldenFight]):
    assert golden_fights, "Expected at least one Golden ELO reopen fight"

    for fight in golden_fights:
        assert fight.bet is True
        assert fight.decision_source == "golden_elo_reopen"
        assert fight.review_tier in {1, "1A", 2, 3}
        assert fight.review_bucket == _expected_bucket_for_tier(fight.review_tier)
        assert fight.review_label.startswith(f"Golden ELO Tier {fight.review_tier}")
        assert "Historical " in fight.review_label
        assert "ROI" in fight.review_label
        assert fight.pick_elo_diff is not None
        assert fight.pick_elo_diff >= 50


def test_local_predict_matches_events_golden_elo_decision(sample_golden_fight: GoldenFight):
    fight_date = _normalize_predict_date(sample_golden_fight.event_date)
    response = requests.post(
        f"{SITE_URL}/api/predict",
        json={
            "fighter1": sample_golden_fight.fighter1,
            "fighter2": sample_golden_fight.fighter2,
            "fighter1_odds": sample_golden_fight.f1_odds,
            "fighter2_odds": sample_golden_fight.f2_odds,
            "fight_date": fight_date,
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()

    for key in (
        "fighter1",
        "fighter2",
        "model_prob_f1",
        "market_prob_f1",
        "edge",
        "bet",
        "decision_source",
        "review_bucket",
        "review_tier",
        "review_label",
        "pick_elo_diff",
        "confidence_score",
        "confidence_historical_win_rate",
    ):
        assert key in payload, f"Missing key {key} in /api/predict response"

    assert payload["bet"] is True
    assert payload["decision_source"] == "golden_elo_reopen"
    assert payload["review_bucket"] == sample_golden_fight.review_bucket
    assert payload["review_tier"] == sample_golden_fight.review_tier
    assert payload["review_label"] == sample_golden_fight.review_label
    assert payload["pick_elo_diff"] == sample_golden_fight.pick_elo_diff
    assert payload["review_bucket"] == _expected_bucket_for_tier(payload["review_tier"])


def test_local_events_page_renders_golden_elo_badge(sample_golden_fight: GoldenFight):
    sync_api = pytest.importorskip("playwright.sync_api")

    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1800})
            _select_event_tab(
                page,
                sample_golden_fight.event_index,
                sample_golden_fight.event_name,
                sample_golden_fight.fighter1,
                sample_golden_fight.fighter2,
            )
            card = _find_fight_card(page, sample_golden_fight.fighter1, sample_golden_fight.fighter2)
            assert card is not None, (
                f"Could not find fight card for {sample_golden_fight.fighter1} vs {sample_golden_fight.fighter2}"
            )
            text = card.inner_text()
            assert f"Golden ELO T{sample_golden_fight.review_tier}" in text
            assert sample_golden_fight.review_label in text
            if sample_golden_fight.pick_elo_diff is not None:
                elo_pattern = rf"ELO Diff:\s*\+?{re.escape(str(sample_golden_fight.pick_elo_diff).rstrip('0').rstrip('.'))}"
                assert re.search(elo_pattern, text), text
        finally:
            browser.close()
