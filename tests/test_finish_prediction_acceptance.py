"""End-to-end acceptance tests against the real, protected ufc_decision_skill
model (https://github.com/blankwall/ufc_decision_skill, pinned commit
da46562), run through fastapi_app.services.finish_prediction_service exactly
as the /api/predict endpoint calls it.

These exercise the actual subprocess + pinned-venv model, so each call takes
several seconds (the skill retrains its walk-forward model on every predict()
call by design). Skipped automatically if the skill isn't installed/bootstrapped
locally, mirroring how test_ufc_328_consistency.py is skipped without
playwright.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from services import finish_prediction_service as svc  # noqa: E402

pytestmark = pytest.mark.skipif(
    not svc.SKILL_CLI.exists(),
    reason=f"ufc_decision_skill not installed/bootstrapped at {svc.SKILL_CLI}",
)


def test_diego_ferreira_vs_billy_quarantillo_matches_golden_fixture():
    result = svc.run_finish_prediction(
        "Diego Ferreira", "Billy Quarantillo",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
    )
    assert result["probabilities"]["finish"] == pytest.approx(0.6270289421081543, abs=1e-9)
    assert result["probabilities"]["decision"] == pytest.approx(0.3729710578918457, abs=1e-9)
    assert result["tier"] == "strong"
    assert result["eligible"] is True
    assert result["method_probabilities"]["ko_tko"] == pytest.approx(0.4737807810306549, abs=1e-9)
    assert result["method_probabilities"]["submission"] == pytest.approx(0.16598257422447205, abs=1e-9)


def test_fighter_order_is_invariant():
    forward = svc.run_finish_prediction(
        "Diego Ferreira", "Billy Quarantillo",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
    )
    reversed_ = svc.run_finish_prediction(
        "Billy Quarantillo", "Diego Ferreira",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
    )
    assert forward["probabilities"] == reversed_["probabilities"]
    assert forward["method_probabilities"] == reversed_["method_probabilities"]


def test_nurgozhay_vs_lopes_is_ineligible_below_sixty_percent():
    result = svc.run_finish_prediction(
        "Diyar Nurgozhay", "Bruno Lopes",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
    )
    assert result["confidence"] == pytest.approx(0.5986, abs=1e-3)
    assert result["eligible"] is False
    assert result["tier"] == "ineligible"


def test_ty_miller_vs_billy_ray_goff_is_ineligible_for_thin_history():
    result = svc.run_finish_prediction(
        "Ty Miller", "Billy Ray Goff",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
    )
    # Confidence clears 60%, but Miller has fewer than two prior fights.
    assert result["confidence"] >= 0.60
    assert result["eligible"] is False
    assert result["tier"] == "ineligible"


def test_market_odds_are_devigged_and_wired_through():
    market_probability = svc.devig_finish_probability(finish_odds=-140, decision_odds=+130)
    result = svc.run_finish_prediction(
        "Diego Ferreira", "Billy Quarantillo",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
        market_finish_probability=market_probability,
    )
    assert result["market"]["available"] is True
    # de-vigged probability must sit strictly between the two raw implied probabilities
    raw_finish = 140 / (140 + 100)
    raw_decision = 100 / (130 + 100)
    assert min(raw_finish, raw_decision) <= market_probability <= max(raw_finish, raw_decision)


def test_missing_market_odds_is_never_actionable_even_when_strong():
    result = svc.run_finish_prediction(
        "Diego Ferreira", "Billy Quarantillo",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
    )
    assert result["tier"] == "strong"
    assert result["market"]["available"] is False
    assert result["market"]["actionable"] is False
    assert result["bet"] is False
