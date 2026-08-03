"""Integration/contract tests for the `finish_prediction` block added to
POST /api/predict by the ufc_decision_skill integration.

These tests monkeypatch `_run_finish_prediction` so they never invoke the
real (slow) subprocess — real parity against the protected model lives in
tests/test_finish_prediction_acceptance.py.

Every test here also asserts the *winner* model fields are present and
untouched, per the requirement that this integration must not modify the
existing winner model, its probabilities, pick, or betting decision.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.routers import predict as predict_router  # noqa: E402


class _DummySession:
    def close(self):
        return None


def _patch_winner_model(monkeypatch, *, model_prob: float = 0.578):
    """Same winner-model mocking as test_predict_response.py, kept minimal
    here since these tests only care about the finish_prediction block."""
    monkeypatch.setattr(predict_router, "_Session", lambda: _DummySession())
    monkeypatch.setattr(
        predict_router,
        "_resolve_fighter",
        lambda _session, name: SimpleNamespace(id=1, name=name, wins=10, losses=2, draws=0),
    )
    monkeypatch.setattr(predict_router, "MatchupFeatureExtractor", lambda _session: object())
    monkeypatch.setattr(
        predict_router,
        "_score_row",
        lambda *_args, **_kwargs: {"model_prob_f1": model_prob, "model_source": "general"},
    )
    monkeypatch.setattr(predict_router, "_fight_count_as_of", lambda *_a, **_k: 5)
    monkeypatch.setattr(predict_router, "_matchup_wmma_flag", lambda *_a, **_k: False)
    monkeypatch.setattr(
        predict_router,
        "_evaluate_bet",
        lambda **_kwargs: {"bet": True, "skip_code": None, "skip_reason": None},
    )
    monkeypatch.setattr(
        predict_router,
        "describe_historical_context",
        lambda **_kwargs: {"primary_bucket": {"label": "Primary", "sample_size": 1, "wins": 1, "losses": 0, "win_rate": 100.0, "roi": 1.0}},
    )


def test_finish_prediction_block_present_and_winner_model_untouched(monkeypatch):
    _patch_winner_model(monkeypatch, model_prob=0.578)

    captured = {}

    def _fake_run(fighter_a, fighter_b, *, fight_date, weight_class, fight_number, market_finish_probability):
        captured.update(
            fighter_a=fighter_a, fighter_b=fighter_b, fight_date=fight_date,
            weight_class=weight_class, fight_number=fight_number,
            market_finish_probability=market_finish_probability,
        )
        return {
            "bet": True,
            "error_code": None,
            "error_message": None,
            "selection": "finish",
            "confidence": 0.6270289421081543,
            "tier": "strong",
            "eligible": True,
            "probabilities": {"finish": 0.6270289421081543, "decision": 0.3729710578918457},
            "method_probabilities": {
                "decision": 0.36023667454719543,
                "ko_tko": 0.4737807810306549,
                "submission": 0.16598257422447205,
            },
            "history": {"fighter_a_prior": 17, "fighter_b_prior": 11},
            "market": {"available": True, "selected_probability": 0.55, "edge": 0.05, "actionable": True},
            "fight_number": 5,
        }

    monkeypatch.setattr(predict_router, "_run_finish_prediction", _fake_run)

    result = asyncio.run(
        predict_router.predict_fight(
            predict_router.PredictRequest(
                fighter1="Diego Ferreira",
                fighter2="Billy Quarantillo",
                fight_date="2026-08-08",
                weight_class="Lightweight",
            )
        )
    )

    # Winner model fields are unaffected by the new block.
    assert result["model_prob_f1"] == 57.8
    assert result["model_prob_f2"] == 42.2
    assert result["model_pick"] == "Diego Ferreira"
    assert result["bet"] is True
    assert result["decision"] == "Bet"

    fp = result["finish_prediction"]
    assert fp["bet"] is True
    assert fp["selection"] == "finish"
    assert fp["confidence"] == 0.6270289421081543
    assert fp["tier"] == "strong"
    assert fp["probabilities"] == {"finish": 0.6270289421081543, "decision": 0.3729710578918457}
    assert fp["method_probabilities"]["ko_tko"] == 0.4737807810306549
    assert fp["method_probabilities"]["submission"] == 0.16598257422447205

    assert captured["fighter_a"] == "Diego Ferreira"
    assert captured["fighter_b"] == "Billy Quarantillo"
    assert captured["weight_class"] == "Lightweight"
    assert captured["fight_number"] is None  # not supplied -> service applies its own default


def test_missing_weight_class_yields_error_without_touching_winner_model(monkeypatch):
    _patch_winner_model(monkeypatch, model_prob=0.62)

    # No monkeypatch of _run_finish_prediction: use the real function so we
    # exercise the actual missing-input guard (no subprocess should run).
    result = asyncio.run(
        predict_router.predict_fight(
            predict_router.PredictRequest(
                fighter1="Diego Ferreira",
                fighter2="Billy Quarantillo",
                fight_date="2026-08-08",
                # weight_class intentionally omitted
            )
        )
    )

    assert result["finish_prediction"]["bet"] == "error"
    assert result["finish_prediction"]["error_code"] == "missing_input"
    # Winner model still ran normally.
    assert result["model_prob_f1"] == 62.0
    assert result["bet"] is True


def test_devig_finish_and_decision_odds_passed_through(monkeypatch):
    _patch_winner_model(monkeypatch)

    captured = {}

    def _fake_run(fighter_a, fighter_b, *, fight_date, weight_class, fight_number, market_finish_probability):
        captured["market_finish_probability"] = market_finish_probability
        return {"bet": False, "market": {"available": True, "actionable": False}}

    monkeypatch.setattr(predict_router, "_run_finish_prediction", _fake_run)

    asyncio.run(
        predict_router.predict_fight(
            predict_router.PredictRequest(
                fighter1="Diego Ferreira",
                fighter2="Billy Quarantillo",
                fight_date="2026-08-08",
                weight_class="Lightweight",
                finish_odds=150,
                decision_odds=-140,
            )
        )
    )

    raw_finish = 100 / (150 + 100)
    raw_decision = 140 / (140 + 100)
    expected = raw_finish / (raw_finish + raw_decision)
    assert captured["market_finish_probability"] == expected


def test_missing_market_odds_never_actionable(monkeypatch):
    _patch_winner_model(monkeypatch)

    def _fake_run(*_a, **_k):
        # Real skill behavior when market_finish_probability is None:
        # market.available is False, so actionable can never be True even
        # if eligible/strong.
        return {
            "bet": False,
            "selection": "finish",
            "confidence": 0.63,
            "tier": "strong",
            "eligible": True,
            "probabilities": {"finish": 0.63, "decision": 0.37},
            "method_probabilities": {"decision": 0.3, "ko_tko": 0.5, "submission": 0.2},
            "history": {"fighter_a_prior": 10, "fighter_b_prior": 10},
            "market": {"available": False, "selected_probability": None, "edge": None, "actionable": False},
            "fight_number": 5,
        }

    monkeypatch.setattr(predict_router, "_run_finish_prediction", _fake_run)

    result = asyncio.run(
        predict_router.predict_fight(
            predict_router.PredictRequest(
                fighter1="Diego Ferreira",
                fighter2="Billy Quarantillo",
                fight_date="2026-08-08",
                weight_class="Lightweight",
            )
        )
    )

    fp = result["finish_prediction"]
    assert fp["eligible"] is True
    assert fp["tier"] == "strong"
    assert fp["market"]["available"] is False
    assert fp["bet"] is False  # high confidence alone never creates a bet
