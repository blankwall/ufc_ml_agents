import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.routers import predict as predict_router  # noqa: E402
from fastapi_app.services import bet_evaluator  # noqa: E402


class _DummySession:
    def close(self):
        return None


def _patch_predict_dependencies(monkeypatch, *, model_prob: float, bet_eval: dict):
    monkeypatch.setattr(predict_router, "_Session", lambda: _DummySession())
    monkeypatch.setattr(
        predict_router,
        "_resolve_fighter",
        lambda _session, name: SimpleNamespace(
            id=1 if "Alex" in name or "Victor" in name else 2,
            name=name,
            wins=10,
            losses=2,
            draws=0,
        ),
    )
    monkeypatch.setattr(predict_router, "MatchupFeatureExtractor", lambda _session: object())
    monkeypatch.setattr(
        predict_router,
        "_score_row",
        lambda *_args, **_kwargs: {"model_prob_f1": model_prob, "model_source": "general"},
    )
    monkeypatch.setattr(
        predict_router,
        "_fight_count_as_of",
        lambda _session, fighter_id, _as_of: 5 if fighter_id == 1 else 7,
    )
    monkeypatch.setattr(predict_router, "_matchup_wmma_flag", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(predict_router, "_evaluate_bet", lambda **_kwargs: bet_eval)
    monkeypatch.setattr(
        predict_router,
        "describe_historical_context",
        lambda **_kwargs: {
            "primary_bucket": {
                "label": "Primary",
                "source": "backtest_2025_2026",
                "criteria": [{"field": "side", "value": "favorite"}],
                "sample_size": 12,
                "wins": 8,
                "losses": 4,
                "win_rate": 66.7,
                "roi": 18.4,
                "profit": 2.2,
                "avg_model_prob": 61.4,
                "avg_edge": 5.1,
                "avg_pick_odds": -150.0,
            },
        },
    )


def test_predict_response_is_minimized_and_static_config_only(monkeypatch):
    _patch_predict_dependencies(
        monkeypatch,
        model_prob=0.578,
        bet_eval={"bet": True, "skip_code": None, "skip_reason": None, "decision_source": "static_config"},
    )

    result = asyncio.run(
        predict_router.predict_fight(
            predict_router.PredictRequest(
                fighter1="Alex Perez",
                fighter2="Charles Johnson",
                fighter1_odds=154,
                fighter2_odds=-180,
            )
        )
    )

    assert result["historical_context"] == {
        "primary_bucket": {
            "label": "Primary",
            "sample_size": 12,
            "wins": 8,
            "losses": 4,
            "win_rate": 66.7,
            "roi": 18.4,
        }
    }
    assert result["decision"] == "Bet"
    assert result["explanation"] == "Bet: the model clears the current betting rules."
    assert result["skip_reason"] is None

    assert "fighter1_db_name" not in result
    assert "fighter2_db_name" not in result
    assert "model_source" not in result
    assert "f1_fight_count" not in result
    assert "f2_fight_count" not in result
    assert "f1_record" not in result
    assert "f2_record" not in result
    assert "skip_code" not in result
    assert "confidence_score" not in result
    assert "confidence_historical_win_rate" not in result
    assert "decision_source" not in result
    assert "review_bucket" not in result
    assert "review_tier" not in result
    assert "review_label" not in result
    assert "pick_elo_diff" not in result


def test_predict_response_static_skip_cannot_be_reopened_by_elo(monkeypatch):
    _patch_predict_dependencies(
        monkeypatch,
        model_prob=0.284,
        bet_eval={"bet": False, "skip_code": "F3", "skip_reason": "Favorite low edge", "decision_source": "static_skip"},
    )

    result = asyncio.run(
        predict_router.predict_fight(
            predict_router.PredictRequest(
                fighter1="Victor Henry",
                fighter2="Bryce Mitchell",
                fighter1_odds=200,
                fighter2_odds=-250,
                fight_date="2026-06-06",
            )
        )
    )

    assert result["bet"] is False
    assert result["skip_reason"] == "Favorite low edge"
    assert result["decision"] == "Pass"
    assert result["explanation"] == "Pass: the favorite edge is too small."
    assert "decision_source" not in result
    assert "review_bucket" not in result
    assert "review_tier" not in result
    assert "review_label" not in result
    assert "pick_elo_diff" not in result


def test_evaluate_bet_decision_uses_only_static_config():
    result = bet_evaluator.evaluate_bet_decision(
        fighter1_name="Pick Fighter",
        fighter2_name="Opp Fighter",
        pick_slot="fighter1",
        pick_model_prob=0.70,
        pick_mkt_prob=0.68,
        pick_odds=-180,
        is_favorite=True,
        is_wmma=False,
        f1_count=10,
        f2_count=10,
        filters={"favorite_confidence_min": 0.65, "edge_min": 0.04},
        wmma_rules={},
        as_of_date="2026-02-01",
    )

    assert result == {
        "bet": False,
        "skip_code": "F3",
        "skip_reason": "Favorite low edge",
        "decision_source": "static_skip",
    }
