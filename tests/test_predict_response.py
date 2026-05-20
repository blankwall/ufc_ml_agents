import asyncio
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.routers import predict as predict_router  # noqa: E402
from fastapi_app.services import bet_evaluator  # noqa: E402


class _DummySession:
    def close(self):
        return None


def test_predict_response_hides_internal_confidence_metadata(monkeypatch):
    monkeypatch.setattr(predict_router, "_Session", lambda: _DummySession())
    monkeypatch.setattr(
        predict_router,
        "_resolve_fighter",
        lambda _session, name: SimpleNamespace(
            id=1 if "Alex" in name else 2,
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
        lambda *_args, **_kwargs: {"model_prob_f1": 0.578, "model_source": "general"},
    )
    monkeypatch.setattr(
        predict_router,
        "_fight_count_as_of",
        lambda _session, fighter_id, _as_of: 5 if fighter_id == 1 else 7,
    )
    monkeypatch.setattr(predict_router, "_matchup_wmma_flag", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        predict_router,
        "_evaluate_bet",
        lambda **_kwargs: {
            "bet": True,
            "skip_code": None,
            "skip_reason": None,
            "decision_source": "golden_elo_reopen",
            "review_bucket": "golden_elo_not_expensive",
            "review_tier": 2,
            "review_label": "Golden ELO Tier 2 · Historical 20-7 · +17.2% ROI",
            "pick_elo_diff": 118.0,
        },
    )
    monkeypatch.setattr(
        predict_router,
        "describe_confidence",
        lambda _pick_prob: {
            "confidence_score": 4,
            "confidence_method": "backtest_pick_prob_decile",
            "confidence_prob_min": 57.2,
            "confidence_prob_max": 60.3,
            "confidence_historical_win_rate": 58.7,
            "confidence_sample_size": 42,
        },
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

    assert result["confidence_score"] == 4
    assert result["confidence_historical_win_rate"] == 58.7
    assert result["decision_source"] == "golden_elo_reopen"
    assert result["review_tier"] == 2
    assert result["review_label"] == "Golden ELO Tier 2 · Historical 20-7 · +17.2% ROI"
    assert result["pick_elo_diff"] == 118.0
    assert "confidence_method" not in result
    assert "confidence_prob_min" not in result
    assert "confidence_prob_max" not in result
    assert "confidence_avg_prob" not in result
    assert "confidence_sample_size" not in result


def test_evaluate_bet_forwards_as_of_date(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        predict_router,
        "_load_betting_filters",
        lambda: {"filters": {}, "wmma": {}},
    )

    def fake_evaluate_bet_decision(**kwargs):
        captured.update(kwargs)
        return {
            "bet": True,
            "skip_code": None,
            "skip_reason": None,
            "decision_source": "golden_elo_reopen",
            "review_bucket": "golden_elo_not_expensive",
            "review_tier": 2,
            "review_label": "Golden ELO Tier 2 · Historical 8-1 · +41.9% ROI",
            "pick_elo_diff": 118.0,
        }

    monkeypatch.setattr(predict_router, "evaluate_bet_decision", fake_evaluate_bet_decision)

    as_of = datetime(2026, 2, 1)
    result = predict_router._evaluate_bet(
        fighter1_name="Alexander Volkanovski",
        fighter2_name="Diego Lopes",
        pick_slot="fighter1",
        pick_model_prob=0.58,
        pick_mkt_prob=0.52,
        pick_odds=-120,
        is_favorite=True,
        is_wmma=False,
        f1_count=10,
        f2_count=8,
        as_of_date=as_of,
    )

    assert result["decision_source"] == "golden_elo_reopen"
    assert captured["as_of_date"] == as_of


def test_predict_response_keeps_pick_elo_diff_on_static_skip(monkeypatch):
    monkeypatch.setattr(predict_router, "_Session", lambda: _DummySession())
    monkeypatch.setattr(
        predict_router,
        "_resolve_fighter",
        lambda _session, name: SimpleNamespace(
            id=1 if "Victor" in name else 2,
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
        lambda *_args, **_kwargs: {"model_prob_f1": 0.284, "model_source": "general"},
    )
    monkeypatch.setattr(
        predict_router,
        "_fight_count_as_of",
        lambda _session, fighter_id, _as_of: 7 if fighter_id == 1 else 12,
    )
    monkeypatch.setattr(predict_router, "_matchup_wmma_flag", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        predict_router,
        "_evaluate_bet",
        lambda **_kwargs: {
            "bet": False,
            "skip_code": "F3",
            "skip_reason": "Favorite low edge",
            "decision_source": "static_skip",
            "review_bucket": "elo_against_50",
            "review_tier": 1,
            "review_label": "ELO Against Tier 1 · Historical 50-25 · +27.1% ROI",
            "pick_elo_diff": -81.0,
        },
    )
    monkeypatch.setattr(
        predict_router,
        "describe_confidence",
        lambda _pick_prob: {
            "confidence_score": 8,
            "confidence_method": "backtest_pick_prob_decile",
            "confidence_prob_min": 68.0,
            "confidence_prob_max": 73.0,
            "confidence_historical_win_rate": 74.4,
            "confidence_sample_size": 42,
        },
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
    assert result["decision_source"] == "static_skip"
    assert result["skip_code"] == "F3"
    assert result["pick_elo_diff"] == -81.0
    assert result["review_bucket"] == "elo_against_50"
    assert result["review_tier"] == 1
    assert result["review_label"] == "ELO Against Tier 1 · Historical 50-25 · +27.1% ROI"


def test_evaluate_bet_decision_preserves_review_only_bucket(monkeypatch):
    monkeypatch.setattr(
        bet_evaluator,
        "evaluate_golden_elo_reopen",
        lambda **_kwargs: {
            "reopen": False,
            "review_bucket": "elo_against_tier_1a",
            "review_tier": "-1A",
            "review_label": "ELO Against Tier -1A · Historical 12-3 · +49.1% ROI",
            "review_stats": {"wins": 12, "losses": 3, "roi_pct": 49.1},
            "pick_elo_diff": -60.0,
        },
    )

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

    assert result["bet"] is False
    assert result["decision_source"] == "static_skip"
    assert result["skip_code"] == "F3"
    assert result["review_bucket"] == "elo_against_tier_1a"
    assert result["review_tier"] == "-1A"
    assert result["review_label"] == "ELO Against Tier -1A · Historical 12-3 · +49.1% ROI"
    assert result["pick_elo_diff"] == -60.0
