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
            "decision_source": "static_config",
            "review_candidate": False,
            "review_bucket": None,
            "review_label": None,
            "review_reason": None,
            "signal_bucket": "skip_50_65_elo_100_plus",
            "signal_label": "ELO 100+ confidence bucket",
            "signal_reason": "Strongest bucket surfaced.",
            "signal_tags": ["skip_50_65_elo_100_plus", "elo_100_plus"],
            "pick_elo_diff": 118.0,
            "cardio_score_diff": 11.0,
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
    assert result["signal_label"] == "ELO 100+ confidence bucket"
    assert result["pick_elo_diff"] == 118.0
    assert result["review_candidate"] is False
    assert "confidence_method" not in result
    assert "confidence_prob_min" not in result
    assert "confidence_prob_max" not in result
    assert "confidence_avg_prob" not in result
    assert "confidence_sample_size" not in result
