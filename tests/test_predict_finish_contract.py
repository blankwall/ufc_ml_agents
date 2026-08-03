"""The finish model is intentionally not part of the one-off predict endpoint."""

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


def test_predict_remains_winner_only(monkeypatch):
    monkeypatch.setattr(predict_router, "_Session", lambda: _DummySession())
    monkeypatch.setattr(
        predict_router,
        "_resolve_fighter",
        lambda _session, name: SimpleNamespace(id=1, name=name),
    )
    monkeypatch.setattr(predict_router, "MatchupFeatureExtractor", lambda _session: object())
    monkeypatch.setattr(
        predict_router,
        "_score_row",
        lambda *_args, **_kwargs: {"model_prob_f1": 0.578, "model_source": "general"},
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
        lambda **_kwargs: {
            "primary_bucket": {
                "label": "Primary",
                "sample_size": 1,
                "wins": 1,
                "losses": 0,
                "win_rate": 100.0,
                "roi": 1.0,
            }
        },
    )

    result = asyncio.run(
        predict_router.predict_fight(
            predict_router.PredictRequest(
                fighter1="Diego Ferreira",
                fighter2="Billy Quarantillo",
                fight_date="2026-08-08",
            )
        )
    )

    assert result["model_prob_f1"] == 57.8
    assert result["model_prob_f2"] == 42.2
    assert result["model_pick"] == "Diego Ferreira"
    assert result["bet"] is True
    assert "finish_prediction" not in result
