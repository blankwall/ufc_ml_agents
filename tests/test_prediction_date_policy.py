import asyncio
import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.routers import predict as predict_router  # noqa: E402
from fastapi_app.services import predict_service  # noqa: E402
from services import predict_context_service as predict_context_service  # noqa: E402


class _DummySession:
    def close(self):
        return None


def test_api_predict_uses_previous_day_snapshot_for_explicit_fight_date(monkeypatch):
    captured = {"fight_count_as_of": []}

    monkeypatch.setattr(predict_router, "_Session", lambda: _DummySession())
    monkeypatch.setattr(
        predict_router,
        "_resolve_fighter",
        lambda _session, name: SimpleNamespace(id=1 if "Daniel" in name else 2, name=name, wins=10, losses=2, draws=0),
    )
    monkeypatch.setattr(predict_router, "MatchupFeatureExtractor", lambda _session: object())

    def fake_score_row(*_args, **kwargs):
        captured["score_as_of"] = kwargs["as_of_date"]
        return {"model_prob_f1": 0.58, "model_source": "general"}

    def fake_fight_count_as_of(_session, _fighter_id, as_of):
        captured["fight_count_as_of"].append(as_of)
        return 5

    monkeypatch.setattr(predict_router, "_score_row", fake_score_row)
    monkeypatch.setattr(predict_router, "_fight_count_as_of", fake_fight_count_as_of)
    monkeypatch.setattr(predict_router, "_matchup_wmma_flag", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        predict_router,
        "_evaluate_bet",
        lambda **_kwargs: {"bet": True, "skip_code": None, "skip_reason": None},
    )
    monkeypatch.setattr(
        predict_router,
        "describe_confidence",
        lambda _pick_prob: {"confidence_score": 4, "confidence_historical_win_rate": 58.7},
    )

    result = asyncio.run(
        predict_router.predict_fight(
            predict_router.PredictRequest(
                fighter1="Daniel Santos",
                fighter2="Doo Ho Choi",
                fight_date=date(2026, 5, 17),
                fighter1_odds=150,
                fighter2_odds=-170,
            )
        )
    )

    assert result["fight_date"] == "2026-05-17"
    assert captured["score_as_of"] == datetime(2026, 5, 16)
    assert captured["fight_count_as_of"] == [datetime(2026, 5, 16), datetime(2026, 5, 16)]


def test_build_prediction_frame_uses_previous_day_snapshot(monkeypatch):
    captured = {"fight_count_as_of": []}

    monkeypatch.setattr(
        predict_context_service,
        "_resolve_fighter",
        lambda _session, name: SimpleNamespace(id=1 if name == "Pick" else 2, name=name),
    )
    monkeypatch.setattr(predict_context_service, "MatchupFeatureExtractor", lambda _session: object())

    def fake_score_row(*_args, **kwargs):
        captured["score_as_of"] = kwargs["as_of_date"]
        return {"model_prob_f1": 0.58, "model_source": "general"}

    def fake_fight_count_as_of(_session, _fighter_id, as_of):
        captured["fight_count_as_of"].append(as_of)
        return 7

    monkeypatch.setattr(predict_context_service, "_score_row", fake_score_row)
    monkeypatch.setattr(predict_context_service, "_fight_count_as_of", fake_fight_count_as_of)

    frame = predict_context_service.build_prediction_frame(
        fighter1="Pick",
        fighter2="Opponent",
        fight_date=date(2026, 3, 1),
        fighter1_odds=-120,
        fighter2_odds=100,
        session=_DummySession(),
    )

    assert frame["request"]["fight_date"] == "2026-03-01"
    assert frame["as_of"] == datetime(2026, 2, 28)
    assert captured["score_as_of"] == datetime(2026, 2, 28)
    assert captured["fight_count_as_of"] == [datetime(2026, 2, 28), datetime(2026, 2, 28)]


def test_events_prediction_loop_uses_previous_day_snapshot_for_event_date(monkeypatch):
    captured = {"fight_count_as_of": []}

    monkeypatch.setattr(predict_service, "get_bet_placed_map", lambda: {})
    monkeypatch.setattr(
        predict_service,
        "_resolve_fighter",
        lambda _session, name: SimpleNamespace(id=1 if "Song" in name else 2, name=name),
    )

    def fake_score_row(*_args, **kwargs):
        captured["score_as_of"] = kwargs["as_of_date"]
        return {"model_prob_f1": 0.512, "model_source": "general"}

    def fake_fight_count_as_of(_session, _fighter_id, as_of):
        captured["fight_count_as_of"].append(as_of)
        return 9

    monkeypatch.setattr(predict_service, "_score_row", fake_score_row)
    monkeypatch.setattr(predict_service, "_fight_count_as_of", fake_fight_count_as_of)
    monkeypatch.setattr(predict_service, "_is_wmma", lambda *_args, **_kwargs: False)

    odds_df = predict_service.pd.DataFrame(
        [
            {
                "event_name": "MMA Card · 2026-05-30",
                "event_date": "2026-05-30",
                "event_url": "",
                "fighter1": "Song Yadong",
                "fighter2": "Deiveson Figueiredo",
                "fighter1_odds": -125,
                "fighter2_odds": 105,
                "fighter1_prob": 0.5556,
                "fighter2_prob": 0.4878,
                "source_type": "the_odds_api",
                "source_file": "the_odds_api_new_events.csv",
            }
        ]
    )

    events_map, _ = predict_service._run_prediction_loop(
        odds_df=odds_df,
        outcomes=predict_service.pd.DataFrame(),
        cache={},
        session=_DummySession(),
        extractor=object(),
    )

    fight = events_map["the_odds_api|2026-05-30"]["fights"][0]
    assert fight["model_prob_f1"] == 51.2
    assert captured["score_as_of"] == datetime(2026, 5, 29)
    assert captured["fight_count_as_of"] == [datetime(2026, 5, 29), datetime(2026, 5, 29)]
