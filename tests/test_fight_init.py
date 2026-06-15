from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mcp_server import fight_init  # noqa: E402
from mcp_server import ufc_context_server  # noqa: E402
from backtest import deterministic_signal_filter  # noqa: E402


class _DummySession:
    def close(self):
        return None


def test_init_fight_analysis_builds_structured_payload(monkeypatch):
    monkeypatch.setattr(fight_init, "_Session", lambda: _DummySession())

    def fake_resolve(_session, name):
        fighters = {
            "King Green": SimpleNamespace(id=10, name="King Green", wins=32, losses=16, draws=1),
            "Charles Johnson": SimpleNamespace(id=20, name="Charles Johnson", wins=17, losses=6, draws=0),
        }
        return fighters.get(name)

    monkeypatch.setattr(fight_init, "_resolve_fighter", fake_resolve)
    monkeypatch.setattr(fight_init, "MatchupFeatureExtractor", lambda _session: object())
    monkeypatch.setattr(
        fight_init,
        "_score_row",
        lambda *_args, **_kwargs: {"model_prob_f1": 0.61, "model_source": "general"},
    )
    monkeypatch.setattr(
        fight_init,
        "_fight_count_as_of",
        lambda _session, fighter_id, _as_of: 4 if fighter_id == 10 else 9,
    )
    monkeypatch.setattr(fight_init, "_is_wmma", lambda _session, _fighter_id: False)
    monkeypatch.setattr(fight_init, "_load_all_odds", lambda: pd.DataFrame())
    monkeypatch.setattr(
        fight_init,
        "build_fighter_snapshot",
        lambda fighter_name, **_kwargs: {
            "resolved": True,
            "identity": {"resolved_name": "King Green" if fighter_name == "Bobby Green" else fighter_name},
            "elo": {
                "available": True,
                "elo_current": 1200,
                "elo_peak": 1250,
                "recent_fights": [{"fight_date": "2025-01-01"}],
            },
            "recent_results": [{"date": "2025-01-01", "result": "win"}],
        },
    )
    monkeypatch.setattr(
        fight_init,
        "describe_confidence",
        lambda _pick_prob: {
            "confidence_score": 7,
            "confidence_method": "backtest_pick_prob_decile",
            "confidence_prob_min": 59.0,
            "confidence_prob_max": 63.0,
            "confidence_avg_prob": 61.0,
            "confidence_historical_win_rate": 62.5,
            "confidence_sample_size": 88,
        },
    )

    result = fight_init.init_fight_analysis(
        fighter1="Bobby Green",
        fighter2="Charles Johnson",
        fight_date="2025-05-10",
        fighter1_odds=120,
        fighter2_odds=-140,
    )

    assert result["status"] == "ok"
    assert result["validation"]["ok"] is True
    assert result["resolution"]["fighter1"]["alias_applied"] is True
    assert result["resolution"]["fighter1"]["lookup_name"] == "King Green"
    assert result["resolution"]["fighter1"]["resolved_name"] == "King Green"
    assert result["resolution"]["fight_date"]["parsed"] == "2025-05-10"
    assert result["market"]["normalization_method"] == "vig_normalized"
    assert result["market"]["pricing_context"]["edge_type"] == "market_edge"
    assert result["market"]["pricing_context"]["pricing_context_degraded"] is False
    assert result["market"]["normalized_probabilities_pct"]["fighter1"] == 43.8
    assert result["prediction"]["pick"]["slot"] == "fighter1"
    assert result["prediction"]["pick"]["edge_pct"] == 17.2
    assert result["prediction"]["confidence"]["score"] == 7
    assert result["prediction"]["fighter_metadata"]["thin_data_warning"] is False
    assert result["fighters"]["fighter1"]["identity"]["resolved_name"] == "King Green"
    assert result["fighters"]["fighter2"]["elo"]["available"] is True
    assert "recent_fights" not in result["fighters"]["fighter1"]["elo"]
    assert result["fighters"]["fighter1"]["recent_results"][0]["result"] == "win"
    assert result["provenance"]["steps"]["model_prediction"] == "completed"


def test_init_fight_analysis_returns_validation_errors_before_scoring():
    result = fight_init.init_fight_analysis(
        fighter1="",
        fighter2="Same Fighter",
        fight_date="not-a-date",
        fighter1_odds=0,
    )

    assert result["status"] == "invalid"
    assert result["validation"]["ok"] is False
    error_codes = {error["code"] for error in result["validation"]["errors"]}
    assert {"missing_fighter", "invalid_fight_date", "invalid_odds"} <= error_codes
    assert result["fighters"] is None
    assert result["prediction"] is None
    assert result["provenance"]["steps"]["model_prediction"] == "not_run"


def test_init_fight_analysis_reports_unresolved_fighters(monkeypatch):
    monkeypatch.setattr(fight_init, "_Session", lambda: _DummySession())
    monkeypatch.setattr(fight_init, "_load_all_odds", lambda: pd.DataFrame())
    monkeypatch.setattr(fight_init, "_resolve_fighter", lambda _session, _name: None)

    result = fight_init.init_fight_analysis(
        fighter1="Unknown One",
        fighter2="Unknown Two",
        fight_date="2025-01-01",
    )

    assert result["status"] == "invalid"
    assert result["validation"]["ok"] is False
    assert {error["code"] for error in result["validation"]["errors"]} == {"fighter_not_found"}
    assert result["resolution"]["fighter1"]["resolved"] is False
    assert result["resolution"]["fighter2"]["resolved"] is False
    assert result["fighters"] is None
    assert result["market"]["pricing_context"]["market_missing"] is True
    assert result["market"]["pricing_context"]["edge_type"] == "neutral_line_edge"
    assert any(warning["code"] == "market_missing" for warning in result["validation"]["warnings"])
    assert result["provenance"]["steps"]["fighter_resolution"] == "completed"


def test_init_fight_analysis_falls_back_to_app_odds_lookup(monkeypatch):
    monkeypatch.setattr(fight_init, "_Session", lambda: _DummySession())
    monkeypatch.setattr(
        fight_init,
        "_load_all_odds",
        lambda: pd.DataFrame(
            [
                {
                    "fighter1": "Alex Perez",
                    "fighter2": "Su Mudaerji",
                    "fighter1_odds": -110,
                    "fighter2_odds": -110,
                    "event_date": "2026-05-30",
                    "event_name": "UFC Fight Night",
                    "event_url": "https://example.test/event",
                    "source_type": "the_odds_api",
                    "source_file": "the_odds_api_new_events.csv",
                    "last_update": "2026-05-19T13:00:00Z",
                }
            ]
        ),
    )

    def fake_resolve(_session, name):
        fighters = {
            "Alex Perez": SimpleNamespace(id=10, name="Alex Perez", wins=25, losses=9, draws=0),
            "Su Mudaerji": SimpleNamespace(id=20, name="Su Mudaerji", wins=17, losses=7, draws=0),
            "Sumudaerji": SimpleNamespace(id=20, name="Sumudaerji", wins=17, losses=7, draws=0),
        }
        return fighters.get(name)

    monkeypatch.setattr(fight_init, "_resolve_fighter", fake_resolve)
    monkeypatch.setattr(fight_init, "MatchupFeatureExtractor", lambda _session: object())
    monkeypatch.setattr(
        fight_init,
        "_score_row",
        lambda *_args, **_kwargs: {"model_prob_f1": 0.55, "model_source": "general"},
    )
    monkeypatch.setattr(fight_init, "_fight_count_as_of", lambda *_args, **_kwargs: 10)
    monkeypatch.setattr(fight_init, "_is_wmma", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        fight_init,
        "describe_confidence",
        lambda _pick_prob: {
            "confidence_score": 5,
            "confidence_method": "backtest_pick_prob_decile",
            "confidence_prob_min": 54.0,
            "confidence_prob_max": 56.0,
            "confidence_avg_prob": 55.0,
            "confidence_historical_win_rate": 55.0,
            "confidence_sample_size": 50,
        },
    )
    monkeypatch.setattr(
        fight_init,
        "build_fighter_snapshot",
        lambda fighter_name, **_kwargs: {
            "resolved": True,
            "identity": {"resolved_name": fighter_name},
            "elo": {"available": True, "elo_current": 1200, "elo_peak": 1230, "recent_fights": []},
            "recent_results": [],
        },
    )

    result = fight_init.init_fight_analysis(
        fighter1="Alex Perez",
        fighter2="Su Mudaerji",
        fight_date="2026-05-30",
    )

    assert result["status"] == "ok"
    assert result["market"]["odds"]["fighter1"] == -110
    assert result["market"]["odds"]["fighter2"] == -110
    assert result["market"]["provenance"]["source"] == "app_odds_lookup"
    assert result["market"]["provenance"]["lookup"]["source_file"] == "the_odds_api_new_events.csv"
    assert result["market"]["pricing_context"]["pricing_context_degraded"] is False


def test_normalize_market_odds_marks_neutral_line_as_degraded():
    result = fight_init.normalize_market_odds(None, None)

    assert result["normalization_method"] == "even_money_default"
    assert result["pricing_context"] == {
        "has_real_market": False,
        "has_two_sided_market": False,
        "market_missing": True,
        "pricing_context_degraded": True,
        "edge_type": "neutral_line_edge",
        "market_completeness": "missing_market",
        "warning_codes": ["market_missing", "pricing_context_degraded"],
    }


def test_mcp_tool_forwards_to_helper(monkeypatch):
    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(ufc_context_server, "build_init_fight_analysis", fake_init)

    result = ufc_context_server.init_fight_analysis(
        fighter1="A",
        fighter2="B",
        fight_date="2025-06-01",
        fighter1_odds=-110,
        fighter2_odds=-110,
    )

    assert result == {"status": "ok"}
    assert captured == {
        "fighter1": "A",
        "fighter2": "B",
        "fight_date": "2025-06-01",
        "fighter1_odds": -110,
        "fighter2_odds": -110,
    }


def test_get_elo_market_signal_scores_support(monkeypatch):
    monkeypatch.setattr(
        fight_init,
        "init_fight_analysis",
        lambda **_kwargs: {
            "status": "ok",
            "request": {"fighter1": "A", "fighter2": "B", "fight_date": "2026-05-30"},
            "market": {"odds": {"fighter1": 120, "fighter2": -140}},
            "prediction": {
                "pick": {
                    "slot": "fighter1",
                    "fighter_name": "A",
                    "probability": 0.58,
                    "probability_pct": 58.0,
                    "market_probability": 0.455,
                    "market_probability_pct": 45.5,
                }
            },
            "fighters": {
                "fighter1": {"elo": {"elo_current": 1325}},
                "fighter2": {"elo": {"elo_current": 1210}},
            },
        },
    )
    captured_patterns = []

    def fake_pattern_stats(pattern_names, *, context_pool_path):
        captured_patterns.extend(pattern_names)
        return {
            "available": True,
            "matched_patterns": [
                {
                    "pattern_name": pattern_names[0],
                    "sample_size": 10,
                    "win_rate": 0.7,
                    "roi": 0.12,
                }
            ],
            "missing_patterns": [],
        }

    monkeypatch.setattr(deterministic_signal_filter, "_load_pattern_stats", fake_pattern_stats)

    result = fight_init.get_elo_market_signal(fighter1="A", fighter2="B", fight_date="2026-05-30")

    assert result["status"] == "ok"
    assert result["elo"]["pick_elo_diff"] == 115
    assert result["historical_signal"]["boost_points"] >= 2
    assert any(item["tag"] == "elo_100_plus" for item in result["historical_signal"]["triggers"])
    assert any(item["tag"] == "underdog_elo_support" for item in result["historical_signal"]["triggers"])
    assert "skip_50_65_elo_100_plus" in captured_patterns
    assert "underdog_elo_support" in captured_patterns
    assert result["historical_signal"]["historical_evidence"]["matched_patterns"][0]["roi"] == 0.12


def test_get_elo_market_signal_wrapper_forwards(monkeypatch):
    captured = {}

    def fake_signal(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "historical_signal": {"tier": "mild_boost"}}

    monkeypatch.setattr(ufc_context_server, "build_elo_market_signal", fake_signal)

    result = ufc_context_server.get_elo_market_signal(
        fighter1="A",
        fighter2="B",
        fight_date="2025-06-01",
        fighter1_odds=110,
        fighter2_odds=-130,
    )

    assert result["status"] == "ok"
    assert captured == {
        "fighter1": "A",
        "fighter2": "B",
        "fight_date": "2025-06-01",
        "fighter1_odds": 110,
        "fighter2_odds": -130,
    }


def test_deterministic_signal_filter_wrapper_forwards(monkeypatch):
    captured = {}

    def fake_filter(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "filter_version": "deterministic_elo_cardio_v1"}

    monkeypatch.setattr(ufc_context_server, "build_deterministic_signal_filter", fake_filter)

    result = ufc_context_server.get_deterministic_signal_filter(
        fighter1="A",
        fighter2="B",
        fight_date="2025-06-01",
        fighter1_odds=110,
        fighter2_odds=-130,
    )

    assert result["filter_version"] == "deterministic_elo_cardio_v1"
    assert captured == {
        "fighter1": "A",
        "fighter2": "B",
        "fight_date": "2025-06-01",
        "fighter1_odds": 110,
        "fighter2_odds": -130,
    }
