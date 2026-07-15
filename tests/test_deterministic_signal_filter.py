from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from backtest import deterministic_signal_filter as signals  # noqa: E402


def _analysis(*, elo_diff=115, cardio_diff=12, odds=120):
    pick_elo = 1325
    opp_elo = pick_elo - elo_diff
    pick_cardio = 62
    opp_cardio = pick_cardio - cardio_diff
    return {
        "status": "ok",
        "request": {"fighter1": "A", "fighter2": "B", "fight_date": "2026-05-30"},
        "market": {"odds": {"fighter1": odds, "fighter2": -140}},
        "prediction": {
            "pick": {
                "slot": "fighter1",
                "fighter_name": "A",
                "probability": 0.58,
                "probability_pct": 58.0,
                "market_probability": 0.455,
                "market_probability_pct": 45.5,
                "edge": 0.125,
            }
        },
        "fighters": {
            "fighter1": {
                "elo": {"elo_current": pick_elo},
                "qualitative": {
                    "available": True,
                    "cardio_score": pick_cardio,
                    "trait_confidence": 0.9,
                },
            },
            "fighter2": {
                "elo": {"elo_current": opp_elo},
                "qualitative": {
                    "available": True,
                    "cardio_score": opp_cardio,
                    "trait_confidence": 0.85,
                },
            },
        },
    }


def test_evaluate_elo_market_signal_returns_historical_bucket_evidence(monkeypatch):
    captured_patterns = []

    def fake_pattern_stats(pattern_names, *, context_pool_path):
        captured_patterns.extend(pattern_names)
        return {
            "available": True,
            "matched_patterns": [
                {
                    "pattern_name": "underdog_elo_support",
                    "sample_size": 12,
                    "win_rate": 0.67,
                    "roi": 0.22,
                }
            ],
            "missing_patterns": [],
        }

    monkeypatch.setattr(signals, "_load_pattern_stats", fake_pattern_stats)

    result = signals.evaluate_elo_market_signal(_analysis())

    assert result["status"] == "ok"
    assert result["elo"]["pick_elo_diff"] == 115
    assert "skip_50_65_elo_100_plus" in captured_patterns
    assert "underdog_elo_support" in captured_patterns
    assert result["historical_signal"]["historical_evidence"]["matched_patterns"][0]["roi"] == 0.22


def test_evaluate_cardio_signal_flags_validated_cardio_advantage():
    result = signals.evaluate_cardio_signal(_analysis(cardio_diff=14))

    assert result["status"] == "ok"
    assert result["cardio"]["cardio_score_diff"] == 14
    assert result["signal"]["tier"] == "cardio_support"
    assert result["signal"]["validation"]["status"] == "first_pass_aligned"
    assert result["signal"]["triggers"][0]["tag"] == "validated_cardio_advantage"


def test_deterministic_signal_filter_combines_elo_and_cardio(monkeypatch):
    monkeypatch.setattr(
        signals,
        "_load_pattern_stats",
        lambda pattern_names, *, context_pool_path: {
            "available": True,
            "matched_patterns": [],
            "missing_patterns": [],
        },
    )

    result = signals.evaluate_deterministic_signal_filter(_analysis())

    assert result["filter_version"] == "deterministic_elo_cardio_v1"
    assert result["summary"]["action"] == "positive_filter_review"
    assert "validated_cardio_advantage" in result["summary"]["support_flags"]
    assert result["summary"]["not_a_recommendation"] is True
