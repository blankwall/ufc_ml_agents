from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.services import bet_evaluator  # noqa: E402


class _StubSession:
    query = object()


def test_review_gate_surfaces_broad_elo_override(monkeypatch):
    snapshots = {
        "Pick Fighter": {
            "resolved": True,
            "identity": {"canonical_name": "Pick Fighter"},
            "elo": {"elo_current": 1280},
            "qualitative": {"cardio_score": 62},
        },
        "Opp Fighter": {
            "resolved": True,
            "identity": {"canonical_name": "Opp Fighter"},
            "elo": {"elo_current": 1210},
            "qualitative": {"cardio_score": 54},
        },
    }
    monkeypatch.setattr(
        bet_evaluator,
        "build_fighter_snapshot",
        lambda fighter_name, **_kwargs: snapshots[fighter_name],
    )

    result = bet_evaluator.evaluate_bet_decision(
        fighter1_name="Pick Fighter",
        fighter2_name="Opp Fighter",
        pick_slot="fighter1",
        pick_model_prob=0.58,
        pick_mkt_prob=0.52,
        pick_odds=-150,
        is_favorite=True,
        is_wmma=False,
        f1_count=5,
        f2_count=6,
        as_of=None,
        session=_StubSession(),
        snapshot_cache={},
    )

    assert result["bet"] is False
    assert result["skip_code"] == "F1"
    assert result["review_candidate"] is True
    assert result["review_bucket"] == "golden_elo_not_expensive"
    assert result["signal_bucket"] == "skip_50_65_elo_50_plus_not_expensive"
    assert result["pick_elo_diff"] == 70.0


def test_strongest_signal_bucket_is_still_surfaced(monkeypatch):
    snapshots = {
        "Pick Fighter": {
            "resolved": True,
            "identity": {"canonical_name": "Pick Fighter"},
            "elo": {"elo_current": 1335},
            "qualitative": {"cardio_score": 64},
        },
        "Opp Fighter": {
            "resolved": True,
            "identity": {"canonical_name": "Opp Fighter"},
            "elo": {"elo_current": 1210},
            "qualitative": {"cardio_score": 50},
        },
    }
    monkeypatch.setattr(
        bet_evaluator,
        "build_fighter_snapshot",
        lambda fighter_name, **_kwargs: snapshots[fighter_name],
    )

    result = bet_evaluator.evaluate_bet_decision(
        fighter1_name="Pick Fighter",
        fighter2_name="Opp Fighter",
        pick_slot="fighter1",
        pick_model_prob=0.62,
        pick_mkt_prob=0.50,
        pick_odds=-145,
        is_favorite=True,
        is_wmma=False,
        f1_count=9,
        f2_count=8,
        as_of=None,
        session=_StubSession(),
        snapshot_cache={},
    )

    assert result["bet"] is False
    assert result["review_candidate"] is True
    assert result["signal_bucket"] == "skip_50_65_elo_100_plus"
    assert result["signal_label"] == "ELO 100+ confidence bucket"
    assert "skip_50_65_elo_50_plus_not_expensive" in result["signal_tags"]
