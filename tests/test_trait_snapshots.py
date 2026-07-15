import json

from backtest.build_trait_snapshots import (
    FightSide,
    compute_snapshot,
    duration_minutes,
    parse_control_seconds,
    parse_fraction,
)


def _side(**overrides):
    base = {
        "main_fight_id": 1,
        "fight_date": "2026-01-01",
        "fighter_id": 10,
        "fighter_name": "Fighter A",
        "opponent_id": 20,
        "opponent_name": "Fighter B",
        "sergey_fight_id": 100,
        "sergey_fighter_id": 200,
        "result": "win",
        "method": "Decision - Unanimous",
        "duration_min": 15.0,
        "reached_round_3": True,
        "sig_landed": 50,
        "sig_attempted": 100,
        "sig_absorbed": 30,
        "sig_attempted_against": 90,
        "knockdowns_for": 0,
        "knockdowns_against": 0,
        "takedowns_landed": 2,
        "takedowns_attempted": 4,
        "opponent_takedowns_landed": 1,
        "opponent_takedowns_attempted": 5,
        "submission_attempts": 1,
        "control_seconds": 300,
        "control_conceded_seconds": 60,
    }
    base.update(overrides)
    return FightSide(**base)


def test_trait_parsers_handle_ufcstats_strings():
    assert parse_fraction("42 of 60") == (42.0, 60.0)
    assert parse_fraction("---") == (0.0, 0.0)
    assert parse_control_seconds("6:02") == 362.0
    assert duration_minutes(3, "2:30", 3) == 12.5


def test_compute_snapshot_uses_prior_history_only():
    target = _side(main_fight_id=4, fight_date="2026-04-01")
    history = [
        _side(main_fight_id=1, fight_date="2025-01-01", result="win", method="Submission"),
        _side(main_fight_id=2, fight_date="2025-06-01", result="loss", method="KO/TKO", knockdowns_against=1),
        _side(main_fight_id=3, fight_date="2025-12-01", result="win", method="Decision - Unanimous"),
    ]

    snapshot = compute_snapshot(target=target, history=history, created_at="2026-01-01T00:00:00+00:00")

    assert snapshot["main_fight_id"] == 4
    assert snapshot["fight_count"] == 3
    assert snapshot["wins"] == 2
    assert snapshot["losses"] == 1
    assert snapshot["recent3_win_rate"] == 2 / 3
    assert snapshot["source_fights_json"]
    source_fights = json.loads(snapshot["source_fights_json"])
    assert [fight["fight_id"] for fight in source_fights] == [1, 2, 3]
    assert 0 <= snapshot["offensive_control_score"] <= 100
    assert 0 <= snapshot["durability_risk_score"] <= 100
