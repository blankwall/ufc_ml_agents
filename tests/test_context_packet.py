import json
import sqlite3

from backtest.context_packet import build_pattern_score, filter_pattern_rows, pattern_allowed_for_target_score
from backtest.context_candidates import build_candidates
from backtest.validate_context_pipeline import build_validation_row, summarize
from backtest.build_context_pool import (
    POOL_COLUMNS,
    build_evidence_items,
    build_pattern_stats,
    build_trait_evidence_items,
    create_agent_views,
    create_schema,
    elo_implied_probability,
    insert_evidence_items,
    insert_pattern_stats,
    insert_pool_rows,
    quality_metrics,
    triangle_label,
)


def _pattern(name, sample_size, wins, losses, win_rate, roi, *, ungraded=0):
    return {
        "pattern_name": name,
        "sample_size": sample_size,
        "graded_sample_size": sample_size,
        "ungraded_sample_size": ungraded,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "roi": roi,
        "avg_confidence": 0.57,
        "avg_edge": -0.10,
        "avg_elo_diff": 127,
        "last_graded_date": "2026-04-11",
    }


def test_pattern_score_prefers_specific_pattern_on_score_tie():
    score = build_pattern_score(
        {"edge": -0.03, "bet": False},
        [
            _pattern("model_pick_higher_elo", 204, 158, 46, 0.775, 0.196),
            _pattern("skip_50_65_elo_50_plus", 64, 51, 13, 0.797, 0.184, ungraded=4),
        ],
    )

    assert score["score"] == 8
    assert score["source_pattern"] == "skip_50_65_elo_50_plus"
    assert score["basis"]["sample_size"] == 64
    assert score["basis"]["graded_sample_size"] == 64
    assert score["basis"]["ungraded_sample_size"] == 4


def test_skipped_score_ignores_broad_model_elo_patterns():
    score = build_pattern_score(
        {"edge": -0.03, "bet": False},
        [_pattern("model_pick_higher_elo", 204, 158, 46, 0.775, 0.196)],
    )

    assert score["score"] == 5
    assert score["source_pattern"] is None
    assert pattern_allowed_for_target_score({"bet": False}, "model_pick_higher_elo") is False
    assert pattern_allowed_for_target_score({"bet": False}, "skip_50_65_elo_50_plus") is True


def test_filter_pattern_rows_defaults_to_graded_only():
    base_row = {
        "bet": False,
        "pick_prob": 0.55,
        "pick_elo_diff": 75,
        "model_agrees_with_elo": True,
        "pick_odds": -120,
    }
    rows = [
        {**base_row, "pick_correct": True},
        {**base_row, "pick_correct": False},
        {**base_row, "pick_correct": None},
        {**base_row, "pick_prob": 0.70, "pick_correct": True},
    ]

    graded = filter_pattern_rows(rows, pattern_name="skip_50_65_elo_50_plus", include_pending=False)
    with_pending = filter_pattern_rows(rows, pattern_name="skip_50_65_elo_50_plus", include_pending=True)

    assert len(graded) == 2
    assert len(with_pending) == 3


class _FakeConn:
    pass


def test_build_candidates_filters_and_sorts(monkeypatch):
    patterns_by_pick = {
        "Strong Pick": [_pattern("skip_50_65_elo_50_plus", 64, 51, 13, 0.797, 0.184)],
        "Weak Pick": [_pattern("model_pick_higher_elo", 204, 158, 46, 0.775, 0.196)],
    }

    def fake_pattern_payload(conn, row):
        return patterns_by_pick[row["pick"]]

    monkeypatch.setattr("backtest.context_candidates.pattern_payload", fake_pattern_payload)
    rows = [
        {
            "date": "2026-06-01",
            "fighter1": "Strong A",
            "fighter2": "Strong B",
            "pick": "Strong Pick",
            "pick_prob": 0.55,
            "pick_odds": -120,
            "edge": -0.03,
            "pick_elo_diff": 100,
            "skip_reason": "favorite confidence",
            "bet": False,
        },
        {
            "date": "2026-06-01",
            "fighter1": "Weak A",
            "fighter2": "Weak B",
            "pick": "Weak Pick",
            "pick_prob": 0.55,
            "pick_odds": -120,
            "edge": -0.03,
            "pick_elo_diff": 100,
            "skip_reason": "favorite confidence",
            "bet": False,
        },
    ]

    candidates = build_candidates(_FakeConn(), rows, min_score=7)

    assert [candidate["pick"] for candidate in candidates] == ["Strong Pick"]
    assert candidates[0]["score"] == 8
    assert candidates[0]["source_pattern"] == "skip_50_65_elo_50_plus"


def test_quality_metrics_are_point_in_time_summary():
    metrics = quality_metrics(
        [
            {"own_elo": 1200, "opponent_elo": 1100, "won": True},
            {"own_elo": 1250, "opponent_elo": 1300, "won": False},
            {"own_elo": 1230, "opponent_elo": 1400, "won": True},
        ],
        current_elo=1220,
    )

    assert metrics["prior_fight_count"] == 3
    assert metrics["avg_prior_opponent_elo"] == (1100 + 1300 + 1400) / 3
    assert metrics["recent3_prior_opponent_elo"] == (1100 + 1300 + 1400) / 3
    assert len(metrics["recent_fights"]) == 3
    assert metrics["recent_fights"][-1]["opponent_elo"] == 1400
    assert metrics["best_win_opponent_elo"] == 1400
    assert metrics["peak_elo_as_of"] == 1250
    assert metrics["current_vs_peak_decline"] == 30


def test_elo_triangle_helpers():
    implied = elo_implied_probability(141)

    assert round(implied, 3) == 0.692
    assert triangle_label(-0.10, -0.05) == "model_and_market_under_elo"
    assert triangle_label(0.10, 0.05) == "model_and_market_over_elo"
    assert triangle_label(0.10, -0.05) == "model_over_market_under_elo"
    assert triangle_label(-0.10, 0.05) == "model_under_market_over_elo"


def test_validation_row_uses_leave_one_out_evidence():
    base_row = {
        "date": "2026-01-01",
        "season": 2026,
        "fighter1": "A",
        "fighter2": "B",
        "pick": "A",
        "pick_prob": 0.55,
        "pick_odds": -320,
        "edge": 0.01,
        "pick_elo_diff": 75,
        "bet": False,
        "skip_reason": "favorite confidence",
        "model_agrees_with_elo": True,
        "join_status": "matched",
    }
    rows = [
        {**base_row, "id": 1, "pick_correct": True, "actual_pnl": 0.83},
        {**base_row, "id": 2, "pick_correct": True, "actual_pnl": 0.83},
        {**base_row, "id": 3, "pick_correct": False, "actual_pnl": -1.0},
    ]

    validation = build_validation_row(rows, rows[0], mode="leave-one-out")

    assert validation["source_pattern"] == "skip_50_65_elo_50_plus"
    assert validation["source_n"] == 2


def test_summarize_returns_roi():
    stats = summarize([
        {"pick_correct": True, "actual_pnl": 1.0},
        {"pick_correct": False, "actual_pnl": -1.0},
        {"pick_correct": True, "actual_pnl": 0.5},
    ])

    assert stats["n"] == 3
    assert stats["wins"] == 2
    assert stats["losses"] == 1
    assert stats["profit"] == 0.5
    assert stats["roi"] == 0.5 / 3


def test_context_pool_materializes_agent_evidence_views():
    row = {column: None for column in POOL_COLUMNS}
    row.update(
        {
            "season": 2026,
            "source_results": "test.csv",
            "row_num": 1,
            "source_row_key": "test.csv:1",
            "date": "2026-01-01",
            "fighter1": "Fighter A",
            "fighter2": "Fighter B",
            "pick": "Fighter A",
            "winner": "Fighter A",
            "pick_prob": 0.56,
            "pick_odds": -120,
            "pick_correct": 1,
            "actual_pnl": 0.83,
            "bet": 0,
            "skip_reason": "favorite confidence",
            "female": 0,
            "edge": -0.02,
            "join_status": "matched",
            "join_method": "unit_test",
            "pick_elo": 1300,
            "opponent_elo": 1200,
            "pick_elo_diff": 100,
            "abs_elo_diff": 100,
            "model_agrees_with_elo": 1,
            "pick_prior_fight_count": 3,
            "opponent_prior_fight_count": 3,
            "pick_avg_prior_opponent_elo": 1250,
            "opponent_avg_prior_opponent_elo": 1210,
            "pick_opponent_quality_diff": 40,
            "pick_recent_fights_json": json.dumps(
                [{"fight_id": 7, "date": "2025-01-01", "opponent": "Prior Opp", "opponent_elo": 1240, "result": "W"}]
            ),
            "opponent_recent_fights_json": "[]",
            "market_implied_prob": 0.545,
            "elo_implied_prob": elo_implied_probability(100),
            "model_minus_elo_prob": -0.08,
            "market_minus_elo_prob": -0.095,
            "model_market_elo_triangle": "model_and_market_under_elo",
        }
    )

    conn = sqlite3.connect(":memory:")
    try:
        create_schema(conn)
        insert_pool_rows(conn, [row])
        insert_pattern_stats(conn, build_pattern_stats([row]))
        insert_evidence_items(conn, build_evidence_items(conn))
        create_agent_views(conn)

        roles = {
            role
            for (role,) in conn.execute(
                "SELECT DISTINCT evidence_role FROM v_agent_packet_evidence WHERE fight_pool_id = 1"
            )
        }
        pattern_names = {
            name
            for (name,) in conn.execute(
                "SELECT pattern_name FROM v_pattern_evidence WHERE fight_pool_id = 1"
            )
        }
        recent_count = conn.execute("SELECT COUNT(*) FROM v_recent_fight_evidence WHERE fight_pool_id = 1").fetchone()[0]

        assert {"target", "context_metric", "aggregate_pattern", "audit_detail"} <= roles
        assert "skip_50_65_elo_50_plus" in pattern_names
        assert recent_count == 1
    finally:
        conn.close()


def test_context_pool_materializes_trait_delta_evidence(tmp_path):
    row = {column: None for column in POOL_COLUMNS}
    row.update(
        {
            "season": 2026,
            "source_results": "test.csv",
            "row_num": 1,
            "source_row_key": "test.csv:1",
            "date": "2026-01-01",
            "main_fight_id": 99,
            "fighter1": "Sean Omalley",
            "fighter2": "Opponent B",
            "pick": "Sean Omalley",
            "winner": None,
            "pick_prob": 0.56,
            "pick_odds": -120,
            "pick_correct": None,
            "actual_pnl": None,
            "bet": 0,
            "skip_reason": "favorite confidence",
            "female": 0,
            "edge": -0.02,
            "join_status": "unmatched",
            "pick_recent_fights_json": "[]",
            "opponent_recent_fights_json": "[]",
        }
    )
    traits_path = tmp_path / "traits.sqlite"
    trait_conn = sqlite3.connect(traits_path)
    try:
        trait_conn.execute(
            """
            CREATE TABLE v_trait_pair_deltas (
                main_fight_id INTEGER,
                fighter_id INTEGER,
                fighter_name TEXT,
                opponent_name TEXT,
                fight_count INTEGER,
                opponent_fight_count INTEGER,
                trait_confidence REAL,
                opponent_trait_confidence REAL,
                experience_score_diff REAL,
                recent_form_score_diff REAL,
                cardio_score_diff REAL,
                durability_risk_score_diff REAL,
                defensive_exposure_score_diff REAL,
                offensive_control_score_diff REAL,
                anti_control_score_diff REAL,
                scramble_score_diff REAL,
                striking_pressure_score_diff REAL,
                striking_efficiency_score_diff REAL,
                grappling_threat_score_diff REAL,
                finishing_threat_score_diff REAL,
                variance_score_diff REAL
            )
            """
        )
        trait_conn.execute(
            """
            INSERT INTO v_trait_pair_deltas VALUES (
                99, 10, 'Sean O''Malley', 'Opponent B', 8, 5, 1.0, 1.0,
                10, 5, 12, -4, 3, 8, 6, 7, 9, 11, 2, 4, -1
            )
            """
        )
        trait_conn.commit()
    finally:
        trait_conn.close()

    conn = sqlite3.connect(":memory:")
    try:
        create_schema(conn)
        insert_pool_rows(conn, [row])
        trait_evidence = build_trait_evidence_items(conn, traits_path)
        insert_evidence_items(conn, trait_evidence)
        create_agent_views(conn)

        evidence = conn.execute(
            "SELECT summary, data_json FROM v_agent_packet_evidence WHERE evidence_type = 'trait_delta'"
        ).fetchone()
        payload = json.loads(evidence[1])

        assert "cardio=+12.0" in evidence[0]
        assert payload["deltas"]["cardio_score_diff"] == 12
        assert payload["validation_notes"]["cardio_score_diff"]["status"] == "first_pass_aligned"
    finally:
        conn.close()
