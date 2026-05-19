import json
import sqlite3
from pathlib import Path

import pytest

from backtest.build_context_pool import (
    POOL_COLUMNS,
    build_evidence_items,
    build_pattern_stats,
    create_schema,
    insert_evidence_items,
    insert_pattern_stats,
    insert_pool_rows,
)
from backtest.context_packet import row_to_dict
from backtest.historical_evidence import (
    find_similar_elo_gap_fights,
    find_similar_fighter_profiles as build_similar_fighter_profiles,
    find_similar_market_fights,
    find_trait_matchup_examples,
    get_historical_pattern_summary,
)
from mcp_server import ufc_context_server


def _pool_row(**overrides):
    row = {column: None for column in POOL_COLUMNS}
    row.update(
        {
            "season": 2026,
            "source_results": "backtest/test_results.csv",
            "row_num": 1,
            "source_row_key": "backtest/test_results.csv:1",
            "date": "2026-01-01",
            "fighter1": "Alpha",
            "fighter2": "Beta",
            "pick": "Alpha",
            "winner": "Alpha",
            "pick_prob": 0.61,
            "pick_odds": -145,
            "pick_correct": 1,
            "actual_pnl": 0.69,
            "bet": 0,
            "skip_reason": "favorite confidence",
            "female": 0,
            "edge": 0.08,
            "join_status": "matched",
            "join_method": "unit_test",
            "fighter1_elo": 1360,
            "fighter2_elo": 1240,
            "pick_elo": 1360,
            "opponent_elo": 1240,
            "pick_elo_diff": 120,
            "abs_elo_diff": 120,
            "model_agrees_with_elo": 1,
            "pick_prior_fight_count": 5,
            "opponent_prior_fight_count": 5,
            "pick_avg_prior_opponent_elo": 1260,
            "opponent_avg_prior_opponent_elo": 1230,
            "pick_recent_fights_json": "[]",
            "opponent_recent_fights_json": "[]",
            "market_implied_prob": 0.592,
            "elo_implied_prob": 0.666,
            "model_minus_elo_prob": -0.056,
            "market_minus_elo_prob": -0.074,
            "model_market_elo_triangle": "model_and_market_under_elo",
        }
    )
    row.update(overrides)
    return row


def _trait_evidence_row(fight_pool_id, fighter_name, opponent_name, deltas):
    return {
        "fight_pool_id": fight_pool_id,
        "evidence_type": "trait_delta",
        "evidence_role": "context_metric",
        "summary": f"Trait delta for {fighter_name} vs {opponent_name}",
        "data_json": json.dumps(
            {
                "trait_version": "trait_v0_1_stats_totals",
                "fighter_name": fighter_name,
                "opponent_name": opponent_name,
                "fight_count": 8,
                "opponent_fight_count": 8,
                "trait_confidence": 1.0,
                "opponent_trait_confidence": 1.0,
                "deltas": deltas,
                "validation_notes": {},
            },
            sort_keys=True,
        ),
        "source_table": "trait_snapshots.v_trait_pair_deltas",
        "source_key": f"{fight_pool_id}:1",
        "created_at": "2026-01-01T00:00:00+00:00",
    }


@pytest.fixture
def historical_pool():
    rows = [
        _pool_row(),
        _pool_row(
            season=2025,
            row_num=2,
            source_row_key="backtest/test_results.csv:2",
            date="2025-09-01",
            fighter1="Gamma",
            fighter2="Delta",
            pick="Gamma",
            winner="Gamma",
            pick_prob=0.60,
            pick_odds=-145,
            pick_correct=1,
            actual_pnl=0.69,
            edge=0.06,
            fighter1_elo=1345,
            fighter2_elo=1230,
            pick_elo=1345,
            opponent_elo=1230,
            pick_elo_diff=115,
            abs_elo_diff=115,
            market_implied_prob=0.592,
            elo_implied_prob=0.660,
            model_minus_elo_prob=-0.060,
            market_minus_elo_prob=-0.068,
        ),
        _pool_row(
            season=2025,
            row_num=3,
            source_row_key="backtest/test_results.csv:3",
            date="2025-08-01",
            fighter1="Epsilon",
            fighter2="Zeta",
            pick="Epsilon",
            winner="Zeta",
            pick_prob=0.58,
            pick_odds=-150,
            pick_correct=0,
            actual_pnl=-1.0,
            edge=0.07,
            fighter1_elo=1290,
            fighter2_elo=1255,
            pick_elo=1290,
            opponent_elo=1255,
            pick_elo_diff=35,
            abs_elo_diff=35,
            market_implied_prob=0.600,
            elo_implied_prob=0.550,
            model_minus_elo_prob=0.030,
            market_minus_elo_prob=0.050,
            model_market_elo_triangle="model_and_market_over_elo",
        ),
        _pool_row(
            season=2025,
            row_num=4,
            source_row_key="backtest/test_results.csv:4",
            date="2025-07-01",
            fighter1="Eta",
            fighter2="Theta",
            pick="Eta",
            winner="Eta",
            pick_prob=0.55,
            pick_odds=120,
            pick_correct=1,
            actual_pnl=1.2,
            edge=0.11,
            fighter1_elo=1330,
            fighter2_elo=1245,
            pick_elo=1330,
            opponent_elo=1245,
            pick_elo_diff=85,
            abs_elo_diff=85,
            market_implied_prob=0.455,
            elo_implied_prob=0.620,
            model_minus_elo_prob=-0.070,
            market_minus_elo_prob=-0.165,
            model_market_elo_triangle="model_and_market_under_elo",
        ),
    ]

    uri = "file:historical-evidence-tests?mode=memory&cache=shared"
    keeper = sqlite3.connect(uri, uri=True)
    keeper.row_factory = sqlite3.Row
    create_schema(keeper)
    insert_pool_rows(keeper, rows)
    insert_pattern_stats(keeper, build_pattern_stats(rows))
    insert_evidence_items(keeper, build_evidence_items(keeper))
    insert_evidence_items(
        keeper,
        [
            _trait_evidence_row(
                1,
                "Alpha",
                "Beta",
                {
                    "cardio_score_diff": 9.0,
                    "offensive_control_score_diff": 12.0,
                    "grappling_threat_score_diff": 10.0,
                    "durability_risk_score_diff": -11.0,
                    "defensive_exposure_score_diff": -8.0,
                },
            ),
            _trait_evidence_row(
                2,
                "Gamma",
                "Delta",
                {
                    "cardio_score_diff": 8.0,
                    "offensive_control_score_diff": 13.0,
                    "grappling_threat_score_diff": 11.0,
                    "durability_risk_score_diff": -10.0,
                    "defensive_exposure_score_diff": -9.0,
                },
            ),
            _trait_evidence_row(
                3,
                "Epsilon",
                "Zeta",
                {
                    "striking_efficiency_score_diff": 12.0,
                    "defensive_exposure_score_diff": -10.0,
                    "durability_risk_score_diff": -2.0,
                },
            ),
            _trait_evidence_row(
                4,
                "Eta",
                "Theta",
                {
                    "offensive_control_score_diff": 9.0,
                    "grappling_threat_score_diff": 12.0,
                    "durability_risk_score_diff": -9.0,
                    "defensive_exposure_score_diff": -7.0,
                },
            ),
        ],
    )
    keeper.commit()
    try:
        yield keeper, uri
    finally:
        keeper.close()


def _target_row(conn):
    return row_to_dict(conn.execute("SELECT * FROM backtest_fight_pool WHERE id = 1").fetchone())


def test_find_similar_elo_gap_fights_returns_structured_examples(historical_pool):
    conn, _ = historical_pool
    result = find_similar_elo_gap_fights(conn, target=_target_row(conn), limit=2)

    assert result["summary"]["example_count"] == 2
    assert result["examples"][0]["fight_pool_id"] == 2
    assert "ELO gap" in result["examples"][0]["match_reason"]
    assert result["examples"][0]["provenance"]["source_table"] == "backtest_fight_pool"


def test_find_similar_market_fights_uses_market_profile(historical_pool):
    conn, _ = historical_pool
    result = find_similar_market_fights(conn, target=_target_row(conn), limit=2)

    assert result["examples"][0]["fight_pool_id"] == 2
    assert "price -145" in result["examples"][0]["match_reason"]
    assert result["summary"]["average_pick_odds"] == -147.5


def test_find_trait_matchup_examples_supports_target_signature_and_archetypes(historical_pool):
    conn, _ = historical_pool

    target_result = find_trait_matchup_examples(conn, target=_target_row(conn), limit=2)
    archetype_result = find_trait_matchup_examples(conn, archetype="weak_chin_vs_wrestler", limit=2)
    wrestler_striker_result = find_trait_matchup_examples(conn, archetype="wrestler_vs_striker", limit=2)
    grappling_striking_result = find_trait_matchup_examples(
        conn,
        archetype="grappling_control_vs_striking_efficiency",
        limit=2,
    )

    assert target_result["query"]["signature"]["type"] == "target_signature"
    assert [example["fight_pool_id"] for example in target_result["examples"]] == [2, 4]
    assert archetype_result["query"]["signature"]["archetype"] == "weak_chin_vs_wrestler"
    assert wrestler_striker_result["query"]["signature"]["archetype"] == "wrestler_vs_striker"
    assert grappling_striking_result["query"]["signature"]["archetype"] == "grappling_control_vs_striking_efficiency"
    assert all(example["provenance"]["trait_evidence"]["source_table"] == "trait_snapshots.v_trait_pair_deltas" for example in archetype_result["examples"])


def test_find_trait_matchup_examples_accepts_synthetic_trait_payload(historical_pool):
    conn, _ = historical_pool

    result = find_trait_matchup_examples(
        conn,
        target_payload={
            "deltas": {
                "cardio_score_diff": 7.0,
                "offensive_control_score_diff": 12.0,
                "grappling_threat_score_diff": 10.0,
                "durability_risk_score_diff": -11.0,
                "defensive_exposure_score_diff": -8.0,
            }
        },
        limit=2,
    )

    assert result["query"]["signature"]["type"] == "target_signature"
    assert [example["fight_pool_id"] for example in result["examples"]] == [1, 2]


def test_find_similar_fighter_profiles_uses_trait_and_quantitative_neighbors():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE fighter_trait_snapshots (
            snapshot_id INTEGER PRIMARY KEY,
            main_fight_id INTEGER,
            as_of_date TEXT,
            fighter_id INTEGER,
            fighter_name TEXT,
            opponent_name TEXT,
            fight_count INTEGER,
            recent3_win_rate REAL,
            recent5_win_rate REAL,
            finish_win_rate REAL,
            finish_loss_rate REAL,
            ko_loss_rate REAL,
            avg_sig_landed_per_min REAL,
            avg_sig_absorbed_per_min REAL,
            avg_sig_diff_per_min REAL,
            avg_control_diff_minutes_per_15 REAL,
            experience_score REAL,
            recent_form_score REAL,
            cardio_score REAL,
            durability_risk_score REAL,
            defensive_exposure_score REAL,
            offensive_control_score REAL,
            anti_control_score REAL,
            scramble_score REAL,
            striking_pressure_score REAL,
            striking_efficiency_score REAL,
            grappling_threat_score REAL,
            finishing_threat_score REAL,
            variance_score REAL,
            trait_confidence REAL,
            trait_version TEXT
        )
        """
    )
    rows = [
        (1, 100, "2025-01-01", 10, "Analog One", "Opponent A", 9, 0.67, 0.60, 0.30, 0.10, 0.05, 3.1, 2.4, 0.7, 1.8, 60, 58, 62, 35, 40, 64, 55, 57, 61, 63, 66, 59, 45, 0.9, "trait_v0"),
        (2, 101, "2025-01-01", 11, "Far Away", "Opponent B", 2, 0.00, 0.20, 0.10, 0.40, 0.30, 1.0, 5.0, -4.0, -3.0, 20, 25, 20, 80, 85, 15, 15, 15, 20, 20, 20, 20, 80, 0.8, "trait_v0"),
        (3, 102, "2025-02-01", 12, "Analog Two", "Opponent C", 8, 0.50, 0.55, 0.25, 0.12, 0.06, 3.3, 2.6, 0.6, 2.0, 59, 56, 60, 36, 41, 62, 53, 58, 59, 61, 65, 57, 46, 0.9, "trait_v0"),
    ]
    conn.executemany(
        "INSERT INTO fighter_trait_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    target_snapshot = {
        "resolved": True,
        "query_name": "Target",
        "identity": {"main_fighter_id": 99, "resolved_name": "Target"},
        "record": {"fight_count_as_of": 8},
        "stats": {
            "sig_strikes_landed_per_min": 3.2,
            "sig_strikes_absorbed_per_min": 2.5,
        },
        "qualitative": {
            "available": True,
            "fight_count": 8,
            "recent5_win_rate": 0.58,
            "ko_loss_rate": 0.05,
            "avg_sig_diff_per_min": 0.65,
            "avg_control_diff_minutes_per_15": 1.9,
            "experience_score": 60,
            "recent_form_score": 57,
            "cardio_score": 61,
            "durability_risk_score": 35,
            "defensive_exposure_score": 40,
            "offensive_control_score": 63,
            "anti_control_score": 54,
            "scramble_score": 57,
            "striking_pressure_score": 60,
            "striking_efficiency_score": 62,
            "grappling_threat_score": 66,
            "finishing_threat_score": 58,
            "variance_score": 45,
        },
    }

    result = build_similar_fighter_profiles(conn, target_snapshot=target_snapshot, limit=2, min_fight_count=3)

    assert result["summary"]["evidence_lanes"] == ["qualitative_trait_scores", "quantitative_performance_stats"]
    assert [example["fighter_name"] for example in result["examples"]] == ["Analog One", "Analog Two"]
    assert "qualitative" in result["examples"][0]["matched_fields"]
    assert result["examples"][0]["provenance"]["source_table"] == "fighter_trait_snapshots"


def test_get_historical_pattern_summary_returns_structured_patterns(historical_pool):
    conn, _ = historical_pool
    result = get_historical_pattern_summary(conn, target=_target_row(conn))

    assert result["target"]["fight_pool_id"] == 1
    assert result["summary"]["support_level"] is not None
    assert result["patterns"]
    assert result["patterns"][0]["provenance"]["source_table"] == "pattern_stats"
    assert result["provenance"]["source_table"] == "backtest_fight_pool"


def test_mcp_wrappers_use_structured_historical_helpers(monkeypatch, historical_pool):
    _, uri = historical_pool

    def fake_resolve_database_path(database):
        assert database == "context_pool"
        return Path("unused")

    def fake_readonly_connection(_path):
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(ufc_context_server, "resolve_database_path", fake_resolve_database_path)
    monkeypatch.setattr(ufc_context_server, "readonly_connection", fake_readonly_connection)

    result = ufc_context_server.find_similar_elo_gap_fights(fight_pool_id=1, limit=2)

    assert result["query"]["from_target"]["fight_pool_id"] == 1
    assert result["examples"][0]["fight_pool_id"] == 2


def test_mcp_historical_wrappers_build_dynamic_targets_without_pool_row(monkeypatch, historical_pool):
    _, uri = historical_pool

    def fake_resolve_database_path(database):
        assert database == "context_pool"
        return Path("unused")

    def fake_readonly_connection(_path):
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def fake_init(**_kwargs):
        return {
            "status": "ok",
            "request": {"fighter1": "Future Alpha", "fighter2": "Future Beta", "fight_date": "2026-05-30"},
            "resolution": {
                "fighter1": {"resolved_name": "Future Alpha"},
                "fighter2": {"resolved_name": "Future Beta"},
                "fight_date": {"parsed": "2026-05-30"},
            },
            "market": {
                "odds": {"fighter1": -145, "fighter2": 120},
                "provenance": {"source": "unit_test"},
            },
            "prediction": {
                "pick": {
                    "slot": "fighter1",
                    "fighter_name": "Future Alpha",
                    "probability": 0.60,
                    "market_probability": 0.592,
                    "edge": 0.08,
                },
                "fighter_metadata": {"is_wmma": False},
            },
            "fighters": {
                "fighter1": {
                    "record": {"fight_count_as_of": 8},
                    "identity": {"resolved_name": "Future Alpha"},
                    "elo": {"elo_current": 1360},
                    "qualitative": {
                        "available": True,
                        "fight_count": 8,
                        "trait_version": "trait_v0",
                        "trait_confidence": 0.9,
                        "cardio_score": 60,
                        "offensive_control_score": 62,
                        "grappling_threat_score": 63,
                        "durability_risk_score": 30,
                        "defensive_exposure_score": 35,
                    },
                },
                "fighter2": {
                    "record": {"fight_count_as_of": 9},
                    "identity": {"resolved_name": "Future Beta"},
                    "elo": {"elo_current": 1240},
                    "qualitative": {
                        "available": True,
                        "fight_count": 9,
                        "trait_version": "trait_v0",
                        "trait_confidence": 0.9,
                        "cardio_score": 52,
                        "offensive_control_score": 50,
                        "grappling_threat_score": 52,
                        "durability_risk_score": 42,
                        "defensive_exposure_score": 44,
                    },
                },
            },
        }

    monkeypatch.setattr(ufc_context_server, "resolve_database_path", fake_resolve_database_path)
    monkeypatch.setattr(ufc_context_server, "readonly_connection", fake_readonly_connection)
    monkeypatch.setattr(ufc_context_server, "build_init_fight_analysis", fake_init)
    monkeypatch.setattr(
        ufc_context_server,
        "resolve_target_row",
        lambda *_args, **_kwargs: pytest.fail("dynamic fighter/date calls must not require a context-pool target row"),
    )

    elo_result = ufc_context_server.find_similar_elo_gap_fights(
        fighter1="Future Alpha",
        fighter2="Future Beta",
        date="2026-05-30",
        limit=2,
    )
    market_result = ufc_context_server.find_similar_market_fights(
        fighter1="Future Alpha",
        fighter2="Future Beta",
        date="2026-05-30",
        limit=2,
    )
    trait_result = ufc_context_server.find_trait_matchup_examples(
        fighter1="Future Alpha",
        fighter2="Future Beta",
        date="2026-05-30",
        limit=2,
    )
    pattern_result = ufc_context_server.get_historical_pattern_summary(
        fighter1="Future Alpha",
        fighter2="Future Beta",
        date="2026-05-30",
    )

    assert elo_result["query"]["from_target"]["target_type"] == "dynamic_synthetic"
    assert elo_result["examples"][0]["fight_pool_id"] == 1
    assert market_result["examples"][0]["fight_pool_id"] == 1
    assert trait_result["query"]["signature"]["type"] == "target_signature"
    assert pattern_result["target"]["target_type"] == "dynamic_synthetic"
    assert pattern_result["dynamic_source"]["source"] == "init_fight_analysis"
