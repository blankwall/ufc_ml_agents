import json
import sqlite3
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.routers import predict as predict_router  # noqa: E402
from services import predict_context_service as svc  # noqa: E402


class _DummySession:
    def close(self):
        return None


TEST_DB = Path(__file__).resolve().parent / ".predict_context_pool.sqlite"


def _write_context_pool() -> Path:
    if TEST_DB.exists():
        TEST_DB.unlink()
    conn = sqlite3.connect(TEST_DB)
    conn.executescript(
        """
        CREATE TABLE backtest_fight_pool (
            id INTEGER PRIMARY KEY,
            date TEXT,
            fighter1 TEXT,
            fighter2 TEXT,
            pick TEXT,
            winner TEXT,
            pick_prob REAL,
            pick_odds INTEGER,
            pick_correct INTEGER,
            actual_pnl REAL,
            bet INTEGER,
            skip_reason TEXT,
            female INTEGER,
            edge REAL,
            pick_elo_diff REAL,
            model_agrees_with_elo INTEGER
        );
        CREATE TABLE pattern_stats (
            pattern_name TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            filters_json TEXT NOT NULL,
            sample_size INTEGER NOT NULL,
            graded_sample_size INTEGER NOT NULL,
            ungraded_sample_size INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            win_rate REAL,
            profit REAL NOT NULL,
            roi REAL,
            avg_confidence REAL,
            avg_edge REAL,
            avg_elo_diff REAL,
            last_graded_date TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE evidence_items (
            evidence_id INTEGER PRIMARY KEY,
            fight_pool_id INTEGER NOT NULL,
            evidence_type TEXT NOT NULL,
            evidence_role TEXT NOT NULL,
            summary TEXT NOT NULL,
            data_json TEXT NOT NULL,
            source_table TEXT NOT NULL,
            source_key TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    rows = [
        (1, "2026-01-01", "A", "B", "A", "A", 0.58, -120, 1, 0.83, 0, "favorite confidence", 0, 0.07, 75, 1),
        (2, "2026-01-08", "C", "D", "C", "D", 0.59, -110, 0, -1.00, 0, "favorite confidence", 0, 0.04, 110, 1),
        (3, "2026-02-01", "Dog", "Fav", "Dog", "Dog", 0.57, 150, 1, 1.50, 1, None, 0, 0.17, -80, 0),
        (4, "2026-02-08", "Dog2", "Fav2", "Dog2", "Fav2", 0.56, 130, 0, -1.00, 1, None, 0, 0.13, -60, 0),
    ]
    conn.executemany(
        """
        INSERT INTO backtest_fight_pool
        (id, date, fighter1, fighter2, pick, winner, pick_prob, pick_odds, pick_correct,
         actual_pnl, bet, skip_reason, female, edge, pick_elo_diff, model_agrees_with_elo)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    pattern_rows = [
        ("skip_50_65_elo_50_plus_not_expensive", "Golden ELO not-expensive bucket", "{}", 2, 2, 0, 1, 1, 0.5, -0.17, -0.085, 0.585, 0.055, 92.5, "2026-01-08", "now"),
        ("underdog_elo_against", "Plus-money model picks against ELO", "{}", 2, 2, 0, 1, 1, 0.5, 0.50, 0.25, 0.565, 0.15, -70.0, "2026-02-08", "now"),
        ("model_pick_lower_elo", "Model pick has lower ELO", "{}", 2, 2, 0, 1, 1, 0.5, 0.50, 0.25, 0.565, 0.15, -70.0, "2026-02-08", "now"),
    ]
    conn.executemany(
        """
        INSERT INTO pattern_stats
        (pattern_name, description, filters_json, sample_size, graded_sample_size, ungraded_sample_size,
         wins, losses, win_rate, profit, roi, avg_confidence, avg_edge, avg_elo_diff, last_graded_date, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        pattern_rows,
    )
    evidence = [
        (3, {"trait_confidence": 0.9, "opponent_trait_confidence": 0.88, "deltas": {"cardio_score_diff": 12, "striking_efficiency_score_diff": 11, "defensive_exposure_score_diff": -12, "scramble_score_diff": 10}}),
        (4, {"trait_confidence": 0.8, "opponent_trait_confidence": 0.8, "deltas": {"striking_pressure_score_diff": 12}}),
    ]
    conn.executemany(
        """
        INSERT INTO evidence_items
        (fight_pool_id, evidence_type, evidence_role, summary, data_json, source_table, source_key, created_at)
        VALUES (?, 'trait_delta', 'context_metric', 'trait summary', ?, 'test', 'test', 'now')
        """,
        [(fight_id, json.dumps(payload)) for fight_id, payload in evidence],
    )
    conn.commit()
    conn.close()
    return TEST_DB


def _frame(*, pick_prob=0.58, pick_odds=-120, edge=0.07):
    return {
        "request": {"fighter1": "Pick", "fighter2": "Opponent", "fight_date": "2026-03-01", "fighter1_odds": pick_odds, "fighter2_odds": None},
        "as_of": None,
        "resolved": {"fighter1_db_name": "Pick", "fighter2_db_name": "Opponent", "model_pick_db_name": "Pick", "opponent_db_name": "Opponent"},
        "fight_counts": {"fighter1": 5, "fighter2": 6},
        "model_context": {
            "model_pick": "Pick",
            "model_pick_db_name": "Pick",
            "pick_slot": "fighter1",
            "opponent_slot": "fighter2",
            "pick_prob": pick_prob,
            "pick_odds": pick_odds,
            "market_prob": round(pick_prob - edge, 4),
            "edge": edge,
            "model_prob_f1": pick_prob,
            "model_prob_f2": round(1 - pick_prob, 4),
            "model_source": "test",
        },
    }


def _snapshot(*, elo, cardio=50, striking_eff=50, exposure=50, scramble=50, trait_conf=0.9):
    return {
        "resolved": True,
        "identity": {"canonical_name": "fighter"},
        "elo": {"elo_current": elo},
        "qualitative": {
            "available": True,
            "trait_confidence": trait_conf,
            "cardio_score": cardio,
            "striking_efficiency_score": striking_eff,
            "defensive_exposure_score": exposure,
            "scramble_score": scramble,
            "offensive_control_score": 50,
            "anti_control_score": 50,
            "striking_pressure_score": 50,
            "finishing_threat_score": 50,
        },
    }


def _patch_frame_and_snapshots(monkeypatch, frame, pick_snapshot, opp_snapshot):
    monkeypatch.setattr(svc, "build_prediction_frame", lambda **_kwargs: frame)

    def fake_snapshot(name, **_kwargs):
        return pick_snapshot if name == "Pick" else opp_snapshot

    monkeypatch.setattr(svc, "build_fighter_snapshot", fake_snapshot)


def test_golden_elo_not_expensive_returns_pattern_hit_and_stats(monkeypatch):
    db_path = _write_context_pool()
    try:
        _patch_frame_and_snapshots(
            monkeypatch,
            _frame(pick_prob=0.58, pick_odds=-120, edge=0.07),
            _snapshot(elo=1500),
            _snapshot(elo=1425),
        )

        result = svc.build_predict_context(
            fighter1="Pick",
            fighter2="Opponent",
            fighter1_odds=-120,
            session=_DummySession(),
            context_pool_path=db_path,
        )

        labels = result["pattern_hits"]["labels"]
        assert "golden_elo_not_expensive" in labels
        assert "skip_50_65_elo_50_plus_not_expensive" in labels
        assert result["historical_bucket_stats"]["golden_elo_not_expensive"]["n"] == 2
        assert result["historical_bucket_stats"]["skip_50_65_elo_50_plus_not_expensive"]["source"] == "pattern_stats"
        assert result["similar_rows"]
    finally:
        if db_path.exists():
            db_path.unlink()


def test_underdog_elo_against_with_trait_support_is_offset_not_blanket_fade(monkeypatch):
    db_path = _write_context_pool()
    try:
        _patch_frame_and_snapshots(
            monkeypatch,
            _frame(pick_prob=0.57, pick_odds=150, edge=0.17),
            _snapshot(elo=1400, cardio=62, striking_eff=61, exposure=38, scramble=60),
            _snapshot(elo=1480, cardio=50, striking_eff=50, exposure=50, scramble=50),
        )

        result = svc.build_predict_context(
            fighter1="Pick",
            fighter2="Opponent",
            fighter1_odds=150,
            session=_DummySession(),
            context_pool_path=db_path,
        )

        labels = result["pattern_hits"]["labels"]
        assert "underdog_elo_against" in labels
        assert "model_pick_lower_elo" in labels
        assert "trait_offset_elo_against" in labels
        assert "primary_trait_support_any" in result["pattern_hits"]["trait_support"]
        assert "elo_disagrees_moderate" in result["risk_flags"]
        assert "blanket_fade" not in labels
        assert result["historical_bucket_stats"]["underdog_elo_against"]["n"] == 2
        primary_overlay = next(row for row in result["trait_overlays"] if row["name"] == "primary_support_any")
        assert primary_overlay["cohort"] == "plus_money_elo_against"
        assert primary_overlay["n"] == 1
    finally:
        if db_path.exists():
            db_path.unlink()


def test_missing_context_pool_returns_unavailable_data_quality(monkeypatch):
    missing_path = Path(__file__).resolve().parent / ".missing_predict_context_pool.sqlite"
    if missing_path.exists():
        missing_path.unlink()
    _patch_frame_and_snapshots(
        monkeypatch,
        _frame(),
        _snapshot(elo=1500),
        _snapshot(elo=1425),
    )

    result = svc.build_predict_context(
        fighter1="Pick",
        fighter2="Opponent",
        session=_DummySession(),
        context_pool_path=missing_path,
    )

    assert result["data_quality"]["context_pool_available"] is False
    assert result["historical_bucket_stats"] == {}
    assert result["similar_rows"] == []


def test_predict_context_endpoint_response_shape(monkeypatch):
    monkeypatch.setattr(predict_router, "_Session", lambda: _DummySession())
    monkeypatch.setattr(
        predict_router,
        "build_predict_context",
        lambda **_kwargs: {
            "request": {"fighter1": "A", "fighter2": "B"},
            "model_context": {"model_pick": "A"},
            "current_context": {},
            "pattern_hits": {"labels": []},
            "historical_bucket_stats": {},
            "similar_rows": [],
            "trait_overlays": [],
            "risk_flags": [],
            "data_quality": {"context_pool_available": False},
        },
    )
    app = FastAPI()
    app.include_router(predict_router.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/predict/context",
        json={"fighter1": "A", "fighter2": "B", "fighter1_odds": 120, "fighter2_odds": -140},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "request",
        "model_context",
        "current_context",
        "pattern_hits",
        "historical_bucket_stats",
        "similar_rows",
        "trait_overlays",
        "risk_flags",
        "data_quality",
    }
