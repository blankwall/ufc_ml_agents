import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from services import golden_elo_service as svc  # noqa: E402


def _write_sidecar(path: Path, *, pick_elo: float = 1500, opp_elo: float = 1400):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE fighters (
            fighter_id INTEGER PRIMARY KEY,
            full_name TEXT,
            elo_current REAL
        );
        """
    )
    conn.executemany(
        "INSERT INTO fighters (fighter_id, full_name, elo_current) VALUES (?, ?, ?)",
        [
            (1, "Pick Fighter", pick_elo),
            (2, "Opp Fighter", opp_elo),
        ],
    )
    conn.commit()
    conn.close()


def _write_main_db(path: Path):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE fighters (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        "INSERT INTO fighters (id, name) VALUES (?, ?)",
        [
            (1, "Pick Fighter"),
            (2, "Opp Fighter"),
        ],
    )
    conn.commit()
    conn.close()


def _write_traits(path: Path):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE fighter_trait_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            main_fight_id INTEGER NOT NULL,
            as_of_date TEXT NOT NULL,
            fighter_id INTEGER NOT NULL,
            fighter_name TEXT NOT NULL,
            opponent_id INTEGER NOT NULL,
            opponent_name TEXT NOT NULL,
            cardio_score REAL,
            defensive_exposure_score REAL,
            striking_efficiency_score REAL,
            trait_confidence REAL NOT NULL
        );
        CREATE VIEW v_trait_pair_deltas AS
        SELECT
            a.main_fight_id,
            a.as_of_date,
            a.fighter_id,
            a.fighter_name,
            a.opponent_id,
            a.opponent_name,
            a.cardio_score - b.cardio_score AS cardio_score_diff,
            a.defensive_exposure_score - b.defensive_exposure_score AS defensive_exposure_score_diff,
            a.striking_efficiency_score - b.striking_efficiency_score AS striking_efficiency_score_diff,
            a.trait_confidence,
            b.trait_confidence AS opponent_trait_confidence
        FROM fighter_trait_snapshots a
        JOIN fighter_trait_snapshots b
          ON b.main_fight_id = a.main_fight_id
         AND b.fighter_id = a.opponent_id;
        """
    )
    conn.executemany(
        """
        INSERT INTO fighter_trait_snapshots (
            main_fight_id,
            as_of_date,
            fighter_id,
            fighter_name,
            opponent_id,
            opponent_name,
            cardio_score,
            defensive_exposure_score,
            striking_efficiency_score,
            trait_confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (10, "2026-01-31", 1, "Pick Fighter", 2, "Opp Fighter", 65, 38, 60, 0.90),
            (10, "2026-01-31", 2, "Opp Fighter", 1, "Pick Fighter", 50, 50, 48, 0.85),
        ],
    )
    conn.commit()
    conn.close()


def _write_negative_traits(path: Path):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE fighter_trait_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            main_fight_id INTEGER NOT NULL,
            as_of_date TEXT NOT NULL,
            fighter_id INTEGER NOT NULL,
            fighter_name TEXT NOT NULL,
            opponent_id INTEGER NOT NULL,
            opponent_name TEXT NOT NULL,
            cardio_score REAL,
            defensive_exposure_score REAL,
            striking_efficiency_score REAL,
            trait_confidence REAL NOT NULL
        );
        CREATE VIEW v_trait_pair_deltas AS
        SELECT
            a.main_fight_id,
            a.as_of_date,
            a.fighter_id,
            a.fighter_name,
            a.opponent_id,
            a.opponent_name,
            a.cardio_score - b.cardio_score AS cardio_score_diff,
            a.defensive_exposure_score - b.defensive_exposure_score AS defensive_exposure_score_diff,
            a.striking_efficiency_score - b.striking_efficiency_score AS striking_efficiency_score_diff,
            a.trait_confidence,
            b.trait_confidence AS opponent_trait_confidence
        FROM fighter_trait_snapshots a
        JOIN fighter_trait_snapshots b
          ON b.main_fight_id = a.main_fight_id
         AND b.fighter_id = a.opponent_id;
        """
    )
    conn.executemany(
        """
        INSERT INTO fighter_trait_snapshots (
            main_fight_id,
            as_of_date,
            fighter_id,
            fighter_name,
            opponent_id,
            opponent_name,
            cardio_score,
            defensive_exposure_score,
            striking_efficiency_score,
            trait_confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (20, "2026-01-31", 1, "Pick Fighter", 2, "Opp Fighter", 50, 50, 48, 0.85),
            (20, "2026-01-31", 2, "Opp Fighter", 1, "Pick Fighter", 65, 38, 60, 0.90),
        ],
    )
    conn.commit()
    conn.close()


def _write_negative_defensive_only_traits(path: Path):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE fighter_trait_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            main_fight_id INTEGER NOT NULL,
            as_of_date TEXT NOT NULL,
            fighter_id INTEGER NOT NULL,
            fighter_name TEXT NOT NULL,
            opponent_id INTEGER NOT NULL,
            opponent_name TEXT NOT NULL,
            cardio_score REAL,
            defensive_exposure_score REAL,
            striking_efficiency_score REAL,
            trait_confidence REAL NOT NULL
        );
        CREATE VIEW v_trait_pair_deltas AS
        SELECT
            a.main_fight_id,
            a.as_of_date,
            a.fighter_id,
            a.fighter_name,
            a.opponent_id,
            a.opponent_name,
            a.cardio_score - b.cardio_score AS cardio_score_diff,
            a.defensive_exposure_score - b.defensive_exposure_score AS defensive_exposure_score_diff,
            a.striking_efficiency_score - b.striking_efficiency_score AS striking_efficiency_score_diff,
            a.trait_confidence,
            b.trait_confidence AS opponent_trait_confidence
        FROM fighter_trait_snapshots a
        JOIN fighter_trait_snapshots b
          ON b.main_fight_id = a.main_fight_id
         AND b.fighter_id = a.opponent_id;
        """
    )
    conn.executemany(
        """
        INSERT INTO fighter_trait_snapshots (
            main_fight_id,
            as_of_date,
            fighter_id,
            fighter_name,
            opponent_id,
            opponent_name,
            cardio_score,
            defensive_exposure_score,
            striking_efficiency_score,
            trait_confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (30, "2026-01-31", 1, "Pick Fighter", 2, "Opp Fighter", 50, 50, 48, 0.85),
            (30, "2026-01-31", 2, "Opp Fighter", 1, "Pick Fighter", 50, 38, 48, 0.90),
        ],
    )
    conn.commit()
    conn.close()


def _write_pool(path: Path):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE backtest_fight_pool (
            id INTEGER PRIMARY KEY,
            pick_correct INTEGER,
            bet INTEGER,
            pick_prob REAL,
            pick_elo_diff REAL,
            pick_odds INTEGER,
            actual_pnl REAL
        );
        CREATE TABLE evidence_items (
            id INTEGER PRIMARY KEY,
            fight_pool_id INTEGER NOT NULL,
            evidence_type TEXT NOT NULL,
            data_json TEXT
        );
        """
    )
    conn.executemany(
        """
        INSERT INTO backtest_fight_pool (id, pick_correct, bet, pick_prob, pick_elo_diff, pick_odds, actual_pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (1, 1, 0, 0.58, 110, -120, 0.83),
            (2, 0, 0, 0.59, 70, -110, -1.00),
            (3, 1, 0, 0.57, 120, -115, 0.95),
            (4, 1, 0, 0.56, 140, -130, 1.20),
            (5, 0, 0, 0.61, 115, -125, -1.00),
        ],
    )
    conn.executemany(
        """
        INSERT INTO evidence_items (fight_pool_id, evidence_type, data_json)
        VALUES (?, 'trait_delta', ?)
        """,
        [
            (1, '{"trait_confidence":0.90,"opponent_trait_confidence":0.85,"deltas":{"cardio_score_diff":0,"striking_efficiency_score_diff":0,"defensive_exposure_score_diff":0}}'),
            (2, '{"trait_confidence":0.90,"opponent_trait_confidence":0.85,"deltas":{"cardio_score_diff":0,"striking_efficiency_score_diff":0,"defensive_exposure_score_diff":0}}'),
            (3, '{"trait_confidence":0.92,"opponent_trait_confidence":0.82,"deltas":{"cardio_score_diff":0,"striking_efficiency_score_diff":12,"defensive_exposure_score_diff":0}}'),
            (4, '{"trait_confidence":0.93,"opponent_trait_confidence":0.88,"deltas":{"cardio_score_diff":12,"striking_efficiency_score_diff":0,"defensive_exposure_score_diff":0}}'),
            (5, '{"trait_confidence":0.91,"opponent_trait_confidence":0.84,"deltas":{"cardio_score_diff":0,"striking_efficiency_score_diff":11,"defensive_exposure_score_diff":0}}'),
        ],
    )
    conn.commit()
    conn.close()


def _write_favorite_caution_pool(path: Path):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE backtest_fight_pool (
            id INTEGER PRIMARY KEY,
            pick_correct INTEGER,
            bet INTEGER,
            pick_prob REAL,
            pick_elo_diff REAL,
            pick_odds INTEGER,
            actual_pnl REAL
        );
        CREATE TABLE evidence_items (
            id INTEGER PRIMARY KEY,
            fight_pool_id INTEGER NOT NULL,
            evidence_type TEXT NOT NULL,
            data_json TEXT
        );
        """
    )
    conn.executemany(
        """
        INSERT INTO backtest_fight_pool (id, pick_correct, bet, pick_prob, pick_elo_diff, pick_odds, actual_pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (1, 1, 0, 0.70, -80, -180, 0.56),
            (2, 0, 0, 0.68, -90, -200, -1.00),
        ],
    )
    conn.executemany(
        """
        INSERT INTO evidence_items (fight_pool_id, evidence_type, data_json)
        VALUES (?, 'trait_delta', ?)
        """,
        [
            (1, '{"trait_confidence":0.90,"opponent_trait_confidence":0.85,"deltas":{"cardio_score_diff":0,"striking_efficiency_score_diff":0}}'),
            (2, '{"trait_confidence":0.90,"opponent_trait_confidence":0.85,"deltas":{"cardio_score_diff":0,"striking_efficiency_score_diff":0}}'),
        ],
    )
    conn.commit()
    conn.close()


def test_golden_elo_reopen_returns_tier_3_with_trait_and_cardio_support(tmp_path):
    sidecar = tmp_path / "sergey_sidecar.sqlite"
    pool = tmp_path / "context_pool.sqlite"
    main_db = tmp_path / "ufc_database.db"
    traits = tmp_path / "trait_snapshots.sqlite"
    _write_sidecar(sidecar)
    _write_pool(pool)
    _write_main_db(main_db)
    _write_traits(traits)

    result = svc.evaluate_golden_elo_reopen(
        fighter1_name="Pick Fighter",
        fighter2_name="Opp Fighter",
        pick_slot="fighter1",
        pick_model_prob=0.58,
        pick_odds=-120,
        as_of_date="2026-02-01",
        sidecar_path=sidecar,
        context_pool_path=pool,
        main_db_path=main_db,
        trait_snapshot_path=traits,
    )

    assert result["reopen"] is True
    assert result["review_tier"] == 3
    assert result["pick_elo_diff"] == 100.0
    assert result["trait_support"] is True
    assert result["cardio_support"] is True
    assert result["review_label"] == "Golden ELO Tier 3 · Historical 1-0 · +120.0% ROI"


def test_golden_elo_reopen_uses_tier_1_when_traits_missing(tmp_path):
    sidecar = tmp_path / "sergey_sidecar.sqlite"
    pool = tmp_path / "context_pool.sqlite"
    _write_sidecar(sidecar)
    _write_pool(pool)

    result = svc.evaluate_golden_elo_reopen(
        fighter1_name="Pick Fighter",
        fighter2_name="Opp Fighter",
        pick_slot="fighter1",
        pick_model_prob=0.58,
        pick_odds=-120,
        sidecar_path=sidecar,
        context_pool_path=pool,
    )

    assert result["reopen"] is True
    assert result["review_tier"] == 1
    assert result["trait_support"] is False
    assert result["cardio_support"] is False
    assert result["review_label"] == "Golden ELO Tier 1 · Historical 3-2 · +19.6% ROI"


def test_low_positive_elo_with_cardio_support_sets_tier_1a(tmp_path):
    sidecar = tmp_path / "sergey_sidecar.sqlite"
    pool = tmp_path / "context_pool.sqlite"
    main_db = tmp_path / "ufc_database.db"
    traits = tmp_path / "trait_snapshots.sqlite"
    _write_sidecar(sidecar, pick_elo=1460, opp_elo=1400)
    _write_pool(pool)
    _write_main_db(main_db)
    _write_traits(traits)

    result = svc.evaluate_golden_elo_reopen(
        fighter1_name="Pick Fighter",
        fighter2_name="Opp Fighter",
        pick_slot="fighter1",
        pick_model_prob=0.58,
        pick_odds=-120,
        as_of_date="2026-02-01",
        sidecar_path=sidecar,
        context_pool_path=pool,
        main_db_path=main_db,
        trait_snapshot_path=traits,
    )

    assert result["reopen"] is True
    assert result["pick_elo_diff"] == 60.0
    assert result["review_bucket"] == "golden_elo_tier_1a"
    assert result["review_tier"] == "1A"
    assert result["review_label"] == "Golden ELO Tier 1A"


def test_golden_elo_reopen_falls_back_cleanly_when_data_missing(tmp_path):
    missing_sidecar = tmp_path / "missing_sidecar.sqlite"
    missing_pool = tmp_path / "missing_pool.sqlite"

    result = svc.evaluate_golden_elo_reopen(
        fighter1_name="Pick Fighter",
        fighter2_name="Opp Fighter",
        pick_slot="fighter1",
        pick_model_prob=0.58,
        pick_odds=-120,
        sidecar_path=missing_sidecar,
        context_pool_path=missing_pool,
    )

    assert result["reopen"] is False
    assert result["pick_elo_diff"] is None


def test_golden_elo_returns_pick_elo_diff_even_when_outside_reopen_band(tmp_path):
    sidecar = tmp_path / "sergey_sidecar.sqlite"
    _write_sidecar(sidecar)

    result = svc.evaluate_golden_elo_reopen(
        fighter1_name="Pick Fighter",
        fighter2_name="Opp Fighter",
        pick_slot="fighter2",
        pick_model_prob=0.716,
        pick_odds=-250,
        sidecar_path=sidecar,
    )

    assert result["reopen"] is False
    assert result["pick_elo_diff"] == -100.0
    assert result["review_bucket"] == "favorite_negative_elo_midprice_no_offset"
    assert result["review_tier"] == "F-"


def test_negative_elo_with_trait_support_sets_trait_offset_review_bucket(tmp_path):
    sidecar = tmp_path / "sergey_sidecar.sqlite"
    pool = tmp_path / "context_pool.sqlite"
    main_db = tmp_path / "ufc_database.db"
    traits = tmp_path / "trait_snapshots.sqlite"
    _write_sidecar(sidecar)
    _write_pool(pool)
    _write_main_db(main_db)
    _write_negative_traits(traits)

    result = svc.evaluate_golden_elo_reopen(
        fighter1_name="Pick Fighter",
        fighter2_name="Opp Fighter",
        pick_slot="fighter2",
        pick_model_prob=0.716,
        pick_odds=-250,
        as_of_date="2026-02-01",
        sidecar_path=sidecar,
        context_pool_path=pool,
        main_db_path=main_db,
        trait_snapshot_path=traits,
    )

    assert result["reopen"] is False
    assert result["pick_elo_diff"] == -100.0
    assert result["review_bucket"] == "trait_offset_elo_against"
    assert result["review_tier"] == 3
    assert result["review_label"].startswith("Trait Offset Tier 3")


def test_low_negative_elo_with_offset_trait_support_sets_tier_negative_1a(tmp_path):
    sidecar = tmp_path / "sergey_sidecar.sqlite"
    pool = tmp_path / "context_pool.sqlite"
    main_db = tmp_path / "ufc_database.db"
    traits = tmp_path / "trait_snapshots.sqlite"
    _write_sidecar(sidecar, pick_elo=1500, opp_elo=1440)
    _write_pool(pool)
    _write_main_db(main_db)
    _write_negative_traits(traits)

    result = svc.evaluate_golden_elo_reopen(
        fighter1_name="Pick Fighter",
        fighter2_name="Opp Fighter",
        pick_slot="fighter2",
        pick_model_prob=0.716,
        pick_odds=-250,
        as_of_date="2026-02-01",
        sidecar_path=sidecar,
        context_pool_path=pool,
        main_db_path=main_db,
        trait_snapshot_path=traits,
    )

    assert result["reopen"] is False
    assert result["pick_elo_diff"] == -60.0
    assert result["review_bucket"] == "elo_against_tier_1a"
    assert result["review_tier"] == "-1A"
    assert result["review_label"] == "ELO Against Tier -1A"


def test_negative_elo_defensive_only_support_does_not_set_trait_offset(tmp_path):
    sidecar = tmp_path / "sergey_sidecar.sqlite"
    pool = tmp_path / "context_pool.sqlite"
    main_db = tmp_path / "ufc_database.db"
    traits = tmp_path / "trait_snapshots.sqlite"
    _write_sidecar(sidecar)
    _write_pool(pool)
    _write_main_db(main_db)
    _write_negative_defensive_only_traits(traits)

    result = svc.evaluate_golden_elo_reopen(
        fighter1_name="Pick Fighter",
        fighter2_name="Opp Fighter",
        pick_slot="fighter2",
        pick_model_prob=0.716,
        pick_odds=-250,
        as_of_date="2026-02-01",
        sidecar_path=sidecar,
        context_pool_path=pool,
        main_db_path=main_db,
        trait_snapshot_path=traits,
    )

    assert result["reopen"] is False
    assert result["pick_elo_diff"] == -100.0
    assert result["review_bucket"] == "favorite_negative_elo_midprice_no_offset"
    assert result["review_tier"] == "F-"
    assert result["trait_support"] is True
    assert result["offset_trait_support"] is False


def test_midpriced_negative_elo_favorite_without_offset_gets_fade_label(tmp_path):
    sidecar = tmp_path / "sergey_sidecar.sqlite"
    pool = tmp_path / "context_pool.sqlite"
    _write_sidecar(sidecar, pick_elo=1320, opp_elo=1400)
    _write_favorite_caution_pool(pool)

    result = svc.evaluate_golden_elo_reopen(
        fighter1_name="Pick Fighter",
        fighter2_name="Opp Fighter",
        pick_slot="fighter1",
        pick_model_prob=0.70,
        pick_odds=-180,
        sidecar_path=sidecar,
        context_pool_path=pool,
    )

    assert result["reopen"] is False
    assert result["pick_elo_diff"] == -80.0
    assert result["review_bucket"] == "favorite_negative_elo_midprice_no_offset"
    assert result["review_tier"] == "F-"
    assert result["review_label"] == "Favorite ELO Fade · Historical 1-1 · -22.0% ROI"
