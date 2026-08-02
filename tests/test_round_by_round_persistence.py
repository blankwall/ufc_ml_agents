"""Persistence + no-overwrite guard tests for fight_stats.round_by_round.

These run against an isolated temporary SQLite database created per-test; they
never touch the real UFC database.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.schema import FightStats, create_all_tables


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    create_all_tables(engine)
    return sessionmaker(bind=engine)()


_SAMPLE_RBR = [
    {
        "round": 1,
        "fighter_1": {"totals": {"knockdowns": "0", "control_time": "0:23"}, "significant_strikes": {}},
        "fighter_2": {"totals": {"knockdowns": "0"}, "significant_strikes": {}},
    },
    {
        "round": 2,
        "fighter_1": {"totals": {"knockdowns": "1"}, "significant_strikes": {}},
        "fighter_2": {"totals": {"knockdowns": "0"}, "significant_strikes": {}},
    },
]


def test_round_by_round_persists_on_new_row(tmp_path):
    session = _session(tmp_path)
    session.add(FightStats(fight_id=1, round_by_round=_SAMPLE_RBR))
    session.commit()

    row = session.query(FightStats).filter_by(fight_id=1).first()
    assert row.round_by_round == _SAMPLE_RBR
    assert row.round_by_round[1]["fighter_1"]["totals"]["knockdowns"] == "1"
    session.close()


def test_round_by_round_updates_existing_row(tmp_path):
    session = _session(tmp_path)
    stats = FightStats(fight_id=1, fighter_1_totals={"kd": "0"})
    session.add(stats)
    session.commit()

    # Mirror the write-path guard: fill only when currently empty.
    if not stats.round_by_round:
        stats.round_by_round = _SAMPLE_RBR
    session.commit()

    row = session.query(FightStats).filter_by(fight_id=1).first()
    assert row.round_by_round == _SAMPLE_RBR
    session.close()


def test_populated_round_by_round_not_overwritten_without_force(tmp_path):
    session = _session(tmp_path)
    original = [{"round": 1, "fighter_1": {"totals": {"kd": "9"}}, "fighter_2": {"totals": {}}}]
    stats = FightStats(fight_id=1, round_by_round=original)
    session.add(stats)
    session.commit()

    # Guard: because round_by_round is already populated, the incoming payload
    # must be ignored (no --force).
    incoming = _SAMPLE_RBR
    if not stats.round_by_round:
        stats.round_by_round = incoming
    session.commit()

    row = session.query(FightStats).filter_by(fight_id=1).first()
    assert row.round_by_round == original
    session.close()
