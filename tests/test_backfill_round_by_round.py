"""End-to-end dry-run test for scripts/backfill_round_by_round.py.

Runs the real backfill entrypoint against an isolated temporary SQLite database
and a temp cache dir seeded with a cached multi-round fixture. The production
database is never touched.
"""
import shutil
import sys
from pathlib import Path

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.schema import Event, Fighter, Fight, FightStats, create_all_tables  # noqa: E402
from scripts.backfill_round_by_round import main  # noqa: E402

FIGHT_HASH = "8bee5192baec71d8"
CACHED_FIXTURE = ROOT / "data" / "raw" / "events" / f"fight_{FIGHT_HASH}.html"


def _write_temp_config(tmp_path: Path, db_path: Path, cache_root: Path) -> Path:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    config["database"]["type"] = "sqlite"
    config["database"]["sqlite_path"] = str(db_path)
    config["scraping"]["cache_dir"] = str(cache_root)
    config["scraping"]["cache_enabled"] = True
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    return cfg_path


def _seed_db(db_path: Path, populated: bool = False) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    create_all_tables(engine)
    session = sessionmaker(bind=engine)()

    f1 = Fighter(fighter_id="db-fighter-1", name="Andre Fili")
    f2 = Fighter(fighter_id="db-fighter-2", name="Jose Delgado")
    event = Event(event_id="evt-1", name="Test Card", date="July 12, 2025")
    session.add_all([f1, f2, event])
    session.flush()

    fight = Fight(
        fight_id=FIGHT_HASH,
        event_id=event.id,
        fighter_1_id=f1.id,
        fighter_2_id=f2.id,
        fight_detail_url=f"http://ufcstats.com/fight-details/{FIGHT_HASH}",
    )
    session.add(fight)
    session.flush()

    session.add(
        FightStats(
            fight_id=fight.id,
            fighter_1_totals={"kd": "0"},
            round_by_round=[{"round": 1, "sentinel": True}] if populated else None,
        )
    )
    session.commit()
    session.close()


def _prep(tmp_path: Path, populated: bool = False):
    db_path = tmp_path / "test.db"
    cache_root = tmp_path / "cache"
    events_dir = cache_root / "events"
    events_dir.mkdir(parents=True)
    shutil.copy2(CACHED_FIXTURE, events_dir / f"fight_{FIGHT_HASH}.html")
    _seed_db(db_path, populated=populated)
    cfg = _write_temp_config(tmp_path, db_path, cache_root)
    return db_path, cfg


def _read_rbr(db_path: Path):
    session = sessionmaker(bind=create_engine(f"sqlite:///{db_path}"))()
    row = session.query(FightStats).first()
    value = row.round_by_round
    session.close()
    return value


def test_dry_run_reports_candidate_without_mutating(tmp_path):
    db_path, cfg = _prep(tmp_path)

    summary = main(["--config", str(cfg), "--dry-run"])

    assert summary["would_update"] == 1
    assert summary["updated"] == 0
    assert summary["multi_round_fights"] == 1
    # DB row must remain empty after a dry-run.
    assert _read_rbr(db_path) is None
    # No backup should be produced on a dry-run (backups live next to the DB).
    assert not list((db_path.parent / "backups").glob("*.db"))


def test_apply_populates_and_backs_up(tmp_path):
    db_path, cfg = _prep(tmp_path)

    summary = main(["--config", str(cfg), "--apply"])

    assert summary["updated"] == 1
    rbr = _read_rbr(db_path)
    assert isinstance(rbr, list)
    assert [r["round"] for r in rbr] == [1, 2, 3]
    # A backup of the DB was created before the write.
    backups = list((db_path.parent / "backups").glob("ufc_database_before_round_by_round_*.db"))
    assert len(backups) == 1


def test_apply_skips_populated_row_without_force(tmp_path):
    db_path, cfg = _prep(tmp_path, populated=True)

    summary = main(["--config", str(cfg), "--apply"])

    assert summary["already_populated_skipped"] == 1
    assert summary["updated"] == 0
    # Existing sentinel payload is preserved.
    assert _read_rbr(db_path) == [{"round": 1, "sentinel": True}]


def test_force_overwrites_populated_row(tmp_path):
    db_path, cfg = _prep(tmp_path, populated=True)

    summary = main(["--config", str(cfg), "--apply", "--force"])

    assert summary["updated"] == 1
    rbr = _read_rbr(db_path)
    assert [r["round"] for r in rbr] == [1, 2, 3]
