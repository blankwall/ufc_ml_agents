"""Unit tests for the ingest service (fighter preview/save) and the Sherdog
display-name splitter it relies on.

Uses an isolated temp SQLite DB and a mocked Sherdog scraper so no network or
real-DB writes happen.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from database.db_manager import DatabaseManager  # noqa: E402
from database.schema import Fighter  # noqa: E402
from fastapi_app.services import fighter_alias_service as alias_service  # noqa: E402
from fastapi_app.services import ingest_service  # noqa: E402
from fastapi_app.services import sherdog_recovery_service as recovery_service  # noqa: E402
from scrapers import sherdog_scraper  # noqa: E402


# ── _split_name_nickname ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ('Richard "The Hammer" Harris', ("Richard Harris", "The Hammer")),
        ('Kangjie "The Hypnotist / One Punch Man" Zhu',
         ("Kangjie Zhu", "The Hypnotist / One Punch Man")),
        ("Yi Sak Lee", ("Yi Sak Lee", None)),
        ("", ("", None)),
        (None, ("", None)),
        ('\u201cCurly\u201d Quotes Guy', ("Quotes Guy", "Curly")),
    ],
)
def test_split_name_nickname(raw, expected):
    assert recovery_service._split_name_nickname(raw) == expected


# ── fixtures ──────────────────────────────────────────────────────────────────

def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "database": {"type": "sqlite", "sqlite_path": "data/ufc_database.db"},
                "scraping": {
                    "user_agent": "pytest",
                    "timeout": 5,
                    "cache_enabled": True,
                    "cache_dir": "data/raw",
                },
            }
        )
    )
    return config_path


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolated config, DB, recovery state, and alias store."""
    config_path = _write_config(tmp_path)
    state_path = tmp_path / "data" / "future_fight_odds" / "sherdog_recovery.json"
    alias_file = tmp_path / "fighter_aliases.json"
    alias_file.write_text("{}")

    monkeypatch.setattr(recovery_service, "CONFIG_PATH", config_path)
    monkeypatch.setattr(recovery_service, "STATE_PATH", state_path)
    monkeypatch.setattr(ingest_service, "CONFIG_PATH", config_path)
    monkeypatch.setattr(sherdog_scraper, "BASE_URL", "https://www.sherdog.com")
    monkeypatch.setattr(alias_service, "ALIAS_FILE", alias_file)
    alias_service.load_aliases()

    yield {"config_path": config_path, "state_path": state_path, "alias_file": alias_file}

    monkeypatch.undo()
    alias_service.load_aliases()


def _mock_scrape(monkeypatch, name='Richard "The Hammer" Harris', fid="429363"):
    monkeypatch.setattr(
        sherdog_scraper.SherdogScraper,
        "scrape_fighter",
        lambda self, _url, fighter_id=None, bust_cache=False: {
            "fighter_id": fighter_id or fid,
            "source": "sherdog",
            "url": f"https://www.sherdog.com/fighter/Richard-Harris-{fid}",
            "scraped_at": "2026-05-19T00:00:00Z",
            "name": name,
            "height": "6'0\"",
            "weight": "155 lbs",
            "date_of_birth": "Jan 11, 1996",
            "age": 30,
            "method_breakdown": {"wins": {"total": 10}, "losses": {"total": 2}},
            "fight_history": [],
        },
    )


URL = "https://www.sherdog.com/fighter/Richard-Harris-429363"


# ── preview ───────────────────────────────────────────────────────────────────

def test_preview_writes_nothing_and_cleans_name(env, monkeypatch):
    _mock_scrape(monkeypatch)
    result = ingest_service.preview_fighter(URL, requested_name="RJ Harris")

    assert result["status"] == "ok"
    assert result["scraped_name"] == "Richard Harris"  # nickname stripped
    assert result["fighter_id"] == "sherdog:429363"
    assert result["already_in_db"] is False
    assert result["requested_name"] == "RJ Harris"
    # suggested alias because requested != scraped
    assert result["suggested_alias"] == {"alias": "RJ Harris", "canonical": "Richard Harris"}
    assert result["alias_needed"] is True

    # nothing persisted
    db = DatabaseManager(config_path=str(env["config_path"]))
    session = db.get_session()
    try:
        assert session.query(Fighter).filter_by(fighter_id="sherdog:429363").first() is None
    finally:
        session.close()


def test_preview_no_alias_when_names_match(env, monkeypatch):
    _mock_scrape(monkeypatch, name="Yi Sak Lee", fid="390123")
    result = ingest_service.preview_fighter(
        "https://www.sherdog.com/fighter/Yi-Sak-Lee-390123",
        requested_name="Yi Sak Lee",
    )
    assert result["scraped_name"] == "Yi Sak Lee"
    assert result["suggested_alias"] is None
    assert result["alias_needed"] is False


def test_preview_detects_existing_fighter(env, monkeypatch):
    _mock_scrape(monkeypatch)
    db = DatabaseManager(config_path=str(env["config_path"]))
    session = db.get_session()
    try:
        db.add_fighter(session, {"fighter_id": "sherdog:429363", "name": "Richard Harris"})
        session.commit()
    finally:
        session.close()

    result = ingest_service.preview_fighter(URL, requested_name="RJ Harris")
    assert result["already_in_db"] is True
    assert result["db_name"] == "Richard Harris"


# ── save ──────────────────────────────────────────────────────────────────────

def test_save_commits_fighter_and_alias(env, monkeypatch):
    _mock_scrape(monkeypatch)
    result = ingest_service.save_fighter(
        sherdog_url=URL,
        requested_name="RJ Harris",
        write_alias=True,
        alias_from="RJ Harris",
        alias_to="Richard Harris",
    )

    assert result["status"] == "saved"
    assert result["fighter"]["db_name"] == "Richard Harris"
    assert result["fighter"]["nickname"] == "The Hammer"
    assert result["alias"] == {"alias": "RJ Harris", "canonical": "Richard Harris"}

    # fighter in DB with split name + nickname
    db = DatabaseManager(config_path=str(env["config_path"]))
    session = db.get_session()
    try:
        f = session.query(Fighter).filter_by(fighter_id="sherdog:429363").first()
        assert f is not None
        assert f.name == "Richard Harris"
        assert f.nickname == "The Hammer"
        assert f.wins == 10 and f.losses == 2
    finally:
        session.close()

    # alias persisted to disk
    assert json.loads(env["alias_file"].read_text())["RJ Harris"] == "Richard Harris"


def test_save_without_alias_does_not_write_alias(env, monkeypatch):
    _mock_scrape(monkeypatch)
    result = ingest_service.save_fighter(
        sherdog_url=URL, requested_name="RJ Harris", write_alias=False,
    )
    assert result["alias"] is None
    assert json.loads(env["alias_file"].read_text()) == {}


def test_save_alias_defaults_from_requested_and_scraped(env, monkeypatch):
    _mock_scrape(monkeypatch)
    # no explicit alias_from/alias_to -> derive from requested + scraped clean name
    result = ingest_service.save_fighter(
        sherdog_url=URL, requested_name="RJ Harris", write_alias=True,
    )
    assert result["alias"] == {"alias": "RJ Harris", "canonical": "Richard Harris"}


def test_save_skips_selfloop_alias(env, monkeypatch):
    _mock_scrape(monkeypatch, name="Yi Sak Lee", fid="390123")
    result = ingest_service.save_fighter(
        sherdog_url="https://www.sherdog.com/fighter/Yi-Sak-Lee-390123",
        requested_name="Yi Sak Lee",
        write_alias=True,
    )
    # alias_from == alias_to -> no alias written
    assert result["alias"] is None
