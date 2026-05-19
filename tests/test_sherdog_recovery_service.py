import csv
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from database.db_manager import DatabaseManager
from database.schema import Fighter
from fastapi_app.services import sherdog_recovery_service as recovery_service
from scrapers import sherdog_scraper


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "database": {
                    "type": "sqlite",
                    "sqlite_path": "data/ufc_database.db",
                },
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


def test_recover_missing_fighters_from_odds_inserts_sherdog_fighter(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path)
    state_path = tmp_path / "data" / "future_fight_odds" / "sherdog_recovery.json"

    monkeypatch.setattr(recovery_service, "CONFIG_PATH", config_path)
    monkeypatch.setattr(recovery_service, "STATE_PATH", state_path)
    monkeypatch.setattr(sherdog_scraper, "BASE_URL", "https://www.sherdog.com")

    db = DatabaseManager(config_path=str(config_path))
    session = db.get_session()
    try:
        existing = db.add_fighter(
            session,
            {
                "fighter_id": "ufcstats-existing-1",
                "name": "Known Fighter",
                "wins": 5,
                "losses": 1,
                "draws": 0,
                "url": "http://example.com/known",
            },
        )
        session.commit()
        assert existing.name == "Known Fighter"
    finally:
        session.close()

    odds_df = pd.DataFrame(
        [
            {
                "event_name": "MMA Card · 2026-05-30",
                "event_date": "2026-05-30",
                "fighter1": "Yi Sak Lee",
                "fighter2": "Known Fighter",
                "source_type": "the_odds_api",
            }
        ]
    )

    monkeypatch.setattr(
        sherdog_scraper.SherdogScraper,
        "search_fighter",
        lambda self, _name: {
            "name": "Yi Sak Lee",
            "url": "https://www.sherdog.com/fighter/Yi-Sak-Lee-390123",
            "fighter_id": "390123",
        },
    )
    monkeypatch.setattr(
        sherdog_scraper.SherdogScraper,
        "scrape_fighter",
        lambda self, _url, fighter_id=None, bust_cache=False: {
            "fighter_id": fighter_id or "390123",
            "source": "sherdog",
            "url": "https://www.sherdog.com/fighter/Yi-Sak-Lee-390123",
            "scraped_at": "2026-05-19T00:00:00Z",
            "name": "Yi Sak Lee",
            "height": "6'0\"",
            "weight": "185 lbs",
            "date_of_birth": "Jan 11, 2000",
            "age": 26,
            "method_breakdown": {
                "wins": {"total": 8},
                "losses": {"total": 1},
            },
            "fight_history": [],
        },
    )

    result = recovery_service.recover_missing_fighters_from_odds(
        odds_df=odds_df,
        trigger="pytest",
    )

    assert result["queued"] == 1
    assert result["attempted"] == 1
    assert result["recovered"] == 1
    assert result["search_misses"] == 0
    assert result["errors"] == 0
    assert result["processed"][0]["status"] == "recovered"

    session = db.get_session()
    try:
        recovered = session.query(Fighter).filter_by(fighter_id="sherdog:390123").first()
        assert recovered is not None
        assert recovered.name == "Yi Sak Lee"
        assert recovered.wins == 8
        assert recovered.losses == 1
        assert recovered.age == 26
        assert recovered.date_of_birth == "Jan 11, 2000"
    finally:
        session.close()

    state = json.loads(state_path.read_text())
    fighter_state = state["fighters"]["yi sak lee"]
    assert fighter_state["status"] == "recovered"
    assert fighter_state["sherdog_fighter_id"] == "390123"
    assert fighter_state["db_fighter_id"] == "sherdog:390123"
    assert fighter_state["source_rows"][0]["event_name"] == "MMA Card · 2026-05-30"


def test_recover_missing_fighters_from_odds_records_search_miss(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path)
    state_path = tmp_path / "data" / "future_fight_odds" / "sherdog_recovery.json"

    monkeypatch.setattr(recovery_service, "CONFIG_PATH", config_path)
    monkeypatch.setattr(recovery_service, "STATE_PATH", state_path)

    db = DatabaseManager(config_path=str(config_path))
    session = db.get_session()
    try:
        db.add_fighter(
            session,
            {
                "fighter_id": "ufcstats-existing-1",
                "name": "Known Fighter",
                "wins": 5,
                "losses": 1,
                "draws": 0,
                "url": "http://example.com/known",
            },
        )
        session.commit()
    finally:
        session.close()

    monkeypatch.setattr(
        sherdog_scraper.SherdogScraper,
        "search_fighter",
        lambda self, _name: None,
    )

    odds_df = pd.DataFrame(
        [
            {
                "event_name": "MMA Card · 2026-05-30",
                "event_date": "2026-05-30",
                "fighter1": "Mystery Prospect",
                "fighter2": "Known Fighter",
                "source_type": "the_odds_api",
            }
        ]
    )

    result = recovery_service.recover_missing_fighters_from_odds(
        odds_df=odds_df,
        trigger="pytest",
    )

    assert result["queued"] == 1
    assert result["attempted"] == 1
    assert result["recovered"] == 0
    assert result["search_misses"] == 1
    state = json.loads(state_path.read_text())
    assert state["fighters"]["mystery prospect"]["status"] == "search_not_found"
