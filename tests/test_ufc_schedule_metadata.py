import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.services import ufc_schedule_service as schedule


def test_find_upcoming_bout_reads_metadata_rich_cache(monkeypatch):
    monkeypatch.setattr(
        schedule,
        "load_allowlist",
        lambda: {
            "events": [{
                "date": "2026-08-08",
                "name": "UFC Fight Night",
                "url": "http://ufcstats.test/event",
                "bouts": [{
                    "fighter1": "Diego Ferreira",
                    "fighter2": "Billy Quarantillo",
                    "weight_class": "Lightweight",
                    "fight_number": 3,
                }],
            }],
        },
    )

    result = schedule.find_upcoming_bout(
        "Billy Quarantillo", "Diego Ferreira", "2026-08-08"
    )

    assert result["weight_class"] == "Lightweight"
    assert result["fight_number"] == 3


def test_legacy_allowlist_bout_still_matches(monkeypatch):
    monkeypatch.setattr(
        schedule,
        "load_allowlist",
        lambda: {
            "events": [{
                "date": "2026-08-08",
                "name": "UFC Fight Night",
                "bouts": [["Diego Ferreira", "Billy Quarantillo"]],
            }],
        },
    )

    result = schedule.find_upcoming_bout(
        "Diego Ferreira", "Billy Quarantillo", "2026-08-08"
    )

    assert result is not None
    assert result["weight_class"] is None
    assert result["fight_number"] is None


def test_find_upcoming_bout_applies_shared_fighter_aliases(monkeypatch):
    monkeypatch.setitem(
        schedule.FIGHTER_ALIASES,
        "Carlos Diego Ferreira",
        "Diego Ferreira",
    )
    monkeypatch.setitem(
        schedule.FIGHTER_ALIASES,
        "Yadier DelValle",
        "Yadier del Valle",
    )
    monkeypatch.setattr(
        schedule,
        "load_allowlist",
        lambda: {
            "events": [{
                "date": "2026-08-08",
                "name": "UFC Fight Night",
                "bouts": [
                    {
                        "fighter1": "Diego Ferreira",
                        "fighter2": "Billy Quarantillo",
                        "weight_class": "Lightweight",
                        "fight_number": 2,
                    },
                    {
                        "fighter1": "Darren Elkins",
                        "fighter2": "Yadier del Valle",
                        "weight_class": "Featherweight",
                        "fight_number": 3,
                    },
                ],
            }],
        },
    )

    diego = schedule.find_upcoming_bout(
        "Carlos Diego Ferreira", "Billy Quarantillo", "2026-08-09"
    )
    del_valle = schedule.find_upcoming_bout(
        "Darren Elkins", "Yadier DelValle", "2026-08-09"
    )

    assert diego["weight_class"] == "Lightweight"
    assert diego["fight_number"] == 2
    assert del_valle["weight_class"] == "Featherweight"
    assert del_valle["fight_number"] == 3
