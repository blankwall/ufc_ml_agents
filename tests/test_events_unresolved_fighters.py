import sys
from pathlib import Path

from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.main import app
from routers import events as events_router


def test_unresolved_fighters_endpoint_returns_only_name_resolution_failures(monkeypatch):
    monkeypatch.setattr(
        events_router,
        "get_events_data",
        lambda: [
            {
                "event_name": "MMA Card · 2026-05-30",
                "event_date": "2026-05-30",
                "source_type": "the_odds_api",
                "fights": [
                    {
                        "fighter1": "Luis Felipe Dias",
                        "fighter2": "Yi Sak Lee",
                        "model_source": "not_found",
                        "error": "Fighter not found: Yi Sak Lee",
                    },
                    {
                        "fighter1": "Known Fighter",
                        "fighter2": "Another Fighter",
                        "model_source": "general",
                        "error": None,
                    },
                ],
            },
            {
                "event_name": "Other Event",
                "event_date": "2026-06-01",
                "source_type": "csv",
                "fights": [
                    {
                        "fighter1": "Broken A",
                        "fighter2": "Broken B",
                        "model_source": "error",
                        "error": "Some other scoring failure",
                    }
                ],
            },
        ],
    )

    client = TestClient(app)

    response = client.get("/api/events/unresolved-fighters")

    assert response.status_code == 200
    assert response.json() == [
        {
            "event_name": "MMA Card · 2026-05-30",
            "event_date": "2026-05-30",
            "source_type": "the_odds_api",
            "fighter1": "Luis Felipe Dias",
            "fighter2": "Yi Sak Lee",
            "matchup": "Luis Felipe Dias vs Yi Sak Lee",
            "error": "Fighter not found: Yi Sak Lee",
            "unresolved_fighters": ["Yi Sak Lee"],
        }
    ]
