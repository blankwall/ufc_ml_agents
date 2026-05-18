import asyncio
import sys
from pathlib import Path

from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.main import app
from fastapi_app.routers import database as database_router


def test_fighter_profile_returns_display_friendly_summary():
    profile = asyncio.run(database_router.get_fighter_profile("Carlos Prates"))

    assert profile["name"] == "Carlos Prates"
    assert profile["ufc_bout_count"] == len(profile["fight_history"])
    assert len(profile["recent_form"]) <= 5
    assert profile["striking_accuracy_pct"] is None or profile["striking_accuracy_pct"] > 1
    assert profile["takedown_accuracy_pct"] is None or profile["takedown_accuracy_pct"] >= 0
    assert set(profile["ufc_record"].keys()) == {"wins", "losses", "draws", "no_contests"}


def test_backtest_and_ingest_routes_redirect_to_events():
    client = TestClient(app)

    backtest = client.get("/backtest", follow_redirects=False)
    ingest = client.get("/ingest", follow_redirects=False)

    assert backtest.status_code in {302, 307}
    assert ingest.status_code in {302, 307}
    assert backtest.headers["location"] == "/events"
    assert ingest.headers["location"] == "/events"
