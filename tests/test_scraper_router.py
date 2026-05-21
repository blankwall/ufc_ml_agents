import sys
from pathlib import Path

from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.main import app  # noqa: E402
from fastapi_app.routers import scraper as scraper_router  # noqa: E402


def test_recover_fighter_route_propagates_validation_error(monkeypatch):
    monkeypatch.setattr(
        scraper_router,
        "recover_fighter_from_url",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("Sherdog URL must be a fighter profile link")),
    )

    client = TestClient(app)
    response = client.post(
        "/api/recover-fighter",
        json={"sherdog_url": "https://www.sherdog.com/not-a-fighter"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Sherdog URL must be a fighter profile link"}
