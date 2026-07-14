"""API tests for the ingest router: /api/ingest/preview, /api/ingest/save, and
the /api/aliases CRUD endpoints. Service layer is mocked so these assert routing,
validation, and status-code behavior only.
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.main import app  # noqa: E402
# NOTE: main.py imports the router as ``routers.ingest`` (fastapi_app is on
# sys.path), so the LIVE router object is ``routers.ingest`` -- patch that one,
# not ``fastapi_app.routers.ingest`` (a separate module object; split-brain).
from routers import ingest as ingest_router  # noqa: E402
from fastapi_app.services import fighter_alias_service as alias_service  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def isolated_aliases(monkeypatch, tmp_path):
    alias_file = tmp_path / "fighter_aliases.json"
    alias_file.write_text(json.dumps({"Bobby Green": "King Green"}))
    monkeypatch.setattr(alias_service, "ALIAS_FILE", alias_file)
    alias_service.load_aliases()
    yield alias_file
    monkeypatch.undo()
    alias_service.load_aliases()


# ── /api/ingest/preview ───────────────────────────────────────────────────────

def test_preview_ok(client, monkeypatch):
    monkeypatch.setattr(
        ingest_router, "preview_fighter",
        lambda url, name=None: {"status": "ok", "scraped_name": "Richard Harris"},
    )
    r = client.post("/api/ingest/preview",
                    json={"sherdog_url": "https://www.sherdog.com/fighter/X-1", "requested_name": "RJ Harris"})
    assert r.status_code == 200
    assert r.json()["scraped_name"] == "Richard Harris"


def test_preview_validation_error_maps_422(client, monkeypatch):
    def boom(*a, **k):
        raise ValueError("Sherdog URL must point to sherdog.com")
    monkeypatch.setattr(ingest_router, "preview_fighter", boom)
    r = client.post("/api/ingest/preview", json={"sherdog_url": "http://example.com"})
    assert r.status_code == 422
    assert r.json() == {"detail": "Sherdog URL must point to sherdog.com"}


def test_preview_unexpected_error_maps_500(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("scrape exploded")
    monkeypatch.setattr(ingest_router, "preview_fighter", boom)
    r = client.post("/api/ingest/preview", json={"sherdog_url": "https://www.sherdog.com/fighter/X-1"})
    assert r.status_code == 500
    assert "scrape exploded" in r.json()["detail"]


def test_preview_requires_url(client):
    r = client.post("/api/ingest/preview", json={})
    assert r.status_code == 422  # pydantic validation


# ── /api/ingest/save ──────────────────────────────────────────────────────────

def test_save_ok_passes_through_args(client, monkeypatch):
    captured = {}

    def fake_save(**kwargs):
        captured.update(kwargs)
        return {"status": "saved", "fighter": {"db_name": "Richard Harris"}, "alias": None}

    monkeypatch.setattr(ingest_router, "save_fighter", fake_save)
    r = client.post("/api/ingest/save", json={
        "sherdog_url": "https://www.sherdog.com/fighter/X-1",
        "requested_name": "RJ Harris",
        "write_alias": True,
        "alias_from": "RJ Harris",
        "alias_to": "Richard Harris",
    })
    assert r.status_code == 200
    assert r.json()["status"] == "saved"
    assert captured["write_alias"] is True
    assert captured["alias_from"] == "RJ Harris"
    assert captured["alias_to"] == "Richard Harris"


def test_save_validation_error_maps_422(client, monkeypatch):
    def boom(**k):
        raise ValueError("bad")
    monkeypatch.setattr(ingest_router, "save_fighter", boom)
    r = client.post("/api/ingest/save", json={"sherdog_url": "https://www.sherdog.com/fighter/X-1"})
    assert r.status_code == 422


# ── /api/aliases CRUD ─────────────────────────────────────────────────────────

def test_list_aliases(client, isolated_aliases):
    r = client.get("/api/aliases")
    assert r.status_code == 200
    assert r.json() == {"Bobby Green": "King Green"}


def test_upsert_alias_via_api(client, isolated_aliases):
    r = client.post("/api/aliases", json={"alias": "RJ Harris", "canonical": "Richard Harris"})
    assert r.status_code == 200
    assert r.json() == {"alias": "RJ Harris", "canonical": "Richard Harris"}
    # reflected in a subsequent GET and on disk
    assert client.get("/api/aliases").json()["RJ Harris"] == "Richard Harris"
    assert json.loads(isolated_aliases.read_text())["RJ Harris"] == "Richard Harris"


def test_upsert_alias_selfloop_422(client, isolated_aliases):
    r = client.post("/api/aliases", json={"alias": "Same", "canonical": "Same"})
    assert r.status_code == 422


def test_delete_alias(client, isolated_aliases):
    r = client.delete("/api/aliases/Bobby Green")
    assert r.status_code == 200
    assert r.json() == {"status": "deleted", "alias": "Bobby Green"}
    assert "Bobby Green" not in client.get("/api/aliases").json()


def test_delete_missing_alias_404(client, isolated_aliases):
    r = client.delete("/api/aliases/Nonexistent Person")
    assert r.status_code == 404


def test_ingest_page_renders(client):
    r = client.get("/ingest")
    assert r.status_code == 200
    body = r.text
    assert "Fighter Ingest" in body
    assert "ingest_fighter.js" in body
