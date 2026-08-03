from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.main import app
from fastapi_app.services import decision_card_service as service
from routers import decision_cards as router


def _raw_result() -> dict:
    return {
        "binary": {
            "probabilities": {"finish": 0.627, "decision": 0.373},
            "selection": "finish",
            "confidence": 0.627,
            "tier": "strong",
            "eligible": True,
            "market": {
                "available": False,
                "selected_probability": None,
                "edge": None,
                "actionable": False,
            },
        },
        "method": {"decision": 0.36, "ko_tko": 0.47, "submission": 0.17},
        "history": {"fighter_a_prior": 17, "fighter_b_prior": 11},
    }


def test_run_card_persists_shaped_results(monkeypatch, tmp_path):
    cache_path = tmp_path / "decision_cards.json"
    skill_python = tmp_path / "python"
    batch_script = tmp_path / "batch.py"
    skill_python.write_text("")
    batch_script.write_text("")
    monkeypatch.setattr(service, "CACHE_PATH", cache_path)
    monkeypatch.setattr(service, "SKILL_PYTHON", skill_python)
    monkeypatch.setattr(service, "BATCH_SCRIPT", batch_script)
    monkeypatch.setattr(
        service,
        "_resolve_fights",
        lambda *_a, **_k: [{
            "request": {"fighter1": "Diego Ferreira", "fighter2": "Billy Quarantillo"},
            "metadata": {
                "fighter1": "Diego Ferreira",
                "fighter2": "Billy Quarantillo",
                "event_date": "2026-08-08",
                "weight_class": "Lightweight",
                "fight_number": 3,
            },
        }],
    )
    output = json.dumps({"type": "result", "index": 0, "result": _raw_result()})
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    service._run_card("card-key", {
        "event_name": "UFC Fight Night",
        "event_date": "2026-08-08",
        "fights": [{"fighter1": "Diego Ferreira", "fighter2": "Billy Quarantillo"}],
        "created_at": "2026-08-03T00:00:00+00:00",
    })

    card = service.get_card_analysis("card-key")
    assert card["status"] == "complete"
    assert card["fights"][0]["weight_class"] == "Lightweight"
    assert card["fights"][0]["fight_number"] == 3
    assert card["fights"][0]["result"]["tier"] == "strong"


def test_decision_card_endpoints_delegate(monkeypatch):
    queued = {"card_key": "abc", "status": "queued", "event_date": "2026-08-08"}
    monkeypatch.setattr(router, "start_card_analysis", lambda **_kwargs: queued)
    monkeypatch.setattr(router, "get_card_analysis", lambda key: {**queued, "card_key": key})
    monkeypatch.setattr(router, "get_card_analysis_by_date", lambda date: {**queued, "event_date": date})
    client = TestClient(app)

    started = client.post("/api/decision-cards/analyze", json={
        "event_name": "UFC Fight Night",
        "event_date": "2026-08-08",
        "fights": [{"fighter1": "Diego Ferreira", "fighter2": "Billy Quarantillo"}],
    })
    assert started.status_code == 200
    assert started.json()["status"] == "queued"
    assert client.get("/api/decision-cards/abc").status_code == 200
    assert client.get("/api/decision-cards?event_date=2026-08-08").status_code == 200


def test_decision_card_get_views_filter_signals_and_actionable(monkeypatch):
    card = {
        "card_key": "abc",
        "status": "complete",
        "event_date": "2026-08-08",
        "fights": [
            {"fighter1": "Strong", "result": {"eligible": True, "tier": "strong", "bet": False}},
            {"fighter1": "Bet", "result": {"eligible": True, "tier": "eligible", "bet": True}},
            {"fighter1": "Below", "result": {"eligible": False, "tier": "ineligible", "bet": False}},
            {"fighter1": "Error", "result": {"eligible": None, "tier": None, "bet": "error"}},
        ],
    }
    monkeypatch.setattr(router, "get_card_analysis_by_date", lambda _date: card)
    client = TestClient(app)

    signals = client.get(
        "/api/decision-cards?event_date=2026-08-08&view=signals"
    ).json()
    actionable = client.get(
        "/api/decision-cards?event_date=2026-08-08&view=actionable"
    ).json()

    assert signals["returned_fights"] == 2
    assert [fight["fighter1"] for fight in signals["fights"]] == ["Strong", "Bet"]
    assert actionable["returned_fights"] == 1
    assert actionable["fights"][0]["fighter1"] == "Bet"
    assert signals["summary"] == {
        "total": 4,
        "signals": 2,
        "strong": 1,
        "actionable": 1,
        "errors": 1,
    }


def test_orphaned_running_job_is_marked_interrupted(monkeypatch, tmp_path):
    cache_path = tmp_path / "decision_cards.json"
    cache_path.write_text(json.dumps({
        "version": service.CACHE_VERSION,
        "cards": {
            "orphan": {
                "card_key": "orphan",
                "event_date": "2026-08-08",
                "status": "running",
                "owner_pid": 99999999,
            }
        },
    }))
    monkeypatch.setattr(service, "CACHE_PATH", cache_path)

    card = service.get_card_analysis("orphan")

    assert card["status"] == "error"
    assert card["error_code"] == "interrupted"


def test_failed_refresh_restores_previous_complete_result(monkeypatch, tmp_path):
    cache_path = tmp_path / "decision_cards.json"
    skill_python = tmp_path / "python"
    batch_script = tmp_path / "batch.py"
    skill_python.write_text("")
    batch_script.write_text("")
    monkeypatch.setattr(service, "CACHE_PATH", cache_path)
    monkeypatch.setattr(service, "SKILL_PYTHON", skill_python)
    monkeypatch.setattr(service, "BATCH_SCRIPT", batch_script)
    monkeypatch.setattr(
        service,
        "_resolve_fights",
        lambda *_a, **_k: [{
            "request": {"fighter1": "A", "fighter2": "B"},
            "metadata": {
                "fighter1": "A",
                "fighter2": "B",
                "event_date": "2026-08-08",
                "weight_class": "Lightweight",
                "fight_number": 3,
            },
        }],
    )
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("refresh failed")),
    )
    previous = {
        "card_key": "card-key",
        "event_date": "2026-08-08",
        "status": "complete",
        "fights": [{"fighter1": "A", "fighter2": "B", "result": {"tier": "strong"}}],
        "completed_at": "2026-08-03T00:00:00+00:00",
    }

    service._run_card("card-key", {
        "event_name": "UFC",
        "event_date": "2026-08-08",
        "fights": [{"fighter1": "A", "fighter2": "B"}],
        "created_at": "2026-08-03T00:00:00+00:00",
        "previous_complete": previous,
    })

    card = service.get_card_analysis("card-key")
    assert card["status"] == "complete"
    assert card["fights"] == previous["fights"]
    assert card["refresh_error_message"] == "refresh failed"
