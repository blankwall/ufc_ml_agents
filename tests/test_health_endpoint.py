import sys
from pathlib import Path

from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app import main as main_module
from fastapi_app.main import app


def test_health_endpoint_reports_runtime_and_background_jobs(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "get_runtime_health",
        lambda: {
            "started_at": "2026-05-19T12:00:00Z",
            "now": "2026-05-19T12:10:00Z",
            "uptime_seconds": 600,
            "jobs": {"the_odds_api_sync": {"task_active": True}},
        },
    )
    monkeypatch.setattr(
        main_module,
        "odds_health_status",
        lambda: {"enabled": True, "task_active": True, "runs_since_launch": 2},
    )
    monkeypatch.setattr(
        main_module,
        "sherdog_recovery_health_status",
        lambda: {"enabled": True, "task_active": False, "runs_since_launch": 2},
    )
    monkeypatch.setattr(
        main_module,
        "ufcstats_health_status",
        lambda: {"enabled": False, "task_active": False, "runs_since_launch": 0},
    )

    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()

    # Overall status is now computed from job health + data freshness.
    assert body["status"] in {"ok", "degraded", "error"}
    assert isinstance(body["issues"], list)
    assert isinstance(body["data_freshness"], dict)
    assert "age_days" in body["data_freshness"]

    assert body["app"] == {
        "started_at": "2026-05-19T12:00:00Z",
        "now": "2026-05-19T12:10:00Z",
        "uptime_seconds": 600,
        "jobs": {"the_odds_api_sync": {"task_active": True}},
    }

    jobs = body["background_jobs"]
    assert set(jobs) == {"the_odds_api_sync", "sherdog_recovery", "ufcstats_completed_sync"}
    # Original per-job fields are preserved, plus an injected health verdict.
    assert jobs["the_odds_api_sync"]["runs_since_launch"] == 2
    assert jobs["ufcstats_completed_sync"]["enabled"] is False
    for job in jobs.values():
        assert job["health"]["level"] in {"ok", "degraded", "error"}


def test_health_page_renders_styled_dashboard(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "_health_payload",
        lambda: {
            "status": "ok",
            "app": {
                "started_at": "2026-05-19T12:00:00Z",
                "now": "2026-05-19T12:10:00Z",
                "uptime_seconds": 600,
            },
            "background_jobs": {
                "the_odds_api_sync": {
                    "enabled": True,
                    "task_active": True,
                    "checks_since_launch": 4,
                    "runs_since_launch": 2,
                    "successes_since_launch": 2,
                    "failures_since_launch": 0,
                    "last_check_at": "2026-05-19T12:09:00Z",
                    "last_run_started_at": "2026-05-19T12:05:00Z",
                    "last_run_finished_at": "2026-05-19T12:06:00Z",
                    "last_success_at": "2026-05-19T12:06:00Z",
                    "last_error_at": None,
                    "last_trigger": "scheduler",
                    "task_stop_reason": None,
                    "state_file": "data/future_fight_odds/the_odds_api_sync.json",
                    "last_summary": {"new_rows_added": 3},
                },
                "ufcstats_completed_sync": {
                    "enabled": True,
                    "task_active": False,
                    "checks_since_launch": 1,
                    "runs_since_launch": 1,
                    "successes_since_launch": 1,
                    "failures_since_launch": 0,
                    "last_check_at": "2026-05-19T12:09:00Z",
                    "last_run_started_at": "2026-05-19T12:05:00Z",
                    "last_run_finished_at": "2026-05-19T12:06:00Z",
                    "last_success_at": "2026-05-19T12:06:00Z",
                    "last_error_at": None,
                    "last_trigger": "scheduler",
                    "task_stop_reason": None,
                    "state_file": "data/ufcstats_sync_state.json",
                    "persisted_last_result": {"validation_failed": 1},
                    "recent_failed_events": [
                        {
                            "event_id": "evt-123",
                            "event_name": "UFC Test Card",
                            "status": "validation_failed_restored",
                            "reason": "validation failed; restored DB backup",
                            "retry_command": ".venv/bin/python scrapers/event_populator.py --event-id evt-123 --include-fight-stats --force-refresh-fighters --validate --validate-details",
                        }
                    ],
                },
            },
        },
    )

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Background job health" in response.text
    assert "/api/health" in response.text
    assert "the odds api sync" in response.text
    assert "validation failed; restored DB backup" in response.text
    assert "scrapers/event_populator.py" in response.text
