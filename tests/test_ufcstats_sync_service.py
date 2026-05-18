import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.services import ufcstats_sync_service as sync_service  # noqa: E402


class _FakePopulator:
    def __init__(self, dry_run_summary, commit_summary=None, validation=None):
        self.dry_run_summary = dry_run_summary
        self.commit_summary = commit_summary or dry_run_summary
        self.validation = validation or {"fights_checked": 10, "missing_in_db": 0, "mismatches": 0}
        self.calls = []

    def populate_event_from_url(self, url, *, options):
        self.calls.append(("populate", url, options.commit))
        return self.commit_summary if options.commit else self.dry_run_summary

    def validate_event_against_db(self, url, *, use_fight_details=False):
        self.calls.append(("validate", url, use_fight_details))
        return self.validation


def test_sync_completed_ufcstats_events_records_synced_event(monkeypatch, tmp_path):
    state_path = tmp_path / "ufcstats_sync_state.json"
    monkeypatch.setattr(sync_service, "STATE_PATH", state_path)
    monkeypatch.setattr(
        sync_service,
        "_recent_completed_events",
        lambda now=None: [
            {
                "event_id": "evt-123",
                "name": "UFC Test Card",
                "url": "http://ufcstats.com/event-details/evt-123",
                "date": "May 17, 2026",
                "location": "Las Vegas",
            }
        ],
    )
    monkeypatch.setattr(sync_service, "_known_event_ids_from_db", lambda: set())
    monkeypatch.setattr(sync_service, "_create_db_backup", lambda: "data/backups/ufc_database_20260518.db")
    fake = _FakePopulator(
        dry_run_summary={
            "event_name": "UFC Test Card",
            "fights_total": 12,
            "fights_upserted": 12,
            "fighters_failed": 0,
        },
        commit_summary={
            "event_name": "UFC Test Card",
            "fights_total": 12,
            "fights_upserted": 12,
            "fighters_failed": 0,
            "committed": True,
        },
    )
    monkeypatch.setattr(sync_service, "_build_populator", lambda: fake)

    result = sync_service.sync_completed_ufcstats_events()

    assert result["synced_events"] == 1
    assert result["validation_failed"] == 0
    assert result["backup_path"] == "data/backups/ufc_database_20260518.db"
    state = json.loads(state_path.read_text())
    assert state["events"]["evt-123"]["status"] == "synced"
    assert ("populate", "http://ufcstats.com/event-details/evt-123", False) in fake.calls
    assert ("populate", "http://ufcstats.com/event-details/evt-123", True) in fake.calls
    assert ("validate", "http://ufcstats.com/event-details/evt-123", True) in fake.calls


def test_sync_completed_ufcstats_events_stops_on_dry_run_failure(monkeypatch, tmp_path):
    state_path = tmp_path / "ufcstats_sync_state.json"
    monkeypatch.setattr(sync_service, "STATE_PATH", state_path)
    monkeypatch.setattr(
        sync_service,
        "_recent_completed_events",
        lambda now=None: [
            {
                "event_id": "evt-123",
                "name": "UFC Test Card",
                "url": "http://ufcstats.com/event-details/evt-123",
                "date": "May 17, 2026",
                "location": "Las Vegas",
            }
        ],
    )
    monkeypatch.setattr(sync_service, "_known_event_ids_from_db", lambda: set())
    backup_calls = []
    monkeypatch.setattr(sync_service, "_create_db_backup", lambda: backup_calls.append(True))
    fake = _FakePopulator(
        dry_run_summary={
            "event_name": "UFC Test Card",
            "fights_total": 2,
            "fights_upserted": 2,
            "fighters_failed": 0,
        }
    )
    monkeypatch.setattr(sync_service, "_build_populator", lambda: fake)

    result = sync_service.sync_completed_ufcstats_events()

    assert result["synced_events"] == 0
    assert result["dry_run_failed"] == 1
    assert backup_calls == []
    state = json.loads(state_path.read_text())
    assert state["events"]["evt-123"]["status"] == "dry_run_failed"
    assert fake.calls == [("populate", "http://ufcstats.com/event-details/evt-123", False)]


def test_sync_completed_ufcstats_events_skips_known_and_synced(monkeypatch, tmp_path):
    state_path = tmp_path / "ufcstats_sync_state.json"
    state_path.write_text(
        json.dumps({"events": {"evt-synced": {"status": "synced"}}})
    )
    monkeypatch.setattr(sync_service, "STATE_PATH", state_path)
    monkeypatch.setattr(
        sync_service,
        "_recent_completed_events",
        lambda now=None: [
            {"event_id": "evt-known", "name": "Known", "url": "http://ufcstats.com/event-details/evt-known", "date": "May 17, 2026", "location": ""},
            {"event_id": "evt-synced", "name": "Synced", "url": "http://ufcstats.com/event-details/evt-synced", "date": "May 17, 2026", "location": ""},
        ],
    )
    monkeypatch.setattr(sync_service, "_known_event_ids_from_db", lambda: {"evt-known"})
    monkeypatch.setattr(sync_service, "_build_populator", lambda: (_ for _ in ()).throw(AssertionError("should not ingest")))

    result = sync_service.sync_completed_ufcstats_events()

    assert result["candidates_considered"] == 0
    assert result["skipped_existing"] == 2
    state = json.loads(state_path.read_text())
    assert state["events"]["evt-synced"]["status"] == "synced"


def test_sync_completed_ufcstats_events_restores_backup_on_validation_failure(monkeypatch, tmp_path):
    state_path = tmp_path / "ufcstats_sync_state.json"
    monkeypatch.setattr(sync_service, "STATE_PATH", state_path)
    monkeypatch.setattr(
        sync_service,
        "_recent_completed_events",
        lambda now=None: [
            {
                "event_id": "evt-123",
                "name": "UFC Test Card",
                "url": "http://ufcstats.com/event-details/evt-123",
                "date": "May 17, 2026",
                "location": "Las Vegas",
            }
        ],
    )
    monkeypatch.setattr(sync_service, "_known_event_ids_from_db", lambda: set())
    monkeypatch.setattr(sync_service, "_create_db_backup", lambda: "data/backups/safe.db")
    restore_calls = []
    monkeypatch.setattr(sync_service, "_restore_db_backup", lambda path: restore_calls.append(path) or True)
    fake = _FakePopulator(
        dry_run_summary={
            "event_name": "UFC Test Card",
            "fights_total": 12,
            "fights_upserted": 12,
            "fighters_failed": 0,
        },
        commit_summary={
            "event_name": "UFC Test Card",
            "fights_total": 12,
            "fights_upserted": 12,
            "fighters_failed": 0,
            "committed": True,
        },
        validation={"fights_checked": 12, "missing_in_db": 0, "mismatches": 2},
    )
    monkeypatch.setattr(sync_service, "_build_populator", lambda: fake)

    result = sync_service.sync_completed_ufcstats_events()

    assert result["validation_failed"] == 1
    assert restore_calls == ["data/backups/safe.db"]
    state = json.loads(state_path.read_text())
    assert state["events"]["evt-123"]["status"] == "validation_failed_restored"


def test_scheduler_disabled_by_default(monkeypatch):
    monkeypatch.delenv(sync_service.AUTO_SYNC_ENV, raising=False)
    assert sync_service.scheduler_enabled() is False


def test_sync_due_respects_recent_success(monkeypatch, tmp_path):
    state_path = tmp_path / "ufcstats_sync_state.json"
    state_path.write_text(json.dumps({"last_success_at": "2026-05-18T12:00:00Z"}))
    monkeypatch.setattr(sync_service, "STATE_PATH", state_path)
    assert sync_service.sync_due(now=sync_service._parse_iso("2026-05-18T18:00:00Z")) is False
