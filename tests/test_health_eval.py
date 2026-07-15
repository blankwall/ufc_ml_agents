import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.services import health_eval


def _iso_ago(**kwargs) -> str:
    return (
        (datetime.now(timezone.utc) - timedelta(**kwargs))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def test_humanize_age_formats():
    assert health_eval.humanize_age(None) == "—"
    assert health_eval.humanize_age(90) == "1m"
    assert health_eval.humanize_age(3 * 3600 + 5 * 60) == "3h 5m"
    assert health_eval.humanize_age(2 * 86400 + 3 * 3600) == "2d 3h"


def test_evaluate_job_flags_failures_as_error():
    result = health_eval.evaluate_job(
        "ufcstats_completed_sync",
        {"enabled": True, "failures_since_launch": 2, "last_success_at": _iso_ago(hours=1)},
    )
    assert result["level"] == "error"
    assert any("failed run" in issue for issue in result["issues"])


def test_evaluate_job_flags_stale_success_as_degraded():
    result = health_eval.evaluate_job(
        "ufcstats_completed_sync",
        {"enabled": True, "failures_since_launch": 0, "persisted_last_success_at": _iso_ago(days=5)},
    )
    assert result["level"] == "degraded"
    assert result["stale"] is True


def test_evaluate_job_flags_silent_no_op():
    result = health_eval.evaluate_job(
        "ufcstats_completed_sync",
        {
            "enabled": True,
            "failures_since_launch": 0,
            "last_success_at": _iso_ago(hours=1),
            "persisted_last_result": {"synced_events": 0, "recent_events_seen": 0},
        },
    )
    assert result["level"] == "degraded"
    assert any("0 events" in issue for issue in result["issues"])


def test_evaluate_job_disabled_is_not_alarmed():
    result = health_eval.evaluate_job(
        "ufcstats_completed_sync",
        {"enabled": False, "failures_since_launch": 0, "last_success_at": None},
    )
    assert result["level"] == "ok"


def test_data_freshness_stale_detection(monkeypatch, tmp_path):
    import sqlite3

    db = tmp_path / "ufc_database.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE events (name TEXT, date TEXT)")
    old = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%B %d, %Y")
    con.execute("INSERT INTO events VALUES (?, ?)", ("Old Card", old))
    con.commit()
    con.close()
    monkeypatch.setattr(health_eval, "DB_PATH", db)

    fresh = health_eval.db_data_freshness()
    assert fresh["stale"] is True
    assert fresh["age_days"] >= 59


def test_resolve_evidence_rejects_unknown_id():
    assert health_eval.resolve_evidence_file("../../etc/passwd") is None
    assert health_eval.resolve_evidence_file("not_a_real_artifact") is None


def test_list_evidence_includes_backups_dir():
    artifacts = health_eval.list_evidence()
    kinds = {a["id"]: a for a in artifacts}
    assert "db_backups" in kinds
    assert kinds["db_backups"]["kind"] == "dir"
    assert "ufcstats_sync_state" in kinds
