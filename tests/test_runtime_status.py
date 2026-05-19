import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.services import runtime_status


def test_runtime_status_tracks_task_and_run_lifecycle():
    job_name = "pytest_runtime_job"
    runtime_status.configure_job(job_name, enabled=True)
    runtime_status.mark_task_started(job_name)
    runtime_status.mark_check(job_name)
    runtime_status.mark_run_started(job_name, trigger="pytest")
    runtime_status.mark_run_finished(
        job_name,
        success=True,
        summary={"new_rows_added": 2, "processed": [1, 2, 3]},
    )
    runtime_status.mark_task_stopped(job_name, reason="pytest_done")

    status = runtime_status.get_job_status(job_name)

    assert status["enabled"] is True
    assert status["task_active"] is False
    assert status["checks_since_launch"] >= 1
    assert status["runs_since_launch"] >= 1
    assert status["successes_since_launch"] >= 1
    assert status["last_trigger"] == "pytest"
    assert status["task_stop_reason"] == "pytest_done"
    assert status["last_summary"]["new_rows_added"] == 2
    assert status["last_summary"]["processed_count"] == 3
