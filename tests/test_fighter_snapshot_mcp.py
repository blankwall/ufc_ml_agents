import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.services.fighter_snapshot import _parse_date_any, build_fighter_snapshot
from mcp_server.ufc_context_server import get_fighter_snapshot

_MAIN_DB = _ROOT / "data" / "ufc_database.db"
_SIDECAR = _ROOT / "data" / "enrichment" / "sergey_sidecar.sqlite"

pytestmark = pytest.mark.skipif(
    not _MAIN_DB.exists() or not _SIDECAR.exists(),
    reason="fighter snapshot integration data not available",
)


def test_build_fighter_snapshot_preserves_canonical_main_name_for_alias_input():
    snapshot = build_fighter_snapshot("Konklak Suphisara", recent_elo_fights=2)

    assert snapshot["resolved"] is True
    assert snapshot["identity"]["resolved_name"] == "Loma Lookboonmee"
    assert snapshot["identity"]["alias_input"] == "Konklak Suphisara"
    assert snapshot["elo"]["available"] is True
    assert len(snapshot["elo"]["recent_fights"]) <= 2



def test_build_fighter_snapshot_as_of_filters_history_strictly_before_date():
    current = build_fighter_snapshot("Conor McGregor", recent_elo_fights=3)
    assert current["recent_results"], "expected Conor McGregor to have recent results in DB"

    as_of_date = current["recent_results"][0]["date"]
    historical = build_fighter_snapshot("Conor McGregor", as_of=as_of_date, recent_elo_fights=3)
    cutoff = _parse_date_any(as_of_date)

    assert historical["record"]["fight_count_as_of"] < current["record"]["fight_count_as_of"]
    assert all(_parse_date_any(fight["date"]) < cutoff for fight in historical["recent_results"])
    assert all(fight["fight_date"] < cutoff.date().isoformat() for fight in historical["elo"]["recent_fights"])
    assert historical["elo"]["elo_current_source"] in {
        "next_fight_pre_elo",
        "fighters_current",
        "latest_visible_pre_fight_elo",
        "unavailable",
    }



def test_mcp_get_fighter_snapshot_returns_structured_payload():
    snapshot = get_fighter_snapshot("Conor McGregor", recent_elo_fights=2)

    assert snapshot["resolved"] is True
    assert snapshot["identity"]["resolved_name"] == "Conor McGregor"
    assert snapshot["record"]["fight_count_as_of"] >= len(snapshot["recent_results"])
    assert snapshot["recent_results_summary"]["window"] == 5
    assert len(snapshot["elo"]["recent_fights"]) <= 2
    if snapshot["elo"].get("elo_current") is not None and snapshot["elo"].get("elo_peak") is not None:
        assert snapshot["elo"]["elo_decline_from_peak"] == (
            snapshot["elo"]["elo_peak"] - snapshot["elo"]["elo_current"]
        )
