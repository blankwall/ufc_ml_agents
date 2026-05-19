from pathlib import Path

import pytest

from mcp_server.ufc_context_server import (
    ROOT_DIR,
    DEFAULT_SERGEY_DB,
    allowed_file,
    ensure_read_only_query,
    get_fight_elo_context,
    get_fight_historical_patterns,
    get_fight_model_market,
    get_fight_style_flags,
    get_fighter_elo_history,
    get_context_packet,
    resolve_whitelisted_file,
    _parse_elo_window,
)


def test_ensure_read_only_query_allows_select_and_with():
    assert ensure_read_only_query("SELECT 1") == "SELECT 1"
    assert ensure_read_only_query("WITH t AS (SELECT 1) SELECT * FROM t") == "WITH t AS (SELECT 1) SELECT * FROM t"


def test_ensure_read_only_query_rejects_writes_and_multi_statement():
    with pytest.raises(ValueError, match="read-only"):
        ensure_read_only_query("DELETE FROM fights")
    with pytest.raises(ValueError, match="single"):
        ensure_read_only_query("SELECT 1; SELECT 2")


def test_allowed_file_accepts_whitelisted_paths():
    assert allowed_file(ROOT_DIR / "README.md") is True
    assert allowed_file(ROOT_DIR / "backtest" / "context_packet.py") is True


def test_resolve_whitelisted_file_rejects_outside_root(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    with pytest.raises(ValueError, match="whitelist"):
        resolve_whitelisted_file(str(outside))


def test_resolve_whitelisted_file_returns_repo_file():
    path = resolve_whitelisted_file("backtest/context_packet.py")
    assert path == Path(ROOT_DIR / "backtest" / "context_packet.py").resolve()


# ---------------------------------------------------------------------------
# _parse_elo_window unit tests (no DB required)
# ---------------------------------------------------------------------------

def test_parse_elo_window_all_sentinel():
    assert _parse_elo_window("all") is None
    assert _parse_elo_window("ALL") is None
    assert _parse_elo_window("All") is None


def test_parse_elo_window_integer_values():
    assert _parse_elo_window(3) == 3
    assert _parse_elo_window(5) == 5
    assert _parse_elo_window("10") == 10


def test_parse_elo_window_rejects_invalid():
    with pytest.raises(ValueError):
        _parse_elo_window("last5")
    with pytest.raises(ValueError):
        _parse_elo_window(0)
    with pytest.raises(ValueError):
        _parse_elo_window(-1)


# ---------------------------------------------------------------------------
# get_fighter_elo_history integration tests (require sergey_sidecar.sqlite)
# ---------------------------------------------------------------------------

SIDECAR_AVAILABLE = DEFAULT_SERGEY_DB.exists()
skip_no_sidecar = pytest.mark.skipif(
    not SIDECAR_AVAILABLE,
    reason="Sergey sidecar DB not present in this environment",
)


@skip_no_sidecar
def test_get_fighter_elo_history_known_fighter():
    result = get_fighter_elo_history("Conor McGregor", window=5)

    assert result["mapped"] is True
    assert result["resolved_name"] == "Conor McGregor"
    assert result["returned_fights"] <= 5
    assert result["total_fights_in_db"] >= 5

    first = result["fights"][0]
    assert "fight_date" in first
    assert "opponent_name" in first
    assert first["result"] in {"win", "loss", "draw", "no_contest", "unknown"}
    assert "fighter_pre_elo" in first
    assert "opponent_pre_elo" in first
    assert "elo_diff" in first
    assert "method" in first
    assert "division" in first


@skip_no_sidecar
def test_get_fighter_elo_history_elo_diff_consistency():
    result = get_fighter_elo_history("Conor McGregor", window=10)
    for fight in result["fights"]:
        if fight["fighter_pre_elo"] is not None and fight["opponent_pre_elo"] is not None:
            assert fight["elo_diff"] == fight["fighter_pre_elo"] - fight["opponent_pre_elo"]


@skip_no_sidecar
def test_get_fighter_elo_history_fights_ordered_most_recent_first():
    result = get_fighter_elo_history("Conor McGregor", window="all")
    dates = [f["fight_date"] for f in result["fights"] if f["fight_date"]]
    assert dates == sorted(dates, reverse=True)


@skip_no_sidecar
def test_get_fighter_elo_history_window_all_returns_full_history():
    result_all = get_fighter_elo_history("Conor McGregor", window="all")
    result_n = get_fighter_elo_history("Conor McGregor", window=result_all["total_fights_in_db"])
    assert result_all["returned_fights"] == result_n["returned_fights"]


@skip_no_sidecar
def test_get_fighter_elo_history_unknown_fighter():
    result = get_fighter_elo_history("Zzz Nonexistent Qqq Fighter 9999")
    assert result["mapped"] is False
    assert result["returned_fights"] == 0
    assert result["fights"] == []
    assert "note" in result


@skip_no_sidecar
def test_get_fighter_elo_history_ambiguous_name_returns_candidates():
    # "jones" matches many fighters — should trigger ambiguous path
    result = get_fighter_elo_history("jones")
    if not result["mapped"] and result.get("ambiguous"):
        assert len(result["candidates"]) > 1
        assert "note" in result
    # If the sidecar happens to have exactly one Jones, that's also fine
    else:
        assert result["mapped"] is True


@skip_no_sidecar
def test_get_fighter_elo_history_includes_metadata_fields():
    result = get_fighter_elo_history("Khabib Nurmagomedov", window=3)
    assert "elo_current" in result
    assert "elo_peak" in result
    assert "fighter_id" in result
    assert isinstance(result["fighter_id"], int)


def test_get_fight_model_market_for_known_context_row():
    result = get_fight_model_market(fight_pool_id=443)
    assert result["fight_pool_id"] == 443
    assert result["fighter1"] == "Dominick Reyes"
    assert "pick_prob" in result
    assert "odds_provenance" in result


def test_get_fight_historical_patterns_for_known_context_row():
    result = get_fight_historical_patterns(fight_pool_id=443)
    assert result["fight_pool_id"] == 443
    assert "pattern_score_v0" in result
    assert isinstance(result["patterns"], list)
    assert result["pattern_score_v0"]["score"] >= 0


def test_get_fight_style_flags_for_known_context_row():
    result = get_fight_style_flags(fight_pool_id=443)
    assert result["fight_pool_id"] == 443
    assert "flags" in result
    assert "support" in result["flags"]
    assert "risk" in result["flags"]


def test_missing_context_target_raises_value_error_instead_of_exiting():
    with pytest.raises(ValueError, match="No context-pool row found"):
        get_fight_elo_context(
            fighter1="Alex Perez",
            fighter2="Su Mudaerji",
            date="2026-05-30",
        )


def test_get_context_packet_missing_target_raises_value_error_instead_of_exiting():
    with pytest.raises(ValueError, match="No context-pool row found"):
        get_context_packet(
            fighter1="Alex Perez",
            fighter2="Su Mudaerji",
            date="2026-05-30",
        )
