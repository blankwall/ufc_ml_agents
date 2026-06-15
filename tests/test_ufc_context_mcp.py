import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from mcp_server import ufc_context_server as context_server
from mcp_server.ufc_context_server import (
    ROOT_DIR,
    DEFAULT_SERGEY_DB,
    allowed_file,
    ensure_read_only_query,
    get_age_experience_context,
    get_historical_fight_deep_dive,
    get_fight_stats,
    get_fight_elo_context,
    get_fight_historical_patterns,
    get_fight_model_market,
    get_fight_style_flags,
    get_fighter_elo_history,
    get_context_packet,
    query_fragility_cases,
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


def test_query_fragility_cases_filters_by_fighter_and_failure_mode(monkeypatch, tmp_path):
    cases_path = tmp_path / "fragility_cases.jsonl"
    cases_path.write_text(
        "\n".join(
            [
                '{"fight_id":"case-1","event_date":"2026-05-16","event_name":"UFC Test","fighters":{"pick":"Tuco Tokkos","opponent":"Ivan Erslan"},"fragility_flags":["tiny_sample_confidence"],"post_fight":{"failure_mode_tags":["grappling_path_collapsed"],"why_pick_lost":"Takedowns failed."},"lesson":{"summary":"Downgrade tiny samples."}}',
                '{"fight_id":"case-2","event_date":"2026-06-01","event_name":"UFC Other","fighters":{"pick":"Other Fighter","opponent":"Opponent"},"fragility_flags":["market_resistance"],"post_fight":{"failure_mode_tags":["striking_gap"],"why_pick_lost":"Striking gap."}}',
            ]
        )
    )
    monkeypatch.setattr(context_server, "FRAGILITY_CASES_FILE", cases_path)

    result = query_fragility_cases(
        fighter="Tokkos",
        failure_mode="grappling_path",
        limit=5,
    )

    assert result["total_cases"] == 2
    assert result["matched_cases"] == 1
    assert result["cases"][0]["fight_id"] == "case-1"
    assert "fighter" in result["cases"][0]["match_reasons"]
    assert "failure_mode" in result["cases"][0]["match_reasons"]


def test_query_fragility_cases_supports_text_and_date_range(monkeypatch, tmp_path):
    cases_path = tmp_path / "fragility_cases.jsonl"
    cases_path.write_text(
        "\n".join(
            [
                '{"fight_id":"old","event_date":"April 18, 2026","fighters":{"pick":"A","opponent":"B"},"review_notes":["control risk"],"post_fight":{"failure_mode_tags":["control_without_damage"]}}',
                '{"fight_id":"new","event_date":"2026-06-01","fighters":{"pick":"C","opponent":"D"},"review_notes":["control risk"],"post_fight":{"failure_mode_tags":["control_without_damage"]}}',
            ]
        )
    )
    monkeypatch.setattr(context_server, "FRAGILITY_CASES_FILE", cases_path)

    result = query_fragility_cases(
        text_query="control risk",
        min_date="2026-04-01",
        max_date="2026-04-30",
    )

    assert result["matched_cases"] == 1
    assert result["cases"][0]["fight_id"] == "old"


def test_get_age_experience_context_returns_direct_buckets(monkeypatch):
    def _fake_snapshot(name, *, as_of=None, recent_elo_fights=2):
        if name == "Prospect":
            return {
                "resolved": True,
                "identity": {"resolved_name": "Prospect"},
                "profile": {"age": 24, "date_of_birth": "Jan 01, 2002"},
                "record": {"fight_count_as_of": 3},
            }
        return {
            "resolved": True,
            "identity": {"resolved_name": "Veteran"},
            "profile": {"age": 35, "date_of_birth": "Jan 01, 1991"},
            "record": {"fight_count_as_of": 15},
        }

    monkeypatch.setattr(context_server, "build_fighter_snapshot", _fake_snapshot)

    result = get_age_experience_context("Prospect", "Veteran", as_of_date="2026-05-21")

    assert result["mapped"] is True
    assert result["fighters"]["fighter1"]["age"] == 24
    assert result["fighters"]["fighter2"]["age"] == 35
    assert result["deltas"]["fighter1_minus_fighter2_age"] == -11
    assert result["deltas"]["fighter1_minus_fighter2_fight_count"] == -12
    assert result["buckets"]["age_gap_bucket"] == "extreme_age_gap"
    assert result["buckets"]["experience_gap_bucket"] == "large_experience_gap"
    assert result["buckets"]["prospect_vs_veteran_bucket"] == "fighter1_prospect_vs_fighter2_veteran"


def test_get_age_experience_context_reports_unmapped_fighter(monkeypatch):
    def _fake_snapshot(name, *, as_of=None, recent_elo_fights=2):
        return {
            "resolved": name == "Known",
            "identity": {"resolved_name": "Known"} if name == "Known" else {},
            "profile": {"age": 30},
            "record": {"fight_count_as_of": 5},
        }

    monkeypatch.setattr(context_server, "build_fighter_snapshot", _fake_snapshot)

    result = get_age_experience_context("Known", "Unknown", as_of_date="2026-05-21")

    assert result["mapped"] is False
    assert result["fighters"]["fighter1"]["resolved"] is True
    assert result["fighters"]["fighter2"]["resolved"] is False


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
MAIN_DB = ROOT_DIR / "data" / "ufc_database.db"
MAIN_DB_AVAILABLE = MAIN_DB.exists()
skip_no_sidecar = pytest.mark.skipif(
    not SIDECAR_AVAILABLE,
    reason="Sergey sidecar DB not present in this environment",
)
skip_no_main_db = pytest.mark.skipif(
    not MAIN_DB_AVAILABLE,
    reason="Main DB not present in this environment",
)


def _sample_main_db_fight_with_stats():
    conn = sqlite3.connect(MAIN_DB)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT
                f.fight_id,
                e.date AS event_date,
                f1.name AS fighter1_name,
                f2.name AS fighter2_name
            FROM fights f
            JOIN events e ON e.id = f.event_id
            JOIN fighters f1 ON f1.id = f.fighter_1_id
            JOIN fighters f2 ON f2.id = f.fighter_2_id
            JOIN fight_stats fs ON fs.fight_id = f.id
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            pytest.skip("No fight_stats rows available in main DB")
        return dict(row)
    finally:
        conn.close()


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


def test_get_context_packet_missing_target_returns_dynamic_packet(monkeypatch):
    captured = {}

    def fake_dynamic_target(**kwargs):
        captured.update(kwargs)
        target = {
            "id": "dynamic:alpha:beta:2026-05-30",
            "season": 2026,
            "date": "2026-05-30",
            "fighter1": "Alpha Fighter",
            "fighter2": "Beta Fighter",
            "pick": "Alpha Fighter",
            "winner": None,
            "pick_correct": None,
            "actual_pnl": None,
            "pick_prob": 0.58,
            "pick_odds": 120,
            "market_implied_prob": 0.455,
            "edge": 0.125,
            "bet": None,
            "skip_reason": "dynamic_synthetic_target_no_config_decision",
            "fighter1_elo": 1325,
            "fighter2_elo": 1210,
            "pick_elo": 1325,
            "opponent_elo": 1210,
            "pick_elo_diff": 115,
            "abs_elo_diff": 115,
            "model_agrees_with_elo": True,
            "join_status": "matched",
            "join_method": "dynamic_fighter_snapshot",
            "elo_implied_prob": 0.6598,
            "model_minus_elo_prob": -0.0798,
            "market_minus_elo_prob": -0.2048,
            "model_market_elo_triangle": "model_and_market_under_elo",
        }
        analysis = {
            "request": {
                "fighter1": "Alpha Fighter",
                "fighter2": "Beta Fighter",
                "fight_date": "2026-05-30",
                "fighter1_odds": 120,
                "fighter2_odds": -140,
            },
            "resolution": {"fight_date": {"parsed": "2026-05-30"}},
            "validation": {"ok": True, "warnings": []},
            "market": {
                "odds": {"fighter1": 120, "fighter2": -140},
                "provenance": {"source": "user_input"},
                "pricing_context": {
                    "has_real_market": True,
                    "has_two_sided_market": True,
                    "market_missing": False,
                    "pricing_context_degraded": False,
                    "edge_type": "market_edge",
                    "market_completeness": "two_sided_market",
                    "warning_codes": [],
                },
            },
            "prediction": {"pick": {"fighter_name": "Alpha Fighter", "probability": 0.58}},
            "fighters": {"fighter1": {}, "fighter2": {}},
            "provenance": {"source": "pytest"},
        }
        return target, None, analysis

    monkeypatch.setattr(context_server, "_dynamic_synthetic_target", fake_dynamic_target)

    result = get_context_packet(
        fighter1="No Context Alpha",
        fighter2="No Context Beta",
        date="2026-05-30",
        fighter1_odds=120,
        fighter2_odds=-140,
    )

    assert result["packet_type"] == "dynamic_future_fight"
    assert result["source"]["dynamic_reason"] == "missing_context_pool_row"
    assert result["source"]["exact_context_pool_row"] is False
    assert result["source"]["historical_pool_role"] == "evidence_library"
    assert result["pricing_context"]["edge_type"] == "market_edge"
    assert result["model_market"]["market_provenance"]["source"] == "user_input"
    assert "historical_lookup_error" in result["source"]
    assert captured == {
        "fighter1": "No Context Alpha",
        "fighter2": "No Context Beta",
        "date": "2026-05-30",
        "fighter1_odds": 120,
        "fighter2_odds": -140,
    }


@skip_no_main_db
def test_get_fight_stats_returns_structured_payload_for_known_fight():
    sample = _sample_main_db_fight_with_stats()

    result = get_fight_stats(
        fighter1=sample["fighter1_name"],
        fighter2=sample["fighter2_name"],
        date=sample["event_date"],
    )

    assert result["mapped"] is True
    assert result["fight_id"] == sample["fight_id"]
    assert result["event"]["event_date"] == sample["event_date"]
    assert result["fighters"]["fighter1"] == sample["fighter1_name"]
    assert result["fighters"]["fighter2"] == sample["fighter2_name"]
    assert "fighter1_totals" in result["stats"]
    assert "fighter2_totals" in result["stats"]
    assert isinstance(result["fight_details"]["stats_available"], bool)


def test_get_fight_stats_returns_unmapped_payload_for_missing_fight():
    result = get_fight_stats(
        fighter1="Nope Fighter Alpha",
        fighter2="Nope Fighter Beta",
        date="2099-01-01",
    )

    assert result["mapped"] is False
    assert "note" in result


@skip_no_main_db
def test_get_fight_stats_matches_iso_date_against_human_readable_event_date():
    sample = _sample_main_db_fight_with_stats()
    iso_date = datetime.strptime(sample["event_date"], "%B %d, %Y").strftime("%Y-%m-%d")

    result = get_fight_stats(
        fighter1=sample["fighter1_name"],
        fighter2=sample["fighter2_name"],
        date=iso_date,
    )

    assert result["mapped"] is True
    assert result["fight_id"] == sample["fight_id"]


def test_get_historical_fight_deep_dive_passes_fight_date_to_pre_fight_analysis(monkeypatch):
    fake_row = {
        "id": 42,
        "fight_id": "fight-123",
        "event_id": "event-123",
        "event_name": "UFC Test Card",
        "event_date": "May 17, 2026",
        "event_url": "http://ufcstats.com/event-details/event-123",
        "fight_number": 3,
        "weight_class": "Lightweight",
        "is_title_fight": 0,
        "scheduled_rounds": 3,
        "result": "fighter_1",
        "method": "Decision",
        "method_detail": "Unanimous",
        "round_finished": 3,
        "time": "5:00",
        "fight_detail_url": "http://ufcstats.com/fight-details/fight-123",
        "fighter1_name": "Alpha Fighter",
        "fighter2_name": "Beta Fighter",
        "winner_name": "Alpha Fighter",
        "fighter_1_totals": '{"sig_strikes":"10 of 20"}',
        "fighter_2_totals": '{"sig_strikes":"8 of 18"}',
        "round_by_round": None,
        "significant_strikes": None,
    }
    captured = {}

    class _DummyConn:
        def close(self):
            return None

    monkeypatch.setattr("mcp_server.ufc_context_server.readonly_connection", lambda _path: _DummyConn())
    monkeypatch.setattr("mcp_server.ufc_context_server.resolve_database_path", lambda _db: Path("/tmp/fake.db"))
    monkeypatch.setattr("mcp_server.ufc_context_server._fight_stats_candidates", lambda *_args, **_kwargs: [fake_row])
    monkeypatch.setattr(
        "mcp_server.ufc_context_server._historical_market_odds",
        lambda *_args, **_kwargs: {"fighter_1_odds": -150, "fighter_2_odds": 130, "bookmaker": "pytest"},
    )

    def _fake_build_init_fight_analysis(**kwargs):
        captured.update(kwargs)
        return {
            "status": "ok",
            "fighters": {
                "fighter1": {"qualitative": {"available": False}},
                "fighter2": {"qualitative": {"available": False}},
            },
            "prediction": {"pick": {"slot": "fighter1", "fighter_name": "Alpha Fighter", "probability": 0.55, "market_probability": 0.52}},
        }

    monkeypatch.setattr("mcp_server.ufc_context_server.build_init_fight_analysis", _fake_build_init_fight_analysis)

    result = get_historical_fight_deep_dive(fight_id="fight-123")

    assert result["mapped"] is True
    assert captured["fight_date"] == "May 17, 2026"
    assert captured["fighter1"] == "Alpha Fighter"
    assert captured["fighter2"] == "Beta Fighter"
    assert captured["fighter1_odds"] == -150
    assert captured["fighter2_odds"] == 130
    assert result["pre_fight"]["analysis_cutoff"] == "May 17, 2026"
    assert result["actual_fight"]["fight_id"] == "fight-123"


@skip_no_main_db
def test_get_historical_fight_deep_dive_returns_actual_and_pre_fight_payloads():
    sample = _sample_main_db_fight_with_stats()

    result = get_historical_fight_deep_dive(
        fighter1=sample["fighter1_name"],
        fighter2=sample["fighter2_name"],
        date=sample["event_date"],
    )

    assert result["mapped"] is True
    assert result["fight_id"] == sample["fight_id"]
    assert result["actual_fight"]["event"]["event_date"] == sample["event_date"]
    assert result["pre_fight"]["analysis_cutoff"] == sample["event_date"]
    assert "analysis" in result["pre_fight"]
    assert "provenance" in result
