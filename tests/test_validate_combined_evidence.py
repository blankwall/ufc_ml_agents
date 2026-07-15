import pytest

from backtest import validate_combined_evidence as combined


def _matching_row(row_id, *, main_fight_id, date="2026-01-01", won=True, pnl=1.0):
    return {
        "id": row_id,
        "main_fight_id": main_fight_id,
        "date": date,
        "season": 2026,
        "fighter1": f"Fighter {row_id}A",
        "fighter2": f"Fighter {row_id}B",
        "pick": f"Fighter {row_id}A",
        "pick_prob": 0.58,
        "pick_odds": -120,
        "edge": -0.02,
        "pick_elo_diff": 90,
        "bet": False,
        "skip_reason": "favorite confidence",
        "pick_correct": won,
        "actual_pnl": pnl,
        "cardio_score_diff": 12,
        "striking_efficiency_score_diff": None,
        "defensive_exposure_score_diff": None,
        "trait_delta": None,
    }


def test_dedupe_by_main_fight_keeps_earliest_line_row():
    rows = [
        _matching_row(3, main_fight_id=101, date="2026-01-02", won=True, pnl=0.5),
        _matching_row(2, main_fight_id="100", date="2026-01-01", won=False, pnl=-1.0),
        _matching_row(1, main_fight_id=100, date="2026-01-01", won=True, pnl=0.83),
    ]

    unique = combined.dedupe_by_main_fight(rows)

    assert [row["id"] for row in unique] == [1, 3]
    assert list(combined.duplicate_main_fights(rows)) == [100]


def test_build_rule_rows_can_report_unique_main_fight_stats(monkeypatch):
    rows = [
        _matching_row(1, main_fight_id=100, won=True, pnl=0.83),
        _matching_row(2, main_fight_id=100, won=False, pnl=-1.0),
        _matching_row(3, main_fight_id=101, won=True, pnl=1.0),
    ]

    def fake_validation_row(all_rows, target, *, mode):
        assert mode == "leave-one-out"
        return {"score": 0}

    monkeypatch.setattr(combined, "build_validation_row", fake_validation_row)

    raw = combined.build_rule_rows(rows, mode="leave-one-out", skips_only=False, dedupe_main_fight=False)
    unique = combined.build_rule_rows(rows, mode="leave-one-out", skips_only=False, dedupe_main_fight=True)

    raw_rule = next(row for row in raw if row["rule"] == "golden_elo_not_expensive_plus_trait_support")
    unique_rule = next(row for row in unique if row["rule"] == "golden_elo_not_expensive_plus_trait_support")

    assert raw_rule["n"] == 3
    assert raw_rule["wins"] == 2
    assert raw_rule["losses"] == 1
    assert raw_rule["profit"] == pytest.approx(0.83)
    assert unique_rule["n"] == 2
    assert unique_rule["wins"] == 2
    assert unique_rule["losses"] == 0
    assert unique_rule["profit"] == pytest.approx(1.83)


def test_source_result_status_verifies_csv_line_number(tmp_path):
    results = tmp_path / "results.csv"
    results.write_text(
        "date,fighter1,fighter2,pick,pick_odds,pick_prob\n"
        "2026-01-01,A,B,A,-120,0.58\n"
    )
    row = {
        "source_results": str(results),
        "row_num": 2,
        "date": "2026-01-01",
        "fighter1": "A",
        "fighter2": "B",
        "pick": "A",
        "pick_odds": -120,
        "pick_prob": 0.58,
    }

    assert combined.source_result_status(row) == "verified"

    row["pick_odds"] = -125
    assert combined.source_result_status(row) == "mismatch"


def test_filtered_targets_respects_date_window():
    rows = [
        _matching_row(1, main_fight_id=100, date="2025-12-31"),
        _matching_row(2, main_fight_id=101, date="2026-01-01"),
        _matching_row(3, main_fight_id=102, date="2026-03-01"),
    ]

    filtered = combined.filtered_targets(rows, skips_only=False, min_date="2026-01-01", max_date="2026-02-01")

    assert [row["id"] for row in filtered] == [2]


def test_matching_rows_for_rule_annotates_temporal_segments(monkeypatch):
    rows = [
        _matching_row(1, main_fight_id=100, date="2026-01-01"),
        _matching_row(2, main_fight_id=101, date="2026-02-01"),
    ]

    def fake_validation_row(all_rows, target, *, mode):
        assert mode == "temporal"
        return {"score": 0}

    monkeypatch.setattr(combined, "build_validation_row", fake_validation_row)

    matching = combined.matching_rows_for_rule(
        rows,
        mode="temporal",
        skips_only=False,
        rule_name="golden_elo_not_expensive_plus_cardio",
        min_date="2026-01-01",
        max_date="2026-12-31",
    )

    assert [row["temporal_half"] for row in matching] == ["first_half", "second_half"]
    assert [row["temporal_quartile"] for row in matching] == ["Q1", "Q3"]
