from types import SimpleNamespace

from database.db_manager import DatabaseManager


def _fight(db_f1_id: str, db_f2_id: str):
    return SimpleNamespace(
        fight_id="fight-123",
        fighter_1=SimpleNamespace(fighter_id=db_f1_id),
        fighter_2=SimpleNamespace(fighter_id=db_f2_id),
    )


def test_remap_fight_details_to_db_slots_swaps_when_detail_page_order_differs():
    details = {
        "fighter_1_id": "kyler",
        "fighter_2_id": "charles",
        "totals": {
            "fighter_1": {"control_time": "7:13"},
            "fighter_2": {"control_time": "0:21"},
        },
        "significant_strikes": {
            "fighter_1": {"sig_strikes_total": "36 of 77"},
            "fighter_2": {"sig_strikes_total": "58 of 104"},
        },
        "round_by_round": [
            {
                "round": 1,
                "fighter_1": {"totals": {"knockdowns": "0"}, "significant_strikes": {}},
                "fighter_2": {"totals": {"knockdowns": "1"}, "significant_strikes": {}},
            }
        ],
    }

    f1_totals, f2_totals, sig, round_by_round = DatabaseManager.remap_fight_details_to_db_slots(
        details,
        _fight("charles", "kyler"),
    )

    assert f1_totals == {"control_time": "0:21"}
    assert f2_totals == {"control_time": "7:13"}
    assert sig["fighter_1"] == {"sig_strikes_total": "58 of 104"}
    assert sig["fighter_2"] == {"sig_strikes_total": "36 of 77"}
    # Every round payload swaps fighter slots too; round number is preserved.
    assert round_by_round[0]["round"] == 1
    assert round_by_round[0]["fighter_1"]["totals"] == {"knockdowns": "1"}
    assert round_by_round[0]["fighter_2"]["totals"] == {"knockdowns": "0"}


def test_remap_fight_details_to_db_slots_keeps_aligned_payload():
    details = {
        "fighter_1_id": "charles",
        "fighter_2_id": "kyler",
        "totals": {
            "fighter_1": {"control_time": "0:21"},
            "fighter_2": {"control_time": "7:13"},
        },
        "significant_strikes": {
            "fighter_1": {"sig_strikes_total": "58 of 104"},
            "fighter_2": {"sig_strikes_total": "36 of 77"},
        },
        "round_by_round": [
            {
                "round": 1,
                "fighter_1": {"totals": {"knockdowns": "1"}, "significant_strikes": {}},
                "fighter_2": {"totals": {"knockdowns": "0"}, "significant_strikes": {}},
            }
        ],
    }

    f1_totals, f2_totals, sig, round_by_round = DatabaseManager.remap_fight_details_to_db_slots(
        details,
        _fight("charles", "kyler"),
    )

    assert f1_totals == {"control_time": "0:21"}
    assert f2_totals == {"control_time": "7:13"}
    assert sig["fighter_1"] == {"sig_strikes_total": "58 of 104"}
    assert sig["fighter_2"] == {"sig_strikes_total": "36 of 77"}
    # Aligned order: round payloads pass through untouched.
    assert round_by_round[0]["fighter_1"]["totals"] == {"knockdowns": "1"}
    assert round_by_round[0]["fighter_2"]["totals"] == {"knockdowns": "0"}
