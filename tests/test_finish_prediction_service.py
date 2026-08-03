"""Unit tests for fastapi_app/services/finish_prediction_service.py.

These tests never invoke the real ufc_decision_skill subprocess — they mock
subprocess.run and SKILL_CLI existence so they run fast and don't require the
skill to be installed. Real end-to-end parity against the protected model is
covered separately in tests/test_finish_prediction_acceptance.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from services import finish_prediction_service as svc  # noqa: E402


def _fake_skill_response(**overrides) -> dict:
    payload = {
        "binary": {
            "probabilities": {"finish": 0.6270289421081543, "decision": 0.3729710578918457},
            "selection": "finish",
            "confidence": 0.6270289421081543,
            "tier": "strong",
            "eligible": True,
            "history_eligible": True,
            "confidence_eligible": True,
            "market": {
                "available": False,
                "selected_probability": None,
                "edge": None,
                "actionable": False,
            },
        },
        "method": {
            "decision": 0.36023667454719543,
            "ko_tko": 0.4737807810306549,
            "submission": 0.16598257422447205,
            "finish": 0.639763355255127,
        },
        "history": {"fighter_a_prior": 17, "fighter_b_prior": 11},
    }
    payload.update(overrides)
    return payload


def test_missing_inputs_return_error_without_subprocess(monkeypatch):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(1))

    result = svc.run_finish_prediction(
        None, "Billy Quarantillo", fight_date=date(2026, 8, 8), weight_class="Lightweight"
    )
    assert result["bet"] == "error"
    assert result["error_code"] == "missing_input"

    result = svc.run_finish_prediction(
        "Diego Ferreira", "Billy Quarantillo", fight_date=None, weight_class="Lightweight"
    )
    assert result["bet"] == "error"
    assert result["error_code"] == "missing_input"

    result = svc.run_finish_prediction(
        "Diego Ferreira", "Billy Quarantillo", fight_date=date(2026, 8, 8), weight_class=None
    )
    assert result["bet"] == "error"
    assert result["error_code"] == "missing_input"

    assert not called  # never shells out when required inputs are absent


def test_skill_not_installed_reports_error(monkeypatch, tmp_path):
    monkeypatch.setattr(svc, "SKILL_CLI", tmp_path / "does-not-exist")

    result = svc.run_finish_prediction(
        "Diego Ferreira", "Billy Quarantillo",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
    )
    assert result["bet"] == "error"
    assert result["error_code"] == "skill_not_installed"


def test_successful_run_is_shaped_correctly(monkeypatch, tmp_path):
    fake_cli = tmp_path / "ufc-decision"
    fake_cli.write_text("#!/bin/sh\n")
    fake_cli.chmod(0o755)
    monkeypatch.setattr(svc, "SKILL_CLI", fake_cli)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=json.dumps(_fake_skill_response()), stderr=""
        ),
    )

    result = svc.run_finish_prediction(
        "Diego Ferreira", "Billy Quarantillo",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
    )
    assert result["bet"] is False  # eligible/strong but no market price => not actionable
    assert result["selection"] == "finish"
    assert result["confidence"] == 0.6270289421081543
    assert result["tier"] == "strong"
    assert result["eligible"] is True
    assert result["probabilities"] == {
        "finish": 0.6270289421081543,
        "decision": 0.3729710578918457,
    }
    assert result["method_probabilities"]["ko_tko"] == 0.4737807810306549
    assert result["method_probabilities"]["submission"] == 0.16598257422447205
    assert result["history"] == {"fighter_a_prior": 17, "fighter_b_prior": 11}
    assert result["fight_number"] == svc.DEFAULT_FIGHT_NUMBER


def test_market_actionable_requires_price_and_positive_edge(monkeypatch, tmp_path):
    fake_cli = tmp_path / "ufc-decision"
    fake_cli.write_text("#!/bin/sh\n")
    fake_cli.chmod(0o755)
    monkeypatch.setattr(svc, "SKILL_CLI", fake_cli)

    response = _fake_skill_response()
    response["binary"]["market"] = {
        "available": True,
        "selected_probability": 0.55,
        "edge": 0.0770289421081543,
        "actionable": True,
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr=""),
    )

    result = svc.run_finish_prediction(
        "Diego Ferreira", "Billy Quarantillo",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
        market_finish_probability=0.55,
    )
    assert result["bet"] is True
    assert result["market"]["actionable"] is True
    assert result["market"]["edge"] == 0.0770289421081543


def test_nonzero_exit_maps_to_subprocess_error(monkeypatch, tmp_path):
    fake_cli = tmp_path / "ufc-decision"
    fake_cli.write_text("#!/bin/sh\n")
    fake_cli.chmod(0o755)
    monkeypatch.setattr(svc, "SKILL_CLI", fake_cli)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )

    result = svc.run_finish_prediction(
        "Diego Ferreira", "Billy Quarantillo",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
    )
    assert result["bet"] == "error"
    assert result["error_code"] == "subprocess_error"
    assert "boom" in result["error_message"]


def test_fighter_not_found_maps_to_dedicated_error_code(monkeypatch, tmp_path):
    fake_cli = tmp_path / "ufc-decision"
    fake_cli.write_text("#!/bin/sh\n")
    fake_cli.chmod(0o755)
    monkeypatch.setattr(svc, "SKILL_CLI", fake_cli)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="", stderr="ValueError: fighter not found: Nobody Real"
        ),
    )

    result = svc.run_finish_prediction(
        "Nobody Real", "Billy Quarantillo",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
    )
    assert result["bet"] == "error"
    assert result["error_code"] == "fighter_not_found"


def test_timeout_maps_to_dedicated_error_code(monkeypatch, tmp_path):
    fake_cli = tmp_path / "ufc-decision"
    fake_cli.write_text("#!/bin/sh\n")
    fake_cli.chmod(0o755)
    monkeypatch.setattr(svc, "SKILL_CLI", fake_cli)

    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ufc-decision", timeout=1)

    monkeypatch.setattr(subprocess, "run", _raise)

    result = svc.run_finish_prediction(
        "Diego Ferreira", "Billy Quarantillo",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
        timeout=1,
    )
    assert result["bet"] == "error"
    assert result["error_code"] == "timeout"


def test_invalid_json_output_maps_to_dedicated_error_code(monkeypatch, tmp_path):
    fake_cli = tmp_path / "ufc-decision"
    fake_cli.write_text("#!/bin/sh\n")
    fake_cli.chmod(0o755)
    monkeypatch.setattr(svc, "SKILL_CLI", fake_cli)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="not json", stderr=""),
    )

    result = svc.run_finish_prediction(
        "Diego Ferreira", "Billy Quarantillo",
        fight_date=date(2026, 8, 8), weight_class="Lightweight",
    )
    assert result["bet"] == "error"
    assert result["error_code"] == "invalid_output"


def test_devig_finish_probability_matches_manual_calculation():
    # +150 finish, -140 decision
    raw_finish = 100 / (150 + 100)
    raw_decision = 140 / (140 + 100)
    expected = raw_finish / (raw_finish + raw_decision)
    assert svc.devig_finish_probability(150, -140) == expected
