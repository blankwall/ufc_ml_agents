from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.services import marco_service as marco  # noqa: E402
from fastapi_app.services import marco_warm_service as warmer  # noqa: E402
from fastapi_app.services import predict_service  # noqa: E402
from fastapi_app.services import ufc_schedule_service as schedule_service  # noqa: E402


def _marco_result(*, pick: str = "Pick", prior: int = 5) -> dict:
    return {
        "status": "complete",
        "pick": pick,
        "history": {"fighter1_prior": prior, "fighter2_prior": prior},
    }


def _resurrect(**overrides):
    values = {
        "marco": _marco_result(),
        "model_pick": "Pick",
        "original_bet": False,
        "skip_code": "F1",
        "skip_reason": "skipped",
        "pick_model_prob": 0.58,
        "pick_market_prob": 0.53,
        "pick_odds": -110,
    }
    values.update(overrides)
    return marco.evaluate_resurrection(**values)


def test_favorite_resurrection_boundaries():
    assert _resurrect(pick_model_prob=0.55, pick_market_prob=0.55)["resurrected"] is True
    assert _resurrect(pick_model_prob=0.649, pick_market_prob=0.55)["resurrected"] is True
    assert _resurrect(pick_model_prob=0.65, pick_market_prob=0.55)["resurrected"] is False
    assert _resurrect(pick_model_prob=0.54, pick_market_prob=0.50)["resurrected"] is False
    assert _resurrect(pick_model_prob=0.58, pick_market_prob=0.60)["resurrected"] is False


def test_underdog_resurrection_boundaries():
    assert _resurrect(
        skip_code="U1",
        pick_model_prob=0.40,
        pick_market_prob=0.35,
        pick_odds=400,
    )["resurrected"] is True
    assert _resurrect(
        skip_code="U1",
        pick_model_prob=0.40,
        pick_market_prob=0.351,
        pick_odds=400,
    )["resurrected"] is False
    assert _resurrect(
        skip_code="U3",
        pick_model_prob=0.40,
        pick_market_prob=0.35,
        pick_odds=401,
    )["resurrected"] is False


def test_resurrection_never_flips_or_reopens_disallowed_bets():
    assert _resurrect(original_bet=True)["reason_code"] == "existing_bet"
    assert _resurrect(skip_code="D1")["reason_code"] == "insufficient_history"
    assert _resurrect(marco=_marco_result(prior=1))["reason_code"] == "insufficient_history"
    assert _resurrect(marco=_marco_result(pick="Opponent"))["reason_code"] == "marco_disagrees"
    assert _resurrect(pick_odds=None)["reason_code"] == "missing_odds"


def test_resurrection_can_be_disabled_without_hiding_agreement():
    result = _resurrect(enabled=False)

    assert result["resurrected"] is False
    assert result["final_bet"] is False
    assert result["agreement"] is True
    assert result["reason_code"] == "disabled"


def test_marco_prediction_uses_persistent_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "marco_cache.json"
    script = tmp_path / "marco" / "predict.py"
    script.parent.mkdir()
    script.write_text("")
    monkeypatch.setattr(marco, "CACHE_PATH", cache_path)
    monkeypatch.setattr(marco, "LOCK_PATH", tmp_path / "marco_cache.lock")
    monkeypatch.setattr(marco, "KEY_LOCK_DIR", tmp_path / ".marco_locks")
    monkeypatch.setattr(marco, "_predict_script", lambda: script)
    monkeypatch.setattr(marco, "_marco_root", lambda: tmp_path)
    monkeypatch.setattr(marco, "_marco_python", lambda: "python3")
    monkeypatch.setattr(marco, "_runtime_identity", lambda: {"identity": "test"})
    monkeypatch.setattr(marco, "resolve_canonical_names", lambda *_args: ("A", "B"))
    monkeypatch.setattr(
        marco,
        "resolve_marco_metadata",
        lambda *_args: {
            "fight_date": "2026-08-08",
            "weight_class": "Lightweight",
            "lbs": 155,
            "rounds": 3,
            "fight_number": 2,
            "is_title_fight": False,
            "metadata_source": "ufcstats",
        },
    )
    raw = {
        "p_A": 0.61,
        "p_B": 0.39,
        "a_fights_prior": 8,
        "b_fights_prior": 6,
    }
    calls = []

    def _run(*_args, **_kwargs):
        calls.append(True)
        return SimpleNamespace(returncode=0, stdout=json.dumps(raw), stderr="")

    monkeypatch.setattr(marco.subprocess, "run", _run)

    first = marco.run_marco_prediction("A", "B", fight_date="2026-08-08")
    second = marco.run_marco_prediction("A", "B", fight_date="2026-08-08")

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["pick"] == "A"
    assert len(calls) == 1


def test_warmer_only_processes_cards_after_today(monkeypatch):
    monkeypatch.setattr(schedule_service, "load_allowlist", lambda: {"events": []})
    monkeypatch.setattr(
        predict_service,
        "get_events_data",
        lambda: [
            {
                "event_date": "2026-08-04",
                "fights": [{"fighter1": "Today A", "fighter2": "Today B"}],
            },
            {
                "event_date": "2026-08-08",
                "fights": [
                    {"fighter1": "Future A", "fighter2": "Future B"},
                    {"fighter1": "Future C", "fighter2": "Future D"},
                ],
            },
        ],
    )
    calls = []
    monkeypatch.setattr(
        warmer,
        "run_marco_prediction",
        lambda a, b, **_kwargs: calls.append((a, b)) or {
            "status": "complete",
            "cache_hit": len(calls) == 2,
        },
    )
    monkeypatch.setattr(warmer, "scheduler_enabled", lambda: True)

    summary = warmer.warm_future_cards(today=date(2026, 8, 4))

    assert calls == [("Future A", "Future B"), ("Future C", "Future D")]
    assert summary == {
        "cards": 1,
        "fights": 2,
        "warmed": 1,
        "cached": 1,
        "unavailable": 0,
        "errors": 0,
    }
