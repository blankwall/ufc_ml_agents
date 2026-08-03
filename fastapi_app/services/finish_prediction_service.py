"""
Integration wrapper around the protected `ufc_decision_skill` package
(https://github.com/blankwall/ufc_decision_skill, pinned commit da46562).

This module never re-implements or duplicates the skill's model logic
(feature extraction, gating, XGBoost training, or de-vig math for its own
`market_finish_probability` handling). It shells out to the skill's own CLI
entry point — which is a thin wrapper around
`ufc_decision_skill.inference.predict()` — running inside the skill's own
pinned virtualenv. This keeps the skill's exactly-reproduced dependency
versions (numpy/pandas/scikit-learn/xgboost/scipy) fully isolated from this
app's own dependency set, so the winner model is never put at risk of a
transitive version bump, and the skill's validated walk-forward parity
(see ~/ufc_decision_skill/reports/validation_report.json) stays intact.

Error codes returned under `error_code` (all mean `bet` is the string
"error", never a substituted probability):

- ``missing_input``        — fighter_a/fighter_b/fight_date/weight_class not supplied
- ``skill_not_installed``  — the skill's repo/venv isn't present at the configured root
- ``timeout``              — the subprocess did not finish within the timeout
- ``invalid_output``       — stdout was not valid JSON
- ``fighter_not_found``    — the skill could not resolve a fighter (ValueError from the skill)
- ``subprocess_error``     — any other non-zero exit from the skill CLI
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional


def _default_skill_root() -> Path:
    override = os.environ.get("UFC_DECISION_SKILL_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / "ufc_decision_skill"


SKILL_ROOT = _default_skill_root()
SKILL_CLI = SKILL_ROOT / ".venv" / "bin" / "ufc-decision"

DEFAULT_FIGHT_NUMBER = 5  # non-main-event card slot; mirrors the skill's own CLI default
SUBPROCESS_TIMEOUT_SECONDS = 45


def _american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def devig_finish_probability(finish_odds: int, decision_odds: int) -> float:
    """Vig-normalize a pair of American odds (finish market vs. decision market)
    into a single de-vigged probability that the fight finishes.
    """
    raw_finish = _american_to_prob(finish_odds)
    raw_decision = _american_to_prob(decision_odds)
    overround = raw_finish + raw_decision
    return raw_finish / overround


def _error(error_code: str, error_message: str) -> dict[str, Any]:
    return {
        "bet": "error",
        "error_code": error_code,
        "error_message": error_message,
        "selection": None,
        "confidence": None,
        "tier": None,
        "eligible": None,
        "probabilities": None,
        "method_probabilities": None,
        "history": None,
        "market": None,
        "fight_number": None,
    }


def _shape_success(raw: dict[str, Any], fight_number: int) -> dict[str, Any]:
    binary = raw["binary"]
    method = raw["method"]
    market = binary["market"]
    return {
        "bet": bool(market["actionable"]),
        "error_code": None,
        "error_message": None,
        "selection": binary["selection"],
        "confidence": binary["confidence"],
        "tier": binary["tier"],
        "eligible": binary["eligible"],
        "probabilities": {
            "finish": binary["probabilities"]["finish"],
            "decision": binary["probabilities"]["decision"],
        },
        "method_probabilities": {
            "decision": method["decision"],
            "ko_tko": method["ko_tko"],
            "submission": method["submission"],
        },
        "history": {
            "fighter_a_prior": raw["history"]["fighter_a_prior"],
            "fighter_b_prior": raw["history"]["fighter_b_prior"],
        },
        "market": {
            "available": market["available"],
            "selected_probability": market["selected_probability"],
            "edge": market["edge"],
            "actionable": market["actionable"],
        },
        "fight_number": fight_number,
    }


def run_finish_prediction(
    fighter_a: Optional[str],
    fighter_b: Optional[str],
    *,
    fight_date,
    weight_class: Optional[str],
    fight_number: Optional[int] = None,
    market_finish_probability: Optional[float] = None,
    timeout: int = SUBPROCESS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the protected finish/decision model for one matchup.

    Never substitutes a probability and never suppresses a failure: any
    problem running the model returns ``{"bet": "error", "error_code": ...}``.
    """
    if not fighter_a or not fighter_b:
        return _error("missing_input", "Both fighter names are required.")
    if fight_date is None:
        return _error("missing_input", "fight_date is required for point-in-time feature extraction.")
    if not weight_class:
        return _error("missing_input", "weight_class is required.")

    resolved_fight_number = fight_number if fight_number is not None else DEFAULT_FIGHT_NUMBER

    if not SKILL_CLI.exists():
        return _error(
            "skill_not_installed",
            f"ufc_decision_skill CLI not found at {SKILL_CLI}. "
            "Install/bootstrap it per ufc_decision_skill/README.md.",
        )

    date_str = fight_date.isoformat() if hasattr(fight_date, "isoformat") else str(fight_date)

    cmd = [
        str(SKILL_CLI),
        str(fighter_a),
        str(fighter_b),
        "--date", date_str,
        "--weight-class", str(weight_class),
        "--fight-number", str(resolved_fight_number),
    ]
    if market_finish_probability is not None:
        cmd += ["--market-finish-probability", str(market_finish_probability)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return _error(
            "skill_not_installed",
            f"ufc_decision_skill CLI not executable at {SKILL_CLI}.",
        )
    except subprocess.TimeoutExpired:
        return _error(
            "timeout",
            f"ufc_decision_skill did not respond within {timeout}s.",
        )

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        if "fighter not found" in stderr.lower():
            return _error("fighter_not_found", stderr.splitlines()[-1] if stderr else "Fighter not found.")
        return _error(
            "subprocess_error",
            stderr.splitlines()[-1] if stderr else f"ufc_decision_skill exited with code {proc.returncode}.",
        )

    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return _error("invalid_output", f"ufc_decision_skill returned non-JSON output: {exc}")

    try:
        return _shape_success(raw, resolved_fight_number)
    except (KeyError, TypeError) as exc:
        return _error("invalid_output", f"ufc_decision_skill returned an unexpected shape: {exc}")
