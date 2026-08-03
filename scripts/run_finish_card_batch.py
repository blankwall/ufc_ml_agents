#!/usr/bin/env python3
"""Run one card through ufc_decision_skill with date-level memoization.

This script is executed with the protected skill's own Python interpreter. It
still calls ``ufc_decision_skill.inference.predict`` for every matchup; the only
optimization is retaining immutable data/frame/state work shared by fights on
the same card inside this short-lived process.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path


def _emit(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def main() -> int:
    skill_root = Path(
        os.environ.get("UFC_DECISION_SKILL_ROOT", Path.home() / "ufc_decision_skill")
    ).expanduser()
    sys.path.insert(0, str(skill_root / "src"))

    from ufc_decision_skill import features, inference

    features.load_rows = lru_cache(maxsize=1)(features.load_rows)
    inference.prepare_frame = lru_cache(maxsize=1)(inference.prepare_frame)
    inference.build_states = lru_cache(maxsize=4)(inference.build_states)

    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError) as exc:
        _emit({"type": "fatal", "error_code": "invalid_request", "error_message": str(exc)})
        return 2

    fights = request.get("fights")
    if not isinstance(fights, list):
        _emit({
            "type": "fatal",
            "error_code": "invalid_request",
            "error_message": "fights must be a list",
        })
        return 2

    for index, fight in enumerate(fights):
        try:
            result = inference.predict(
                str(fight["fighter1"]),
                str(fight["fighter2"]),
                date=str(fight["fight_date"]),
                weight_class=str(fight["weight_class"]),
                fight_number=int(fight["fight_number"]),
                market_finish_probability=fight.get("market_finish_probability"),
            )
            _emit({"type": "result", "index": index, "result": result})
        except ValueError as exc:
            message = str(exc)
            code = "fighter_not_found" if "fighter not found" in message.lower() else "model_error"
            _emit({
                "type": "error",
                "index": index,
                "error_code": code,
                "error_message": message,
            })
        except Exception as exc:  # noqa: BLE001 - each fight must report an explicit failure
            _emit({
                "type": "error",
                "index": index,
                "error_code": "model_error",
                "error_message": f"{type(exc).__name__}: {exc}",
            })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
