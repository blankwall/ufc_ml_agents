"""Exhaustive coverage of every skip code.

For each code, build a synthetic _evaluate_bet input that should trigger it,
plus a happy-path bet=True case. Catches accidental ordering changes that
let early checks shadow later ones.
"""
import json, sys, pytest
from pathlib import Path

# fastapi_app uses bare imports (`from services...`) so the CWD must be
# fastapi_app/ when its router modules import.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.routers.predict import _evaluate_bet, SKIP_REASONS  # noqa: E402

CFG = json.loads((_ROOT / "config/betting_config.json").read_text())
F   = CFG.get("filters", {})
W   = CFG.get("wmma_rules", {})

FAV_CONF   = F.get("favorite_confidence_min", 0.65)
UD_CONF    = F.get("underdog_confidence_min", 0.53)
FAV_CAP    = F.get("favorite_odds_cap", -300)
UD_CAP     = F.get("underdog_odds_cap", 300)
EDGE_MIN   = F.get("edge_min", 0.04)
UD_EDGE    = F.get("underdog_edge_min", EDGE_MIN)
WMMA_EDGE  = W.get("min_edge", 0.10)
MIN_FIGHTS = F.get("min_fights", 2)


# (code, kwargs) — each row designed to *just* trip its target code while
# clearing all earlier checks.
CASES = [
    # D1 fires first regardless of anything else
    ("D1", dict(pick_model_prob=0.80, pick_mkt_prob=0.50, pick_odds=-150,
                is_favorite=True, is_wmma=False,
                f1_count=MIN_FIGHTS - 1, f2_count=MIN_FIGHTS)),

    # W1 — wmma fight, edge below WMMA min but above general edge
    ("W1", dict(pick_model_prob=0.66, pick_mkt_prob=0.60, pick_odds=-150,
                is_favorite=True, is_wmma=True,
                f1_count=10, f2_count=10)),

    # F1 — favorite low confidence
    ("F1", dict(pick_model_prob=FAV_CONF - 0.05, pick_mkt_prob=FAV_CONF - 0.10,
                pick_odds=-150, is_favorite=True, is_wmma=False,
                f1_count=10, f2_count=10)),

    # F2 — fav over odds cap (more negative than cap)
    ("F2", dict(pick_model_prob=0.80, pick_mkt_prob=0.70, pick_odds=FAV_CAP - 50,
                is_favorite=True, is_wmma=False, f1_count=10, f2_count=10)),

    # F3 — fav meets confidence + cap, edge below floor
    ("F3", dict(pick_model_prob=FAV_CONF + 0.02,
                pick_mkt_prob=FAV_CONF + 0.02 - (EDGE_MIN - 0.01),
                pick_odds=-150, is_favorite=True, is_wmma=False,
                f1_count=10, f2_count=10)),

    # U1 — underdog low confidence
    ("U1", dict(pick_model_prob=UD_CONF - 0.05, pick_mkt_prob=UD_CONF - 0.10,
                pick_odds=200, is_favorite=False, is_wmma=False,
                f1_count=10, f2_count=10)),

    # U3 — underdog odds beyond cap
    ("U3", dict(pick_model_prob=UD_CONF + 0.05, pick_mkt_prob=UD_CONF - 0.10,
                pick_odds=UD_CAP + 50, is_favorite=False, is_wmma=False,
                f1_count=10, f2_count=10)),

    # U2 — underdog confidence ok, odds within cap, edge below floor
    ("U2", dict(pick_model_prob=UD_CONF + 0.02,
                pick_mkt_prob=UD_CONF + 0.02 - (UD_EDGE - 0.01),
                pick_odds=200, is_favorite=False, is_wmma=False,
                f1_count=10, f2_count=10)),
]


@pytest.mark.parametrize("expected_code,kwargs", CASES, ids=[c[0] for c in CASES])
def test_skip_code_fires(expected_code, kwargs):
    res = _evaluate_bet(**kwargs)
    assert res["bet"] is False, f"Expected bet=False for {expected_code}, got {res}"
    assert res["skip_code"] == expected_code, (
        f"Expected {expected_code}, got {res['skip_code']} ({res['skip_reason']}) "
        f"for inputs {kwargs}")
    assert res["skip_reason"] == SKIP_REASONS[expected_code]


def test_happy_path_favorite_bets():
    res = _evaluate_bet(
        pick_model_prob=FAV_CONF + 0.10,
        pick_mkt_prob=FAV_CONF + 0.10 - (EDGE_MIN + 0.05),
        pick_odds=-150, is_favorite=True, is_wmma=False,
        f1_count=10, f2_count=10,
    )
    assert res["bet"] is True
    assert res["skip_code"] is None


def test_happy_path_underdog_bets():
    res = _evaluate_bet(
        pick_model_prob=UD_CONF + 0.05,
        pick_mkt_prob=UD_CONF + 0.05 - (UD_EDGE + 0.05),
        pick_odds=200, is_favorite=False, is_wmma=False,
        f1_count=10, f2_count=10,
    )
    assert res["bet"] is True
    assert res["skip_code"] is None
