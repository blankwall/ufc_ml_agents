"""Bet-sizing buckets — config invariants & boundary correctness.

Mirrors the JS rule in events.js:108-138:
    edgeFrac in [b.min_edge, b.max_edge)  → multiplier or 'skip'
    WMMA cap: multiplier = min(multiplier, wmma.max_multiplier)
    WMMA floor: edgeFrac < wmma.min_edge → null

These tests pin the spec the user verbally agreed on:
    0–5%   skip
    5–10%  1.0×
    10–20% 1.5×
    20%+   2.0×
"""
import json, math
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CFG  = json.loads((_ROOT / "config/betting_config.json").read_text())


# ─── tiny Python port of the JS bucket loop ──────────────────────────────────

def py_multiplier(edge_frac: float, is_wmma: bool = False) -> float | None:
    mult = None
    for b in _CFG.get("edge_buckets", []):
        if b["min_edge"] <= edge_frac < b["max_edge"]:
            if b.get("action") == "skip":
                mult = None
            else:
                mult = b.get("multiplier")
            break
    wmma = _CFG.get("wmma_rules", {})
    if is_wmma and wmma.get("enabled"):
        if edge_frac < wmma.get("min_edge", 0.10):
            mult = None
        elif mult is not None:
            mult = min(mult, wmma.get("max_multiplier", 1.0))
    return mult


# ─── config invariants ───────────────────────────────────────────────────────

def test_buckets_cover_zero_to_one_continuous():
    """Edge buckets must cover [0, 1.0] with no gaps and no overlap."""
    buckets = sorted(_CFG["edge_buckets"], key=lambda b: b["min_edge"])
    assert buckets[0]["min_edge"] == 0.0,    "first bucket must start at 0"
    assert buckets[-1]["max_edge"] >= 1.0,   "last bucket must reach 1.0"
    for prev, nxt in zip(buckets, buckets[1:]):
        assert math.isclose(prev["max_edge"], nxt["min_edge"]), (
            f"gap or overlap between {prev} and {nxt}")


def test_documented_spec_matches_config():
    """Pin the specific bucket boundaries the user agreed on."""
    expected = [
        (0.00, 0.05, "skip", None),
        (0.05, 0.10, None,   1.0),
        (0.10, 0.20, None,   1.5),
        (0.20, 1.00, None,   2.0),
    ]
    actual = [(b["min_edge"], b["max_edge"], b.get("action"), b.get("multiplier"))
              for b in _CFG["edge_buckets"]]
    assert actual == expected, f"Edge buckets drifted from spec.\n  got: {actual}\n  want: {expected}"


# ─── boundary tests (parametrised) ───────────────────────────────────────────

# (edge_pct, expected_multiplier_for_non_wmma)
BOUNDARIES = [
    (0.0,    None),
    (0.0499, None),     # just below 5% → skip
    (0.05,   1.0),      # exactly 5% → 1.0× (min_edge inclusive)
    (0.0799, 1.0),
    (0.0999, 1.0),
    (0.10,   1.5),      # exactly 10% → tier up
    (0.1999, 1.5),
    (0.20,   2.0),      # exactly 20% → top tier
    (0.50,   2.0),
]


@pytest.mark.parametrize("edge,expected", BOUNDARIES,
                         ids=[f"edge={e:.4f}" for e, _ in BOUNDARIES])
def test_non_wmma_multiplier(edge, expected):
    assert py_multiplier(edge, is_wmma=False) == expected


def test_wmma_floor_blocks_below_10pct():
    for e in [0.0, 0.04, 0.05, 0.099]:
        assert py_multiplier(e, is_wmma=True) is None, f"WMMA edge {e} should skip"


def test_wmma_cap_at_1x():
    for e in [0.10, 0.15, 0.20, 0.50]:
        assert py_multiplier(e, is_wmma=True) == 1.0, (
            f"WMMA edge {e} must cap at 1.0× (got {py_multiplier(e, True)})")
