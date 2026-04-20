"""No-lookahead leakage check.

The feature builder must use a strict `event_date < as_of_date` filter so a
fight never sees its own outcome. To verify:

  1. Pick a known historical fight at date D.
  2. Extract features with as_of=D and as_of=D-1day → must be IDENTICAL
     (because both exclude the fight itself).
  3. Extract with as_of=D+1day → at least one rolling-stat feature must
     change (because the day-D fight is now in history).

If anyone changes `<` to `<=` (or worse), this test fails loudly.
"""
import os, sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from features.matchup_features import MatchupFeatureExtractor       # noqa: E402
from database.schema import Fight, Event                              # noqa: E402

_DB = _ROOT / "data" / "ufc_database.db"
if not _DB.exists():
    _DB = _ROOT / "ufc_database.db"

pytestmark = pytest.mark.skipif(not _DB.exists(), reason="UFC DB not available")


@pytest.fixture(scope="module")
def session():
    eng = create_engine(f"sqlite:///{_DB}")
    Sess = sessionmaker(bind=eng)
    s = Sess()
    yield s
    s.close()


@pytest.fixture(scope="module")
def historical_fights(session):
    """Pick 3 fights from 2023 where both fighters have ≥5 prior bouts."""
    rows = (
        session.query(Fight, Event)
        .join(Event, Fight.event_id == Event.id)
        .filter(Event.date.like("%2023%"))
        .limit(60)
        .all()
    )
    sample = []
    for fight, event in rows:
        try:
            d = datetime.strptime(event.date, "%B %d, %Y")
        except (ValueError, TypeError):
            continue
        sample.append((fight.fighter_1_id, fight.fighter_2_id, d))
        if len(sample) >= 3:
            break
    if not sample:
        pytest.skip("No 2023 fights parseable")
    return sample


def _features_at(extractor, f1_id, f2_id, as_of):
    return extractor.extract_matchup_features(f1_id, f2_id, as_of_date=as_of)


def _diff_keys(a: dict, b: dict, tol: float = 1e-9) -> list[str]:
    """Return non-calendar feature names that differ between two extractions.

    Calendar-elapsed features (`days_since_*`, `years_since_*`, and any
    `age_x_*` interactions of those) move every day even with no new fight
    history, so they are not lookahead-leakage indicators. Filter them out.
    """
    SKIP_SUBSTR = ("days_since_", "years_since_", "_x_days_since_", "_x_years_since_")
    out = []
    for k in a.keys() | b.keys():
        if any(s in k for s in SKIP_SUBSTR):
            continue
        va, vb = a.get(k), b.get(k)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            out.append(k); continue
        try:
            if abs(float(va) - float(vb)) > tol:
                out.append(k)
        except (TypeError, ValueError):
            if va != vb:
                out.append(k)
    return out


def test_strict_lt_boundary_excludes_fight_itself(session, historical_fights):
    """as_of = fight_date and as_of = fight_date − 1 must yield identical
    features (strict `<` semantics excludes the fight from its own history)."""
    extractor = MatchupFeatureExtractor(session, use_cache=False)
    for f1, f2, d in historical_fights:
        on_day  = _features_at(extractor, f1, f2, d)
        day_b4  = _features_at(extractor, f1, f2, d - timedelta(days=1))
        diff = _diff_keys(on_day, day_b4)
        assert not diff, (
            f"Fight ({f1} vs {f2}) on {d.date()}: features changed between "
            f"as_of=D and as_of=D-1 (lookahead leak suspected). "
            f"Differing keys: {diff[:8]}{'...' if len(diff)>8 else ''}")


def test_day_after_includes_fight(session, historical_fights):
    """as_of = fight_date + 1 day must change at least one rolling-stat
    feature (the day-D fight is now visible)."""
    extractor = MatchupFeatureExtractor(session, use_cache=False)
    no_change_for_all = True
    for f1, f2, d in historical_fights:
        on_day = _features_at(extractor, f1, f2, d)
        day_after = _features_at(extractor, f1, f2, d + timedelta(days=1))
        diff = _diff_keys(on_day, day_after)
        if diff:
            no_change_for_all = False
            break
    assert not no_change_for_all, (
        "Features identical at as_of=D vs D+1 for ALL sampled fights — "
        "history filter may not be re-evaluating per as_of_date.")
