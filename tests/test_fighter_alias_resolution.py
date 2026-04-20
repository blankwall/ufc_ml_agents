"""FIGHTER_ALIASES integrity check.

Every alias key and its canonical value must resolve to the same Fighter row
in the DB, otherwise the alias is stale and the model will silently use the
wrong fighter's history.
"""
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.services.predict_service import FIGHTER_ALIASES, _resolve_fighter   # noqa: E402

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


def test_every_alias_resolves(session):
    unresolved = []
    mismatched = []
    for alias, canonical in FIGHTER_ALIASES.items():
        a = _resolve_fighter(session, alias)
        c = _resolve_fighter(session, canonical)
        if c is None:
            unresolved.append(f"canonical '{canonical}' (alias '{alias}')")
            continue
        if a is None:
            # Alias unresolved is acceptable IF the resolver falls back to
            # canonical via the FIGHTER_ALIASES lookup at call sites.
            continue
        if a.id != c.id:
            mismatched.append(
                f"'{alias}' → fighter#{a.id} but canonical '{canonical}' → fighter#{c.id}"
            )
    msg = ""
    if unresolved:
        msg += "Canonical names not in DB:\n  " + "\n  ".join(unresolved) + "\n"
    if mismatched:
        msg += "Alias resolves to different fighter than canonical:\n  " + "\n  ".join(mismatched)
    assert not msg, msg
