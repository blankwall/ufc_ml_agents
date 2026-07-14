"""Unit tests for the runtime-writable fighter alias store.

Covers load/persist/upsert/remove, atomic file writes, in-place mutation of the
shared ALIASES dict (so importers like predict_service see live updates), and
input validation.
"""
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.services import fighter_alias_service as alias_service  # noqa: E402


@pytest.fixture
def isolated_store(monkeypatch, tmp_path):
    """Point the alias service at a temp file seeded with two entries.

    Restores the real on-disk aliases into the shared dict at teardown so other
    tests/modules that already imported ALIASES are unaffected.
    """
    alias_file = tmp_path / "fighter_aliases.json"
    alias_file.write_text(json.dumps({"Bobby Green": "King Green"}))
    monkeypatch.setattr(alias_service, "ALIAS_FILE", alias_file)
    alias_service.load_aliases()
    yield alias_file
    # teardown: restore real state
    monkeypatch.undo()
    alias_service.load_aliases()


def test_load_populates_shared_dict(isolated_store):
    assert alias_service.get_aliases() == {"Bobby Green": "King Green"}
    assert alias_service.ALIASES["Bobby Green"] == "King Green"


def test_load_mutates_in_place_same_object(isolated_store):
    before = alias_service.ALIASES  # identity
    alias_service.load_aliases()
    assert alias_service.ALIASES is before  # never reassigned


def test_predict_service_sees_live_updates(isolated_store):
    # predict_service.FIGHTER_ALIASES must be the SAME object the service mutates.
    from fastapi_app.services.predict_service import FIGHTER_ALIASES
    assert FIGHTER_ALIASES is alias_service.ALIASES
    alias_service.upsert_alias("RJ Harris", "Richard Harris")
    assert FIGHTER_ALIASES.get("RJ Harris") == "Richard Harris"


def test_upsert_adds_and_persists(isolated_store):
    result = alias_service.upsert_alias("RJ Harris", "Richard Harris")
    assert result == {"alias": "RJ Harris", "canonical": "Richard Harris"}
    # in memory
    assert alias_service.get_aliases()["RJ Harris"] == "Richard Harris"
    # on disk
    on_disk = json.loads(isolated_store.read_text())
    assert on_disk["RJ Harris"] == "Richard Harris"


def test_upsert_updates_existing(isolated_store):
    alias_service.upsert_alias("Bobby Green", "Robert Green")
    assert alias_service.get_aliases()["Bobby Green"] == "Robert Green"


def test_upsert_strips_whitespace(isolated_store):
    alias_service.upsert_alias("  Spacey Name  ", "  Canonical Name  ")
    aliases = alias_service.get_aliases()
    assert "Spacey Name" in aliases
    assert aliases["Spacey Name"] == "Canonical Name"


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("", "Canonical"),
        ("Alias", ""),
        ("   ", "Canonical"),
        ("Same", "Same"),
    ],
)
def test_upsert_rejects_invalid(isolated_store, alias, canonical):
    with pytest.raises(ValueError):
        alias_service.upsert_alias(alias, canonical)


def test_remove_existing_returns_true_and_persists(isolated_store):
    assert alias_service.remove_alias("Bobby Green") is True
    assert "Bobby Green" not in alias_service.get_aliases()
    assert json.loads(isolated_store.read_text()) == {}


def test_remove_missing_returns_false(isolated_store):
    assert alias_service.remove_alias("Nonexistent Person") is False


def test_get_returns_copy_not_live_ref(isolated_store):
    snapshot = alias_service.get_aliases()
    snapshot["Injected"] = "Should Not Persist"
    assert "Injected" not in alias_service.ALIASES


def test_load_tolerates_missing_file(monkeypatch, tmp_path):
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(alias_service, "ALIAS_FILE", missing)
    alias_service.load_aliases()
    assert alias_service.get_aliases() == {}
    monkeypatch.undo()
    alias_service.load_aliases()


def test_load_tolerates_corrupt_json(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    monkeypatch.setattr(alias_service, "ALIAS_FILE", bad)
    alias_service.load_aliases()
    assert alias_service.get_aliases() == {}
    monkeypatch.undo()
    alias_service.load_aliases()


def test_persist_is_atomic_no_tmp_left_behind(isolated_store):
    alias_service.upsert_alias("A One", "B Two")
    tmp = isolated_store.with_suffix(isolated_store.suffix + ".tmp")
    assert not tmp.exists()


def test_seed_file_matches_repo_config():
    """The committed seed file must be valid JSON with the known entries."""
    seed = _ROOT / "config" / "fighter_aliases.json"
    data = json.loads(seed.read_text())
    assert isinstance(data, dict)
    assert data.get("Bobby Green") == "King Green"
    assert len(data) >= 20
