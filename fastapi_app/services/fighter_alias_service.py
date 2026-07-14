"""
Fighter Alias Service
=====================
Runtime-writable store for fighter name aliases (odds-source name → canonical
DB name). Replaces the previously hardcoded FIGHTER_ALIASES dict in
predict_service.py so aliases can be added/updated at runtime (e.g. from the
ingest UI) and persisted to config/fighter_aliases.json.

The module-level ``ALIASES`` dict is the single shared instance. It is mutated
**in place** on every load/upsert/remove so that other modules which did
``from ...predict_service import FIGHTER_ALIASES`` keep seeing a live view.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ALIAS_FILE = ROOT_DIR / "config" / "fighter_aliases.json"

_lock = threading.RLock()

# Shared, mutated-in-place mapping: alias (as seen in a data source) → canonical DB name.
ALIASES: dict[str, str] = {}


def load_aliases() -> dict[str, str]:
    """(Re)load aliases from disk into the shared ALIASES dict (in place)."""
    with _lock:
        ALIASES.clear()
        if ALIAS_FILE.exists():
            try:
                data = json.loads(ALIAS_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            if isinstance(data, dict):
                for key, value in data.items():
                    k, v = str(key).strip(), str(value).strip()
                    if k and v:
                        ALIASES[k] = v
        return dict(ALIASES)


def get_aliases() -> dict[str, str]:
    """Return a copy of the current alias mapping."""
    with _lock:
        return dict(ALIASES)


def _persist() -> None:
    """Atomically write the current ALIASES to disk (sorted for stable diffs)."""
    ALIAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ALIAS_FILE.with_suffix(ALIAS_FILE.suffix + ".tmp")
    payload = json.dumps(ALIASES, indent=2, ensure_ascii=False, sort_keys=True)
    tmp_path.write_text(payload + "\n", encoding="utf-8")
    tmp_path.replace(ALIAS_FILE)


def upsert_alias(alias: str, canonical: str) -> dict[str, str]:
    """
    Add or update a single alias → canonical mapping and persist it.

    Raises ValueError if either side is empty or if the mapping is a self-loop.
    """
    alias = str(alias).strip()
    canonical = str(canonical).strip()
    if not alias or not canonical:
        raise ValueError("Both 'alias' and 'canonical' must be non-empty.")
    if alias == canonical:
        raise ValueError("Alias and canonical name must differ.")
    with _lock:
        ALIASES[alias] = canonical
        _persist()
    return {"alias": alias, "canonical": canonical}


def remove_alias(alias: str) -> bool:
    """Delete an alias. Returns True if it existed, False otherwise."""
    alias = str(alias).strip()
    with _lock:
        existed = alias in ALIASES
        if existed:
            del ALIASES[alias]
            _persist()
        return existed


# Populate on import so the shared dict is ready for callers.
load_aliases()
