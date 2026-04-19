"""
Shared helpers for resolving model artifact paths.

New training runs store all artifacts under models/saved/{model_name}/.
Older runs used a flat layout (models/saved/{model_name}.json).
resolve_model_dir() checks the subdirectory first so new runs take
precedence, then falls back to the flat layout so old artifacts keep
working without any file migration.
"""

from pathlib import Path


def resolve_model_dir(base_dir: Path, model_name: str) -> Path:
    """Return the directory that contains artifacts for *model_name*.

    Checks ``base_dir/model_name/`` first (new per-run layout), falls back
    to ``base_dir/`` if the subdirectory does not contain the model JSON
    (legacy flat layout).
    """
    subdir = base_dir / model_name
    if (subdir / f"{model_name}.json").exists():
        return subdir
    return base_dir
