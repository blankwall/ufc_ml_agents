from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> object:
    """
    Read and parse JSON file, with better error messages and basic cleanup for common agent mistakes.
    """
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to fix common issues: + signs in numbers (e.g., +0.123 -> 0.123)
        import re
        # Match patterns like ": +0.123" or ", +0.123" and remove the +
        # This handles the common case where agents write +0.0322 in JSON
        fixed = re.sub(r'([:,]\s*)\+(\d+\.?\d*)', r'\1\2', text)
        if fixed != text:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
        
        # Show helpful error context
        lines = text.split('\n')
        error_line = lines[e.lineno - 1] if e.lineno <= len(lines) else ""
        raise RuntimeError(
            f"Invalid JSON in {path} at line {e.lineno}, column {e.colno}:\n"
            f"  {error_line}\n"
            f"  {' ' * (e.colno - 1)}^\n"
            f"  Error: {e.msg}\n"
            f"  Hint: Check for trailing commas, unquoted strings, or invalid number formats (e.g., +0.123 should be 0.123)"
        ) from e


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    if cwd is not None:
        cwd.mkdir(parents=True, exist_ok=True)
    final_env = os.environ.copy()
    if env:
        final_env.update(env)

    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(stdout_path, "w", encoding="utf-8")
    else:
        out_f = subprocess.PIPE

    if stderr_path:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        err_f = open(stderr_path, "w", encoding="utf-8")
    else:
        err_f = subprocess.PIPE

    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            env=final_env,
            text=True,
            stdout=out_f,
            stderr=err_f,
            check=False,
        )
    finally:
        if stdout_path:
            out_f.close()
        if stderr_path:
            err_f.close()

    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def backup_paths(paths: Iterable[Path], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if not p.exists():
            continue
        rel = p.as_posix().lstrip("/")
        dest = backup_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if p.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(p, dest)
        else:
            shutil.copy2(p, dest)


def restore_paths(backup_dir: Path, repo_root: Path) -> None:
    """
    Restore all files/directories found under backup_dir back into repo_root.
    """
    for item in backup_dir.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(backup_dir)
        dest = repo_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dest)

    # Restore directories that were copied as directories (copytree will include files above)
    # For simplicity, we only overwrite files; callers should include all relevant directories/files.


def newest_matching(path: Path, glob_pat: str) -> Optional[Path]:
    files = list(path.glob(glob_pat))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


