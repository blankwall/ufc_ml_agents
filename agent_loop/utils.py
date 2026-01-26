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
    return json.loads(path.read_text(encoding="utf-8"))


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


