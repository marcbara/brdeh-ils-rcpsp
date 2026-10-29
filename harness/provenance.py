"""Provenance of a run: which solver commit and harness commit produced it."""

from __future__ import annotations

import subprocess
import warnings
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


@lru_cache(maxsize=None)
def submodule_info(path: str) -> dict:
    """Checked-out commit of a solver submodule, the commit the superproject
    pins, and whether the checkout is dirty. Warns on dirty or off-pin."""
    sub = REPO_ROOT / path
    commit = _git("rev-parse", "HEAD", cwd=sub)
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no", cwd=sub))
    pinned = _git("ls-tree", "HEAD", path, cwd=REPO_ROOT).split()[2]
    if dirty:
        warnings.warn(f"solver submodule {path} is DIRTY: results not attributable to {commit}")
    if commit != pinned:
        warnings.warn(f"solver submodule {path} at {commit} but standalone artifact pins {pinned}")
    return {"path": path, "commit": commit, "pinned_commit": pinned, "dirty": dirty}


@lru_cache(maxsize=None)
def harness_info() -> dict:
    commit = _git("rev-parse", "HEAD", cwd=REPO_ROOT)
    status = _git("status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all", cwd=REPO_ROOT)
    return {"commit": commit, "dirty": bool(status)}
