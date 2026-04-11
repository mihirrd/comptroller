"""Clone a SWE-bench task repository at ``base_commit`` (no Docker)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def github_clone_url(repo: str) -> str:
    r = repo.strip()
    if "/" not in r or ".." in r or r.startswith("/"):
        raise ValueError(f"invalid SWE-bench repo field: {repo!r}")
    return f"https://github.com/{r}.git"


def clone_instance_workspace(instance: dict[str, Any], dest: Path) -> None:
    """Clone ``https://github.com/{repo}`` and check out ``base_commit``.

    Tries a partial clone + shallow fetch of the commit first; falls back to a full
    clone + checkout if the server does not allow fetching that object shallowly.

    Parameters
    ----------
    instance:
        Dataset row with ``repo`` (``owner/name``) and ``base_commit`` (sha).
    dest:
        Target directory; must not exist (removed on partial failure before fallback).
    """
    repo = instance["repo"]
    commit = str(instance["base_commit"]).strip()
    url = github_clone_url(str(repo))
    dest = dest.resolve()
    if dest.exists():
        raise FileExistsError(f"clone destination already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _run(args: list[str]) -> None:
        subprocess.run(args, check=True, capture_output=True, text=True)

    try:
        _run(
            [
                "git",
                "-c",
                "advice.detachedHead=false",
                "clone",
                "--quiet",
                "--filter=blob:none",
                "--no-checkout",
                url,
                str(dest),
            ]
        )
        _run(["git", "-C", str(dest), "fetch", "--quiet", "--depth", "1", "origin", commit])
        _run(["git", "-C", str(dest), "checkout", "--quiet", "FETCH_HEAD"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        _run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet", url, str(dest)])
        _run(["git", "-C", str(dest), "checkout", "--quiet", commit])
