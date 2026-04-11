"""Collect ``model_patch`` via ``git diff`` from a prepared workspace."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def git_diff(repo: Path, *, timeout_sec: int = 300) -> str:
    """Return unified diff of working tree vs index/HEAD (no color, suitable for SWE-bench).

    Empty string if there are no changes or git fails (check logs).
    """
    root = repo.resolve()
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        proc = subprocess.run(
            [
                "git",
                "-c",
                "core.pager=cat",
                "diff",
                "--no-ext-diff",
                "--no-color",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.error("git diff failed to run in %s: %s", root, e)
        return ""

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        logger.warning("git diff exit %s in %s: %s", proc.returncode, root, err[:2000])
        return ""

    out = (proc.stdout or "").strip()
    return out
