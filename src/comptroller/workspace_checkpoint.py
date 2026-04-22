"""Git-backed workspace snapshots for rollback (detached refs, no branch movement)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_GIT_TIMEOUT_SEC = 120
_REF_PREFIX = "refs/comptroller"


def is_git_workspace(root: Path) -> bool:
    """True if ``root`` contains a ``.git`` directory (or gitfile)."""
    p = root.expanduser().resolve()
    if (p / ".git").exists():
        return True
    return False


def _run_git(
    root: Path,
    args: list[str],
    *,
    input_text: str | None = None,
    timeout: int = _GIT_TIMEOUT_SEC,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
    )


def ref_name_for_checkpoint(session_id: str, seq: int) -> str:
    """Detached ref path under ``refs/comptroller/``."""
    safe = session_id.replace("/", "_")
    return f"{_REF_PREFIX}/{safe}/{seq}"


def create_git_checkpoint(
    root: Path,
    session_id: str,
    seq: int,
    message: str = "comptroller checkpoint",
) -> tuple[str | None, str | None, str | None]:
    """Stage working tree, create a dangling commit, update ref, reset index.

    Returns ``(commit_sha, parent_sha, error)``. ``error`` is set on failure;
    ``commit_sha`` and ``parent_sha`` are None when skipped or failed.
    """
    root = root.expanduser().resolve()
    if not is_git_workspace(root):
        return None, None, "not a git workspace"

    # Dirty check: porcelain
    st = _run_git(root, ["status", "--porcelain"])
    if st.returncode != 0:
        return None, None, (st.stderr or st.stdout or "git status failed").strip()

    parent_sha: str | None = None
    head = _run_git(root, ["rev-parse", "--verify", "HEAD"])
    if head.returncode == 0:
        parent_sha = head.stdout.strip()

    add = _run_git(root, ["add", "-A"])
    if add.returncode != 0:
        return None, None, (add.stderr or add.stdout or "git add failed").strip()

    wt = _run_git(root, ["write-tree"])
    if wt.returncode != 0:
        _run_git(root, ["reset", "HEAD"])  # best-effort restore index
        return None, None, (wt.stderr or wt.stdout or "git write-tree failed").strip()
    tree = wt.stdout.strip()

    if parent_sha:
        ct = _run_git(
            root,
            ["commit-tree", tree, "-p", parent_sha, "-m", message],
        )
    else:
        ct = _run_git(root, ["commit-tree", tree, "-m", message])
    if ct.returncode != 0:
        _run_git(root, ["reset", "HEAD"])
        return None, None, (ct.stderr or ct.stdout or "git commit-tree failed").strip()
    commit_sha = ct.stdout.strip()

    ref = ref_name_for_checkpoint(session_id, seq)
    up = _run_git(root, ["update-ref", ref, commit_sha])
    if up.returncode != 0:
        _run_git(root, ["reset", "HEAD"])
        return None, None, (up.stderr or up.stdout or "git update-ref failed").strip()

    reset = _run_git(root, ["reset", "HEAD"])
    if reset.returncode != 0:
        logger.warning("git reset HEAD failed after checkpoint: %s", reset.stderr)

    return commit_sha, parent_sha, None


def diff_commits(root: Path, parent_sha: str | None, commit_sha: str) -> str:
    """Unified diff from ``parent_sha`` (or empty tree) to ``commit_sha``."""
    root = root.expanduser().resolve()
    if parent_sha:
        proc = _run_git(root, ["diff", "--no-color", parent_sha, commit_sha])
    else:
        proc = _run_git(root, ["show", "--no-color", "--format=", commit_sha])
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout or "(diff failed)").strip()
    return proc.stdout or ""


def restore_worktree(
    root: Path,
    commit_sha: str,
    mode: Literal["restore", "hard"] = "restore",
) -> str | None:
    """Restore tracked files under ``root`` to match ``commit_sha``.

    ``restore`` uses ``git restore --source=... --worktree --staged .``.
    ``hard`` uses ``git reset --hard`` to ``commit_sha`` (moves HEAD).
    Returns an error string or ``None`` on success.
    """
    root = root.expanduser().resolve()
    if not is_git_workspace(root):
        return "not a git workspace"

    if mode == "hard":
        proc = _run_git(root, ["reset", "--hard", commit_sha])
        if proc.returncode != 0:
            return (proc.stderr or proc.stdout or "git reset --hard failed").strip()
        return None

    proc = _run_git(
        root,
        ["restore", "--source", commit_sha, "--worktree", "--staged", "."],
    )
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout or "git restore failed").strip()
    return None
