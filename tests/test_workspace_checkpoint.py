"""Git workspace checkpoint helpers (real git binary)."""

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest

from comptroller import workspace_checkpoint as wscp

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GIT_TMP_BASE = _REPO_ROOT / ".pytest_git_workspace"


def _git(repo: Path, *args: str) -> None:
    p = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr + p.stdout


@pytest.fixture
def git_repo() -> Path:
    """Repo under the project tree so sandboxed CI can write ``.git`` (not system tmp)."""
    _GIT_TMP_BASE.mkdir(parents=True, exist_ok=True)
    repo = Path(tempfile.mkdtemp(prefix="wc_", dir=str(_GIT_TMP_BASE)))
    tpl = repo.parent / f"_tpl_{uuid.uuid4().hex}"
    (tpl / "info").mkdir(parents=True)
    (tpl / "info" / "exclude").write_text("")
    (tpl / "description").write_text("test repository\n")
    (tpl / "HEAD").write_text("ref: refs/heads/main\n")
    try:
        p = subprocess.run(
            ["git", "init", "--template", str(tpl)],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if p.returncode != 0 and "not permitted" in (p.stderr or ""):
            shutil.rmtree(repo, ignore_errors=True)
            pytest.skip("git init blocked by sandbox; run tests with full permissions for git coverage")
        assert p.returncode == 0, p.stderr + p.stdout
        shutil.rmtree(tpl, ignore_errors=True)
        _git(repo, "config", "user.email", "test@test")
        _git(repo, "config", "user.name", "test")
        (repo / "a.txt").write_text("v1\n")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-m", "init")
        yield repo
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_is_git_workspace(git_repo: Path) -> None:
    assert wscp.is_git_workspace(git_repo) is True
    assert wscp.is_git_workspace(git_repo.parent) is False


def test_create_checkpoint_and_restore(git_repo: Path) -> None:
    (git_repo / "a.txt").write_text("v2\n")
    sha, parent, err = wscp.create_git_checkpoint(git_repo, "sess1", 1, message="cp1")
    assert err is None
    assert sha is not None
    assert parent is not None

    (git_repo / "a.txt").write_text("v3\n")
    err2 = wscp.restore_worktree(git_repo, sha, mode="restore")
    assert err2 is None
    assert (git_repo / "a.txt").read_text() == "v2\n"


def test_diff_commits(git_repo: Path) -> None:
    (git_repo / "a.txt").write_text("v2\n")
    sha, parent, err = wscp.create_git_checkpoint(git_repo, "s", 1)
    assert err is None
    diff = wscp.diff_commits(git_repo, parent, sha)
    assert "v1" in diff or "v2" in diff


def test_store_workspace_checkpoints_and_pending_resume() -> None:
    from comptroller import store

    _GIT_TMP_BASE.mkdir(parents=True, exist_ok=True)
    db = str(_GIT_TMP_BASE / f"chk_{uuid.uuid4().hex}.db")
    try:
        store.init_db(db)
        store.create_session("s1", "task", 100, db)
        assert store.next_workspace_checkpoint_seq("s1", db) == 1
        store.insert_workspace_checkpoint(
            "s1",
            1,
            "halt",
            "deadbeef",
            "cafe",
            "lc1",
            "",
            3,
            str(_REPO_ROOT),
            db,
        )
        rows = store.list_workspace_checkpoints("s1", db)
        assert len(rows) == 1
        assert rows[0]["git_commit_sha"] == "deadbeef"
        assert store.get_workspace_checkpoint("s1", 1, db)["seq"] == 1
        store.set_session_pending_resume("s1", "lc1", "", db)
        assert store.get_session_pending_resume("s1", db) == ("lc1", "")
        store.clear_session_pending_resume("s1", db)
        assert store.get_session_pending_resume("s1", db) == (None, None)
    finally:
        Path(db).unlink(missing_ok=True)
