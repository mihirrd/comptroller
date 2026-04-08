import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import tools as tools_mod
from tools import (
    WORKSPACE_ROOT_ENV,
    apply_patch,
    glob_files,
    grep,
    read_file,
    search_replace,
    set_workspace_root_for_tools,
)


@pytest.fixture(autouse=True)
def clear_workspace_env():
    os.environ.pop(WORKSPACE_ROOT_ENV, None)
    yield
    os.environ.pop(WORKSPACE_ROOT_ENV, None)


def test_search_replace_single_occurrence():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "f.txt"
        p.write_text("hello world\n")
        out = search_replace.invoke({"path": str(p), "old_string": "world", "new_string": "there"})
        assert "Successfully" in out
        assert p.read_text() == "hello there\n"


def test_search_replace_rejects_ambiguous():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "f.txt"
        p.write_text("aa aa\n")
        out = search_replace.invoke({"path": str(p), "old_string": "aa", "new_string": "b"})
        assert "Error" in out


def test_search_replace_replace_all():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "f.txt"
        p.write_text("aa aa\n")
        out = search_replace.invoke(
            {"path": str(p), "old_string": "aa", "new_string": "b", "replace_all": True}
        )
        assert "Successfully" in out
        assert p.read_text() == "b b\n"


def test_workspace_jail_blocks_escape():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "root"
        root.mkdir()
        outside = Path(d) / "secret.txt"
        outside.write_text("x")
        set_workspace_root_for_tools(str(root))
        out = read_file.invoke({"path": str(outside)})
        assert "escapes workspace_root" in out


def test_grep_finds_pattern():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.py"
        p.write_text("foo = 1\nbar = 2\n")
        out = grep.invoke({"pattern": r"foo", "path": str(d)})
        assert "foo" in out
        assert "a.py" in out or str(p.name) in out


def test_grep_uses_system_grep_when_ripgrep_unavailable(monkeypatch):
    """When rg is absent, grep tool should use system grep (if installed)."""
    real_which = shutil.which

    def fake_which(name: str):
        if name == "rg":
            return None
        return real_which(name)

    monkeypatch.setattr(tools_mod.shutil, "which", fake_which)
    if not shutil.which("grep"):
        pytest.skip("system grep not available")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.py"
        p.write_text("unique_marker_xyz\n")
        out = grep.invoke({"pattern": r"unique_marker_xyz", "path": str(d)})
        assert "unique_marker_xyz" in out


@pytest.mark.skipif(not shutil.which("git"), reason="git not installed")
def test_apply_patch_uses_git_apply():
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        init = subprocess.run(
            ["git", "init"],
            cwd=root,
            env=git_env,
            capture_output=True,
            text=True,
        )
        if init.returncode != 0:
            err = (init.stderr or "") + (init.stdout or "")
            if "Operation not permitted" in err or "not permitted" in err.lower():
                pytest.skip("git init blocked (e.g. sandbox .git/hooks)")
            init.check_returncode()
        (root / "hello.txt").write_text("old\n")
        subprocess.run(["git", "add", "hello.txt"], cwd=root, check=True, env=git_env, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=root,
            check=True,
            env=git_env,
            capture_output=True,
        )
        patch = """diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-old
+new
"""
        set_workspace_root_for_tools(str(root))
        out = apply_patch.invoke({"patch_text": patch})
        assert "Successfully" in out
        assert (root / "hello.txt").read_text() == "new\n"


@pytest.mark.skipif(not shutil.which("patch"), reason="patch not installed")
def test_apply_patch_uses_patch_when_no_git():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "f.txt").write_text("a\n")
        patch = """diff --git a/f.txt b/f.txt
--- a/f.txt
+++ b/f.txt
@@ -1 +1 @@
-a
+b
"""
        set_workspace_root_for_tools(str(root))
        out = apply_patch.invoke({"patch_text": patch})
        assert "Successfully" in out
        assert (root / "f.txt").read_text() == "b\n"


def test_glob_files_respects_pattern():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "one.py").write_text("x")
        (Path(d) / "two.txt").write_text("y")
        out = glob_files.invoke({"pattern": "*.py", "root_directory": str(d)})
        assert "one.py" in out
        assert "two.txt" not in out


def test_glob_files_finds_nested_basename():
    """Plain 'cart.py' should match cart/cart.py, not only the workspace root."""
    with tempfile.TemporaryDirectory() as d:
        sub = Path(d) / "cart"
        sub.mkdir()
        (sub / "cart.py").write_text("# cart\n")
        out = glob_files.invoke({"pattern": "cart.py", "root_directory": str(d)})
        assert "cart.py" in out
        assert str(sub / "cart.py") in out


def test_grep_no_match_hints_glob_for_filename_like_pattern():
    with tempfile.TemporaryDirectory() as d:
        out = grep.invoke({"pattern": "cart.py", "path": str(d)})
        assert "No matches" in out
        assert "glob_files" in out
