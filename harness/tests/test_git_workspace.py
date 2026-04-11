"""Tests for git workspace clone helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from comptroller_swebench.git_workspace import clone_instance_workspace, github_clone_url


def test_github_clone_url() -> None:
    assert github_clone_url("astropy/astropy") == "https://github.com/astropy/astropy.git"


def test_github_clone_url_rejects_bad_repo() -> None:
    with pytest.raises(ValueError):
        github_clone_url("../evil")
    with pytest.raises(ValueError):
        github_clone_url("nonslash")


def test_clone_instance_workspace_shallow_path(tmp_path: Path) -> None:
    dest = tmp_path / "repo"
    instance = {"repo": "octocat/Hello-World", "base_commit": "7fd1a60b01f91b314f59955a4e4d4e80d8edf11d"}
    with patch("comptroller_swebench.git_workspace.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        clone_instance_workspace(instance, dest)
    assert run.call_count == 3
    args_lists = [c.args[0] for c in run.call_args_list]
    assert args_lists[0][0] == "git" and "clone" in args_lists[0]
    assert args_lists[1][:4] == ["git", "-C", str(dest), "fetch"]
    assert args_lists[2][:4] == ["git", "-C", str(dest), "checkout"]


def test_clone_instance_workspace_fallback_on_fetch_failure(tmp_path: Path) -> None:
    dest = tmp_path / "repo"
    instance = {"repo": "octocat/Hello-World", "base_commit": "7fd1a60b01f91b314f59955a4e4d4e80d8edf11d"}

    def fake_run(args, **kwargs):
        if args[1] == "clone" and "--no-checkout" in args:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args=args, returncode=0)

    with patch("comptroller_swebench.git_workspace.subprocess.run", side_effect=fake_run):
        clone_instance_workspace(instance, dest)
