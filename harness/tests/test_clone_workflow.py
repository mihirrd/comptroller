"""Tests for clone-workflow helpers and CLI wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

import comptroller_swebench.clone_workflow as cw


def test_find_instance_returns_matching_row() -> None:
    ds = [
        {"instance_id": "a", "repo": "org/r1"},
        {"instance_id": "b", "repo": "org/r2"},
    ]
    out = cw._find_instance(ds, "b")
    assert out["repo"] == "org/r2"


def test_find_instance_raises_for_missing_id() -> None:
    with pytest.raises(KeyError):
        cw._find_instance([{"instance_id": "x"}], "missing")


def test_run_clone_infer_cleanup_writes_jsonl_and_cleans_tmp(monkeypatch, tmp_path: Path) -> None:
    row = {"instance_id": "iid", "repo": "octo/repo", "base_commit": "abc123"}
    output = tmp_path / "preds.jsonl"
    scratch = tmp_path / "scratch"
    observed: dict[str, Path] = {}

    monkeypatch.setattr(cw, "load_dataset", lambda _n, split: [row])

    def fake_clone(instance, clone_root: Path) -> None:
        assert instance == row
        observed["clone_root"] = clone_root
        assert clone_root.parent.exists()

    def fake_run(instance, **kwargs):
        assert instance == row
        assert kwargs["workspace_root"] == str(observed["clone_root"])
        return (
            {
                "instance_id": "iid",
                "model_name_or_path": "m",
                "model_patch": "diff --git",
                "tokens_used": 3,
                "max_tokens": 10,
                "api_dollars_used": 0.0,
                "wall_seconds_used": 0.1,
            },
            "ignored",
        )

    monkeypatch.setattr(cw, "clone_instance_workspace", fake_clone)
    monkeypatch.setattr(cw, "run_single_instance", fake_run)

    pred = cw.run_clone_infer_cleanup(
        "iid",
        dataset_name="dummy/ds",
        split="test",
        output=output,
        model="gpt-4o",
        max_tokens=10,
        model_name_or_path="m",
        scratch_parent=scratch,
        db_parent=None,
        max_wall_seconds=5.0,
        termination_policy="strict_patch_fallback",
        local_model_url="https://example.local/v1",
        local_model_id="openai/gpt-oss:120b",
        local_model_api_key="k",
    )

    assert pred["instance_id"] == "iid"
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["model_patch"] == "diff --git"
    assert not observed["clone_root"].parent.exists()


def test_run_clone_infer_cleanup_cleans_tmp_on_failure(monkeypatch, tmp_path: Path) -> None:
    row = {"instance_id": "iid", "repo": "octo/repo", "base_commit": "abc123"}
    output = tmp_path / "preds.jsonl"
    observed: dict[str, Path] = {}

    monkeypatch.setattr(cw, "load_dataset", lambda _n, split: [row])

    def fake_clone(_instance, clone_root: Path) -> None:
        observed["tmp_root"] = clone_root.parent

    monkeypatch.setattr(cw, "clone_instance_workspace", fake_clone)
    monkeypatch.setattr(cw, "run_single_instance", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        cw.run_clone_infer_cleanup(
            "iid",
            dataset_name="dummy/ds",
            split="test",
            output=output,
            model="gpt-4o",
            max_tokens=10,
            model_name_or_path="m",
            scratch_parent=tmp_path / "scratch",
            db_parent=None,
            max_wall_seconds=None,
            termination_policy=None,
            local_model_url=None,
            local_model_id=None,
            local_model_api_key=None,
        )

    assert "tmp_root" in observed
    assert not observed["tmp_root"].exists()
    assert not output.exists()


def test_clone_run_writes_empty_prediction_on_unhandled_error(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"

    monkeypatch.setattr(cw, "load_dotenv", lambda: None)
    monkeypatch.setattr(cw, "resolve_max_wall_seconds", lambda _v: 12.345)
    monkeypatch.setattr(cw, "run_clone_infer_cleanup", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")))

    with pytest.raises(typer.Exit) as ei:
        cw.clone_run(
            instance_id="inst-1",
            output=output,
            dataset_name="dummy/ds",
            split="test",
            model="gpt-4o",
            max_tokens=123,
            model_name_or_path="custom-model",
            scratch_parent=None,
            db_parent=None,
            verbose=False,
            max_wall_seconds=None,
            local_model_url="https://example.local/v1",
            local_model="openai/gpt-oss:120b",
            local_model_api_key="k",
            termination_policy="strict_patch_fallback",
        )

    assert ei.value.exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert payload["instance_id"] == "inst-1"
    assert payload["model_name_or_path"] == "custom-model"
    assert payload["max_tokens"] == 123
    assert payload["model_patch"] == ""
    assert payload["max_wall_seconds"] == 12.345
    assert payload["termination_policy"] == "strict_patch_fallback"

