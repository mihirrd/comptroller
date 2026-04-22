"""Tests for SWE-bench run_instance helpers."""

from comptroller_swebench.run_instance import resolve_max_wall_seconds


def test_resolve_max_wall_seconds_explicit_positive():
    assert resolve_max_wall_seconds(120.0) == 120.0


def test_resolve_max_wall_seconds_explicit_zero_means_unbounded():
    assert resolve_max_wall_seconds(0.0) is None


def test_resolve_max_wall_seconds_explicit_negative_means_unbounded():
    assert resolve_max_wall_seconds(-1.0) is None


def test_resolve_max_wall_seconds_none_uses_env(monkeypatch):
    monkeypatch.setenv("COMPTROLLER_MAX_WALL_SECONDS", "90")
    assert resolve_max_wall_seconds(None) == 90.0
    monkeypatch.delenv("COMPTROLLER_MAX_WALL_SECONDS", raising=False)


def test_resolve_max_wall_seconds_env_fallback_second_name(monkeypatch):
    monkeypatch.delenv("COMPTROLLER_MAX_WALL_SECONDS", raising=False)
    monkeypatch.setenv("max_wall_seconds", "45")
    assert resolve_max_wall_seconds(None) == 45.0
    monkeypatch.delenv("max_wall_seconds", raising=False)


def test_resolve_max_wall_seconds_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("COMPTROLLER_MAX_WALL_SECONDS", "999")
    assert resolve_max_wall_seconds(10.0) == 10.0
    monkeypatch.delenv("COMPTROLLER_MAX_WALL_SECONDS", raising=False)
