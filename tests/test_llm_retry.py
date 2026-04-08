"""Retry/backoff around LiteLLM transient errors (graph._invoke_with_transient_retries)."""

import litellm
import pytest

import graph as graph_mod


def test_invoke_with_transient_retries_succeeds_after_rate_limits(monkeypatch):
    monkeypatch.setenv("COMPTROLLER_LLM_RETRY_ATTEMPTS", "5")
    monkeypatch.setattr(graph_mod.time, "sleep", lambda _s: None)

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise litellm.RateLimitError("rate limited", llm_provider="test", model="m")
        return "ok"

    assert graph_mod._invoke_with_transient_retries(fn, context="test") == "ok"
    assert calls["n"] == 3


def test_invoke_with_transient_retries_raises_after_exhausted(monkeypatch):
    monkeypatch.setenv("COMPTROLLER_LLM_RETRY_ATTEMPTS", "2")
    monkeypatch.setattr(graph_mod.time, "sleep", lambda _s: None)

    def fn():
        raise litellm.RateLimitError("rate limited", llm_provider="test", model="m")

    with pytest.raises(litellm.RateLimitError):
        graph_mod._invoke_with_transient_retries(fn, context="test")
