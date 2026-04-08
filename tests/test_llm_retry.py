"""Retry/backoff around LiteLLM transient errors (graph._invoke_with_transient_retries)."""

import litellm
import pytest

import comptroller.graph as graph_mod


def test_invoke_with_transient_retries_increments_session_budget(monkeypatch):
    monkeypatch.setenv("COMPTROLLER_LLM_RETRY_ATTEMPTS", "5")
    monkeypatch.setattr(graph_mod.time, "sleep", lambda _s: None)

    budget = {"session_retries_used": 0, "max_session_retries": 10}
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise litellm.RateLimitError("rate limited", llm_provider="test", model="m")
        return "ok"

    assert (
        graph_mod._invoke_with_transient_retries(
            fn, context="test", retry_budget_state=budget
        )
        == "ok"
    )
    assert calls["n"] == 3
    assert budget["session_retries_used"] == 2


def test_invoke_with_transient_retries_session_budget_exhausted(monkeypatch):
    monkeypatch.setenv("COMPTROLLER_LLM_RETRY_ATTEMPTS", "5")
    monkeypatch.setattr(graph_mod.time, "sleep", lambda _s: None)

    budget = {"session_retries_used": 2, "max_session_retries": 2}

    def fn():
        raise litellm.RateLimitError("rate limited", llm_provider="test", model="m")

    with pytest.raises(graph_mod.SessionTransientRetryBudgetExhausted):
        graph_mod._invoke_with_transient_retries(
            fn, context="test", retry_budget_state=budget
        )


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
