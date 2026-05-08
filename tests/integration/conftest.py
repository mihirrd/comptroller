"""Shared integration fixtures for graph policy tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def llm_scenario(monkeypatch):
    """Patch ``comptroller.graph.ChatLiteLLM`` with configurable agent/summary responses."""
    state: dict[str, deque[AIMessage]] = {
        "agent": deque(),
        "summary": deque(),
    }

    class _FakeBoundLLM:
        def invoke(self, _messages):
            if state["agent"]:
                return state["agent"].popleft()
            return AIMessage(content="default-agent-response")

    class _FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def bind_tools(self, _tools):
            return _FakeBoundLLM()

        def invoke(self, _messages):
            if state["summary"]:
                return state["summary"].popleft()
            return AIMessage(content="default-summary-response")

    monkeypatch.setattr("comptroller.graph.ChatLiteLLM", _FakeLLM)

    def _configure(
        *,
        agent: Iterable[AIMessage] = (),
        summary: Iterable[AIMessage] = (),
    ) -> None:
        state["agent"] = deque(agent)
        state["summary"] = deque(summary)

    return _configure
