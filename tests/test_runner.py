"""Tests for the importable :mod:`comptroller.runner` API."""

from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from comptroller.runner import AgentTurnInput, _build_turn_state, run_agent_turn


def test_build_turn_state_includes_recent_files_when_set():
    inp = AgentTurnInput(
        task="t",
        session_id="abc",
        max_tokens=500,
        recent_files=["/tmp/a.py"],
    )
    st = _build_turn_state(inp)
    assert st["recent_files"] == ["/tmp/a.py"]
    assert isinstance(st["messages"][0], HumanMessage)
    assert st["messages"][0].content == "t"


def test_build_turn_state_omits_recent_files_when_none():
    inp = AgentTurnInput(task="t", session_id="abc", max_tokens=500)
    st = _build_turn_state(inp)
    assert "recent_files" not in st


def test_build_turn_state_omits_messages_when_append_disabled():
    inp = AgentTurnInput(
        task="t",
        session_id="abc",
        max_tokens=500,
        append_user_message=False,
    )
    st = _build_turn_state(inp)
    assert "messages" not in st
    assert st.get("interactive_budget") is False


def test_run_agent_turn_no_updates_returns_none_state():
    mock_graph = MagicMock()
    mock_graph.stream.return_value = iter([])

    inp = AgentTurnInput(task="hello", session_id="s1", max_tokens=1000)
    out = run_agent_turn(inp, db_path=":memory:", graph=mock_graph, log_steps_to_store=False)

    assert out.final_state is None
    assert out.streaming_token_budget_hit is False
    mock_graph.stream.assert_called_once()
