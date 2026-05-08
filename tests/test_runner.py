"""Tests for the importable :mod:`comptroller.runner` API."""

from collections import UserDict
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import comptroller.runner as runner_mod
from comptroller.runner import (
    AgentTurnInput,
    _build_turn_state,
    recent_files_from_checkpointer,
    run_agent_turn,
)


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
    mock_snap = MagicMock()
    mock_snap.config = {"configurable": {}}
    mock_graph.get_state.return_value = mock_snap

    inp = AgentTurnInput(task="hello", session_id="s1", max_tokens=1000)
    out = run_agent_turn(inp, db_path=":memory:", graph=mock_graph, log_steps_to_store=False)

    assert out.final_state is None
    assert out.streaming_token_budget_hit is False
    mock_graph.stream.assert_called_once()
    # rehydrate recent_files (skipped when list empty) + _read_langgraph_checkpoint
    assert mock_graph.get_state.call_count == 2


def test_run_agent_turn_rehydrates_recent_files_from_checkpointer():
    """When input omits recent_files, copy non-empty list from get_state before stream."""
    mock_graph = MagicMock()
    mock_graph.stream.return_value = iter([])

    snap_for_files = MagicMock()
    snap_for_files.config = {"configurable": {"thread_id": "s1"}}
    snap_for_files.values = {"recent_files": ["/a.py"]}
    snap_for_lg = MagicMock()
    snap_for_lg.config = {"configurable": {"checkpoint_id": "ck1", "checkpoint_ns": ""}}
    mock_graph.get_state.side_effect = [snap_for_files, snap_for_lg]

    inp = AgentTurnInput(task="hello", session_id="s1", max_tokens=1000)
    run_agent_turn(inp, db_path=":memory:", graph=mock_graph, log_steps_to_store=False)

    stream_args = mock_graph.stream.call_args[0]
    assert stream_args[0].get("recent_files") == ["/a.py"]


def test_recent_files_from_checkpointer_accepts_non_dict_mapping():
    """``snap.values`` can be a Mapping that is not a ``dict`` (e.g. some LangGraph builds)."""
    mock_graph = MagicMock()
    snap = MagicMock()
    snap.values = UserDict(recent_files=["/m.py"])
    mock_graph.get_state.return_value = snap
    out = recent_files_from_checkpointer(
        mock_graph, {"configurable": {"thread_id": "s1"}}
    )
    assert out == ["/m.py"]


def test_run_agent_turn_rehydrates_when_input_empty_list():
    """Empty ``recent_files`` from the caller is treated as unknown; checkpoint can still supply paths."""
    mock_graph = MagicMock()
    mock_graph.stream.return_value = iter([])

    snap_for_files = MagicMock()
    snap_for_files.config = {"configurable": {"thread_id": "s1"}}
    snap_for_files.values = UserDict(recent_files=["/from_ckpt.py"])
    snap_for_lg = MagicMock()
    snap_for_lg.config = {"configurable": {"checkpoint_id": "ck1", "checkpoint_ns": ""}}
    mock_graph.get_state.side_effect = [snap_for_files, snap_for_lg]

    inp = AgentTurnInput(task="hello", session_id="s1", max_tokens=1000, recent_files=[])
    run_agent_turn(inp, db_path=":memory:", graph=mock_graph, log_steps_to_store=False)

    stream_args = mock_graph.stream.call_args[0]
    assert stream_args[0].get("recent_files") == ["/from_ckpt.py"]


def test_run_agent_turn_empty_recent_files_in_checkpoint_does_not_inject():
    """Explicit empty checkpoint list does not set turn_state (no second source of truth)."""
    mock_graph = MagicMock()
    mock_graph.stream.return_value = iter([])

    snap_for_files = MagicMock()
    snap_for_files.config = {"configurable": {"thread_id": "s1"}}
    snap_for_files.values = {"recent_files": []}
    snap_for_lg = MagicMock()
    snap_for_lg.config = {"configurable": {"checkpoint_id": "ck1", "checkpoint_ns": ""}}
    mock_graph.get_state.side_effect = [snap_for_files, snap_for_lg]

    inp = AgentTurnInput(task="hello", session_id="s1", max_tokens=1000)
    run_agent_turn(inp, db_path=":memory:", graph=mock_graph, log_steps_to_store=False)

    stream_args = mock_graph.stream.call_args[0]
    assert "recent_files" not in stream_args[0]


def test_run_agent_turn_passes_resume_checkpoint_config():
    mock_graph = MagicMock()
    mock_graph.stream.return_value = iter([])
    mock_snap = MagicMock()
    mock_snap.config = {"configurable": {}}
    mock_graph.get_state.return_value = mock_snap

    inp = AgentTurnInput(
        task="hello",
        session_id="s1",
        max_tokens=1000,
        resume_langgraph_checkpoint_id="ckpt-1",
        resume_langgraph_checkpoint_ns="",
    )
    run_agent_turn(inp, db_path=":memory:", graph=mock_graph, log_steps_to_store=False)

    config = mock_graph.stream.call_args[0][1]
    assert config["configurable"]["thread_id"] == "s1"
    assert config["configurable"]["checkpoint_id"] == "ckpt-1"
    assert config["configurable"]["checkpoint_ns"] == ""


def test_run_agent_turn_streaming_hit_forces_summarize(monkeypatch):
    mock_graph = MagicMock()
    msg_chunk = SimpleNamespace(content="x")
    mock_graph.stream.return_value = iter(
        [
            ("messages", (msg_chunk, {"langgraph_node": "agent"})),
        ]
    )
    mock_snap = MagicMock()
    mock_snap.config = {"configurable": {"checkpoint_id": "ck1", "checkpoint_ns": ""}}
    mock_graph.get_state.return_value = mock_snap

    summary_state = {
        "status": "complete",
        "summary": "s",
        "tokens_used": 42,
        "api_dollars_used": 0.1,
        "wall_seconds_used": 0.2,
        "session_retries_used": 0,
    }
    monkeypatch.setattr(runner_mod, "summarize_node", lambda _state: summary_state)

    inp = AgentTurnInput(task="hello", session_id="s1", max_tokens=1)
    out = run_agent_turn(inp, db_path=":memory:", graph=mock_graph, log_steps_to_store=False)

    assert out.streaming_token_budget_hit is True
    assert out.final_state == summary_state
    assert out.current_tokens == 42
    assert out.langgraph_checkpoint_id == "ck1"


def test_run_agent_turn_callbacks_and_logging(monkeypatch):
    mock_graph = MagicMock()
    ai = AIMessage(content="answer")
    tool = ToolMessage(content="done", tool_call_id="c1", name="read_file")
    ai_with_call = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "read_file", "args": {"path": "x.py"}}],
    )
    mock_graph.stream.return_value = iter(
        [
            ("updates", {"agent": {"messages": [ai]}}),
            ("updates", {"tools": {"messages": [ai_with_call, tool]}}),
            (
                "updates",
                {
                    "summarize": {
                        "status": "complete",
                        "tokens_used": 9,
                        "api_dollars_used": 0.0,
                        "wall_seconds_used": 0.0,
                        "session_retries_used": 0,
                    }
                },
            ),
        ]
    )
    mock_snap = MagicMock()
    mock_snap.config = {"configurable": {"checkpoint_id": "ck1", "checkpoint_ns": ""}}
    mock_graph.get_state.return_value = mock_snap

    logged: list[tuple[str, int]] = []
    monkeypatch.setattr(
        runner_mod.store,
        "log_step",
        lambda _sid, step, event_type, *_rest: logged.append((event_type, step)),
    )
    monkeypatch.setattr(runner_mod, "tokens_from_llm_message", lambda _msg, _m: 5)
    monkeypatch.setattr(runner_mod, "count_tokens", lambda _s: 3)

    cb_hits = {"agent": 0, "tool": 0, "sum": 0}

    def _after_agent(_msg, _step, _est):
        cb_hits["agent"] += 1

    def _after_tool(_msg, _name, _inp, _content, _step, _est):
        cb_hits["tool"] += 1

    def _on_sum():
        cb_hits["sum"] += 1

    inp = AgentTurnInput(task="hello", session_id="s1", max_tokens=1000)
    run_agent_turn(
        inp,
        db_path=":memory:",
        graph=mock_graph,
        callbacks={
            "after_agent_llm": _after_agent,
            "after_tool": _after_tool,
            "on_summarize_shown": _on_sum,
        },
        log_steps_to_store=True,
    )

    assert cb_hits == {"agent": 1, "tool": 1, "sum": 1}
    assert logged == [("llm_response", 1), ("tool_call", 2)]
