import tempfile
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from comptroller.graph import (
    MAX_TOOL_MSG_CHARS,
    _prepare_messages_for_llm,
    after_tools_gate,
    agent_node,
    budget_gate,
    summarize_node,
    tool_node_wrapper,
)
from comptroller.budget import dollars_from_llm_message
from comptroller.state import AgentState


def test_agent_node_no_tool_calls():
    """With mocked LLM that returns no tool calls, graph should reach END with status complete."""
    # Create initial state
    state: AgentState = {
        "session_id": "test-123",
        "task": "test task",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="test task")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 5000,
        "api_dollars_used": 0.0,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "running",
        "summary": None
    }

    # Mock LLM response with no tool calls
    mock_response = AIMessage(content="Task completed successfully")

    with patch("comptroller.graph.ChatLiteLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        # Run agent node
        result = agent_node(state)

        # Verify status is still running (not summarizing)
        assert result["status"] == "running"
        assert result["tokens_used"] > 0

        # Check budget gate routing
        gate_result = budget_gate(result)
        assert gate_result == "end"


def test_agent_node_with_tool_calls():
    """With mocked LLM that returns tool calls, graph routes through tool node."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "list files",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="list files")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 5000,
        "api_dollars_used": 0.0,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "running",
        "summary": None
    }

    # Mock LLM response with tool calls
    mock_response = AIMessage(
        content="I'll list the files",
        tool_calls=[{
            "name": "list_directory",
            "args": {"path": "."},
            "id": "call_1"
        }]
    )

    with patch("comptroller.graph.ChatLiteLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        # Run agent node
        result = agent_node(state)

        # Check budget gate routing
        gate_result = budget_gate(result)
        assert gate_result == "tools"


def test_after_tools_gate_routes_to_continue_not_end():
    """After tools, last message is ToolMessage — must loop to agent, not END (regression)."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "t",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": "list_directory", "args": {"path": "."}, "id": "1"}],
            ),
            ToolMessage(content="ok", tool_call_id="1"),
        ],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 5000,
        "api_dollars_used": 0.0,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "running",
        "summary": None,
    }
    assert after_tools_gate(state) == "continue"
    # budget_gate would wrongly return "end" here — do not use it after tools
    assert budget_gate(state) == "end"


def test_after_tools_gate_respects_summarizing():
    """When budget hit during tools, route to summarize instead of agent."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "t",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="x")],
        "tool_results": [],
        "tokens_used": 100,
        "max_tokens": 50,
        "api_dollars_used": 0.0,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "summarizing",
        "summary": None,
    }
    assert after_tools_gate(state) == "summarize"


def test_wall_budget_exceeded_triggers_summarize():
    """When max_wall_seconds is exceeded after the agent step, route to summarizing."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "test task",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="test task")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 5000,
        "api_dollars_used": 0.0,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": 1.0,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "running",
        "summary": None,
    }
    mock_response = AIMessage(
        content="Short",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
        response_metadata={"model_name": "gpt-4o"},
    )
    with (
        patch("comptroller.graph.ChatLiteLLM") as mock_llm_class,
        patch("comptroller.graph.time.monotonic", side_effect=[0.0, 5.0]),
    ):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        result = agent_node(state)

    assert result["status"] == "summarizing"
    assert result["wall_seconds_used"] >= state["max_wall_seconds"]
    assert budget_gate(result) == "summarize"


def test_dollar_budget_exceeded_triggers_summarize():
    """When max_api_dollars is already reached after this call, route to summarizing."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "test task",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="test task")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 5000,
        "api_dollars_used": 0.0,
        "max_api_dollars": 0.0001,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "running",
        "summary": None,
    }
    mock_response = AIMessage(
        content="Short",
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
        },
        response_metadata={"model_name": "gpt-4o"},
    )
    with patch("comptroller.graph.ChatLiteLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        result = agent_node(state)

    assert result["status"] == "summarizing"
    assert result["api_dollars_used"] >= state["max_api_dollars"]
    assert budget_gate(result) == "summarize"
    assert dollars_from_llm_message(mock_response, "gpt-4o") > 0


def test_budget_exceeded_triggers_summarize():
    """With token budget set to 1, graph routes to summarize node."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "test task",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="test task")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 1,  # Very low budget
        "api_dollars_used": 0.0,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "running",
        "summary": None
    }

    # Mock LLM response
    mock_response = AIMessage(content="This response will exceed the budget")

    with patch("comptroller.graph.ChatLiteLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        # Run agent node
        result = agent_node(state)

        # Status should be summarizing
        assert result["status"] == "summarizing"
        assert result["tokens_used"] >= result["max_tokens"]

        # Check budget gate routing
        gate_result = budget_gate(result)
        assert gate_result == "summarize"


def test_summarize_node_completes():
    """Summarize node sets status to complete with a summary."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "test task",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="test task")],
        "tool_results": [
            {"content": "tool result 1", "tokens": 10},
            {"content": "tool result 2", "tokens": 10}
        ],
        "tokens_used": 100,
        "max_tokens": 50,
        "api_dollars_used": 0.0,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "summarizing",
        "summary": None
    }

    # Mock LLM response for summary
    mock_summary = AIMessage(content="Task summary: completed some work")

    with patch("comptroller.graph.ChatLiteLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_summary
        mock_llm_class.return_value = mock_llm

        # Run summarize node
        result = summarize_node(state)

        # Status should be complete
        assert result["status"] == "complete"
        assert result["summary"] is not None
        assert len(result["summary"]) > 0


def test_tool_node_tracks_recent_files():
    """After read_file, resolved path appears in recent_files."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"hi")
        raw_path = f.name
    try:
        resolved = str(Path(raw_path).resolve())
        state: AgentState = {
            "session_id": "test-123",
            "task": "read a file",
            "model": "gpt-4o",
            "workspace_root": "/tmp/test-ws",
            "messages": [
                HumanMessage(content="read"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read_file", "args": {"path": raw_path}, "id": "c1"},
                    ],
                ),
            ],
            "tool_results": [],
            "tokens_used": 0,
            "max_tokens": 5000,
            "api_dollars_used": 0.0,
            "max_api_dollars": None,
            "wall_seconds_used": 0.0,
            "max_wall_seconds": None,
            "session_retries_used": 0,
            "max_session_retries": None,
            "status": "running",
            "summary": None,
        }
        tool_msg = ToolMessage(
            content="[1 lines]\nhi",
            tool_call_id="c1",
            name="read_file",
        )
        merged = state["messages"] + [tool_msg]
        mock_exec = MagicMock()
        mock_exec.invoke.return_value = {"messages": merged}

        with patch("comptroller.graph.ToolNode", return_value=mock_exec):
            out = tool_node_wrapper(state)

        assert resolved in out["recent_files"]
    finally:
        Path(raw_path).unlink(missing_ok=True)


def test_prepare_messages_truncates_long_tool_output():
    """Very long ToolMessage content is trimmed only in the copy passed to the LLM."""
    long_body = "x" * (MAX_TOOL_MSG_CHARS + 500)
    tm = ToolMessage(content=long_body, tool_call_id="t1", name="read_file")
    msgs = [HumanMessage(content="h"), tm]
    out = _prepare_messages_for_llm(msgs)
    assert isinstance(out[1], ToolMessage)
    assert len(out[1].content) < len(long_body)
    assert "truncated" in out[1].content
