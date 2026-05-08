import json
import os
import tempfile
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

from comptroller.graph import (
    MAX_TOOL_MSG_CHARS,
    RECENT_FILES_CONTEXT_MSG_ID,
    _find_tool_call,
    _prepare_messages_for_llm,
    _state_messages_with_sliding_window,
    _trim_messages_window,
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


def test_budget_gate_awaiting_budget_routes_to_end():
    state: AgentState = {
        "session_id": "test-123",
        "task": "t",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="hi"), AIMessage(content="done")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 5000,
        "api_dollars_used": 0.0,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "awaiting_budget",
        "summary": None,
    }
    assert budget_gate(state) == "end"


def test_after_tools_gate_awaiting_budget_routes_to_end():
    state: AgentState = {
        "session_id": "test-123",
        "task": "t",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="x")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 5000,
        "api_dollars_used": 0.0,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "awaiting_budget",
        "summary": None,
    }
    assert after_tools_gate(state) == "end"


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


def test_wall_budget_interactive_sets_awaiting_budget():
    """With interactive_budget, wall cap uses awaiting_budget instead of summarizing."""
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
        "interactive_budget": True,
    }
    mock_response = AIMessage(
        content="Short",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
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

    assert result["status"] == "awaiting_budget"
    assert budget_gate(result) == "end"


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


def test_token_budget_interactive_sets_awaiting_budget():
    """With interactive_budget, token cap uses awaiting_budget instead of summarizing."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "test task",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="test task")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 1,
        "api_dollars_used": 0.0,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "running",
        "summary": None,
        "interactive_budget": True,
    }
    mock_response = AIMessage(content="x")
    with patch("comptroller.graph.ChatLiteLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        result = agent_node(state)

    assert result["status"] == "awaiting_budget"
    assert budget_gate(result) == "end"


def test_token_budget_interactive_prefers_hold_over_local_degradation():
    """interactive_budget takes precedence over automatic local model fallback."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "test task",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="test task")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 1,
        "api_dollars_used": 0.0,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "running",
        "summary": None,
        "local_model_url": "http://127.0.0.1:11434/v1",
        "local_model_id": "llama3.2",
        "model_degraded": False,
        "interactive_budget": True,
    }
    mock_response = AIMessage(content="x")
    with patch("comptroller.graph.ChatLiteLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        result = agent_node(state)

    assert result["status"] == "awaiting_budget"
    assert result.get("model_degraded") is not True


def test_token_budget_exceeded_with_local_fallback_degrades_model():
    """When token budget is exceeded and local fallback is set, degrade and keep the same turn running."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "test task",
        "model": "gpt-4o",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="test task")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 1,
        "api_dollars_used": 0.0,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "running",
        "summary": None,
        "local_model_url": "http://127.0.0.1:11434/v1",
        "local_model_id": "llama3.2",
        "model_degraded": False,
    }
    mock_response = AIMessage(content="x")
    with patch("comptroller.graph.ChatLiteLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        result = agent_node(state)

    assert result["status"] == "running"
    assert result.get("model_degraded") is True
    assert result["model"] == "llama3.2"
    assert result["local_model_url"] == "http://127.0.0.1:11434/v1"
    assert result.get("continue_after_degrade") is True
    assert budget_gate(result) == "continue"


def test_degraded_mode_does_not_accrue_tokens_or_dollars():
    """Once degraded, LLM usage is treated as free for budget counters."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "continue on fallback",
        "model": "openai/llama3.2",
        "workspace_root": "/tmp/test-ws",
        "messages": [HumanMessage(content="test task")],
        "tool_results": [],
        "tokens_used": 123,
        "max_tokens": 5000,
        "api_dollars_used": 0.42,
        "max_api_dollars": None,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": None,
        "session_retries_used": 0,
        "max_session_retries": None,
        "status": "running",
        "summary": None,
        "local_model_url": "http://127.0.0.1:11434/v1",
        "local_model_id": "llama3.2",
        "model_degraded": True,
    }
    mock_response = AIMessage(
        content="fallback answer",
        usage_metadata={"input_tokens": 120, "output_tokens": 40, "total_tokens": 160},
        response_metadata={"model_name": "openai/llama3.2"},
    )
    with patch("comptroller.graph.ChatLiteLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        result = agent_node(state)

    assert result["tokens_used"] == state["tokens_used"]
    assert result["api_dollars_used"] == state["api_dollars_used"]


def test_degraded_mode_uses_cli_or_env_local_api_key():
    """Fallback ChatLiteLLM init includes local API key for remote gateways."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "continue on fallback",
        "model": "openai/llama3.2",
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
        "summary": None,
        "local_model_url": "https://example.gateway/v1",
        "local_model_id": "gpt-4o-mini",
        "local_model_api_key": "remote-gateway-key",
        "model_degraded": True,
    }
    mock_response = AIMessage(content="ok")
    with patch("comptroller.graph.ChatLiteLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        agent_node(state)

    kwargs = mock_llm_class.call_args.kwargs
    assert kwargs["api_base"] == "https://example.gateway/v1"
    assert kwargs["api_key"] == "remote-gateway-key"


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


def test_find_tool_call_parses_openai_style_function_arguments():
    """Providers may send OpenAI-style ``function`` + JSON ``arguments``; paths must still resolve."""
    m = AIMessage.model_construct(
        content="",
        tool_calls=[
            {
                "id": "c-json",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps(
                        {
                            "path": "/tmp/x.py",
                            "content": "print(1)",
                        }
                    ),
                },
            }
        ],
    )
    name, args = _find_tool_call([m], "c-json")
    assert name == "write_file"
    assert args.get("path") == "/tmp/x.py"


def test_prepare_messages_truncates_long_tool_output():
    """Very long ToolMessage content is trimmed only in the copy passed to the LLM."""
    long_body = "x" * (MAX_TOOL_MSG_CHARS + 500)
    tm = ToolMessage(content=long_body, tool_call_id="t1", name="read_file")
    msgs = [HumanMessage(content="h"), tm]
    out = _prepare_messages_for_llm(msgs)
    assert isinstance(out[1], ToolMessage)
    assert len(out[1].content) < len(long_body)
    assert "truncated" in out[1].content


def test_trim_messages_window_keeps_system_and_tail():
    sys = SystemMessage(content="sys")
    tail = [HumanMessage(content=str(i)) for i in range(30)]
    msgs = [sys] + tail
    with patch.dict(os.environ, {"COMPTROLLER_MAX_MESSAGE_WINDOW": "10"}):
        out = _trim_messages_window(msgs)
    assert len(out) == 11
    assert isinstance(out[0], SystemMessage)
    assert out[1].content == "20"
    assert out[-1].content == "29"


def test_trim_messages_window_extends_back_over_leading_tool_message():
    sys = SystemMessage(content="sys")
    rest = (
        [HumanMessage(content=str(i)) for i in range(25)]
        + [
            AIMessage(
                content="",
                tool_calls=[{"name": "list_directory", "args": {"path": "."}, "id": "a"}],
            ),
            ToolMessage(content="ok", tool_call_id="a"),
        ]
    )
    msgs = [sys] + rest
    with patch("comptroller.graph._max_message_window", return_value=2):
        out = _trim_messages_window(msgs)
    assert isinstance(out[1], AIMessage)
    assert isinstance(out[2], ToolMessage)


def test_state_messages_with_sliding_window_emits_remove_when_trimmed():
    sys = SystemMessage(content="s")
    tail = [HumanMessage(content=str(i)) for i in range(30)]
    full = [sys] + tail
    with patch.dict(os.environ, {"COMPTROLLER_MAX_MESSAGE_WINDOW": "10"}):
        out = _state_messages_with_sliding_window(full)
    assert isinstance(out[0], RemoveMessage)
    assert len(out) == 12


def test_agent_node_inserts_recent_files_context_for_llm():
    state: AgentState = {
        "session_id": "test-123",
        "task": "do something",
        "model": "gpt-4o",
        "workspace_root": "/tmp/ws",
        "messages": [HumanMessage(content="do something")],
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
        "recent_files": ["/tmp/ws/foo.py"],
    }
    mock_response = AIMessage(content="done")
    with patch("comptroller.graph.ChatLiteLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        agent_node(state)

    sent = mock_llm.bind_tools.return_value.invoke.call_args[0][0]
    ctx = next(m for m in sent if getattr(m, "id", None) == RECENT_FILES_CONTEXT_MSG_ID)
    assert "/tmp/ws/foo.py" in ctx.content
    assert "Recently touched files" in ctx.content


def test_agent_node_recent_files_derived_from_transcript_when_channel_empty():
    """When ``recent_files`` is missing from state but messages include read/write tools, show paths."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "do something",
        "model": "gpt-4o",
        "workspace_root": "/tmp/ws",
        "messages": [
            HumanMessage(content="do something"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "/tmp/ws/maze.py", "content": "x"},
                        "id": "t1",
                    }
                ],
            ),
            ToolMessage(
                content="ok",
                tool_call_id="t1",
                name="write_file",
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
    mock_response = AIMessage(content="done")
    with patch("comptroller.graph.ChatLiteLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        agent_node(state)

    sent = mock_llm.bind_tools.return_value.invoke.call_args[0][0]
    ctx = next(m for m in sent if getattr(m, "id", None) == RECENT_FILES_CONTEXT_MSG_ID)
    assert "maze.py" in ctx.content
    assert "(none recorded yet)" not in ctx.content
