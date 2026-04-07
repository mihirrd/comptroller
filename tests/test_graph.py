import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from comptroller.graph import after_tools_gate, agent_node, budget_gate, summarize_node
from comptroller.state import AgentState


def test_agent_node_no_tool_calls():
    """With mocked LLM that returns no tool calls, graph should reach END with status complete."""
    # Create initial state
    state: AgentState = {
        "session_id": "test-123",
        "task": "test task",
        "model": "gpt-4o",
        "messages": [HumanMessage(content="test task")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 5000,
        "status": "running",
        "summary": None
    }

    # Mock LLM response with no tool calls
    mock_response = AIMessage(content="Task completed successfully")

    with patch('graph.ChatLiteLLM') as mock_llm_class:
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
        "messages": [HumanMessage(content="list files")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 5000,
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

    with patch('graph.ChatLiteLLM') as mock_llm_class:
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
        "messages": [HumanMessage(content="x")],
        "tool_results": [],
        "tokens_used": 100,
        "max_tokens": 50,
        "status": "summarizing",
        "summary": None,
    }
    assert after_tools_gate(state) == "summarize"


def test_budget_exceeded_triggers_summarize():
    """With token budget set to 1, graph routes to summarize node."""
    state: AgentState = {
        "session_id": "test-123",
        "task": "test task",
        "model": "gpt-4o",
        "messages": [HumanMessage(content="test task")],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": 1,  # Very low budget
        "status": "running",
        "summary": None
    }

    # Mock LLM response
    mock_response = AIMessage(content="This response will exceed the budget")

    with patch('graph.ChatLiteLLM') as mock_llm_class:
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
        "messages": [HumanMessage(content="test task")],
        "tool_results": [
            {"content": "tool result 1", "tokens": 10},
            {"content": "tool result 2", "tokens": 10}
        ],
        "tokens_used": 100,
        "max_tokens": 50,
        "status": "summarizing",
        "summary": None
    }

    # Mock LLM response for summary
    mock_summary = AIMessage(content="Task summary: completed some work")

    with patch('graph.ChatLiteLLM') as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_summary
        mock_llm_class.return_value = mock_llm

        # Run summarize node
        result = summarize_node(state)

        # Status should be complete
        assert result["status"] == "complete"
        assert result["summary"] is not None
        assert len(result["summary"]) > 0
