import os
import sqlite3
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from tools import TOOLS
from budget import count_tokens


def agent_node(state: AgentState) -> AgentState:
    """Agent node - calls LLM with tools bound."""
    # Build messages list
    messages = state["messages"].copy()

    # Add system message if this is the first call
    if len(messages) == 0 or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(content=f"You are a helpful assistant. Complete this task: {state['task']}")
        messages.insert(0, system_msg)

    # Initialize LLM with tools
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    # Call LLM
    response = llm_with_tools.invoke(messages)

    # Count tokens in response
    response_text = response.content if hasattr(response, 'content') else str(response)
    tokens = count_tokens(response_text)

    # Update state
    new_state = state.copy()
    new_state["tokens_used"] = state["tokens_used"] + tokens
    new_state["messages"] = state["messages"] + [response]

    # Check if budget exceeded
    if new_state["tokens_used"] >= state["max_tokens"]:
        new_state["status"] = "summarizing"

    return new_state


def tool_node_wrapper(state: AgentState) -> AgentState:
    """Wrapper around ToolNode to track tokens."""
    # Create the tool node
    tool_executor = ToolNode(TOOLS)

    # Execute tools (this only returns updated messages)
    tool_result = tool_executor.invoke(state)

    # Start with a copy of the original state to preserve all fields
    new_state = state.copy()

    # Update messages from tool execution
    new_state["messages"] = tool_result["messages"]

    # Count tokens in tool results
    tool_messages = [msg for msg in tool_result["messages"] if msg not in state["messages"]]
    total_tokens = 0

    for msg in tool_messages:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        tokens = count_tokens(content)
        total_tokens += tokens

        # Log tool result
        result_entry = {
            "content": content,
            "tokens": tokens
        }
        new_state["tool_results"] = new_state.get("tool_results", []) + [result_entry]

    # Update token count
    new_state["tokens_used"] = state["tokens_used"] + total_tokens

    # Check if budget exceeded
    if new_state["tokens_used"] >= state["max_tokens"]:
        new_state["status"] = "summarizing"

    return new_state


def budget_gate(state: AgentState) -> str:
    """Budget gate - decides next step based on state."""
    if state["status"] == "summarizing":
        return "summarize"

    if not state["messages"]:
        return "end"

    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return "end"


def summarize_node(state: AgentState) -> AgentState:
    """Summarize node - generates final summary."""
    # Build summary prompt
    tool_results_text = "\n".join([
        f"- {result.get('content', '')[:100]}..."
        for result in state.get("tool_results", [])
    ])

    summary_prompt = f"""The task was: {state['task']}

Tool calls made:
{tool_results_text if tool_results_text else 'No tools were called'}

Please provide a brief summary of what was accomplished and what remains to be done."""

    # Call LLM for summary (does not check budget)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = llm.invoke([HumanMessage(content=summary_prompt)])

    summary = response.content if hasattr(response, 'content') else str(response)

    # Update state
    new_state = state.copy()
    new_state["status"] = "complete"
    new_state["summary"] = summary

    return new_state


def build_graph(db_path: str = "~/.agent-runtime-mvp/sessions.db"):
    """Build and compile the LangGraph."""
    # Expand path
    db_path = str(Path(db_path).expanduser())

    # Create SQLite connection and checkpointer
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # Build graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node_wrapper)
    workflow.add_node("summarize", summarize_node)

    # Add edges
    workflow.set_entry_point("agent")

    # From agent, go through budget gate
    workflow.add_conditional_edges(
        "agent",
        budget_gate,
        {
            "tools": "tools",
            "summarize": "summarize",
            "end": END
        }
    )

    # From tools, go through budget gate
    workflow.add_conditional_edges(
        "tools",
        budget_gate,
        {
            "tools": "agent",  # Loop back to agent
            "summarize": "summarize",
            "end": END
        }
    )

    # Summarize always goes to END
    workflow.add_edge("summarize", END)

    # Compile
    return workflow.compile(checkpointer=checkpointer)


def run_graph(task: str, max_tokens: int, session_id: str, db_path: str = "~/.agent-runtime-mvp/sessions.db"):
    """Run the graph and return final state."""
    graph = build_graph(db_path)

    # Initial state
    initial_state: AgentState = {
        "session_id": session_id,
        "task": task,
        "messages": [HumanMessage(content=task)],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": max_tokens,
        "status": "running",
        "summary": None
    }

    # Run graph
    config = {"configurable": {"thread_id": session_id}}
    final_state = None

    for state in graph.stream(initial_state, config):
        # Get the last state
        if isinstance(state, dict):
            # Extract state from the step result
            for node_name, node_state in state.items():
                final_state = node_state

    return final_state if final_state else initial_state
