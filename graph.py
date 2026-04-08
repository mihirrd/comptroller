import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import litellm

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from state import AgentState
from tools import TOOLS, set_workspace_root_for_tools
from budget import count_tokens, tokens_from_llm_message
from prompt_builder import build_system_prompt

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_RECENT_FILES = 20
MAX_TOOL_MSG_CHARS = 12_000
MAX_TOOL_DIGEST_ITEMS = 8


def _truncate(s: str, max_len: int = 6000) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... <truncated {len(s) - max_len} chars>"


def _format_message_for_log(i: int, msg: Any) -> str:
    typ = type(msg).__name__
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        body = _truncate(str(content), 4000)
    else:
        body = _truncate(str(content or ""), 4000)
    lines = [f"  [{i}] {typ}: {body}"]
    if getattr(msg, "tool_calls", None):
        lines.append(f"       tool_calls: {_truncate(str(msg.tool_calls), 2000)}")
    ak = getattr(msg, "additional_kwargs", None) or {}
    if ak:
        lines.append(f"       additional_kwargs keys: {list(ak.keys())}")
    return "\n".join(lines)


def _format_messages_for_log(messages: list[Any]) -> str:
    return "\n".join(_format_message_for_log(i, m) for i, m in enumerate(messages))


def _format_ai_response_for_log(response: Any) -> str:
    lines = [_format_message_for_log(0, response)]
    return "\n".join(lines)


# Transient provider errors we backoff and retry (graph-level; avoids ChatLiteLLM default max_retries=1 = no retries).
_LLM_TRANSIENT_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.BadGatewayError,
)


def _llm_retry_attempts() -> int:
    raw = os.environ.get("COMPTROLLER_LLM_RETRY_ATTEMPTS", "8")
    try:
        n = int(raw)
    except ValueError:
        n = 8
    return max(1, n)


def _llm_backoff_seconds(attempt_index: int) -> float:
    """Exponential backoff: base * 2^attempt, capped (attempt 0 = first retry wait)."""
    try:
        base = float(os.environ.get("COMPTROLLER_LLM_BACKOFF_BASE_SEC", "2.0"))
    except ValueError:
        base = 2.0
    try:
        cap = float(os.environ.get("COMPTROLLER_LLM_BACKOFF_MAX_SEC", "120.0"))
    except ValueError:
        cap = 120.0
    delay = base * (2**attempt_index)
    return min(cap, delay)


def _invoke_with_transient_retries(fn: Callable[[], T], *, context: str) -> T:
    """Run ``fn``; on rate limits and common transient API errors, sleep with exponential backoff and retry."""
    attempts = _llm_retry_attempts()
    for attempt in range(attempts):
        try:
            return fn()
        except _LLM_TRANSIENT_EXCEPTIONS as e:
            if attempt >= attempts - 1:
                logger.error("%s: failed after %s attempts: %s", context, attempts, e)
                raise
            delay = _llm_backoff_seconds(attempt)
            logger.warning(
                "%s: %s (attempt %s/%s); sleeping %.1fs then retrying",
                context,
                type(e).__name__,
                attempt + 1,
                attempts,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError("_invoke_with_transient_retries: unreachable")


def _chat(model: str, temperature: float = 0) -> ChatLiteLLM:
    """LiteLLM-backed chat model; `model` uses LiteLLM naming (provider prefixes optional).

    ``max_retries=1`` so LangChain's inner tenacity does not stack with
    :func:`_invoke_with_transient_retries` (which applies backoff for rate limits and 5xx).
    """
    return ChatLiteLLM(model=model, temperature=temperature, max_retries=1)


def _merge_recent_files(existing: list[str] | None, new_paths: list[str]) -> list[str]:
    """Keep order by recency (last touch wins); cap length."""
    combined = list(existing or [])
    for p in new_paths:
        if p in combined:
            combined.remove(p)
        combined.append(p)
    return combined[-MAX_RECENT_FILES:]


def _find_tool_call(msgs: list[Any], tool_call_id: str) -> tuple[str | None, dict[str, Any]]:
    for msg in reversed(msgs):
        if not isinstance(msg, AIMessage):
            continue
        tcs = getattr(msg, "tool_calls", None) or []
        for tc in tcs:
            if tc.get("id") == tool_call_id:
                args = tc.get("args") or {}
                if not isinstance(args, dict):
                    args = {}
                return tc.get("name"), args
    return None, {}


def _paths_from_patch_text(patch_text: str) -> list[str]:
    """Extract target file paths from a unified diff (+++ b/... lines)."""
    out: list[str] = []
    for line in patch_text.splitlines():
        if not line.startswith("+++ "):
            continue
        rest = line[4:].strip()
        if "dev/null" in rest:
            continue
        path_part = rest.split("\t", 1)[0].strip()
        for prefix in ("b/", "a/"):
            if path_part.startswith(prefix):
                path_part = path_part[len(prefix) :]
                break
        if path_part and path_part != "/dev/null":
            out.append(path_part)
    return out


def _paths_from_tool_messages(
    state: AgentState,
    msgs: list[Any],
    tool_messages: list[ToolMessage],
) -> list[str]:
    paths: list[str] = []
    wr = Path(state["workspace_root"]).expanduser().resolve()
    for tm in tool_messages:
        tid = getattr(tm, "tool_call_id", None)
        if not tid:
            continue
        name, args = _find_tool_call(msgs, tid)
        if name == "apply_patch":
            for p in _paths_from_patch_text(args.get("patch_text") or ""):
                try:
                    if os.path.isabs(p):
                        paths.append(str(Path(p).expanduser().resolve()))
                    else:
                        paths.append(str((wr / p).resolve()))
                except Exception:
                    paths.append(p)
            continue
        if name not in ("read_file", "write_file", "search_replace"):
            continue
        raw = args.get("path")
        if not raw:
            continue
        try:
            paths.append(str(Path(raw).expanduser().resolve()))
        except Exception:
            paths.append(str(raw))
    return paths


def _compact_tool_digest(msgs: list[Any]) -> str:
    """Short bullet list of recent tool calls + result preview (for system prompt only)."""
    lines: list[str] = []
    for msg in reversed(msgs):
        if not isinstance(msg, ToolMessage):
            continue
        tid = getattr(msg, "tool_call_id", None)
        name, args = _find_tool_call(msgs, tid)
        if not name:
            name = getattr(msg, "name", None) or "tool"
        arg_preview = ""
        if isinstance(args, dict) and args:
            keys = list(args)[:2]
            arg_preview = str({k: args[k] for k in keys})
            if len(arg_preview) > 120:
                arg_preview = arg_preview[:117] + "..."
        body = msg.content if isinstance(msg.content, str) else str(msg.content)
        preview = body.replace("\n", " ")[:160]
        if len(body) > 160:
            preview += "..."
        lines.append(f"- `{name}` {arg_preview} → {preview}")
        if len(lines) >= MAX_TOOL_DIGEST_ITEMS:
            break
    return "\n".join(reversed(lines))


def _prepare_messages_for_llm(messages: list[Any]) -> list[Any]:
    """Truncate very long tool outputs in a copy for the LLM only (state keeps full content)."""
    out: list[Any] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content
            if isinstance(content, str) and len(content) > MAX_TOOL_MSG_CHARS:
                trimmed = content[:MAX_TOOL_MSG_CHARS] + "\n... [tool output truncated for context]"
                out.append(
                    ToolMessage(
                        content=trimmed,
                        tool_call_id=m.tool_call_id,
                        name=getattr(m, "name", None),
                    )
                )
            else:
                out.append(m)
        else:
            out.append(m)
    return out


def agent_node(state: AgentState) -> AgentState:
    """Agent node - calls LLM with tools bound."""
    # Build messages list
    messages = state["messages"].copy()

    prompt_state = dict(state)
    prompt_state["tool_digest"] = _compact_tool_digest(state["messages"])

    # Add system message if this is the first call
    if len(messages) == 0 or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(content=build_system_prompt(prompt_state))
        messages.insert(0, system_msg)

    messages_for_llm = _prepare_messages_for_llm(messages)

    # Initialize LLM with tools
    llm = _chat(state["model"])
    llm_with_tools = llm.bind_tools(TOOLS)

    logger.info(
        "LLM request session_id=%s model=%s n_messages=%s",
        state["session_id"],
        state["model"],
        len(messages_for_llm),
    )
    logger.info("LLM request messages:\n%s", _format_messages_for_log(messages_for_llm))

    # Call LLM (truncated tool bodies only in this copy)
    response = _invoke_with_transient_retries(
        lambda: llm_with_tools.invoke(messages_for_llm),
        context=f"LLM agent session_id={state['session_id']}",
    )

    logger.info("LLM response session_id=%s", state["session_id"])
    logger.info("LLM response:\n%s", _format_ai_response_for_log(response))

    tokens = tokens_from_llm_message(response, state["model"])

    # Update state
    new_state = state.copy()
    new_state["tokens_used"] = state["tokens_used"] + tokens
    new_state["messages"] = state["messages"] + [response]

    logger.info(
        "LLM exchange session_id=%s model=%s tokens_this_call=%s tokens_used_total=%s max_tokens=%s",
        state["session_id"],
        state["model"],
        tokens,
        new_state["tokens_used"],
        state["max_tokens"],
    )

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

    logger.info(
        "Tool execution session_id=%s new_messages appended:\n%s",
        state["session_id"],
        _format_messages_for_log(
            [m for m in tool_result["messages"] if m not in state["messages"]]
        ),
    )

    # Start with a copy of the original state to preserve all fields
    new_state = state.copy()

    # Update messages from tool execution
    new_state["messages"] = tool_result["messages"]

    # Count tokens in tool results
    tool_messages = [msg for msg in tool_result["messages"] if msg not in state["messages"]]
    total_tokens = 0

    for msg in tool_messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        tokens = count_tokens(content)
        total_tokens += tokens

        # Log tool result
        result_entry = {
            "content": content,
            "tokens": tokens,
        }
        new_state["tool_results"] = new_state.get("tool_results", []) + [result_entry]

    # Update token count
    new_state["tokens_used"] = state["tokens_used"] + total_tokens

    # Check if budget exceeded
    if new_state["tokens_used"] >= state["max_tokens"]:
        new_state["status"] = "summarizing"

    tm_only = [m for m in tool_messages if isinstance(m, ToolMessage)]
    new_paths = _paths_from_tool_messages(state, tool_result["messages"], tm_only)
    new_state["recent_files"] = _merge_recent_files(state.get("recent_files"), new_paths)

    return new_state


def budget_gate(state: AgentState) -> str:
    """Route from agent node: tools, summarize, or end."""
    if state["status"] == "summarizing":
        return "summarize"

    if not state["messages"]:
        return "end"

    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return "end"


def after_tools_gate(state: AgentState) -> str:
    """Route from tools node: model must run again to consume tool results; do not use budget_gate here.

    After ToolNode, the last message is ToolMessage (no tool_calls), so reusing budget_gate would
    incorrectly return \"end\" and terminate before the agent reads tool output.
    """
    if state["status"] == "summarizing":
        return "summarize"
    return "continue"


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

    logger.info(
        "Summarize request session_id=%s model=%s",
        state["session_id"],
        state["model"],
    )
    logger.info("Summarize prompt:\n%s", _truncate(summary_prompt, 8000))

    # Call LLM for summary (does not check budget)
    llm = _chat(state["model"])
    response = _invoke_with_transient_retries(
        lambda: llm.invoke([HumanMessage(content=summary_prompt)]),
        context=f"LLM summarize session_id={state['session_id']}",
    )

    summary = response.content if hasattr(response, "content") else str(response)

    summary_tokens = tokens_from_llm_message(response, state["model"])

    logger.info(
        "Summarize response session_id=%s summary=%s",
        state["session_id"],
        _truncate(str(summary), 4000),
    )

    # Update state
    new_state = state.copy()
    new_state["status"] = "complete"
    new_state["summary"] = summary
    new_state["tokens_used"] = state["tokens_used"] + summary_tokens

    logger.info(
        "Summarize LLM exchange session_id=%s model=%s tokens_this_call=%s tokens_used_total=%s",
        state["session_id"],
        state["model"],
        summary_tokens,
        new_state["tokens_used"],
    )

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
            "end": END,
        },
    )

    # From tools: always return to agent unless budget requires summarization
    workflow.add_conditional_edges(
        "tools",
        after_tools_gate,
        {
            "continue": "agent",
            "summarize": "summarize",
        },
    )

    # Summarize always goes to END
    workflow.add_edge("summarize", END)

    # Compile
    return workflow.compile(checkpointer=checkpointer)


def run_graph(
    task: str,
    max_tokens: int,
    session_id: str,
    model: str = "gpt-4o",
    db_path: str = "~/.agent-runtime-mvp/sessions.db",
):
    """Run the graph and return final state."""
    graph = build_graph(db_path)

    # Initial state
    initial_state: AgentState = {
        "session_id": session_id,
        "task": task,
        "model": model,
        "workspace_root": str(Path.cwd()),
        "messages": [HumanMessage(content=task)],
        "tool_results": [],
        "tokens_used": 0,
        "max_tokens": max_tokens,
        "status": "running",
        "summary": None,
        "recent_files": [],
    }

    # Run graph
    config = {"configurable": {"thread_id": session_id}}
    final_state = None

    set_workspace_root_for_tools(initial_state["workspace_root"])

    for state in graph.stream(initial_state, config):
        # Get the last state
        if isinstance(state, dict):
            # Extract state from the step result
            for node_name, node_state in state.items():
                final_state = node_state

    return final_state if final_state else initial_state
