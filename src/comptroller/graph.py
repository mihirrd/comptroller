import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import litellm

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

from .state import AgentState
from .tools import TOOLS, set_workspace_root_for_tools
from .budget import count_tokens, dollars_from_llm_message, tokens_from_llm_message
from .prompt_builder import build_system_prompt

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_RECENT_FILES = 20
MAX_TOOL_MSG_CHARS = 12_000
MAX_TOOL_DIGEST_ITEMS = 8
DEFAULT_MAX_MESSAGE_WINDOW = 48
RECENT_FILES_CONTEXT_MSG_ID = "comptroller-recent-files-context"


def _max_message_window() -> int:
    raw = os.environ.get("COMPTROLLER_MAX_MESSAGE_WINDOW", str(DEFAULT_MAX_MESSAGE_WINDOW))
    try:
        n = int(raw)
    except ValueError:
        n = DEFAULT_MAX_MESSAGE_WINDOW
    return max(8, n)


def _recent_files_context_block(paths: list[str] | None) -> str:
    """Human-readable block injected each agent turn (not part of build_system_prompt)."""
    header = "### Recently touched files (session)\n\n"
    if not paths:
        return header + "(none recorded yet)"
    lines = "\n".join(f"- `{p}`" for p in paths)
    return header + lines


def _trim_messages_window(messages: list[Any]) -> list[Any]:
    """Keep the system message (if any) and a bounded suffix of the rest; never split a leading tool run."""
    if not messages:
        return messages
    limit = _max_message_window()
    prefix: list[Any] = []
    rest: list[Any]
    if isinstance(messages[0], SystemMessage):
        prefix = [messages[0]]
        rest = list(messages[1:])
    else:
        rest = list(messages)
    if len(rest) <= limit:
        return messages
    start = len(rest) - limit
    while start > 0 and isinstance(rest[start], ToolMessage):
        start -= 1
    return prefix + rest[start:]


def _state_messages_with_sliding_window(full_messages: list[Any]) -> list[Any]:
    """Return value for ``messages`` state key: replace-all + trimmed list when over the window."""
    trimmed = _trim_messages_window(full_messages)
    if len(trimmed) >= len(full_messages):
        return full_messages
    return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed]


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


class SessionTransientRetryBudgetExhausted(Exception):
    """Raised when ``max_session_retries`` is set and another backoff retry would exceed it."""

    def __init__(self, used: int, max_retries: int, cause: BaseException):
        self.used = used
        self.max_retries = max_retries
        self.cause = cause
        super().__init__(
            f"Session transient retry budget exhausted ({used}/{max_retries} backoff retries used): {cause}"
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


def _invoke_with_transient_retries(
    fn: Callable[[], T],
    *,
    context: str,
    retry_budget_state: AgentState | None = None,
) -> T:
    """Run ``fn``; on rate limits and common transient API errors, sleep with exponential backoff and retry.

    If ``retry_budget_state`` is set and ``max_session_retries`` is not None, each backoff retry
    (after a transient error, before ``sleep``) increments ``session_retries_used``. When the cap
    would be exceeded, raises :class:`SessionTransientRetryBudgetExhausted` instead of retrying.
    """
    attempts = _llm_retry_attempts()
    for attempt in range(attempts):
        try:
            return fn()
        except _LLM_TRANSIENT_EXCEPTIONS as e:
            if attempt >= attempts - 1:
                logger.error("%s: failed after %s attempts: %s", context, attempts, e)
                raise
            if retry_budget_state is not None:
                cap = retry_budget_state.get("max_session_retries")
                if cap is not None:
                    used = int(retry_budget_state.get("session_retries_used", 0))
                    if used >= cap:
                        logger.error(
                            "%s: session transient retry budget exhausted (%s/%s) after %s: %s",
                            context,
                            used,
                            cap,
                            type(e).__name__,
                            e,
                        )
                        raise SessionTransientRetryBudgetExhausted(used, cap, e) from e
                    retry_budget_state["session_retries_used"] = used + 1
                    logger.info(
                        "%s: session transient backoff retry %s/%s after %s",
                        context,
                        retry_budget_state["session_retries_used"],
                        cap,
                        type(e).__name__,
                    )
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


def _litellm_openai_compatible_model(local_model_id: str) -> str:
    """LiteLLM id for an OpenAI-compatible ``api_base`` (Ollama, LM Studio, vLLM, etc.)."""
    lid = local_model_id.strip()
    if lid.startswith("openai/"):
        return lid
    return f"openai/{lid}"


def _api_base_for_llm(state: AgentState) -> str | None:
    if not state.get("model_degraded"):
        return None
    url = (state.get("local_model_url") or "").strip().rstrip("/")
    return url or None


def _chat(
    model: str,
    temperature: float = 0,
    *,
    api_base: str | None = None,
) -> ChatLiteLLM:
    """LiteLLM-backed chat model; `model` uses LiteLLM naming (provider prefixes optional).

    When ``api_base`` is set (local OpenAI-compatible server), ``api_key`` defaults to
    ``COMPTROLLER_LOCAL_API_KEY`` or the placeholder ``dummy`` (Ollama/LM Studio).

    ``max_retries=1`` so LangChain's inner tenacity does not stack with
    :func:`_invoke_with_transient_retries` (which applies backoff for rate limits and 5xx).
    """
    kwargs: dict[str, Any] = {"model": model, "temperature": temperature, "max_retries": 1}
    if api_base:
        kwargs["api_base"] = api_base.rstrip("/")
        key = (os.environ.get("COMPTROLLER_LOCAL_API_KEY") or "").strip() or "dummy"
        kwargs["api_key"] = key
    return ChatLiteLLM(**kwargs)


def _degradation_configured(state: AgentState) -> bool:
    url = (state.get("local_model_url") or "").strip()
    mid = (state.get("local_model_id") or "").strip()
    return bool(url and mid and not state.get("model_degraded"))


def _apply_model_degradation(new_state: AgentState, prev_state: AgentState) -> None:
    url = (prev_state.get("local_model_url") or "").strip().rstrip("/")
    mid = (prev_state.get("local_model_id") or "").strip()
    new_state["model"] = _litellm_openai_compatible_model(mid)
    new_state["local_model_url"] = url
    new_state["local_model_id"] = mid
    new_state["model_degraded"] = True
    new_state["status"] = "running"
    logger.info(
        "Model degradation: using local OpenAI-compatible endpoint api_base=%s litellm_model=%s",
        url,
        new_state["model"],
    )


def _apply_budget_exceeded_status(new_state: AgentState, prev_state: AgentState) -> None:
    """Set ``summarizing`` or ``awaiting_budget``, or switch to local LLM when token/dollar cap hits."""
    interactive = bool(prev_state.get("interactive_budget"))
    degraded = bool(prev_state.get("model_degraded"))
    wcap = prev_state.get("max_wall_seconds")
    if wcap is not None and new_state["wall_seconds_used"] >= wcap:
        new_state["status"] = "awaiting_budget" if interactive else "summarizing"
        return

    token_exceeded = new_state["tokens_used"] >= prev_state["max_tokens"]
    cap = prev_state.get("max_api_dollars")
    dollar_exceeded = cap is not None and new_state["api_dollars_used"] >= cap

    if degraded:
        return

    if token_exceeded or dollar_exceeded:
        if interactive:
            new_state["status"] = "awaiting_budget"
        elif _degradation_configured(prev_state):
            _apply_model_degradation(new_state, prev_state)
        else:
            new_state["status"] = "summarizing"


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
    # Build messages list (recent-files context is injected here only, not persisted in state)
    messages = state["messages"].copy()

    prompt_state = dict(state)
    prompt_state["tool_digest"] = _compact_tool_digest(state["messages"])

    # Add system message if this is the first call
    if len(messages) == 0 or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(content=build_system_prompt(prompt_state))
        messages.insert(0, system_msg)

    recent_ctx = HumanMessage(
        content=_recent_files_context_block(state.get("recent_files")),
        id=RECENT_FILES_CONTEXT_MSG_ID,
    )
    if messages and isinstance(messages[0], SystemMessage):
        messages.insert(1, recent_ctx)
    else:
        messages.insert(0, recent_ctx)

    messages_for_llm = _prepare_messages_for_llm(messages)

    # Initialize LLM with tools
    llm = _chat(state["model"], api_base=_api_base_for_llm(state))
    llm_with_tools = llm.bind_tools(TOOLS)

    logger.info(
        "LLM request session_id=%s model=%s n_messages=%s",
        state["session_id"],
        state["model"],
        len(messages_for_llm),
    )
    logger.info("LLM request messages:\n%s", _format_messages_for_log(messages_for_llm))

    # Call LLM (truncated tool bodies only in this copy)
    _t_wall = time.monotonic()
    response = _invoke_with_transient_retries(
        lambda: llm_with_tools.invoke(messages_for_llm),
        context=f"LLM agent session_id={state['session_id']}",
        retry_budget_state=state,
    )
    _wall_dt = time.monotonic() - _t_wall

    logger.info("LLM response session_id=%s", state["session_id"])
    logger.info("LLM response:\n%s", _format_ai_response_for_log(response))

    tokens = tokens_from_llm_message(response, state["model"])
    call_dollars = dollars_from_llm_message(response, state["model"])

    # Update state
    new_state = state.copy()
    new_state["tokens_used"] = state["tokens_used"] + tokens
    new_state["api_dollars_used"] = state["api_dollars_used"] + call_dollars
    new_state["wall_seconds_used"] = state["wall_seconds_used"] + _wall_dt
    new_state["messages"] = _state_messages_with_sliding_window(state["messages"] + [response])

    cap = state.get("max_api_dollars")
    wcap = state.get("max_wall_seconds")
    logger.info(
        "LLM exchange session_id=%s model=%s degraded=%s tokens_this_call=%s tokens_used_total=%s max_tokens=%s "
        "dollars_this_call=%.6f api_dollars_total=%.6f max_api_dollars=%s wall_step=%.3fs wall_total=%.3fs max_wall=%s",
        state["session_id"],
        state["model"],
        bool(state.get("model_degraded")),
        tokens,
        new_state["tokens_used"],
        state["max_tokens"],
        call_dollars,
        new_state["api_dollars_used"],
        cap,
        _wall_dt,
        new_state["wall_seconds_used"],
        wcap,
    )

    _apply_budget_exceeded_status(new_state, state)

    return new_state


def tool_node_wrapper(state: AgentState) -> AgentState:
    """Wrapper around ToolNode to track tokens."""
    # Create the tool node
    tool_executor = ToolNode(TOOLS)

    # Execute tools (this only returns updated messages)
    _t_wall = time.monotonic()
    tool_result = tool_executor.invoke(state)
    _wall_dt = time.monotonic() - _t_wall

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
    new_state["messages"] = _state_messages_with_sliding_window(tool_result["messages"])

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
    new_state["wall_seconds_used"] = state["wall_seconds_used"] + _wall_dt

    interactive = bool(state.get("interactive_budget"))
    wcap = state.get("max_wall_seconds")
    if wcap is not None and new_state["wall_seconds_used"] >= wcap:
        new_state["status"] = "awaiting_budget" if interactive else "summarizing"
    elif not state.get("model_degraded") and new_state["tokens_used"] >= state["max_tokens"]:
        if interactive:
            new_state["status"] = "awaiting_budget"
        elif _degradation_configured(state):
            _apply_model_degradation(new_state, state)
        else:
            new_state["status"] = "summarizing"

    tm_only = [m for m in tool_messages if isinstance(m, ToolMessage)]
    new_paths = _paths_from_tool_messages(state, tool_result["messages"], tm_only)
    new_state["recent_files"] = _merge_recent_files(state.get("recent_files"), new_paths)

    return new_state


def budget_gate(state: AgentState) -> str:
    """Route from agent node: tools, summarize, or end."""
    if state["status"] == "summarizing":
        return "summarize"
    if state["status"] == "awaiting_budget":
        return "end"

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
    if state["status"] == "awaiting_budget":
        return "end"
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

    # Call LLM for summary (does not check token/dollar budget; wall time still accumulates)
    llm = _chat(state["model"], api_base=_api_base_for_llm(state))
    _t_wall = time.monotonic()
    response = _invoke_with_transient_retries(
        lambda: llm.invoke([HumanMessage(content=summary_prompt)]),
        context=f"LLM summarize session_id={state['session_id']}",
        retry_budget_state=state,
    )
    _wall_dt = time.monotonic() - _t_wall

    summary = response.content if hasattr(response, "content") else str(response)

    summary_tokens = tokens_from_llm_message(response, state["model"])
    summary_dollars = dollars_from_llm_message(response, state["model"])

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
    new_state["api_dollars_used"] = state["api_dollars_used"] + summary_dollars
    new_state["wall_seconds_used"] = state["wall_seconds_used"] + _wall_dt

    logger.info(
        "Summarize LLM exchange session_id=%s model=%s tokens_this_call=%s tokens_used_total=%s "
        "dollars_this_call=%.6f api_dollars_total=%.6f wall_step=%.3fs wall_total=%.3fs",
        state["session_id"],
        state["model"],
        summary_tokens,
        new_state["tokens_used"],
        summary_dollars,
        new_state["api_dollars_used"],
        _wall_dt,
        new_state["wall_seconds_used"],
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
            "end": END,
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
    max_api_dollars: float | None = None,
    max_wall_seconds: float | None = None,
    max_session_retries: int | None = None,
    *,
    local_model_url: str | None = None,
    local_model_id: str | None = None,
    model_degraded: bool = False,
    interactive_budget: bool = False,
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
        "api_dollars_used": 0.0,
        "max_api_dollars": max_api_dollars,
        "wall_seconds_used": 0.0,
        "max_wall_seconds": max_wall_seconds,
        "session_retries_used": 0,
        "max_session_retries": max_session_retries,
        "status": "running",
        "summary": None,
        "recent_files": [],
        "local_model_url": local_model_url,
        "local_model_id": local_model_id,
        "model_degraded": model_degraded,
        "interactive_budget": interactive_budget,
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
