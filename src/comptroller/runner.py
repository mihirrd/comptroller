"""Importable agent loop for CLI, tests, and harness scripts.

Use :func:`run_agent_turn` to execute one LangGraph pass (task → tools → … → end)
with optional hooks instead of Rich UI.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from . import store
from . import tools as tools_mod
from .budget import count_tokens, tokens_from_llm_message
from .graph import SessionTransientRetryBudgetExhausted, build_graph, summarize_node

logger = logging.getLogger(__name__)


class AgentLoopCallbacks(TypedDict, total=False):
    """Optional side effects during :func:`run_agent_turn` (all no-op if omitted)."""

    on_streaming_llm_text: Callable[[str], None]
    """Called for each streamed text chunk from the agent node."""

    after_agent_llm: Callable[[AIMessage, int, int], None]
    """``(message, step_index, estimated_output_tokens)`` after each agent completion."""

    after_tool: Callable[[ToolMessage, str, dict[str, Any], str, int, int], None]
    """``(tool_msg, tool_name, tool_input, content, step_index, estimated_tokens)``."""

    on_summarize_shown: Callable[[], None]
    """Called when the graph enters summarization (budget or stream safeguard)."""


@dataclass(frozen=True)
class AgentTurnInput:
    """One conversational turn: merged into LangGraph state for ``thread_id=session_id``."""

    task: str
    session_id: str
    model: str = "gpt-4o"
    workspace_root: str | None = None
    """Defaults to process current working directory."""

    messages: list[BaseMessage] | None = None
    """If set, used as the turn's messages update. If ``None`` and ``append_user_message`` is True, uses
    ``[HumanMessage(content=task)]``. If ``None`` and ``append_user_message`` is False, ``messages`` is omitted
    so the checkpointer transcript is unchanged."""

    tokens_used: int = 0
    max_tokens: int = 1000
    api_dollars_used: float = 0.0
    max_api_dollars: float | None = None
    wall_seconds_used: float = 0.0
    max_wall_seconds: float | None = None
    session_retries_used: int = 0
    max_session_retries: int | None = None
    recent_files: list[str] | None = None
    local_model_url: str | None = None
    local_model_id: str | None = None
    model_degraded: bool = False
    interactive_budget: bool = False
    append_user_message: bool = True  # False: checkpoint-only continuation (e.g. after awaiting_budget)

    resume_langgraph_checkpoint_id: str | None = None
    """When set, the graph starts from this LangGraph checkpoint (time travel / rollback)."""

    resume_langgraph_checkpoint_ns: str | None = None
    """Namespace for ``resume_langgraph_checkpoint_id`` (empty string for the root graph)."""


@dataclass
class AgentTurnResult:
    """Outcome of :func:`run_agent_turn`."""

    final_state: dict[str, Any] | None
    """Last graph state from stream updates, or summary state after stream safeguard."""

    streaming_token_budget_hit: bool
    """True when streaming was stopped early by the same heuristic as the CLI (chunk counter)."""

    next_step_counter: int
    """Pass back into the next turn as ``step_counter_start`` if you log steps across turns."""

    current_tokens: int
    current_dollars: float
    current_wall_seconds: float
    current_session_retries: int

    langgraph_checkpoint_id: str | None = None
    """Head checkpoint id after this turn (from ``SqliteSaver``), if available."""

    langgraph_checkpoint_ns: str | None = None


def recent_files_from_checkpointer(
    compiled: Any,
    config: dict[str, Any],
) -> list[str] | None:
    """Read ``recent_files`` from the graph head checkpoint (non-empty list only)."""
    try:
        snap = compiled.get_state(config)
        vals = snap.values
        if vals is None or not isinstance(vals, Mapping):
            return None
        prev = vals.get("recent_files")
        if isinstance(prev, list) and len(prev) > 0:
            return list(prev)
    except Exception:
        logger.debug(
            "recent_files: could not read from checkpointer",
            exc_info=True,
        )
    return None


def _read_langgraph_checkpoint(compiled: Any, thread_id: str) -> tuple[str | None, str | None]:
    try:
        snap = compiled.get_state({"configurable": {"thread_id": thread_id}})
        cfg = snap.config.get("configurable") or {}
        return cfg.get("checkpoint_id"), cfg.get("checkpoint_ns")
    except Exception:
        logger.exception("read_langgraph_checkpoint failed for thread_id=%s", thread_id)
        return None, None


def _build_turn_state(inp: AgentTurnInput) -> dict[str, Any]:
    root = inp.workspace_root if inp.workspace_root is not None else str(Path.cwd())
    state: dict[str, Any] = {
        "session_id": inp.session_id,
        "task": inp.task,
        "model": inp.model,
        "workspace_root": root,
        "tool_results": [],
        "tokens_used": inp.tokens_used,
        "max_tokens": inp.max_tokens,
        "api_dollars_used": inp.api_dollars_used,
        "max_api_dollars": inp.max_api_dollars,
        "wall_seconds_used": inp.wall_seconds_used,
        "max_wall_seconds": inp.max_wall_seconds,
        "session_retries_used": inp.session_retries_used,
        "max_session_retries": inp.max_session_retries,
        "status": "running",
        "summary": None,
        "local_model_url": inp.local_model_url,
        "local_model_id": inp.local_model_id,
        "model_degraded": inp.model_degraded,
        "interactive_budget": inp.interactive_budget,
    }
    if inp.messages is not None:
        state["messages"] = inp.messages
    elif inp.append_user_message:
        state["messages"] = [HumanMessage(content=inp.task)]
    if inp.recent_files is not None:
        state["recent_files"] = list(inp.recent_files)
    return state


def run_agent_turn(
    inp: AgentTurnInput,
    *,
    db_path: str,
    graph: Any | None = None,
    callbacks: AgentLoopCallbacks | None = None,
    log_steps_to_store: bool = True,
    step_counter_start: int = 0,
    stream_chunk_budget_matches_max_tokens: bool = True,
) -> AgentTurnResult:
    """Run a single agent graph invocation (one user task until END or budget).

    Parameters
    ----------
    db_path:
        SQLite DB for LangGraph checkpointer (and optional step logging).
    graph:
        Pre-built compiled graph; if ``None``, :func:`build_graph` is used.
    callbacks:
        Optional hooks for UI or harness instrumentation.
    log_steps_to_store:
        When True, append rows via :func:`store.log_step` like the CLI.
    step_counter_start:
        Initial step index for logging (incremented across LLM / tool events).
    stream_chunk_budget_matches_max_tokens:
        When True, abort streaming after ``max_tokens`` *chunks* (legacy CLI behavior).

    Returns
    -------
    AgentTurnResult
        Always returns a result; raises session / API exceptions from the graph unchanged.

    Notes
    -----
    Sets ``COMPTROLLER_WORKSPACE_ROOT`` via :func:`tools.set_workspace_root_for_tools`
    for the duration of the run.
    """
    cb = callbacks or {}
    turn_state = _build_turn_state(inp)
    tools_mod.set_workspace_root_for_tools(turn_state["workspace_root"])

    compiled = graph if graph is not None else build_graph(db_path)
    cfg: dict[str, Any] = {"thread_id": inp.session_id}
    if inp.resume_langgraph_checkpoint_id:
        cfg["checkpoint_id"] = inp.resume_langgraph_checkpoint_id
    if inp.resume_langgraph_checkpoint_ns is not None:
        cfg["checkpoint_ns"] = inp.resume_langgraph_checkpoint_ns
    config: dict[str, Any] = {"configurable": cfg}

    # Persisted channel: re-inject recent_files from the head checkpoint so a new
    # turn's partial update does not drop them when omitted from _build_turn_state.
    # Also treat an explicit ``[]`` from the caller as "unknown" so we can replace with checkpoint data.
    if inp.recent_files is None or (
        isinstance(inp.recent_files, list) and len(inp.recent_files) == 0
    ):
        prev = recent_files_from_checkpointer(compiled, config)
        if prev is not None:
            turn_state["recent_files"] = list(prev)

    final_state: dict[str, Any] | None = None
    step_counter = step_counter_start
    token_count = 0
    ai_message = ""
    streaming_hit = False

    db_resolved = str(Path(db_path).expanduser())

    stream_iter = compiled.stream(
        turn_state,
        config,
        stream_mode=["messages", "updates"],
    )

    for chunk in stream_iter:
        mode, data = chunk
        if mode == "messages":
            message_chunk, metadata = data
            node = metadata.get("langgraph_node")
            if node == "agent" and message_chunk.content:
                ai_message += message_chunk.content
                fn = cb.get("on_streaming_llm_text")
                if fn:
                    fn(message_chunk.content)
                token_count += 1
            if stream_chunk_budget_matches_max_tokens and token_count >= inp.max_tokens:
                streaming_hit = True
                break

        elif mode == "updates":
            if not isinstance(data, dict):
                continue
            for node_name, node_state in data.items():
                final_state = node_state

                if node_name == "agent":
                    if node_state.get("messages"):
                        last_msg = node_state["messages"][-1]
                        if isinstance(last_msg, AIMessage):
                            step_counter += 1
                            content = last_msg.content if last_msg.content else "[Tool calls]"
                            est = tokens_from_llm_message(last_msg, inp.model)
                            fn = cb.get("after_agent_llm")
                            if fn:
                                fn(last_msg, step_counter, est)
                            if log_steps_to_store:
                                store.log_step(
                                    inp.session_id,
                                    step_counter,
                                    "llm_response",
                                    content,
                                    est,
                                    db_resolved,
                                )

                elif node_name == "tools":
                    if node_state.get("messages"):
                        for msg in reversed(node_state["messages"]):
                            if not isinstance(msg, ToolMessage):
                                continue
                            step_counter += 1
                            tool_name = msg.name if hasattr(msg, "name") else "tool"
                            content = msg.content if isinstance(msg.content, str) else str(msg.content)
                            tool_input: dict[str, Any] = {}
                            for prev_msg in reversed(node_state["messages"]):
                                if isinstance(prev_msg, AIMessage) and getattr(
                                    prev_msg, "tool_calls", None
                                ):
                                    for tc in prev_msg.tool_calls or []:
                                        if getattr(msg, "tool_call_id", None) and tc.get(
                                            "id"
                                        ) == msg.tool_call_id:
                                            tool_input = tc.get("args") or {}
                                            if not isinstance(tool_input, dict):
                                                tool_input = {}
                                            tool_name = tc.get("name", tool_name)
                                            break
                                    break
                            est = count_tokens(content)
                            fn = cb.get("after_tool")
                            if fn:
                                fn(msg, tool_name, tool_input, content, step_counter, est)
                            if log_steps_to_store:
                                store.log_step(
                                    inp.session_id,
                                    step_counter,
                                    "tool_call",
                                    f"{tool_name}: {content[:200]}",
                                    est,
                                    db_resolved,
                                )
                            break

                if node_name == "summarize" and node_state.get("status") == "complete":
                    fn = cb.get("on_summarize_shown")
                    if fn:
                        fn()

    if streaming_hit:
        partial_state = {
            **turn_state,
            "messages": turn_state["messages"] + [AIMessage(content=ai_message)],
            "tokens_used": inp.tokens_used + token_count,
            "api_dollars_used": inp.api_dollars_used,
            "max_api_dollars": inp.max_api_dollars,
            "wall_seconds_used": inp.wall_seconds_used,
            "max_wall_seconds": inp.max_wall_seconds,
            "session_retries_used": inp.session_retries_used,
            "max_session_retries": inp.max_session_retries,
            "status": "summarizing",
            "local_model_url": inp.local_model_url,
            "local_model_id": inp.local_model_id,
            "model_degraded": inp.model_degraded,
            "interactive_budget": inp.interactive_budget,
        }
        summary_result = summarize_node(partial_state)
        final_state = summary_result
        lg_id, lg_ns = _read_langgraph_checkpoint(compiled, inp.session_id)
        return AgentTurnResult(
            final_state=final_state,
            streaming_token_budget_hit=True,
            next_step_counter=step_counter,
            current_tokens=int(summary_result["tokens_used"]),
            current_dollars=float(summary_result.get("api_dollars_used", inp.api_dollars_used)),
            current_wall_seconds=float(
                summary_result.get("wall_seconds_used", inp.wall_seconds_used)
            ),
            current_session_retries=int(
                summary_result.get("session_retries_used", inp.session_retries_used)
            ),
            langgraph_checkpoint_id=lg_id,
            langgraph_checkpoint_ns=lg_ns,
        )

    if not final_state:
        logger.warning("run_agent_turn: graph stream produced no update state")
        lg_id, lg_ns = _read_langgraph_checkpoint(compiled, inp.session_id)
        return AgentTurnResult(
            final_state=None,
            streaming_token_budget_hit=False,
            next_step_counter=step_counter,
            current_tokens=inp.tokens_used,
            current_dollars=inp.api_dollars_used,
            current_wall_seconds=inp.wall_seconds_used,
            current_session_retries=inp.session_retries_used,
            langgraph_checkpoint_id=lg_id,
            langgraph_checkpoint_ns=lg_ns,
        )

    lg_id, lg_ns = _read_langgraph_checkpoint(compiled, inp.session_id)
    return AgentTurnResult(
        final_state=final_state,
        streaming_token_budget_hit=False,
        next_step_counter=step_counter,
        current_tokens=int(final_state["tokens_used"]),
        current_dollars=float(final_state.get("api_dollars_used", inp.api_dollars_used)),
        current_wall_seconds=float(final_state.get("wall_seconds_used", inp.wall_seconds_used)),
        current_session_retries=int(
            final_state.get("session_retries_used", inp.session_retries_used)
        ),
        langgraph_checkpoint_id=lg_id,
        langgraph_checkpoint_ns=lg_ns,
    )


# Re-export for harness authors who catch graph-level retry exhaustion
__all__ = [
    "AgentLoopCallbacks",
    "AgentTurnInput",
    "AgentTurnResult",
    "SessionTransientRetryBudgetExhausted",
    "recent_files_from_checkpointer",
    "run_agent_turn",
]
