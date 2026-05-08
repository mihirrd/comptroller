import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path

import litellm
import typer
from typing_extensions import Annotated
from .graph import SessionTransientRetryBudgetExhausted
from . import logging_config
from . import store
from . import ui
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _log_repr_preview(obj: object, max_len: int = 2000) -> str:
    s = repr(obj)
    if len(s) > max_len:
        return s[:max_len] + f"... <truncated {len(s) - max_len} chars>"
    return s


def _available_llm_models() -> list[str]:
    """Models LiteLLM can use with the current environment (provider keys from .env)."""
    from litellm.utils import get_valid_models

    return sorted(get_valid_models())


app = typer.Typer()

DB_PATH = "~/.agent-runtime-mvp/sessions.db"


@app.callback()
def _cli_setup() -> None:
    """Run once per process before any subcommand so file logging is always configured."""
    logging_config.configure_logging()


@app.command()
def clear_sessions():
    """Utility function to clear all sessions from the database."""
    db_path = str(Path(DB_PATH).expanduser())
    if Path(db_path).exists():
        os.remove(db_path)
        print("All sessions cleared.")
    else:
        print("No database found to clear.")
    
@app.command(name="list")
def list_sessions_cmd():
    """List all sessions."""
    db_path = str(Path(DB_PATH).expanduser())

    # Check if database exists
    if not Path(db_path).exists():
        ui.print_error("Database not found. Run 'init' first.")
        raise typer.Exit(1)

    sessions_list = store.list_sessions(db_path)
    ui.print_sessions_table(sessions_list)

@app.command()
def init():
    """Initialize the agent runtime - create database and check environment."""
    ui.print_banner()
    load_dotenv()

    valid_models = _available_llm_models()
    if not valid_models:
        ui.print_error(
            "No LLM provider API keys detected. Set credentials for at least one provider "
            "(e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY). "
            "See LiteLLM provider docs for env variable names."
        )
        raise typer.Exit(1)

    # Create directory
    db_path = str(Path(DB_PATH).expanduser())
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize database
    store.init_db(db_path)

    ui.console.print(f"[{ui.THEME['success']}]✓ Initialized database at {db_path}[/]")
    ui.console.print(
        f"[{ui.THEME['success']}]✓ LLM credentials detected "
        f"([{ui.THEME['accent']}]{len(valid_models):,}[/{ui.THEME['accent']}] models available)[/]"
    )
    ui.console.print()
    ui.print_available_models(valid_models)


def _max_api_dollars_from_env() -> float | None:
    raw = (os.environ.get("COMPTROLLER_MAX_API_DOLLARS") or os.environ.get("max_api_dollars") or "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0 else None


def _max_wall_seconds_from_env() -> float | None:
    raw = (os.environ.get("COMPTROLLER_MAX_WALL_SECONDS") or os.environ.get("max_wall_seconds") or "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0 else None


def _max_session_retries_from_env() -> int | None:
    raw = (os.environ.get("COMPTROLLER_MAX_SESSION_RETRIES") or os.environ.get("max_session_retries") or "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n >= 0 else None


def _resolve_local_model_url(cli: str | None, session_val: str | None = None) -> str | None:
    for candidate in (cli, session_val, os.environ.get("COMPTROLLER_LOCAL_MODEL_URL")):
        if candidate is None:
            continue
        s = str(candidate).strip()
        if s:
            return s
    return None


def _resolve_local_model_id(cli: str | None, session_val: str | None = None) -> str | None:
    for candidate in (cli, session_val, os.environ.get("COMPTROLLER_LOCAL_MODEL")):
        if candidate is None:
            continue
        s = str(candidate).strip()
        if s:
            return s
    return None


def _resolve_local_model_api_key(cli: str | None) -> str | None:
    for candidate in (cli, os.environ.get("COMPTROLLER_LOCAL_API_KEY")):
        if candidate is None:
            continue
        s = str(candidate).strip()
        if s:
            return s
    return None


def _interactive_budget_enabled(cli_flag: bool) -> bool:
    if cli_flag:
        return True
    return os.environ.get("COMPTROLLER_INTERACTIVE_BUDGET", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@app.command()
def run(
    task: Annotated[str, typer.Argument()] = None,
    session: Annotated[str, typer.Option("--session")] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = os.getenv("max_tokens", 1000),
    max_dollars: Annotated[
        float | None,
        typer.Option(
            "--max-dollars",
            help="Maximum estimated API spend (USD) for this session using LiteLLM model pricing.",
        ),
    ] = None,
    max_wall_seconds: Annotated[
        float | None,
        typer.Option(
            "--max-wall-seconds",
            help="Maximum cumulative agent wall time (seconds) per session; counts LLM/tool/summary work only.",
        ),
    ] = None,
    max_session_retries: Annotated[
        int | None,
        typer.Option(
            "--max-session-retries",
            help="Max LLM transient-error backoff retries for this session (counts each sleep+retry). None = unlimited.",
        ),
    ] = None,
    model: Annotated[str, typer.Option("--model")] = os.getenv("model", "gpt-4o"),
    local_model_url: Annotated[
        str | None,
        typer.Option(
            "--local-model-url",
            help="OpenAI-compatible API base when cloud token/dollar budget is exceeded (e.g. http://localhost:11434/v1). "
            "Also COMPTROLLER_LOCAL_MODEL_URL.",
        ),
    ] = None,
    local_model_id: Annotated[
        str | None,
        typer.Option(
            "--local-model",
            help="Model name on the local server (required with --local-model-url). Also COMPTROLLER_LOCAL_MODEL.",
        ),
    ] = None,
    local_model_api_key: Annotated[
        str | None,
        typer.Option(
            "--local-api-key",
            help="API key for the OpenAI-compatible fallback endpoint (for remote gateways). "
            "Also COMPTROLLER_LOCAL_API_KEY.",
        ),
    ] = None,
    interactive_budget: Annotated[
        bool,
        typer.Option(
            "--interactive-budget",
            help="When a budget is exceeded, prompt to add more headroom instead of summarizing immediately. "
            "Also COMPTROLLER_INTERACTIVE_BUDGET=1.",
            is_flag=True,
        ),
    ] = False,
):
    """Run an agent task with token, optional dollar, and optional wall-time budgets. Supports continuous conversation mode."""
    from langchain_core.messages import AIMessage, ToolMessage

    from . import workspace_checkpoint as wscp
    from .graph import build_graph, summarize_node
    from .runner import (
        AgentTurnInput,
        AgentTurnResult,
        recent_files_from_checkpointer,
        run_agent_turn,
    )

    def _workspace_checkpoint_record_reason(r: AgentTurnResult) -> str:
        if r.streaming_token_budget_hit:
            return "streaming_budget"
        if not r.final_state:
            return "no_final_state"
        return "halt"

    def _record_workspace_halt_db(
        *,
        session_id: str,
        db_path: str,
        workspace_root: str,
        result: AgentTurnResult,
    ) -> None:
        root = Path(workspace_root).expanduser().resolve()
        if not wscp.is_git_workspace(root):
            return
        seq = store.next_workspace_checkpoint_seq(session_id, db_path)
        commit_sha, parent_sha, err = wscp.create_git_checkpoint(
            root,
            session_id,
            seq,
            message=f"comptroller {_workspace_checkpoint_record_reason(result)}",
        )
        if err:
            logger.warning("workspace git checkpoint skipped: %s", err)
            return
        store.insert_workspace_checkpoint(
            session_id,
            seq,
            _workspace_checkpoint_record_reason(result),
            commit_sha,
            parent_sha,
            result.langgraph_checkpoint_id,
            result.langgraph_checkpoint_ns,
            result.next_step_counter,
            str(root),
            db_path,
        )

    def _record_workspace_error_checkpoint(
        *,
        session_id: str,
        db_path: str,
        workspace_root: str,
        graph,
    ) -> None:
        root = Path(workspace_root).expanduser().resolve()
        if not wscp.is_git_workspace(root):
            return
        lg_id: str | None = None
        lg_ns: str | None = None
        try:
            snap = graph.get_state({"configurable": {"thread_id": session_id}})
            cfg = snap.config.get("configurable") or {}
            lg_id = cfg.get("checkpoint_id")
            lg_ns = cfg.get("checkpoint_ns")
        except Exception:
            logger.exception("workspace error checkpoint: get_state failed")
        seq = store.next_workspace_checkpoint_seq(session_id, db_path)
        commit_sha, parent_sha, err = wscp.create_git_checkpoint(
            root, session_id, seq, message="comptroller error"
        )
        if err:
            logger.warning("workspace git checkpoint skipped after error: %s", err)
            return
        store.insert_workspace_checkpoint(
            session_id,
            seq,
            "error",
            commit_sha,
            parent_sha,
            lg_id,
            lg_ns,
            None,
            str(root),
            db_path,
        )

    load_dotenv()
    ui.print_banner()

    interactive_loop = _interactive_budget_enabled(interactive_budget)

    max_api_dollars = max_dollars if max_dollars is not None else _max_api_dollars_from_env()
    max_wall_s = (
        max_wall_seconds if max_wall_seconds is not None else _max_wall_seconds_from_env()
    )
    session_retry_cap = (
        max_session_retries
        if max_session_retries is not None
        else _max_session_retries_from_env()
    )

    if not _available_llm_models():
        ui.print_error(
            "No LLM provider API keys detected. Run 'init' and configure credentials "
            "(e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY)."
        )
        raise typer.Exit(1)

    db_path = str(Path(DB_PATH).expanduser())

    effective_model = model
    model_degraded_loop = False
    loc_url: str | None = None
    loc_id: str | None = None
    loc_api_key: str | None = None

    # Resume existing session or create new one
    if session:
        # Resume existing session
        session_data = store.get_session(session, db_path)
        if not session_data:
            ui.print_error(f"Session {session} not found")
            raise typer.Exit(1)

        session_id = session
        tokens_used_so_far = session_data["tokens_used"]
        max_tokens = session_data["max_tokens"]
        dollars_used_so_far = float(session_data.get("api_dollars_used") or 0)
        wall_seconds_so_far = float(session_data.get("wall_seconds_used") or 0)
        session_retries_so_far = int(session_data.get("session_retries_used") or 0)
        resumed_max_dollars = session_data.get("max_api_dollars")
        if resumed_max_dollars is not None:
            max_api_dollars = float(resumed_max_dollars)
        resumed_max_wall = session_data.get("max_wall_seconds")
        if resumed_max_wall is not None:
            max_wall_s = float(resumed_max_wall)
        resumed_max_session_retries = session_data.get("max_session_retries")
        if resumed_max_session_retries is not None:
            session_retry_cap = int(resumed_max_session_retries)

        loc_url = _resolve_local_model_url(local_model_url, session_data.get("local_model_url"))
        loc_id = _resolve_local_model_id(local_model_id, session_data.get("local_model_id"))
        loc_api_key = _resolve_local_model_api_key(local_model_api_key)
        model_degraded_loop = bool(int(session_data.get("model_degraded") or 0))
        effective_model = (session_data.get("active_model") or model).strip() or model
        if model_degraded_loop and (not loc_url or not loc_id):
            ui.print_error(
                f"Session {session_id} is marked as using a local model, but local URL or model id is missing."
            )
            raise typer.Exit(1)

        if tokens_used_so_far >= max_tokens and not model_degraded_loop and not interactive_loop:
            ui.print_error(f"Session {session_id} has exceeded token budget of {max_tokens:,}")
            ui.console.print("[dim]Please reset the session")
            raise typer.Exit(1)
        if max_api_dollars is not None and dollars_used_so_far >= max_api_dollars and not model_degraded_loop and not interactive_loop:
            ui.print_error(
                f"Session {session_id} has exceeded API dollar budget "
                f"(${dollars_used_so_far:.4f} / ${max_api_dollars:.4f})"
            )
            ui.console.print("[dim]Please reset the session")
            raise typer.Exit(1)
        if max_wall_s is not None and wall_seconds_so_far >= max_wall_s and not interactive_loop:
            ui.print_error(
                f"Session {session_id} has reached wall-time budget "
                f"({wall_seconds_so_far:.1f}s / {max_wall_s:.0f}s agent time)"
            )
            ui.console.print("[dim]Please reset the session")
            raise typer.Exit(1)
        original_task = session_data["task"]

        ui.console.print(f"[bold]Resuming session {session_id}[/bold]")
        ui.console.print(f"[dim]Original task: {original_task}[/dim]")
        ui.console.print(f"[dim]Budget: {tokens_used_so_far:,} / {max_tokens:,} tokens used[/dim]")
        if max_api_dollars is not None:
            ui.console.print(
                f"[dim]API spend: ${dollars_used_so_far:.4f} / ${max_api_dollars:.4f} (estimated via LiteLLM)[/dim]"
            )
        if max_wall_s is not None:
            ui.console.print(
                f"[dim]Agent wall time: {ui.format_wall_seconds(wall_seconds_so_far)} / "
                f"{ui.format_wall_seconds(max_wall_s)}[/dim]"
            )
        if session_retry_cap is not None:
            ui.console.print(
                f"[dim]LLM backoff retries used: {session_retries_so_far} / {session_retry_cap}[/dim]"
            )
        ui.console.print()
    else:
        # Create new session
        if not task:
            ui.print_error("Task required for new session. Use: python main.py run \"your task\"")
            raise typer.Exit(1)

        loc_url = _resolve_local_model_url(local_model_url, None)
        loc_id = _resolve_local_model_id(local_model_id, None)
        loc_api_key = _resolve_local_model_api_key(local_model_api_key)
        if (loc_url and not loc_id) or (loc_id and not loc_url):
            ui.print_error(
                "Local model fallback requires both --local-model-url and --local-model "
                "(or COMPTROLLER_LOCAL_MODEL_URL and COMPTROLLER_LOCAL_MODEL)."
            )
            raise typer.Exit(1)

        session_id = str(uuid.uuid4())[:8]
        tokens_used_so_far = 0
        dollars_used_so_far = 0.0
        wall_seconds_so_far = 0.0
        session_retries_so_far = 0
        store.create_session(
            session_id,
            task,
            max_tokens,
            db_path,
            max_api_dollars=max_api_dollars,
            max_wall_seconds=max_wall_s,
            max_session_retries=session_retry_cap,
            primary_model=model,
            active_model=model,
            local_model_url=loc_url,
            local_model_id=loc_id,
        )

        # Print session header
        ui.print_session_header(
            session_id,
            task,
            model,
            max_tokens,
            max_api_dollars=max_api_dollars,
            max_wall_seconds=max_wall_s,
            local_model_url=loc_url,
            local_model_id=loc_id,
        )

    # Track start time
    start_time = time.time()
    step_counter = 0

    try:
        # Build graph
        graph = build_graph(db_path)

        # Carried across turns in this process; also seed on resume from checkpoint
        # so the first follow-up user message after `run --session` sees the same
        # recent paths as the last completed graph turn.
        turn_recent_files: list[str] | None = None
        if session:
            turn_recent_files = recent_files_from_checkpointer(
                graph, {"configurable": {"thread_id": session_id}}
            )

        # Continuous conversation loop
        current_tokens = tokens_used_so_far
        current_dollars = dollars_used_so_far
        current_wall_seconds = wall_seconds_so_far
        current_session_retries = session_retries_so_far
        conversation_active = True

        while conversation_active:
            # Get task for this turn
            if not task:
                ui.console.print()
                remaining = max_tokens - current_tokens
                ui.console.print(f"[{ui.THEME['dim']}]Remaining budget: {remaining:,} tokens[/]")
                if max_api_dollars is not None:
                    rem_usd = max(0.0, max_api_dollars - current_dollars)
                    ui.console.print(
                        f"[{ui.THEME['dim']}]Remaining API spend: ${rem_usd:.4f} / ${max_api_dollars:.4f}[/]"
                    )
                if max_wall_s is not None:
                    rem_w = max(0.0, max_wall_s - current_wall_seconds)
                    ui.console.print(
                        f"[{ui.THEME['dim']}]Remaining agent wall time: "
                        f"{ui.format_wall_seconds(rem_w)} / {ui.format_wall_seconds(max_wall_s)}[/]"
                    )
                if session_retry_cap is not None:
                    rem_r = max(0, session_retry_cap - current_session_retries)
                    ui.console.print(
                        f"[{ui.THEME['dim']}]Remaining LLM backoff retries: {rem_r} / {session_retry_cap}[/]"
                    )
                task = ui.console.input(f"[{ui.THEME['accent']}]Task (or 'quit' to exit): [/]").strip()

                if task.lower() in ['quit', 'exit', 'q', '']:
                    ui.console.print("[dim]Exiting session...[/dim]")
                    break

            prev_degraded_outer = model_degraded_loop

            pending_resume_id, pending_resume_ns = store.get_session_pending_resume(session_id, db_path)
            resume_kw: dict = {}
            if pending_resume_id:
                resume_kw["resume_langgraph_checkpoint_id"] = pending_resume_id
                resume_kw["resume_langgraph_checkpoint_ns"] = pending_resume_ns

            turn_inp = AgentTurnInput(
                task=task,
                session_id=session_id,
                model=effective_model,
                workspace_root=str(Path.cwd()),
                tokens_used=current_tokens,
                max_tokens=max_tokens,
                api_dollars_used=current_dollars,
                max_api_dollars=max_api_dollars,
                wall_seconds_used=current_wall_seconds,
                max_wall_seconds=max_wall_s,
                session_retries_used=current_session_retries,
                max_session_retries=session_retry_cap,
                local_model_url=loc_url,
                local_model_id=loc_id,
                local_model_api_key=loc_api_key,
                model_degraded=model_degraded_loop,
                interactive_budget=interactive_loop,
                recent_files=turn_recent_files,
                **resume_kw,
            )

            ui.console.print()
            status = ui.console.status(f"[{ui.THEME['llm']}]● Working on it...[/]", spinner="dots")
            status.start()

            # Do not print stream chunks to the console: that bypasses the Panel and (on resume) duplicates
            # the same text. Final output is always shown once in print_step_llm (boxed).
            def _on_stream(_chunk: str) -> None:
                pass

            def _after_agent_llm(msg: AIMessage, step: int, est: int) -> None:
                status.stop()
                content = msg.content
                if not isinstance(content, str):
                    content = str(content) if content is not None else ""
                if not content:
                    content = "[Tool calls]"
                ui.print_step_llm(step, content, est)
                if getattr(msg, "tool_calls", None):
                    status.update(f"[{ui.THEME['tool']}]● Executing tools...[/]")
                    status.start()

            def _after_tool(
                _msg: ToolMessage,
                tool_name: str,
                tool_input: dict,
                content: str,
                step: int,
                est: int,
            ) -> None:
                status.stop()
                ui.print_step_tool(step, tool_name, tool_input, content, est)
                status.update(f"[{ui.THEME['llm']}]● Thinking...[/]")
                status.start()

            def _on_summarize_shown() -> None:
                status.stop()
                ui.print_summarizing()

            try:
                try:
                    result = run_agent_turn(
                        turn_inp,
                        db_path=db_path,
                        graph=graph,
                        callbacks={
                            "on_streaming_llm_text": _on_stream,
                            "after_agent_llm": _after_agent_llm,
                            "after_tool": _after_tool,
                            "on_summarize_shown": _on_summarize_shown,
                        },
                        log_steps_to_store=True,
                        step_counter_start=step_counter,
                    )
                except Exception:
                    _record_workspace_error_checkpoint(
                        session_id=session_id,
                        db_path=db_path,
                        workspace_root=str(Path.cwd()),
                        graph=graph,
                    )
                    raise
            finally:
                try:
                    status.stop()
                except Exception:
                    pass

            _record_workspace_halt_db(
                session_id=session_id,
                db_path=db_path,
                workspace_root=str(Path.cwd()),
                result=result,
            )
            if pending_resume_id and result.final_state is not None:
                store.clear_session_pending_resume(session_id, db_path)

            if result.final_state is not None:
                rf = result.final_state.get("recent_files")
                if isinstance(rf, list):
                    turn_recent_files = list(rf)

            step_counter = result.next_step_counter
            final_state = result.final_state
            streaming_budget_hit = result.streaming_token_budget_hit

            if streaming_budget_hit:
                current_tokens = result.current_tokens
                current_dollars = result.current_dollars
                current_wall_seconds = result.current_wall_seconds
                current_session_retries = result.current_session_retries
                ui.print_token_bar(
                    current_tokens,
                    max_tokens,
                    dollars_used=current_dollars,
                    max_dollars=max_api_dollars,
                    wall_seconds_used=current_wall_seconds,
                    max_wall_seconds=max_wall_s,
                )
                conversation_active = False

            if not final_state:
                ui.print_error("Graph execution failed - no final state returned")
                break

            while final_state.get("status") == "awaiting_budget":
                choice = ui.prompt_budget_extension(
                    tokens_used=int(final_state["tokens_used"]),
                    max_tokens=int(final_state["max_tokens"]),
                    dollars_used=float(final_state.get("api_dollars_used") or 0),
                    max_dollars=final_state.get("max_api_dollars"),
                    wall_used=float(final_state.get("wall_seconds_used") or 0),
                    max_wall_seconds=final_state.get("max_wall_seconds"),
                )
                if choice.stop_and_summarize:
                    final_state = summarize_node({**final_state, "status": "summarizing"})
                    break
                max_tokens += choice.extra_tokens
                if max_api_dollars is not None:
                    max_api_dollars += choice.extra_dollars
                if max_wall_s is not None:
                    max_wall_s += choice.extra_wall_seconds
                store.update_session_budget_caps(
                    session_id,
                    db_path,
                    max_tokens=max_tokens,
                    max_api_dollars=max_api_dollars,
                    max_wall_seconds=max_wall_s,
                )
                effective_model = str(final_state.get("model") or effective_model)
                model_degraded_loop = bool(final_state.get("model_degraded"))
                try:
                    result = run_agent_turn(
                        AgentTurnInput(
                            task=task,
                            session_id=session_id,
                            model=effective_model,
                            workspace_root=str(Path.cwd()),
                            tokens_used=int(final_state["tokens_used"]),
                            max_tokens=max_tokens,
                            api_dollars_used=float(final_state.get("api_dollars_used") or 0),
                            max_api_dollars=max_api_dollars,
                            wall_seconds_used=float(final_state.get("wall_seconds_used") or 0),
                            max_wall_seconds=max_wall_s,
                            session_retries_used=int(final_state.get("session_retries_used") or 0),
                            max_session_retries=session_retry_cap,
                            local_model_url=loc_url,
                            local_model_id=loc_id,
                            local_model_api_key=loc_api_key,
                            model_degraded=model_degraded_loop,
                            interactive_budget=interactive_loop,
                            append_user_message=False,
                            recent_files=turn_recent_files,
                        ),
                        db_path=db_path,
                        graph=graph,
                        callbacks={
                            "on_streaming_llm_text": _on_stream,
                            "after_agent_llm": _after_agent_llm,
                            "after_tool": _after_tool,
                            "on_summarize_shown": _on_summarize_shown,
                        },
                        log_steps_to_store=True,
                        step_counter_start=step_counter,
                    )
                except Exception:
                    _record_workspace_error_checkpoint(
                        session_id=session_id,
                        db_path=db_path,
                        workspace_root=str(Path.cwd()),
                        graph=graph,
                    )
                    raise
                _record_workspace_halt_db(
                    session_id=session_id,
                    db_path=db_path,
                    workspace_root=str(Path.cwd()),
                    result=result,
                )
                if result.final_state is not None:
                    rf2 = result.final_state.get("recent_files")
                    if isinstance(rf2, list):
                        turn_recent_files = list(rf2)
                step_counter = result.next_step_counter
                final_state = result.final_state
                if not final_state:
                    break

            if not final_state:
                ui.print_error("Graph execution failed - no final state returned")
                break

            current_tokens = int(final_state["tokens_used"])
            current_dollars = float(final_state.get("api_dollars_used", current_dollars))
            current_wall_seconds = float(final_state.get("wall_seconds_used", current_wall_seconds))
            current_session_retries = int(
                final_state.get("session_retries_used", current_session_retries)
            )

            effective_model = str(final_state.get("model") or effective_model)
            model_degraded_loop = bool(final_state.get("model_degraded"))

            # Print token bar after this turn
            ui.print_token_bar(
                current_tokens,
                max_tokens,
                dollars_used=current_dollars,
                max_dollars=max_api_dollars,
                wall_seconds_used=current_wall_seconds,
                max_wall_seconds=max_wall_s,
            )

            # Update session in database
            store.update_session(
                session_id,
                final_state["status"],
                current_tokens,
                final_state.get("summary"),
                current_dollars,
                current_wall_seconds,
                current_session_retries,
                db_path,
                active_model=effective_model,
                model_degraded=model_degraded_loop,
                local_model_url=loc_url,
                local_model_id=loc_id,
            )

            if model_degraded_loop and not prev_degraded_outer:
                ui.print_local_model_fallback(loc_url or "", effective_model)

            # Check if budget exceeded or summarizing (local fallback keeps the session open)
            degraded_now = model_degraded_loop
            budget_hit = (
                final_state["status"] in ("summarizing", "complete")
                or (max_wall_s is not None and current_wall_seconds >= max_wall_s)
                or (
                    not degraded_now
                    and (
                        current_tokens >= max_tokens
                        or (max_api_dollars is not None and current_dollars >= max_api_dollars)
                    )
                )
            )
            if budget_hit:
                ui.console.print()
                reasons: list[str] = []
                if current_tokens >= max_tokens:
                    reasons.append(f"Token budget exhausted ({current_tokens:,} / {max_tokens:,})")
                if max_api_dollars is not None and current_dollars >= max_api_dollars:
                    reasons.append(
                        f"API dollar budget reached (${current_dollars:.4f} / ${max_api_dollars:.4f} estimated)"
                    )
                if max_wall_s is not None and current_wall_seconds >= max_wall_s:
                    reasons.append(
                        f"Agent wall-time budget reached ({ui.format_wall_seconds(current_wall_seconds)} / "
                        f"{ui.format_wall_seconds(max_wall_s)})"
                    )
                if reasons:
                    ui.print_summarizing(" — ".join(reasons))
                else:
                    ui.print_summarizing()

                if final_state.get("summary"):
                    ui.console.print(f"[{ui.THEME['dim']}]Summary: {final_state['summary']}[/]")

                conversation_active = False
                break

            # Reset task for next iteration (will prompt user)
            task = None

        # Calculate elapsed time
        elapsed = time.time() - start_time

        # Print final session stats
        ui.console.print()
        ui.console.print("─" * 80)
        ui.console.print(f"[bold]Session Complete[/bold] ({elapsed:.1f}s)")
        ui.console.print(f"[bold]Total Tokens:[/bold] {current_tokens:,} / {max_tokens:,}")
        if max_api_dollars is not None:
            ui.console.print(
                f"[bold]Est. API spend:[/bold] ${current_dollars:.4f} / ${max_api_dollars:.4f} (LiteLLM pricing)"
            )
        ui.console.print(
            f"[bold]Agent wall time (LLM/tools/summary):[/bold] "
            f"{ui.format_wall_seconds(current_wall_seconds)} ({current_wall_seconds:.1f}s)"
        )
        if max_wall_s is not None:
            ui.console.print(
                f"[dim]Wall budget: {ui.format_wall_seconds(max_wall_s)} ({max_wall_s:.0f}s)[/dim]"
            )
        if session_retry_cap is not None:
            ui.console.print(
                f"[bold]LLM backoff retries used:[/bold] {current_session_retries} / {session_retry_cap}"
            )
        ui.console.print(f"[{ui.THEME['dim']}]Session ID: {session_id}[/]")
        ui.console.print(f"[{ui.THEME['dim']}]Resume: comptroller run --session {session_id}[/]")
        ui.console.print(
            f"[{ui.THEME['dim']}]Checkpoints: comptroller checkpoints list --session {session_id}[/]"
        )
        ui.console.print("─" * 80)
        ui.console.print()

    except SessionTransientRetryBudgetExhausted as e:
        ui.print_error(str(e))
        logger.exception("Session LLM transient retry budget exhausted")
        raise typer.Exit(1)
    except litellm.RateLimitError:
        ui.print_error(
            "Rate limit exceeded after automatic retries with backoff. "
            "Wait a few minutes, reduce how often you call the API, or check your provider quota."
        )
        logger.exception("LLM rate limit exhausted after retries")
        raise typer.Exit(1)
    except (litellm.APIConnectionError, litellm.Timeout, litellm.ServiceUnavailableError) as e:
        ui.print_error(
            f"Transient API error after retries: {e!s}. Check your network and try again."
        )
        logger.exception("LLM transient error exhausted retries")
        raise typer.Exit(1)
    except Exception as e:
        ui.print_error(f"Error during execution: {str(e)}")
        raise typer.Exit(1)


@app.command()
def sessions():
    """List all sessions."""
    db_path = str(Path(DB_PATH).expanduser())

    # Check if database exists
    if not Path(db_path).exists():
        ui.print_error("Database not found. Run 'init' first.")
        raise typer.Exit(1)

    sessions_list = store.list_sessions(db_path)
    ui.print_sessions_table(sessions_list)


@app.command()
def inspect(session: Annotated[str, typer.Option("--session")]):
    """Inspect a specific session in detail."""
    db_path = str(Path(DB_PATH).expanduser())

    # Check if database exists
    if not Path(db_path).exists():
        ui.print_error("Database not found. Run 'init' first.")
        raise typer.Exit(1)

    session_data = store.get_session(session, db_path)
    steps_data = store.get_steps(session, db_path)

    ui.print_inspect(session_data, steps_data)


checkpoints_app = typer.Typer(help="Git workspace snapshots at agent halts; diff and rollback.")


@checkpoints_app.command("list")
def checkpoints_list(session: Annotated[str, typer.Option("--session")]):
    """List recorded workspace checkpoints for a session."""
    db_path = str(Path(DB_PATH).expanduser())
    if not Path(db_path).exists():
        ui.print_error("Database not found. Run 'init' first.")
        raise typer.Exit(1)
    try:
        rows = store.list_workspace_checkpoints(session, db_path)
    except sqlite3.OperationalError as exc:
        ui.print_error(f"Cannot open database: {exc}")
        raise typer.Exit(1)
    ui.console.print(f"[bold]Session[/bold] {session}")
    ui.print_workspace_checkpoints_table(rows)


@checkpoints_app.command("show")
def checkpoints_show(
    session: Annotated[str, typer.Option("--session")],
    seq: Annotated[int, typer.Option("--seq")],
):
    """Show the unified diff for a checkpoint (parent → commit)."""
    from . import workspace_checkpoint as wscp

    db_path = str(Path(DB_PATH).expanduser())
    if not Path(db_path).exists():
        ui.print_error("Database not found. Run 'init' first.")
        raise typer.Exit(1)
    try:
        row = store.get_workspace_checkpoint(session, seq, db_path)
    except sqlite3.OperationalError as exc:
        ui.print_error(f"Cannot open database: {exc}")
        raise typer.Exit(1)
    if not row:
        ui.print_error(f"No checkpoint {seq} for session {session!r}")
        raise typer.Exit(1)
    root = Path(row["workspace_root"]).expanduser().resolve()
    diff = wscp.diff_commits(root, row.get("git_parent_sha"), row["git_commit_sha"])
    ui.print_workspace_checkpoint_diff(diff, title=f"Checkpoint {seq} ({row.get('reason', '')})")


@checkpoints_app.command("rollback")
def checkpoints_rollback(
    session: Annotated[str, typer.Option("--session")],
    seq: Annotated[int, typer.Option("--seq")],
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation prompt")] = False,
):
    """Restore the workspace git tree to a checkpoint and queue LangGraph resume on next run."""
    from . import workspace_checkpoint as wscp

    db_path = str(Path(DB_PATH).expanduser())
    if not Path(db_path).exists():
        ui.print_error("Database not found. Run 'init' first.")
        raise typer.Exit(1)
    try:
        session_row = store.get_session(session, db_path)
        row = store.get_workspace_checkpoint(session, seq, db_path)
    except sqlite3.OperationalError as exc:
        ui.print_error(f"Cannot open database: {exc}")
        raise typer.Exit(1)
    if not session_row:
        ui.print_error(f"Session {session!r} not found")
        raise typer.Exit(1)
    if not row:
        ui.print_error(f"No checkpoint {seq} for session {session!r}")
        raise typer.Exit(1)
    root = Path(row["workspace_root"]).expanduser().resolve()
    if not root.is_dir():
        ui.print_error(f"Workspace root is not a directory: {root}")
        raise typer.Exit(1)
    if not wscp.is_git_workspace(root):
        ui.print_error(f"Not a git workspace: {root}")
        raise typer.Exit(1)
    short = row["git_commit_sha"][:7]
    msg = (
        f"Restore files under\n  {root}\n"
        f"to git tree {short}… and (if stored) resume LangGraph from that halt?\n"
        "Confirm restore"
    )
    if not yes and not typer.confirm(msg, default=False):
        raise typer.Exit(0)
    err = wscp.restore_worktree(root, row["git_commit_sha"], mode="restore")
    if err:
        ui.print_error(err)
        raise typer.Exit(1)
    cid = row.get("langgraph_checkpoint_id")
    cns = row.get("langgraph_checkpoint_ns")
    if cid:
        store.set_session_pending_resume(session, str(cid), str(cns) if cns is not None else "", db_path)
        ui.console.print(
            f"[{ui.THEME['success']}]Git tree restored. Next "
            f"[bold]comptroller run --session {session}[/bold] resumes LangGraph from the stored checkpoint.[/]"
        )
    else:
        ui.console.print(
            f"[{ui.THEME['warning']}]Git tree restored; no LangGraph checkpoint id on this row — "
            "next run keeps the latest transcript (files-only rollback).[/]"
        )


app.add_typer(checkpoints_app, name="checkpoints")


if __name__ == "__main__":
    load_dotenv()
    app()
