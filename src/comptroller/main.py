import logging
import os
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
):
    """Run an agent task with token, optional dollar, and optional wall-time budgets. Supports continuous conversation mode."""
    from langchain_core.messages import AIMessage, ToolMessage

    from .graph import build_graph
    from .runner import AgentTurnInput, run_agent_turn
    load_dotenv()
    ui.print_banner()

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
        model_degraded_loop = bool(int(session_data.get("model_degraded") or 0))
        effective_model = (session_data.get("active_model") or model).strip() or model
        if model_degraded_loop and (not loc_url or not loc_id):
            ui.print_error(
                f"Session {session_id} is marked as using a local model, but local URL or model id is missing."
            )
            raise typer.Exit(1)

        if tokens_used_so_far >= max_tokens and not model_degraded_loop:
            ui.print_error(f"Session {session_id} has exceeded token budget of {max_tokens:,}")
            ui.console.print("[dim]Please reset the session")
            raise typer.Exit(1)
        if max_api_dollars is not None and dollars_used_so_far >= max_api_dollars and not model_degraded_loop:
            ui.print_error(
                f"Session {session_id} has exceeded API dollar budget "
                f"(${dollars_used_so_far:.4f} / ${max_api_dollars:.4f})"
            )
            ui.console.print("[dim]Please reset the session")
            raise typer.Exit(1)
        if max_wall_s is not None and wall_seconds_so_far >= max_wall_s:
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

            prev_degraded = model_degraded_loop

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
                model_degraded=model_degraded_loop,
            )

            ui.console.print()
            status = ui.console.status(f"[{ui.THEME['llm']}]● Working on it...[/]", spinner="dots")
            status.start()

            def _on_stream(chunk: str) -> None:
                print(chunk, end="")

            def _after_agent_llm(msg: AIMessage, step: int, est: int) -> None:
                status.stop()
                content = msg.content if msg.content else "[Tool calls]"
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
            finally:
                try:
                    status.stop()
                except Exception:
                    pass

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

            current_tokens = result.current_tokens
            current_dollars = result.current_dollars
            current_wall_seconds = result.current_wall_seconds
            current_session_retries = result.current_session_retries

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

            if model_degraded_loop and not prev_degraded:
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
        ui.console.print(f"[{ui.THEME['dim']}]Resume: python main.py run --session {session_id}[/]")
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


if __name__ == "__main__":
    load_dotenv()
    app()
