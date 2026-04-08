import logging
import os
import time
import uuid
from pathlib import Path

import litellm
import typer
from typing_extensions import Annotated
from graph import summarize_node
import logging_config
import store
import tools as tools_mod
import ui
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
    logging_config.configure_logging()
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


@app.command()
def run(
    task: Annotated[str, typer.Argument()] = None,
    session: Annotated[str, typer.Option("--session")] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 100,
    model: Annotated[str, typer.Option("--model")] = "gpt-4o",
):
    """Run an agent task with token budget tracking. Supports continuous conversation mode."""
    from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
    from budget import tokens_from_llm_message
    from graph import build_graph

    load_dotenv()  # Load .env file
    logging_config.configure_logging()
    ui.print_banner()

    if not _available_llm_models():
        ui.print_error(
            "No LLM provider API keys detected. Run 'init' and configure credentials "
            "(e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY)."
        )
        raise typer.Exit(1)

    db_path = str(Path(DB_PATH).expanduser())

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

        if tokens_used_so_far >= max_tokens:
            ui.print_error(f"Session {session_id} has exceeded token budget of {max_tokens:,}")
            ui.console.print("[dim]Please reset the session")
            raise typer.Exit(1)
        original_task = session_data["task"]

        ui.console.print(f"[bold]Resuming session {session_id}[/bold]")
        ui.console.print(f"[dim]Original task: {original_task}[/dim]")
        ui.console.print(f"[dim]Budget: {tokens_used_so_far:,} / {max_tokens:,} tokens used[/dim]")
        ui.console.print()
    else:
        # Create new session
        if not task:
            ui.print_error("Task required for new session. Use: python main.py run \"your task\"")
            raise typer.Exit(1)

        session_id = str(uuid.uuid4())[:8]
        tokens_used_so_far = 0
        store.create_session(session_id, task, max_tokens, db_path)

        # Print session header
        ui.print_session_header(session_id, task, model, max_tokens)

    # Track start time
    start_time = time.time()
    step_counter = 0

    try:
        # Build graph
        graph = build_graph(db_path)
        config = {"configurable": {"thread_id": session_id}}

        # Continuous conversation loop
        current_tokens = tokens_used_so_far
        conversation_active = True

        while conversation_active:
            # Get task for this turn
            if not task:
                ui.console.print()
                remaining = max_tokens - current_tokens
                ui.console.print(f"[{ui.THEME['dim']}]Remaining budget: {remaining:,} tokens[/]")
                task = ui.console.input(f"[{ui.THEME['accent']}]Task (or 'quit' to exit): [/]").strip()

                if task.lower() in ['quit', 'exit', 'q', '']:
                    ui.console.print("[dim]Exiting session...[/dim]")
                    break

            # Prepare state for this turn
            turn_state = {
                "session_id": session_id,
                "task": task,
                "model": model,
                "workspace_root": str(Path.cwd()),
                "messages": [HumanMessage(content=task)],
                "tool_results": [],
                "tokens_used": current_tokens,
                "max_tokens": max_tokens,
                "status": "running",
                "summary": None
            }

            # Stream through graph and display outputs
            final_state = None
            tools_mod.set_workspace_root_for_tools(turn_state["workspace_root"])

            # Show thinking animation at the start
            ui.console.print()
            status = ui.console.status(f"[{ui.THEME['llm']}]● Working on it...[/]", spinner="dots")
            status.start()

            logger.info(
                "Graph stream start session_id=%s model=%s tokens_used=%s max_tokens=%s task=%r",
                session_id,
                model,
                current_tokens,
                max_tokens,
                task,
            )
            logger.info(
                "Initial messages for this turn (before system prompt in graph): %s",
                _log_repr_preview(turn_state["messages"][0]),
            )

            token_count = 0
            ai_message = ""
            budget_exhausted = False
            for chunk in graph.stream(turn_state, config, stream_mode=["messages", "updates"]):
                mode, data = chunk
                logger.debug("stream chunk mode=%s", mode)
                if mode == "messages":
                    message_chunk, metadata = data
                    node = metadata.get("langgraph_node")
                    logger.debug(
                        "stream messages langgraph_node=%s chunk=%s",
                        node,
                        _log_repr_preview(message_chunk),
                    )

                    # Stream text tokens live from agent node only
                    if node == "agent" and message_chunk.content:
                        ai_message += message_chunk.content
                        # print(message_chunk.content, end="")
                        token_count += 1
                    
                    if token_count >= 5:
                        # Stop spinner after receiving some tokens
                        status.stop()
                
                elif mode == "updates":
                    if not isinstance(data, dict):
                        logger.debug(
                            "stream updates non-dict payload: %s",
                            _log_repr_preview(data),
                        )
                        continue
                    logger.debug("stream updates node keys: %s", list(data.keys()))

                    for node_name, node_state in data.items():
                        final_state = node_state

                        # Display outputs based on node type
                        if node_name == "agent":
                            # Stop thinking spinner
                            status.stop()

                            # LLM response
                            if node_state.get("messages"):
                                last_msg = node_state["messages"][-1]
                                if isinstance(last_msg, AIMessage):
                                    step_counter += 1
                                    content = last_msg.content if last_msg.content else "[Tool calls]"
                                    tokens_this_step = tokens_from_llm_message(last_msg, model)
                                    ui.print_step_llm(step_counter, content, tokens_this_step)

                                    # Log to database
                                    store.log_step(session_id, step_counter, "llm_response", content, tokens_this_step, db_path)

                                    # If agent called tools, show "Executing tools..." status
                                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                        status.update(f"[{ui.THEME['tool']}]● Executing tools...[/]")
                                        status.start()

                        elif node_name == "tools":
                            # Stop tool execution spinner
                            status.stop()

                            # Tool execution
                            if node_state.get("messages"):
                                # Find tool messages
                                for msg in reversed(node_state["messages"]):
                                    if isinstance(msg, ToolMessage):
                                        step_counter += 1
                                        tool_name = msg.name if hasattr(msg, 'name') else "tool"
                                        content = msg.content

                                        # Get tool input from previous AI message
                                        tool_input = {}
                                        for prev_msg in reversed(node_state["messages"]):
                                            if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                                                for tc in prev_msg.tool_calls:
                                                    if hasattr(msg, 'tool_call_id') and tc.get('id') == msg.tool_call_id:
                                                        tool_input = tc.get('args', {})
                                                        tool_name = tc.get('name', tool_name)
                                                        break
                                                break

                                        # Estimate tokens
                                        from budget import count_tokens
                                        tokens_this_step = count_tokens(content)

                                        ui.print_step_tool(step_counter, tool_name, tool_input, content, tokens_this_step)

                                        # Log to database
                                        store.log_step(session_id, step_counter, "tool_call", f"{tool_name}: {content[:200]}", tokens_this_step, db_path)
                                        break  # Only show the latest tool message

                                # After tools, agent will think again
                                status.update(f"[{ui.THEME['llm']}]● Thinking...[/]")
                                status.start()

                        if node_name == "summarize":
                            # Stop spinner for summarization
                            status.stop()

                            # Show summarizing notice
                            if node_state.get("status") == "complete":
                                ui.print_summarizing()

            # Make sure spinner is stopped at the end
            try:
                status.stop()
            except:
                pass

            if budget_exhausted:                
                partial_state = {
                    **turn_state,
                    "messages": turn_state["messages"] + [AIMessage(content=ai_message)],
                    "tokens_used": current_tokens + token_count,
                    "status": "summarizing"
                }
                summary_result = summarize_node(partial_state)
                status.stop()
                final_state = summary_result
                current_tokens = partial_state["tokens_used"]
                ui.print_token_bar(current_tokens, max_tokens)
                conversation_active = False

            if not final_state:
                ui.print_error("Graph execution failed - no final state returned")
                break

            # Update current token count
            current_tokens = final_state["tokens_used"]

            # Print token bar after this turn
            ui.print_token_bar(current_tokens, max_tokens)

            # Update session in database
            store.update_session(
                session_id,
                final_state["status"],
                current_tokens,
                final_state.get("summary"),
                db_path,
            )

            # Check if budget exceeded or summarizing
            if final_state["status"] == "summarizing" or current_tokens >= max_tokens:
                ui.console.print()
                ui.console.print(f"[{ui.THEME['warning']}]⚠ Token budget exhausted ({current_tokens:,} / {max_tokens:,})[/]")

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
        ui.console.print(f"[{ui.THEME['dim']}]Session ID: {session_id}[/]")
        ui.console.print(f"[{ui.THEME['dim']}]Resume: python main.py run --session {session_id}[/]")
        ui.console.print("─" * 80)
        ui.console.print()

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
    app()
