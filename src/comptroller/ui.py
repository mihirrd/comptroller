from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import BarColumn, Progress, TextColumn
from datetime import datetime

console = Console()

THEME = {
    "accent": "cyan",
    "dim": "dim white",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "tool": "magenta",
    "llm": "cyan",
}


def format_wall_seconds(seconds: float) -> str:
    """Human-friendly duration for agent wall-time totals."""
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.1f}s"
    total = int(round(s))
    m, sec = divmod(total, 60)
    if m < 60:
        return f"{m}m {sec}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {sec}s"


def print_banner():
    """Print product banner."""
    console.print()
    console.print(Panel(
        "[bold cyan]agent-runtime-mvp[/bold cyan] [dim]v0.1.0[/dim]\n"
        "Budget-aware agent runtime with token tracking",
        border_style=THEME["accent"]
    ))
    console.print()


def print_session_header(
    session_id: str,
    task: str,
    model: str,
    max_tokens: int,
    *,
    max_api_dollars: float | None = None,
    max_wall_seconds: float | None = None,
    local_model_url: str | None = None,
    local_model_id: str | None = None,
):
    """Print session header before execution."""
    lines = [
        f"[bold]Session:[/bold] {session_id}",
        f"[bold]Task:[/bold] {task}",
        f"[bold]Model:[/bold] {model}",
        f"[bold]Max Tokens:[/bold] {max_tokens:,}",
    ]
    if local_model_url and local_model_id:
        lines.append(
            f"[bold]Local fallback:[/bold] {local_model_id} @ {local_model_url} "
            f"(used if token or API dollar budget is exceeded)"
        )
    if max_api_dollars is not None:
        lines.append(f"[bold]Max API spend (est.):[/bold] ${max_api_dollars:.4f} USD (LiteLLM pricing)")
    if max_wall_seconds is not None:
        lines.append(
            f"[bold]Max agent wall time:[/bold] {format_wall_seconds(max_wall_seconds)} "
            f"({max_wall_seconds:.0f}s)"
        )
    console.print(Panel(
        "\n".join(lines),
        title="[bold]Starting Session[/bold]",
        border_style=THEME["accent"]
    ))
    console.print()


def print_step_tool(step: int, tool_name: str, tool_input: dict, result: str, tokens: int):
    """Print tool execution step."""
    # Create input preview (first 50 chars)
    input_str = str(tool_input)
    input_preview = input_str[:50] + "..." if len(input_str) > 50 else input_str

    console.print()
    console.print(
        f"[{THEME['dim']}]#{step}[/] [{THEME['tool']}]▶ {tool_name}[/]   "
        f"[{THEME['dim']}]{input_preview}[/]   "
        f"[{THEME['accent']}]+{tokens} tokens[/]"
    )

    # Print FULL result in a panel for emphasis
    console.print(Panel(
        result,
        border_style=THEME["tool"],
        padding=(0, 1)
    ))
    console.print()


def print_step_llm(step: int, content: str, tokens: int):
    """Print LLM response step."""
    console.print()
    console.print(
        f"[{THEME['dim']}]#{step}[/] [{THEME['llm']}]◆ llm:reason[/]   "
        f"[{THEME['accent']}]+{tokens} tokens[/]"
    )

    # Print full content with emphasis
    console.print(Panel(
        content,
        border_style=THEME["llm"],
        padding=(0, 1)
    ))
    console.print()


def print_token_bar(
    used: int,
    max_tokens: int,
    *,
    dollars_used: float | None = None,
    max_dollars: float | None = None,
    wall_seconds_used: float | None = None,
    max_wall_seconds: float | None = None,
):
    """Print inline token usage bar; optional lines for API dollar and wall-time budgets."""
    pct = (used / max_tokens) * 100 if max_tokens else 0.0
    bar_width = 20
    filled = int((used / max_tokens) * bar_width) if max_tokens else 0
    bar = "█" * filled + "░" * (bar_width - filled)

    # Color based on percentage
    if pct < 70:
        color = "green"
    elif pct < 90:
        color = "yellow"
    else:
        color = "red"

    console.print(
        f"[bold]tokens:[/bold] [{color}]{used:,} / {max_tokens:,}[/{color}]  "
        f"[{color}]{bar}[/{color}]  "
        f"[{color}]{pct:.0f}%[/{color}]"
    )
    if max_dollars is not None and dollars_used is not None and max_dollars > 0:
        dpct = min(100.0, (dollars_used / max_dollars) * 100)
        dfilled = int((dollars_used / max_dollars) * bar_width)
        dbar = "█" * min(bar_width, dfilled) + "░" * max(0, bar_width - min(bar_width, dfilled))
        if dpct < 70:
            dcolor = "green"
        elif dpct < 90:
            dcolor = "yellow"
        else:
            dcolor = "red"
        console.print(
            f"[bold]API $ (est.):[/bold] [{dcolor}]${dollars_used:.4f} / ${max_dollars:.4f}[/{dcolor}]  "
            f"[{dcolor}]{dbar}[/{dcolor}]  "
            f"[{dcolor}]{dpct:.0f}%[/{dcolor}]"
        )
    if max_wall_seconds is not None and wall_seconds_used is not None and max_wall_seconds > 0:
        wpct = min(100.0, (wall_seconds_used / max_wall_seconds) * 100)
        wfilled = int((wall_seconds_used / max_wall_seconds) * bar_width)
        wbar = "█" * min(bar_width, wfilled) + "░" * max(0, bar_width - min(bar_width, wfilled))
        if wpct < 70:
            wcolor = "green"
        elif wpct < 90:
            wcolor = "yellow"
        else:
            wcolor = "red"
        wu = format_wall_seconds(wall_seconds_used)
        wm = format_wall_seconds(max_wall_seconds)
        console.print(
            f"[bold]agent wall:[/bold] [{wcolor}]{wu} / {wm}[/{wcolor}]  "
            f"[{wcolor}]{wbar}[/{wcolor}]  "
            f"[{wcolor}]{wpct:.0f}%[/{wcolor}]"
        )
    console.print()


def print_local_model_fallback(api_base: str, litellm_model: str) -> None:
    """Notify that the session switched to the configured OpenAI-compatible endpoint."""
    console.print()
    console.print(
        Panel(
            f"Cloud token or dollar budget was reached.\n"
            f"Continuing on local model [bold]{litellm_model}[/] at [bold]{api_base}[/].\n"
            f"[dim]Wall-time budget (if any) still applies.[/dim]",
            title=f"[{THEME['warning']}]Model fallback[/]",
            border_style=THEME["warning"],
        )
    )
    console.print()


def print_summarizing(reason: str = "Token budget exceeded"):
    """Print notice that summarization is starting."""
    console.print(Panel(
        f"[bold]{reason} — generating summary...[/bold]",
        border_style=THEME["warning"]
    ))
    console.print()


def print_completion(session_id: str, summary: str, tokens_used: int, max_tokens: int, elapsed: float):
    """Print final completion panel."""
    status_color = THEME["warning"] if tokens_used >= max_tokens else THEME["success"]

    console.print()
    console.print("─" * 80)
    console.print(f"[bold]Session Complete[/bold] ({elapsed:.1f}s)")
    console.print()

    # Print summary as simple text, not in a big panel
    if summary and summary != "Task completed successfully":
        console.print(f"[{THEME['dim']}]Summary:[/]")
        console.print(f"  {summary}")
        console.print()

    console.print(f"[bold]Tokens:[/bold] [{status_color}]{tokens_used:,} / {max_tokens:,}[/{status_color}]")
    console.print(f"[{THEME['dim']}]Inspect: python main.py inspect --session {session_id}[/]")
    console.print("─" * 80)
    console.print()


def print_sessions_table(sessions: list[dict]):
    """Print table of all sessions."""
    if not sessions:
        console.print("[dim]No sessions found[/dim]")
        return

    table = Table(title="Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Task", style="white")
    table.add_column("Status", style="yellow")
    table.add_column("Tokens", justify="right")
    table.add_column("Created", style="dim")

    for session in sessions:
        task_preview = session["task"][:50] + "..." if len(session["task"]) > 50 else session["task"]
        tokens = f"{session['tokens_used']:,} / {session['max_tokens']:,}"
        cap = session.get("max_api_dollars")
        spent = float(session.get("api_dollars_used") or 0)
        if cap is not None:
            tokens += f"\n${spent:.3f}/${float(cap):.2f}"
        wcap = session.get("max_wall_seconds")
        wspent = float(session.get("wall_seconds_used") or 0)
        if wcap is not None:
            tokens += f"\n{format_wall_seconds(wspent)}/{format_wall_seconds(float(wcap))}"
        rcap = session.get("max_session_retries")
        rused = int(session.get("session_retries_used") or 0)
        if rcap is not None:
            tokens += f"\nretries {rused}/{int(rcap)}"
        created = datetime.fromtimestamp(session["created_at"]).strftime("%Y-%m-%d %H:%M")

        table.add_row(
            session["session_id"],
            task_preview,
            session["status"],
            tokens,
            created
        )

    console.print(table)
    console.print()


def print_inspect(session: dict, steps: list[dict]):
    """Print full session detail."""
    if not session:
        console.print(f"[{THEME['error']}]Session not found[/]")
        return

    # Session header
    cap = session.get("max_api_dollars")
    spent = float(session.get("api_dollars_used") or 0)
    dollar_line = ""
    if cap is not None:
        dollar_line = f"\n[bold]Est. API spend:[/bold] ${spent:.4f} / ${float(cap):.4f} (LiteLLM)"
    wcap = session.get("max_wall_seconds")
    wspent = float(session.get("wall_seconds_used") or 0)
    wall_line = ""
    if wcap is not None:
        wall_line = (
            f"\n[bold]Agent wall time:[/bold] {format_wall_seconds(wspent)} / "
            f"{format_wall_seconds(float(wcap))} ({wspent:.1f}s / {float(wcap):.0f}s)"
        )
    elif wspent > 0:
        wall_line = f"\n[bold]Agent wall time:[/bold] {format_wall_seconds(wspent)} ({wspent:.1f}s)"
    rcap = session.get("max_session_retries")
    rused = int(session.get("session_retries_used") or 0)
    retry_line = ""
    if rcap is not None:
        retry_line = f"\n[bold]LLM backoff retries:[/bold] {rused} / {int(rcap)}"
    elif rused > 0:
        retry_line = f"\n[bold]LLM backoff retries used:[/bold] {rused}"
    am = session.get("active_model") or session.get("primary_model")
    model_line = f"\n[bold]Active model:[/bold] {am}" if am else ""
    degraded = int(session.get("model_degraded") or 0)
    loc_u = session.get("local_model_url")
    loc_i = session.get("local_model_id")
    local_line = ""
    if loc_u and loc_i:
        local_line = f"\n[bold]Local fallback:[/bold] {loc_i} @ {loc_u}"
    if degraded:
        local_line += "\n[bold]Model mode:[/bold] local fallback (after cloud budget)"
    console.print(Panel(
        f"[bold]Session ID:[/bold] {session['session_id']}\n"
        f"[bold]Task:[/bold] {session['task']}\n"
        f"[bold]Status:[/bold] {session['status']}\n"
        f"[bold]Tokens:[/bold] {session['tokens_used']:,} / {session['max_tokens']:,}"
        f"{dollar_line}{wall_line}{retry_line}{model_line}{local_line}\n"
        f"[bold]Created:[/bold] {datetime.fromtimestamp(session['created_at']).strftime('%Y-%m-%d %H:%M:%S')}",
        title="[bold]Session Details[/bold]",
        border_style=THEME["accent"]
    ))
    console.print()

    # Steps table
    if steps:
        table = Table(title="Steps")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Type", style="cyan")
        table.add_column("Content", style="white")
        table.add_column("Tokens", justify="right", style="yellow")

        for step in steps:
            content_preview = step["content"][:80] + "..." if len(step["content"]) > 80 else step["content"]
            table.add_row(
                str(step["step"]),
                step["event_type"],
                content_preview,
                str(step["tokens"])
            )

        console.print(table)
        console.print()

    # Summary
    if session.get("summary"):
        console.print(Panel(
            session["summary"],
            title="[bold]Summary[/bold]",
            border_style=THEME["success"]
        ))
        console.print()


def print_error(message: str):
    """Print error panel."""
    console.print(Panel(
        f"[{THEME['error']}]{message}[/]",
        title="[bold]Error[/bold]",
        border_style=THEME["error"]
    ))
    console.print()


def print_available_models(models: list[str], max_rows: int = 60) -> None:
    """Print models available given current credentials (LiteLLM get_valid_models)."""
    if not models:
        console.print(f"[{THEME['dim']}]No models listed (no provider credentials detected).[/]")
        console.print()
        return

    total = len(models)
    shown = models[:max_rows]
    overflow = total - len(shown)

    table = Table(title="Available models (from your credentials)")
    table.add_column("Model", style="cyan")

    for name in shown:
        table.add_row(name)

    console.print(table)
    if overflow > 0:
        console.print(
            f"[{THEME['dim']}]… and {overflow:,} more (see LiteLLM docs for full model ids).[/]"
        )
    console.print()
