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


def print_banner():
    """Print product banner."""
    console.print()
    console.print(Panel(
        "[bold cyan]agent-runtime-mvp[/bold cyan] [dim]v0.1.0[/dim]\n"
        "Budget-aware agent runtime with token tracking",
        border_style=THEME["accent"]
    ))
    console.print()


def print_session_header(session_id: str, task: str, model: str, max_tokens: int):
    """Print session header before execution."""
    console.print(Panel(
        f"[bold]Session:[/bold] {session_id}\n"
        f"[bold]Task:[/bold] {task}\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Max Tokens:[/bold] {max_tokens:,}",
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


def print_token_bar(used: int, max_tokens: int):
    """Print inline token usage bar."""
    pct = (used / max_tokens) * 100
    bar_width = 20
    filled = int((used / max_tokens) * bar_width)
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
    console.print()


def print_summarizing():
    """Print notice that summarization is starting."""
    console.print(Panel(
        "[bold]Token budget exceeded - generating summary...[/bold]",
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
    console.print(Panel(
        f"[bold]Session ID:[/bold] {session['session_id']}\n"
        f"[bold]Task:[/bold] {session['task']}\n"
        f"[bold]Status:[/bold] {session['status']}\n"
        f"[bold]Tokens:[/bold] {session['tokens_used']:,} / {session['max_tokens']:,}\n"
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
