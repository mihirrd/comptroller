# Comptroller - A Budget-aware Agentic Runtime

A minimal, budget-aware agent runtime that tracks token usage across a continuous conversation loop, with SQLite persistence and a rich terminal UI.

## Why this exists

Most “agent” demos ignore resource limits. This MVP makes the budget a first-class constraint: each tool call and LLM response is counted, and the run ends with a summary when the limit is hit.

## Features

- Token budget tracking across LLM calls and tool outputs.
- LangGraph orchestration with a budget gate and summarization fallback.
- SQLite persistence for sessions and step-by-step execution history.
- Rich terminal UI with a live token bar and formatted outputs.
- Built-in tools: read_file, write_file, list_directory, run_shell.
- LiteLLM model support (OpenAI, Anthropic, Gemini, Groq, etc.).

## Requirements

- Python >= 3.13
- `uv` (recommended) or any Python environment manager
- One or more provider API keys supported by LiteLLM

## Installation

```bash
uv sync
export OPENAI_API_KEY=sk-...
uv run comptroller init
```

If you use another provider, set the corresponding environment variable (see LiteLLM docs).

## Usage

### Initialize the runtime
```bash
uv run comptroller init
```

### Start a new session
```bash
uv run comptroller run "list all python files"
```

The session stays open for follow-up tasks until you quit or the budget is exhausted:
```
Task (or 'quit' to exit): count the total lines in all python files
```

### Resume a session
```bash
uv run comptroller run --session abc12345
```

### List sessions
```bash
uv run comptroller sessions
```

### Inspect a session
```bash
uv run comptroller inspect --session abc12345
```

### Clear all sessions
```bash
uv run comptroller clear-sessions
```

### Key options

- `--max-tokens`: Token budget limit (default: 1000 or value from `max_tokens` env var)
- `--model`: LiteLLM model id (default: `gpt-4o` or value from `model` env var)
- `--session`: Resume an existing session

## How it works

1. A task starts a session with a token budget.
2. The agent node calls the LLM with tool bindings.
3. If tool calls are requested, the tool node executes them and counts tokens from outputs.
4. The budget gate routes back to the agent or into summarization when the budget is hit.
5. Steps and session metadata are stored in SQLite throughout the run.

## Data and logs

- Session database: `~/.agent-runtime-mvp/sessions.db`
- Execution log file: `logs.txt` (repo root)

## Project structure

```
.
├── src/
│   └── comptroller/
│       ├── __init__.py
│       ├── main.py              # Typer CLI entry point
│       ├── graph.py             # LangGraph orchestration and nodes
│       ├── state.py             # AgentState TypedDict
│       ├── budget.py            # TokenBudget and token counting
│       ├── tools.py             # Built-in tools
│       ├── store.py             # SQLite persistence
│       ├── ui.py                # Rich terminal UI
│       └── logging_config.py    # File-only logging setup
├── tests/                       # Budget + graph tests
```

## Tests

```bash
uv run pytest -v
```

All tests use mocked LLM calls (no real API requests).

## Notes and limitations

- This MVP focuses on token budget enforcement; it does not track cost or time.
- Tools are intentionally minimal and run in-process.
- SQLite checkpointer is used for LangGraph state persistence.

## License

MIT
