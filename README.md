# agent-runtime-mvp

A minimal, working budget-aware agent runtime MVP in Python.

## Features

- **Token Budget Tracking**: Track and enforce token usage limits across agent execution
- **LangGraph Orchestration**: Agent → Tools → Budget Gate → Loop/Summarize
- **SQLite Persistence**: All sessions and steps saved to database
- **Rich Terminal UI**: Clean, colorful terminal output with live token usage bar
- **Four Built-in Tools**: read_file, write_file, list_directory, run_shell

## Installation

```bash
cd agent-runtime-mvp
uv sync
export OPENAI_API_KEY=sk-...
uv run python main.py init
```

## Usage

### Initialize
```bash
uv run python main.py init
```

### Run a Task (Continuous Conversation Mode)

**Start a new session:**
```bash
uv run python main.py run "list all python files"
```

The agent will execute the task, then **prompt you for more tasks** in the same session:
```
Task (or 'quit' to exit): now count the total lines in all python files
```

Budget is tracked continuously across all tasks in the conversation!

**Resume an existing session:**
```bash
uv run python main.py run --session abc12345
```

Options:
- `--session`: Resume existing session with preserved budget
- `--max-tokens`: Token budget limit (default: 5000)
- `--model`: OpenAI model to use (default: gpt-4o)

### List Sessions
```bash
uv run python main.py sessions
```

### Inspect Session
```bash
uv run python main.py inspect --session abc12345
```

## Project Structure

```
agent-runtime-mvp/
├── pyproject.toml       # Dependencies
├── main.py              # Typer CLI entry point
├── state.py             # AgentState TypedDict
├── budget.py            # TokenBudget class + count_tokens()
├── graph.py             # LangGraph orchestration
├── tools.py             # Built-in tools
├── store.py             # SQLite persistence
├── ui.py                # Rich terminal output
└── tests/
    ├── test_budget.py
    └── test_graph.py
```

## How It Works

1. User runs a task with a token budget
2. LangGraph agent calls LLM with tools
3. LLM returns tool calls → tools execute → results counted against budget
4. Loop continues until:
   - Task completes (LLM returns no tool calls) → Exit 0
   - Budget exceeded → Summarize → Exit 1
5. All steps persisted to SQLite throughout

## Tests

```bash
uv run pytest tests/ -v
```

All tests use mocked LLM calls - no real API requests.

## What's NOT Included

This is an MVP focused on one budget dimension (tokens) and one termination policy (summarize). Explicitly excluded:

- Cost tracking
- Time tracking
- Tool call limits
- Retry limits
- Ask-human policy
- Config files
- Async/threading
- FastAPI server

These are Week 2+ features.

## License

MIT
