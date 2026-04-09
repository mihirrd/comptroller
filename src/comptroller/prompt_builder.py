"""Load and fill the system prompt template; token-safe truncation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from .budget import count_tokens
from .tools import TOOLS

logger = logging.getLogger(__name__)

# Same directory as this file: prompts/system.md
_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "system.md"

DEFAULT_MAX_SYSTEM_PROMPT_TOKENS = 8_000
_TRUNCATION_NOTICE = "\n\n[... system prompt truncated to respect max_prompt_tokens ...]"


def _template_text() -> str:
    if not _TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"System prompt template missing: {_TEMPLATE_PATH}")
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def _format_workspace(workspace_root: str | None) -> str:
    if workspace_root:
        return f"Working directory / workspace: `{workspace_root}`"
    return "Working directory / workspace: *(not set yet — defaults will apply when wired)*"


def _format_tool_catalog() -> str:
    lines: list[str] = []
    for t in TOOLS:
        name = getattr(t, "name", None) or ""
        desc = (getattr(t, "description", None) or "").strip()
        first_line = desc.split("\n", 1)[0] if desc else ""
        lines.append(f"- **{name}**: {first_line}")
    return "\n".join(lines) if lines else "(no tools registered)"


def _format_tool_digest(digest: str | None) -> str:
    if digest and digest.strip():
        return digest.strip()
    return "(none — full tool traces are still in the message list when present)"


def _truncate_to_max_tokens(text: str, max_tokens: int, model: str) -> str:
    if max_tokens <= 0 or count_tokens(text, model=model) <= max_tokens:
        return text
    # Binary search on prefix length to avoid quadratic retokenization.
    lo, hi = 0, len(text)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid] + _TRUNCATION_NOTICE
        if count_tokens(candidate, model=model) <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return text[:best] + _TRUNCATION_NOTICE


def build_system_prompt(
    state: Mapping[str, Any],
    *,
    max_prompt_tokens: int | None = None,
    extra_replacements: Mapping[str, str] | None = None,
) -> str:
    """Build the system prompt from ``prompts/system.md`` and ``state``.

    Recognized optional keys: ``workspace_root``, ``tool_digest`` (short string). Uses
    ``state["model"]`` for tiktoken when counting tokens.

    Parameters
    ----------
    state:
        Must include ``task``. Typically an :class:`AgentState` dict.
    max_prompt_tokens:
        Upper bound on prompt tokens (tiktoken). ``None`` uses
        ``DEFAULT_MAX_SYSTEM_PROMPT_TOKENS``.
    extra_replacements:
        Optional extra ``str.format`` keys for ``prompts/system.md`` (``{name}``).
    """
    task = state.get("task")
    if task is None or (isinstance(task, str) and not task.strip()):
        raise ValueError("build_system_prompt requires a non-empty state['task']")

    model = state.get("model") or "gpt-4o"
    cap = max_prompt_tokens if max_prompt_tokens is not None else DEFAULT_MAX_SYSTEM_PROMPT_TOKENS

    workspace_root = state.get("workspace_root")
    if workspace_root is not None and not isinstance(workspace_root, str):
        workspace_root = str(workspace_root)

    tool_digest = state.get("tool_digest")
    if tool_digest is not None and not isinstance(tool_digest, str):
        tool_digest = str(tool_digest)

    replacements: dict[str, str] = {
        "task": str(task).strip(),
        "workspace_root": _format_workspace(workspace_root),
        "tool_catalog": _format_tool_catalog(),
        "tool_digest": _format_tool_digest(tool_digest),
    }
    if extra_replacements:
        replacements.update({k: str(v) for k, v in extra_replacements.items()})

    template = _template_text()
    try:
        body = template.format(**replacements)
    except KeyError as e:
        raise KeyError(f"system.md references unknown placeholder: {e}") from e

    out = _truncate_to_max_tokens(body, cap, model=model)
    if out != body:
        logger.info(
            "System prompt truncated: tokens before=%s cap=%s model=%s",
            count_tokens(body, model=model),
            cap,
            model,
        )
    return out
