"""Build the agent task string from a Hugging Face SWE-bench row (e.g. Lite)."""

from __future__ import annotations

from typing import Any


def build_task_prompt(instance: dict[str, Any], *, workspace_root: str) -> str:
    """Compose the user task for :class:`comptroller.runner.AgentTurnInput`.

    Parameters
    ----------
    instance:
        A dataset row dict (e.g. from ``princeton-nlp/SWE-bench_Lite``).
    workspace_root:
        Absolute path to the repo checkout the agent must edit (bind-mount or local clone).
    """
    problem = (instance.get("problem_statement") or "").strip()
    hints = (instance.get("hints_text") or "").strip()
    iid = instance.get("instance_id") or "(unknown instance_id)"

    parts: list[str] = [
        f"## Instance\n\n`{iid}`\n",
        "## Problem\n\n",
        problem,
    ]
    if hints:
        parts.extend(["\n\n## Hints\n\n", hints])
    parts.extend(
        [
            "\n\n## Workspace\n\n",
            f"Your working directory is the repository root: `{workspace_root}`.\n",
            "Use tools (`read_file`, `grep`, `glob_files`, etc.) to explore and "
            "`write_file`, `search_replace`, or `apply_patch` to fix the issue.\n",
            "Use `run_shell` to run tests or linters from the correct directory; "
            "prefer targeted commands (e.g. a single test file or `pytest path -q`) when possible.\n",
            "Do not modify `.git` or unrelated generated artifacts.\n",
            "When you are done, ensure the fix is saved in the workspace; the harness will collect `git diff`.\n",
        ]
    )
    return "".join(parts)
