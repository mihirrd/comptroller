"""Run Comptroller once per SWE-bench instance and build a prediction dict."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from comptroller.runner import AgentTurnInput, run_agent_turn

from comptroller_swebench.patch import git_diff
from comptroller_swebench.prompt import build_task_prompt

logger = logging.getLogger(__name__)


def run_single_instance(
    instance: dict[str, Any],
    *,
    workspace_root: str,
    model: str,
    max_tokens: int,
    model_name_or_path: str,
    db_parent: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Execute the agent for one dataset row and return a SWE-bench prediction + patch text.

    Returns
    -------
    prediction
        ``{"instance_id", "model_name_or_path", "model_patch"}`` for JSONL.
    model_patch
        Same as ``prediction["model_patch"]`` (convenience for logging size).
    """
    iid = instance["instance_id"]
    task = build_task_prompt(instance, workspace_root=workspace_root)
    root = Path(workspace_root).resolve()

    tmp_parent = Path(db_parent) if db_parent is not None else Path(tempfile.gettempdir())
    tmp_parent.mkdir(parents=True, exist_ok=True)
    db_dir = Path(tempfile.mkdtemp(prefix=f"cg_{iid}_", dir=str(tmp_parent)))
    db_path = str(db_dir / "sessions.db")

    inp = AgentTurnInput(
        task=task,
        session_id=iid,
        model=model,
        workspace_root=str(root),
        max_tokens=max_tokens,
    )

    try:
        result = run_agent_turn(
            inp,
            db_path=db_path,
            graph=None,
            log_steps_to_store=False,
            stream_chunk_budget_matches_max_tokens=False,
        )
        if result.final_state is None:
            logger.warning("instance %s: graph returned no final state", iid)
        elif result.final_state.get("status") == "summarizing":
            logger.warning(
                "instance %s: ended in summarizing (budget hit); patch may be incomplete",
                iid,
            )
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)

    patch = git_diff(root)
    pred = {
        "instance_id": iid,
        "model_name_or_path": model_name_or_path,
        "model_patch": patch,
    }
    return pred, patch
