"""Run Comptroller once per SWE-bench instance and build a prediction dict."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from comptroller.runner import AgentTurnInput, AgentTurnResult, run_agent_turn

from comptroller_swebench.patch import git_diff
from comptroller_swebench.prompt import build_task_prompt

logger = logging.getLogger(__name__)


def _max_wall_seconds_from_env() -> float | None:
    raw = (
        os.environ.get("COMPTROLLER_MAX_WALL_SECONDS") or os.environ.get("max_wall_seconds") or ""
    ).strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0 else None


def resolve_max_wall_seconds(explicit: float | None) -> float | None:
    """CLI value wins; otherwise same env vars as ``comptroller run``."""
    if explicit is not None:
        return explicit if explicit > 0 else None
    return _max_wall_seconds_from_env()


def run_single_instance(
    instance: dict[str, Any],
    *,
    workspace_root: str,
    model: str,
    max_tokens: int,
    model_name_or_path: str,
    db_parent: Path | None = None,
    max_wall_seconds: float | None = None,
) -> tuple[dict[str, Any], str]:
    """Execute the agent for one dataset row and return a SWE-bench prediction + patch text.

    Returns
    -------
    prediction
        SWE-bench fields plus optional metrics: ``tokens_used``, ``max_tokens``,
        ``api_dollars_used``, ``wall_seconds_used`` (``run_evaluation`` ignores extras).
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

    wall_cap = resolve_max_wall_seconds(max_wall_seconds)

    inp = AgentTurnInput(
        task=task,
        session_id=iid,
        model=model,
        workspace_root=str(root),
        max_tokens=max_tokens,
        max_wall_seconds=wall_cap,
    )

    result: AgentTurnResult | None = None
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
    pred: dict[str, Any] = {
        "instance_id": iid,
        "model_name_or_path": model_name_or_path,
        "model_patch": patch,
    }
    if result is not None:
        pred["tokens_used"] = result.current_tokens
        pred["max_tokens"] = max_tokens
        pred["api_dollars_used"] = round(result.current_dollars, 6)
        pred["wall_seconds_used"] = round(result.current_wall_seconds, 3)
        if wall_cap is not None:
            pred["max_wall_seconds"] = round(wall_cap, 3)
        logger.info(
            "instance %s metrics tokens_used=%s max_tokens=%s api_dollars_used=%s wall_seconds=%s max_wall=%s",
            iid,
            result.current_tokens,
            max_tokens,
            result.current_dollars,
            result.current_wall_seconds,
            wall_cap,
        )
    return pred, patch
