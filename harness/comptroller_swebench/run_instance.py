"""Run Comptroller once per SWE-bench instance and build a prediction dict."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from comptroller.runner import AgentTurnInput, AgentTurnResult, run_agent_turn

from comptroller_swebench.patch import git_diff
from comptroller_swebench.prompt import build_task_prompt

logger = logging.getLogger(__name__)
DEFAULT_TERMINATION_POLICY = "default"


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


def resolve_termination_policy(explicit: str | None) -> str:
    """CLI value wins; otherwise COMPTROLLER_TERMINATION_POLICY; fallback to ``default``."""
    if explicit is not None and explicit.strip():
        return explicit.strip()
    env_val = (os.environ.get("COMPTROLLER_TERMINATION_POLICY") or "").strip()
    if env_val:
        return env_val
    return DEFAULT_TERMINATION_POLICY


@contextmanager
def _temporary_env(name: str, value: str):
    prev = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def run_single_instance(
    instance: dict[str, Any],
    *,
    workspace_root: str,
    model: str,
    max_tokens: int,
    model_name_or_path: str,
    db_parent: Path | None = None,
    max_wall_seconds: float | None = None,
    termination_policy: str | None = None,
    local_model_url: str | None = None,
    local_model_id: str | None = None,
    local_model_api_key: str | None = None,
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
    policy = resolve_termination_policy(termination_policy)

    inp = AgentTurnInput(
        task=task,
        session_id=iid,
        model=model,
        workspace_root=str(root),
        max_tokens=max_tokens,
        max_wall_seconds=wall_cap,
        local_model_url=local_model_url,
        local_model_id=local_model_id,
        local_model_api_key=local_model_api_key,
    )

    result: AgentTurnResult | None = None
    try:
        with _temporary_env("COMPTROLLER_TERMINATION_POLICY", policy):
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
        "termination_policy": policy,
    }
    if local_model_url:
        pred["local_model_url"] = local_model_url
    if local_model_id:
        pred["local_model_id"] = local_model_id
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
