"""Typer CLI: SWE-bench Lite inference → predictions JSONL."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from datasets import load_dataset
from dotenv import load_dotenv

from comptroller_swebench.run_instance import (
    resolve_max_wall_seconds,
    resolve_termination_policy,
    run_single_instance,
)
from comptroller_swebench.workspace_docker import materialize_testbed_to_host


def _parse_instance_ids(csv: Optional[str]) -> Optional[set[str]]:
    if not csv or not csv.strip():
        return None
    return {x.strip() for x in csv.split(",") if x.strip()}


def _run(
    workspace_root: Annotated[
        Path,
        typer.Option(
            "--workspace-root",
            help="Repository root the agent may edit (must match the instance checkout).",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Append JSONL predictions here."),
    ] = Path("predictions.jsonl"),
    dataset_name: Annotated[
        str,
        typer.Option("--dataset-name", help="Hugging Face dataset id."),
    ] = "princeton-nlp/SWE-bench_Lite",
    split: Annotated[str, typer.Option(help="Dataset split.")] = "test",
    instance_ids: Annotated[
        Optional[str],
        typer.Option(
            "--instance-ids",
            help="Comma-separated instance_id values (e.g. astropy__astropy-12907).",
        ),
    ] = None,
    limit: Annotated[
        Optional[int],
        typer.Option("--limit", help="Max instances when --instance-ids is omitted."),
    ] = None,
    model: Annotated[str, typer.Option("--model", help="LiteLLM model id.")] = "gpt-4o",
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Comptroller token budget for the turn."),
    ] = 500_000,
    model_name_or_path: Annotated[
        str,
        typer.Option(
            "--model-name-or-path",
            help="Label stored in JSONL (reporting only).",
        ),
    ] = "comptroller",
    db_parent: Annotated[
        Optional[Path],
        typer.Option(
            "--db-parent",
            help="Directory for temporary LangGraph SQLite dirs (default: system temp).",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    materialize_docker: Annotated[
        bool,
        typer.Option(
            "--materialize-docker",
            help="Build/pull SWE-bench instance image, export container /testbed to --workspace-root (see README).",
        ),
    ] = False,
    overwrite_materialized: Annotated[
        bool,
        typer.Option(
            "--overwrite-materialized",
            help="If materialized destination exists and is non-empty, delete it first.",
        ),
    ] = False,
    force_rebuild_docker_image: Annotated[
        bool,
        typer.Option(
            "--force-rebuild-docker-image",
            help="Force SWE-bench to rebuild the instance Docker image before export.",
        ),
    ] = False,
    docker_run_id: Annotated[
        Optional[str],
        typer.Option(
            "--docker-run-id",
            help="Optional run_id for SWE-bench materialize logs (default: random per process).",
        ),
    ] = None,
    max_wall_seconds: Annotated[
        Optional[float],
        typer.Option(
            "--max-wall-seconds",
            help="Max cumulative agent wall time (s) for LLM/tools/summary; "
            "omit to use COMPTROLLER_MAX_WALL_SECONDS or max_wall_seconds env.",
        ),
    ] = None,
    local_model_url: Annotated[
        Optional[str],
        typer.Option(
            "--local-model-url",
            help="OpenAI-compatible API base for budget-triggered local fallback "
            "(also COMPTROLLER_LOCAL_MODEL_URL).",
        ),
    ] = None,
    local_model: Annotated[
        Optional[str],
        typer.Option(
            "--local-model",
            help="Model id on --local-model-url endpoint (also COMPTROLLER_LOCAL_MODEL).",
        ),
    ] = None,
    local_model_api_key: Annotated[
        Optional[str],
        typer.Option(
            "--local-model-api-key",
            help="Bearer key for --local-model-url (also COMPTROLLER_LOCAL_API_KEY).",
        ),
    ] = None,
    termination_policy: Annotated[
        Optional[str],
        typer.Option(
            "--termination-policy",
            help="Optional termination policy label passed to runtime via "
            "COMPTROLLER_TERMINATION_POLICY and recorded in JSONL.",
        ),
    ] = None,
) -> None:
    """Run Comptroller on Lite instances and append prediction lines to ``--output``."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    want = _parse_instance_ids(instance_ids)
    if want is None and not limit:
        typer.echo("Error: provide --instance-ids and/or --limit.", err=True)
        raise typer.Exit(1)

    root = workspace_root.expanduser().resolve()

    typer.echo(f"Loading dataset {dataset_name!r} split={split!r} …")
    ds = load_dataset(dataset_name, split=split)

    rows: list[dict] = []
    if want is not None:
        found: set[str] = set()
        for row in ds:
            iid = row["instance_id"]
            if iid in want:
                rows.append(row)
                found.add(iid)
        missing = want - found
        if missing:
            typer.echo(f"Warning: instance_id not in dataset: {sorted(missing)}", err=True)
    else:
        assert limit is not None
        for i, row in enumerate(ds):
            if i >= limit:
                break
            rows.append(row)

    if not rows:
        typer.echo("No instances to run.", err=True)
        raise typer.Exit(1)

    if materialize_docker:
        if root.exists() and not root.is_dir():
            typer.echo(
                f"Error: --workspace-root exists but is not a directory: {root}",
                err=True,
            )
            raise typer.Exit(1)
        if len(rows) > 1:
            root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        typer.echo(f"Error: --workspace-root is not a directory: {root}", err=True)
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)

    def _workspace_for_instance(row: dict) -> Path:
        if materialize_docker and len(rows) > 1:
            return root / row["instance_id"]
        return root

    hint = (
        f" (per-instance under {root})" if materialize_docker and len(rows) > 1 else ""
    )
    typer.echo(f"Running {len(rows)} instance(s); workspace base={root}{hint}")
    resolved_wall = resolve_max_wall_seconds(max_wall_seconds)
    resolved_termination_policy = resolve_termination_policy(termination_policy)
    with output.open("a", encoding="utf-8") as f:
        for row in rows:
            iid = row["instance_id"]
            typer.echo(f"--- {iid} ---")
            ws = _workspace_for_instance(row)
            if materialize_docker:
                try:
                    typer.echo(f"Materializing SWE-bench /testbed → {ws} …")
                    materialize_testbed_to_host(
                        row,
                        ws,
                        run_id=docker_run_id,
                        force_rebuild_image=force_rebuild_docker_image,
                        overwrite=overwrite_materialized,
                    )
                except ImportError as e:
                    typer.echo(str(e), err=True)
                    raise typer.Exit(1)
                except Exception as e:
                    typer.echo(f"Materialize failed for {iid}: {e}", err=True)
                    logging.exception("materialize %s", iid)
                    raise typer.Exit(1)
            try:
                pred, _patch = run_single_instance(
                    row,
                    workspace_root=str(ws),
                    model=model,
                    max_tokens=max_tokens,
                    model_name_or_path=model_name_or_path,
                    db_parent=db_parent,
                    max_wall_seconds=max_wall_seconds,
                    termination_policy=termination_policy,
                    local_model_url=local_model_url,
                    local_model_id=local_model,
                    local_model_api_key=local_model_api_key,
                )
            except Exception as e:
                typer.echo(f"Error on {iid}: {e}", err=True)
                pred = {
                    "instance_id": iid,
                    "model_name_or_path": model_name_or_path,
                    "model_patch": "",
                    "tokens_used": 0,
                    "max_tokens": max_tokens,
                    "api_dollars_used": 0.0,
                    "wall_seconds_used": 0.0,
                    "termination_policy": resolved_termination_policy,
                }
                if resolved_wall is not None:
                    pred["max_wall_seconds"] = round(resolved_wall, 3)
                logging.exception("instance %s failed", iid)
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            f.flush()
            typer.echo(f"patch bytes: {len(pred.get('model_patch') or '')}")

    typer.echo(f"Wrote append JSONL to {output}")


def main() -> None:
    typer.run(_run)


if __name__ == "__main__":
    main()
