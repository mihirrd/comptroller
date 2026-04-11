"""Clone task repo at ``base_commit``, run Comptroller, append JSONL, delete clone."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import typer
from datasets import load_dataset
from dotenv import load_dotenv

from comptroller_swebench.git_workspace import clone_instance_workspace
from comptroller_swebench.run_instance import run_single_instance


def _find_instance(ds, instance_id: str) -> dict:
    for row in ds:
        if row["instance_id"] == instance_id:
            return row
    raise KeyError(instance_id)


def run_clone_infer_cleanup(
    instance_id: str,
    *,
    dataset_name: str,
    split: str,
    output: Path,
    model: str,
    max_tokens: int,
    model_name_or_path: str,
    scratch_parent: Optional[Path],
    db_parent: Optional[Path],
) -> dict:
    """Load row, clone to a temp dir, infer, append one JSONL line, remove temp dir."""
    ds = load_dataset(dataset_name, split=split)
    row = _find_instance(ds, instance_id)

    parent = (
        scratch_parent.expanduser().resolve()
        if scratch_parent is not None
        else Path(tempfile.gettempdir())
    )
    parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(
        tempfile.mkdtemp(prefix=f"cg_sweb_{instance_id.replace('__', '_')}_", dir=str(parent))
    )
    clone_root = tmp_root / "repo"
    try:
        logging.info("cloning %s @ %s → %s", row["repo"], row["base_commit"], clone_root)
        clone_instance_workspace(row, clone_root)
        pred, _patch = run_single_instance(
            row,
            workspace_root=str(clone_root),
            model=model,
            max_tokens=max_tokens,
            model_name_or_path=model_name_or_path,
            db_parent=db_parent,
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        logging.info("removed temp workspace %s", tmp_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    return pred


def clone_run(
    instance_id: Annotated[
        str,
        typer.Argument(help="Dataset instance_id (e.g. astropy__astropy-12907)."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Append one JSONL prediction line."),
    ] = Path("predictions.jsonl"),
    dataset_name: Annotated[
        str,
        typer.Option("--dataset-name", help="Hugging Face dataset id."),
    ] = "princeton-nlp/SWE-bench_Lite",
    split: Annotated[str, typer.Option(help="Dataset split.")] = "test",
    model: Annotated[str, typer.Option("--model", help="LiteLLM model id.")] = "gpt-4o",
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Comptroller token budget for the turn."),
    ] = 500_000,
    model_name_or_path: Annotated[
        str,
        typer.Option("--model-name-or-path", help="Label in JSONL (reporting only)."),
    ] = "comptroller",
    scratch_parent: Annotated[
        Optional[Path],
        typer.Option(
            "--scratch-parent",
            help="Directory for the temporary clone (default: system temp).",
        ),
    ] = None,
    db_parent: Annotated[
        Optional[Path],
        typer.Option(
            "--db-parent",
            help="Directory for temporary LangGraph SQLite dirs (default: system temp).",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Clone GitHub repo at ``base_commit``, run inference, delete clone; keep JSONL line."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    empty = {
        "instance_id": instance_id,
        "model_name_or_path": model_name_or_path,
        "model_patch": "",
    }
    try:
        pred = run_clone_infer_cleanup(
            instance_id,
            dataset_name=dataset_name,
            split=split,
            output=output,
            model=model,
            max_tokens=max_tokens,
            model_name_or_path=model_name_or_path,
            scratch_parent=scratch_parent,
            db_parent=db_parent,
        )
    except KeyError:
        typer.echo(f"Error: instance_id not in dataset: {instance_id!r}", err=True)
        raise typer.Exit(1)
    except FileNotFoundError:
        typer.echo("Error: `git` not found on PATH.", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logging.exception("clone-infer workflow failed")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(empty, ensure_ascii=False) + "\n")
        raise typer.Exit(1)

    typer.echo(f"patch bytes: {len(pred.get('model_patch') or '')}")
    typer.echo(f"Appended JSONL to {output}")


def main() -> None:
    typer.run(clone_run)


if __name__ == "__main__":
    main()
