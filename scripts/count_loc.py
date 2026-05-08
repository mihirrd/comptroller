#!/usr/bin/env python3
"""Run pygount summaries for ``src/``, ``harness/``, and both together.

Usage (from repo root, with dev deps synced):

    uv run python scripts/count_loc.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pygount_cmd() -> list[str]:
    """Resolve pygount CLI (same venv as this interpreter when possible)."""
    candidate = Path(sys.executable).resolve().parent / "pygount"
    if candidate.is_file():
        return [str(candidate)]
    return ["pygount"]


def _run_pygount(label: str, paths: list[Path]) -> None:
    existing = [p for p in paths if p.is_dir()]
    if not existing:
        print(f"=== {label} ===\n(skip: no such directory)\n", file=sys.stderr, flush=True)
        return
    print(f"=== {label} ===", flush=True)
    subprocess.run(
        [
            *_pygount_cmd(),
            "--format=summary",
            *[str(p) for p in existing],
        ],
        check=True,
        cwd=ROOT,
    )
    print()


def main() -> None:
    _run_pygount("src", [ROOT / "src"])
    _run_pygount("harness", [ROOT / "harness"])
    _run_pygount("src + harness (sum)", [ROOT / "src", ROOT / "harness"])


if __name__ == "__main__":
    main()
