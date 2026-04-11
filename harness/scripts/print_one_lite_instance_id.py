#!/usr/bin/env python3
"""Print one valid SWE-bench Lite instance_id (requires `datasets` in the active env)."""

from __future__ import annotations

from datasets import load_dataset


def main() -> None:
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(ds[0]["instance_id"])


if __name__ == "__main__":
    main()
