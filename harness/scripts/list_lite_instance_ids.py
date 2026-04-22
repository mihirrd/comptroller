#!/usr/bin/env python3
"""Print SWE-bench Lite instance_id values, optionally filtered by Hugging Face ``repo`` field."""

from __future__ import annotations

import argparse

from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser(
        description="List instance_id from princeton-nlp/SWE-bench_Lite (e.g. batch clone-run)."
    )
    p.add_argument(
        "--dataset-name",
        default="princeton-nlp/SWE-bench_Lite",
        help="Hugging Face dataset id",
    )
    p.add_argument("--split", default="test", help="Dataset split")
    p.add_argument(
        "--repo",
        default=None,
        help='Only rows where repo matches exactly, e.g. "astropy/astropy"',
    )
    p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max lines to print (default: 10)",
    )
    args = p.parse_args()

    ds = load_dataset(args.dataset_name, split=args.split)
    n = 0
    for row in ds:
        if args.repo is not None and row.get("repo") != args.repo:
            continue
        print(row["instance_id"])
        n += 1
        if n >= args.limit:
            break


if __name__ == "__main__":
    main()
