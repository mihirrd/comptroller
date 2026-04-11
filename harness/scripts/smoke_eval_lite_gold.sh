#!/usr/bin/env bash
# Run a minimal SWE-bench_Lite gold evaluation (requires Docker + swebench installed).
set -euo pipefail

: "${INSTANCE_ID:?Set INSTANCE_ID to a SWE-bench_Lite instance_id (see harness/README.md)}"

RUN_ID="${RUN_ID:-validate-gold-lite-smoke}"
MAX_WORKERS="${MAX_WORKERS:-1}"

exec python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path gold \
  --instance_ids "${INSTANCE_ID}" \
  --max_workers "${MAX_WORKERS}" \
  --run_id "${RUN_ID}"
