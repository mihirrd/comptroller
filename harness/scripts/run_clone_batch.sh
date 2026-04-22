#!/usr/bin/env bash
# Run comptroller-swebench-clone-run for each instance_id read from stdin (one per line).
# Example — first 5 Astropy Lite tasks into a fresh JSONL:
#   python scripts/list_lite_instance_ids.py --repo astropy/astropy --limit 5 \
#     | ./scripts/run_clone_batch.sh predictions.astropy5.jsonl -- --model gpt-4o
set -euo pipefail

OUT="${1:?usage: $0 OUTPUT.jsonl -- [extra args passed to comptroller-swebench-clone-run...]}"
shift
if [[ "${1:-}" == "--" ]]; then
  shift
fi

rm -f "${OUT}"
while IFS= read -r iid || [[ -n "${iid}" ]]; do
  [[ -z "${iid// /}" ]] && continue
  echo "=== ${iid} ==="
  comptroller-swebench-clone-run "${iid}" --output "${OUT}" "$@"
done

echo "Done. Lines in ${OUT}: $(wc -l < "${OUT}" | tr -d ' ')"
