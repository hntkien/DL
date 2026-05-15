#!/usr/bin/env bash
# Evaluate a Task 3 snapshot.
# Run from lab5_dqn/ project root:
#     bash code/eval_task3.sh # uses best
#     bash code/eval_task3.sh --model-path results/task3/snapshot_600000.pt --episodes 20
#     bash code/eval_task3.sh --no-video  # screenshot only

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MODEL="${PROJECT_ROOT}/results/task3/best_model.pt"
DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/videos/task3"

if [[ " $* " != *" --model-path "* ]]; then
    set -- --model-path "${DEFAULT_MODEL}" "$@"
fi
if [[ " $* " != *" --output-dir "* ]]; then
    set -- "$@" --output-dir "${DEFAULT_OUTPUT_DIR}"
fi

python "${SCRIPT_DIR}/test_model.py" "$@"