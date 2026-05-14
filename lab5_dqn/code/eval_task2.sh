#!/usr/bin/env bash
# Evaluate a Task 2 (ALE/Pong-v5) snapshot.
# Run from the lab5_dqn/ project root:
#     bash code/eval_task2.sh  # uses best snapshot
#     bash code/eval_task2.sh --model-path results/task2/best_model.pt --episodes 20

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MODEL="${PROJECT_ROOT}/results/task2/best_model.pt"
DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/videos/task2"

if [[ " $* " != *" --model-path "* ]]; then
    set -- --model-path "${DEFAULT_MODEL}" "$@"
fi
if [[ " $* " != *" --output-dir "* ]]; then
    set -- "$@" --output-dir "${DEFAULT_OUTPUT_DIR}"
fi

python "${SCRIPT_DIR}/test_model.py" "$@"