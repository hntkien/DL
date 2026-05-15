#!/usr/bin/env bash
# Evaluate a Task 1 (CartPole-v1) snapshot.
# Run from the lab5_dqn/ project root:
#     bash code/eval_task1.sh  # uses best snapshot
#     bash code/eval_task1.sh --model-path results/task1/best_model.pt --episodes 20
#     bash code/eval_task1.sh --no-video  # skip mp4 saving

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MODEL="${PROJECT_ROOT}/results/task1/best_model.pt"
DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/videos/task1"

if [[ " $* " != *" --model-path "* ]]; then
    set -- --model-path "${DEFAULT_MODEL}" "$@"
fi
if [[ " $* " != *" --output-dir "* ]]; then
    set -- "$@" --output-dir "${DEFAULT_OUTPUT_DIR}"
fi

python "${SCRIPT_DIR}/test_cartpole.py" "$@"