#!/usr/bin/env bash
# Evaluate a Task 1 (CartPole-v1) snapshot.
# Run from the lab5_dqn/ project root:
#     bash code/eval_task1.sh  # uses best snapshot
#     bash code/eval_task1.sh --model-path results/task1/best_model.pt --episodes 20

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MODEL="${PROJECT_ROOT}/results/task1/best_model.pt"

# If user did not pass --model-path, fall back to the default best snapshot.
if [[ " $* " != *" --model-path "* ]]; then
    set -- --model-path "${DEFAULT_MODEL}" "$@"
fi

python3 "${SCRIPT_DIR}/test_cartpole.py" "$@"