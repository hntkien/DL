#!/usr/bin/env bash
# Train Task 1 (Vanilla DQN on CartPole-v1).
# Run from the lab5_dqn/ project root:
#     bash code/train_task1.sh
# Any extra args are forwarded to dqn.py and override values in the YAML, e.g.:
#     bash code/train_task1.sh --lr 1e-3 --wandb_run_name task1-lr1e3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/dqn.py" \
    --config "${SCRIPT_DIR}/configs/task1.yaml" \
    "$@"