#!/usr/bin/env bash
# Train Task 2 (Vanilla DQN on ALE/Pong-v5).
# Run from the lab5_dqn/ project root:
#     bash code/train_task2.sh
# Forwarded args override YAML, e.g.:
#     bash code/train_task2.sh --lr 2.5e-4 --wandb_run_name task2-lr2_5e4

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/dqn.py" \
    --config "${SCRIPT_DIR}/configs/task2.yaml" \
    "$@"