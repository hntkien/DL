#!/usr/bin/env bash
# Train Task 3 (Enhanced DQN: DDQN + PER + n-step on ALE/Pong-v5).
# Run from lab5_dqn/ project root:
#     bash code/train_task3.sh
# Forwarded args override YAML. Useful ablation runs:
#     bash code/train_task3.sh --use_ddqn --use_per --n_step 1 --wandb_run_name task3-no-nstep
#     (toggle individual flags; note --use_ddqn / --use_per are store_true; pass them to enable.)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/dqn.py" \
    --config "${SCRIPT_DIR}/configs/task3.yaml" \
    "$@"