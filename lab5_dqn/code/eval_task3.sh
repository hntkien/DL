#!/usr/bin/env bash
# Evaluate all Task 3 milestone snapshots over 20 seeded episodes each.
# Run from lab5_dqn/ project root:
#     bash code/eval_task3.sh                # screenshots only (fast)
#     bash code/eval_task3.sh --with-video   # also save mp4s
#
# Output goes to results/task3/eval_logs/eval_<step>.txt for report screenshots.
#
# The --disable-sticky-actions and --pong-action-subset flags MUST match the
# wrapper config used at training time (see code/configs/task3.yaml).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SNAP_DIR="${PROJECT_ROOT}/results/task3"
LOG_DIR="${PROJECT_ROOT}/results/task3/eval_logs"
mkdir -p "${LOG_DIR}"

VIDEO_FLAG="--no-video"
if [[ "${1:-}" == "--with-video" ]]; then
    VIDEO_FLAG=""
fi

# Env-wrapper flags must match training. Edit if you retrain with a different config.
WRAP_FLAGS="--disable-sticky-actions --pong-action-subset"

STEPS=(600000 800000 1000000 1250000 1500000 2000000 2500000 3000000)
for s in "${STEPS[@]}"; do
    MODEL="${SNAP_DIR}/snapshot_${s}.pt"
    if [[ ! -f "${MODEL}" ]]; then
        echo "WARN: ${MODEL} not found, skipping."
        continue
    fi
    LOG="${LOG_DIR}/eval_${s}.txt"
    echo "=== Evaluating snapshot_${s}.pt ==="
    python3 "${SCRIPT_DIR}/test_model.py" \
        --model-path "${MODEL}" \
        --output-dir "${PROJECT_ROOT}/videos/task3/snapshot_${s}" \
        --episodes 20 ${VIDEO_FLAG} ${WRAP_FLAGS} \
        | tee "${LOG}"
    echo
done

# Also evaluate best_model.pt and first_score_19.pt if present
for extra in "best_model.pt" "first_score_19.pt"; do
    MODEL="${SNAP_DIR}/${extra}"
    [[ -f "${MODEL}" ]] || continue
    name="${extra%.pt}"
    LOG="${LOG_DIR}/eval_${name}.txt"
    echo "=== Evaluating ${extra} ==="
    python3 "${SCRIPT_DIR}/test_model.py" \
        --model-path "${MODEL}" \
        --output-dir "${PROJECT_ROOT}/videos/task3/${name}" \
        --episodes 20 ${VIDEO_FLAG} ${WRAP_FLAGS} \
        | tee "${LOG}"
    echo
done

echo "All eval logs written to ${LOG_DIR}"