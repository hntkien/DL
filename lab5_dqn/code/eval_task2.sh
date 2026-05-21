#!/usr/bin/env bash
# Evaluate the Task 2 (ALE/Pong-v5 vanilla DQN) snapshot.
#
# Run from the submission root (the directory that contains code/ and the
# LAB5_*_task*.pt files):
#     bash code/eval_task2.sh   # auto-discovers LAB5_*_task2.pt
#     bash code/eval_task2.sh --with-video   # also save mp4 videos
#     bash code/eval_task2.sh --model-path /path/to/file.pt --episodes 20
#
# Default behavior runs 20 episodes with seeds 0..19 and skips mp4 generation.
# Task 2 was trained WITHOUT the Atari env wrappers used in Task 3, so no
# wrapper flags are forwarded here.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ----- Decide whether to default to --no-video -----
VIDEO_FLAG="--no-video"
USER_ARGS=()
USER_SET_MODEL=false
USER_SET_OUTPUT=false
for arg in "$@"; do
    case "$arg" in
        --with-video) VIDEO_FLAG="" ;;
        --no-video)   VIDEO_FLAG="--no-video" ;;
        --model-path) USER_SET_MODEL=true;  USER_ARGS+=("$arg") ;;
        --output-dir) USER_SET_OUTPUT=true; USER_ARGS+=("$arg") ;;
        *)            USER_ARGS+=("$arg") ;;
    esac
done

# ----- Auto-discover the Task 2 model if --model-path was not given -----
if ! $USER_SET_MODEL; then
    shopt -s nullglob
    SUBMISSION_MATCHES=("${PROJECT_ROOT}"/LAB5_*_task2.pt)
    shopt -u nullglob

    if [[ ${#SUBMISSION_MATCHES[@]} -gt 0 ]]; then
        DEFAULT_MODEL="${SUBMISSION_MATCHES[0]}"
    elif [[ -f "${PROJECT_ROOT}/results/task2/best_model.pt" ]]; then
        DEFAULT_MODEL="${PROJECT_ROOT}/results/task2/best_model.pt"
    else
        echo "ERROR: no Task 2 checkpoint found." >&2
        echo "       Looked for: ${PROJECT_ROOT}/LAB5_*_task2.pt" >&2
        echo "                   ${PROJECT_ROOT}/results/task2/best_model.pt" >&2
        echo "       Pass --model-path /path/to/model.pt explicitly." >&2
        exit 1
    fi
    USER_ARGS=(--model-path "${DEFAULT_MODEL}" "${USER_ARGS[@]+"${USER_ARGS[@]}"}")
fi

# ----- Default output dir for any videos that do get produced -----
if ! $USER_SET_OUTPUT; then
    USER_ARGS+=(--output-dir "${PROJECT_ROOT}/videos/task2")
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Video:        ${VIDEO_FLAG:-on}"
echo

# shellcheck disable=SC2086
python3 "${SCRIPT_DIR}/test_model.py" \
    ${VIDEO_FLAG} \
    "${USER_ARGS[@]}"