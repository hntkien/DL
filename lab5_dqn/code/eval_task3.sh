#!/usr/bin/env bash
# Evaluate Task 3 (enhanced DQN on ALE/Pong-v5) snapshots over 20 seeded episodes each.
#
# Run from the submission root (the directory that contains code/ and the
# LAB5_*_task*.pt files).
#
# Three modes, auto-detected from the positional argument:
#
#   1) Submission mode (default — no positional arg):
#      Discovers every LAB5_*_task3_*.pt at the project root, sorts numeric
#      milestones in ascending order, then appends any non-numeric snapshot
#      (e.g. LAB5_*_task3_best.pt) at the end.
#          bash code/eval_task3.sh
#
#   2) Single-file mode (positional arg ends in .pt):
#      Evaluates exactly that one checkpoint.
#          bash code/eval_task3.sh LAB5_R12942173_task3_2500000.pt
#          bash code/eval_task3.sh /abs/path/to/some_snapshot.pt
#
#   3) Legacy directory mode (positional arg is an existing directory):
#      Discovers snapshot_*.pt, best_model.pt, and first_score_*.pt inside
#      that directory. Intended for evaluating training-time output dirs.
#          bash code/eval_task3.sh results/task3
#          bash code/eval_task3.sh results/task3_no_ddqn
#
# Optional flags (any order, may appear before or after the positional arg):
#   --with-video   Save mp4 videos to videos/task3/<snapshot_name>/.
#   --no-wrap      Omit --disable-sticky-actions --pong-action-subset.
#                  Use ONLY for legacy snapshots trained without those Atari
#                  wrappers; mixing wrappers between train and eval produces
#                  meaningless scores because the action-head size differs.
#
# Eval logs are written to eval_logs/task3/eval_<snapshot_name>.txt (modes 1
# and 2) or <dir>/eval_logs/eval_<snapshot_name>.txt (mode 3).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ----- Parse args -----
POS_ARG=""
VIDEO_FLAG="--no-video"
WRAP_FLAGS="--disable-sticky-actions --pong-action-subset"
for arg in "$@"; do
    case "$arg" in
        --with-video) VIDEO_FLAG="" ;;
        --no-video)   VIDEO_FLAG="--no-video" ;;
        --no-wrap)    WRAP_FLAGS="" ;;
        -h|--help)
            sed -n '2,35p' "$0"; exit 0 ;;
        *)
            if [[ -n "$POS_ARG" ]]; then
                echo "ERROR: multiple positional args: '$POS_ARG' and '$arg'" >&2
                exit 1
            fi
            POS_ARG="$arg"
            ;;
    esac
done

# ----- Helper: numeric milestone sorter for submission mode -----
# Reads LAB5_*_task3_<N>.pt-style filenames from stdin (one path per line),
# emits them sorted ascending by N. Non-numeric tails are dropped here and
# handled separately by the caller.
sort_milestones() {
    awk -F'_task3_|\\.pt' '
        $2 ~ /^[0-9]+$/ { print $2 "\t" $0 }
    ' | sort -n | cut -f2-
}

# ----- Resolve snapshots based on mode -----
SNAPSHOTS=()
LOG_DIR=""
VIDEO_BASE=""
MODE=""

if [[ -z "$POS_ARG" ]]; then
    # ----- Mode 1: submission -----
    MODE="submission"
    shopt -s nullglob
    ALL_FILES=("${PROJECT_ROOT}"/LAB5_*_task3_*.pt)
    shopt -u nullglob

    if [[ ${#ALL_FILES[@]} -eq 0 ]]; then
        echo "ERROR: no LAB5_*_task3_*.pt files found at ${PROJECT_ROOT}" >&2
        echo "       Place snapshots there, pass a .pt file, or pass a directory." >&2
        exit 1
    fi

    MILESTONE_FILES=()
    EXTRA_FILES=()
    for f in "${ALL_FILES[@]}"; do
        base="$(basename "$f")"
        tail="${base#*_task3_}"
        tail="${tail%.pt}"
        if [[ "$tail" =~ ^[0-9]+$ ]]; then
            MILESTONE_FILES+=("$f")
        else
            EXTRA_FILES+=("$f")
        fi
    done

    if [[ ${#MILESTONE_FILES[@]} -gt 0 ]]; then
        mapfile -t MILESTONE_FILES < <(printf '%s\n' "${MILESTONE_FILES[@]}" | sort_milestones)
        SNAPSHOTS+=("${MILESTONE_FILES[@]}")
    fi
    if [[ ${#EXTRA_FILES[@]} -gt 0 ]]; then
        SNAPSHOTS+=("${EXTRA_FILES[@]}")
    fi
    LOG_DIR="${PROJECT_ROOT}/eval_logs/task3"
    VIDEO_BASE="${PROJECT_ROOT}/videos/task3"

elif [[ -f "$POS_ARG" && "$POS_ARG" == *.pt ]]; then
    # ----- Mode 2: single file -----
    MODE="single_file"
    if [[ "$POS_ARG" = /* ]]; then
        ABS="$POS_ARG"
    else
        ABS="$(cd "$(dirname "$POS_ARG")" && pwd)/$(basename "$POS_ARG")"
    fi
    SNAPSHOTS=("$ABS")
    LOG_DIR="${PROJECT_ROOT}/eval_logs/task3"
    VIDEO_BASE="${PROJECT_ROOT}/videos/task3"

else
    # ----- Mode 3: legacy directory -----
    MODE="directory"
    if [[ "$POS_ARG" = /* ]]; then
        SNAP_DIR="$POS_ARG"
    else
        SNAP_DIR="${PROJECT_ROOT}/${POS_ARG}"
    fi
    if [[ ! -d "$SNAP_DIR" ]]; then
        echo "ERROR: not a snapshot dir, not a .pt file: $POS_ARG" >&2
        echo "       Resolved to: $SNAP_DIR" >&2
        exit 1
    fi
    shopt -s nullglob
    MILESTONES=("$SNAP_DIR"/snapshot_*.pt)
    EXTRAS=()
    [[ -f "$SNAP_DIR/best_model.pt" ]] && EXTRAS+=("$SNAP_DIR/best_model.pt")
    for f in "$SNAP_DIR"/first_score_*.pt; do
        [[ -f "$f" ]] && EXTRAS+=("$f")
    done
    shopt -u nullglob

    if [[ ${#MILESTONES[@]} -eq 0 && ${#EXTRAS[@]} -eq 0 ]]; then
        echo "ERROR: no .pt snapshots found in $SNAP_DIR" >&2
        exit 1
    fi
    if [[ ${#MILESTONES[@]} -gt 0 ]]; then
        # snapshot_<N>.pt — sort numerically by N
        mapfile -t MILESTONES < <(printf '%s\n' "${MILESTONES[@]}" | \
            awk -F'snapshot_|\\.pt' '{print $2"\t"$0}' | sort -n | cut -f2-)
        SNAPSHOTS+=("${MILESTONES[@]}")
    fi
    if [[ ${#EXTRAS[@]} -gt 0 ]]; then
        SNAPSHOTS+=("${EXTRAS[@]}")
    fi
    LOG_DIR="${SNAP_DIR}/eval_logs"
    VIDEO_BASE="${PROJECT_ROOT}/videos/task3/$(basename "$SNAP_DIR")"
fi

mkdir -p "$LOG_DIR"

echo "Mode:         ${MODE}"
echo "Snapshots:    ${#SNAPSHOTS[@]} file(s)"
echo "Wrap flags:   ${WRAP_FLAGS:-<none>}"
echo "Video:        ${VIDEO_FLAG:-on}"
echo "Log dir:      ${LOG_DIR}"
echo

# ----- Evaluate each -----
for MODEL in "${SNAPSHOTS[@]}"; do
    name="$(basename "${MODEL%.pt}")"
    LOG="${LOG_DIR}/eval_${name}.txt"
    echo "=== Evaluating ${name}.pt ==="
    # shellcheck disable=SC2086
    python3 "${SCRIPT_DIR}/test_model.py" \
        --model-path "${MODEL}" \
        --output-dir "${VIDEO_BASE}/${name}" \
        --episodes 20 ${VIDEO_FLAG} ${WRAP_FLAGS} \
        | tee "${LOG}"
    echo
done

echo "All eval logs written to ${LOG_DIR}"