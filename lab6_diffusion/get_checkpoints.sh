#!/usr/bin/env bash
# get_checkpoints.sh
# Downloads the pretrained ResNet18 evaluator checkpoint from Google Drive.
# Usage: bash get_checkpoints.sh

set -euo pipefail

FILE_ID="17TVWmpqce_8PVd_uuS1WVNZDzzokjtGD"
DEST_DIR="./ckpts"
DEST_FILE="${DEST_DIR}/checkpoint.pth"

mkdir -p "${DEST_DIR}"

echo "[*] Downloading evaluator checkpoint to ${DEST_FILE} ..."

if command -v gdown &>/dev/null; then
    gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${DEST_FILE}"
else
    echo "[!] gdown not found. Falling back to wget."
    echo "    Install gdown for more reliable Google Drive downloads: pip install gdown"
    wget --load-cookies /tmp/gdrive_cookies.txt \
        "https://drive.google.com/uc?export=download&confirm=$(
            wget --quiet --save-cookies /tmp/gdrive_cookies.txt \
                --keep-session-cookies \
                --no-check-certificate \
                "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
                -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p'
        )&id=${FILE_ID}" \
        -O "${DEST_FILE}" \
        --no-check-certificate
    rm -f /tmp/gdrive_cookies.txt
fi

echo "[+] Done. Checkpoint saved to ${DEST_FILE}"