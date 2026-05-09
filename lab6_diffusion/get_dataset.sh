#!/usr/bin/env bash
# get_dataset.sh
# Downloads the iCLEVR image dataset from Google Drive and extracts it.
# Usage: bash get_dataset.sh
#
# Result layout after extraction:
#   ./data/iclevr/*.png   (raw image files)

set -euo pipefail

FILE_ID="1Y-N1O0qltVtYMq95CAzJ1s-y-NM0qd_O"
DEST_DIR="./data"
ZIP_FILE="${DEST_DIR}/iclevr.zip"

mkdir -p "${DEST_DIR}"

echo "[*] Downloading iCLEVR dataset to ${ZIP_FILE} ..."

if command -v gdown &>/dev/null; then
    gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${ZIP_FILE}"
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
        -O "${ZIP_FILE}" \
        --no-check-certificate
    rm -f /tmp/gdrive_cookies.txt
fi

echo "[*] Extracting ${ZIP_FILE} to ${DEST_DIR}/ ..."
unzip -q "${ZIP_FILE}" -d "${DEST_DIR}"

# The zip contains a top-level ./iclevr/ folder, so final path is ./data/iclevr/
echo "[*] Cleaning up zip file ..."
rm -f "${ZIP_FILE}"

IMAGE_COUNT=$(find "${DEST_DIR}/iclevr" -name "*.png" | wc -l)
echo "[+] Done. ${IMAGE_COUNT} images available at ${DEST_DIR}/iclevr/"