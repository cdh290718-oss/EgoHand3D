#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/checkpoint.ckpt [output_dir]" >&2
  exit 1
fi

CHECKPOINT="$1"
OUT_DIR="${2:-release_weights}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

BASE_NAME="$(basename "$CHECKPOINT")"
SPLIT_PREFIX="${OUT_DIR}/${BASE_NAME}.part-"
SHA_PATH="${OUT_DIR}/${BASE_NAME}.sha256"
MANIFEST_PATH="${OUT_DIR}/${BASE_NAME}.manifest.txt"

split -b 1900M "$CHECKPOINT" "$SPLIT_PREFIX"
sha256sum "$CHECKPOINT" > "$SHA_PATH"

{
  echo "checkpoint=${CHECKPOINT}"
  echo "size_bytes=$(stat -c '%s' "$CHECKPOINT")"
  echo "sha256_file=${SHA_PATH}"
  echo "parts:"
  ls -lh "${SPLIT_PREFIX}"*
  echo
  echo "reconstruct:"
  echo "cat ${BASE_NAME}.part-* > ${BASE_NAME}"
  echo "sha256sum -c ${BASE_NAME}.sha256"
} > "$MANIFEST_PATH"

echo "Wrote split checkpoint assets to ${OUT_DIR}"
echo "Upload these files to a GitHub Release, not to normal Git history:"
ls -lh "$OUT_DIR"
