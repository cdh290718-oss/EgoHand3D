#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate.sh

IMG_FOLDER="${1:-examples/images}"
OUT_FOLDER="${2:-outputs/3d_joints}"
CKPT="${3:-pretrained_models/wilor_final.ckpt}"

python export_3d_joints.py \
  --img_folder "${IMG_FOLDER}" \
  --out_folder "${OUT_FOLDER}" \
  --checkpoint "${CKPT}" \
  --cfg pretrained_models/model_config.yaml \
  --detector pretrained_models/detector.pt
