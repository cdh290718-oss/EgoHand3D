#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -n "${CONDA_EXE:-}" ]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
else
  echo "conda command not found. Install Conda before running this script." >&2
  exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_NAME="${EGOHAND3D_CONDA_ENV:-egohand3d}"
FALLBACK_ENV_NAME="${EGOHAND3D_FALLBACK_ENV:-wilor}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "conda env ${ENV_NAME} already exists"
elif conda env list | awk '{print $1}' | grep -qx "${FALLBACK_ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" --clone "${FALLBACK_ENV_NAME}"
else
  conda env create -f environment.yml
fi

conda activate "${ENV_NAME}"
python - <<'PY'
import torch
import pytorch_lightning as pl
import ultralytics
print("egohand3d ready")
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("pytorch_lightning", pl.__version__)
print("ultralytics", ultralytics.__version__)
PY
