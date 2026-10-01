#!/usr/bin/env bash
set -e

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -n "${CONDA_EXE:-}" ]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
else
  echo "conda command not found. Install Conda or activate your environment manually." >&2
  return 1 2>/dev/null || exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_NAME="${EGOHAND3D_CONDA_ENV:-egohand3d}"
FALLBACK_ENV_NAME="${EGOHAND3D_FALLBACK_ENV:-wilor}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda activate "${ENV_NAME}"
elif conda env list | awk '{print $1}' | grep -qx "${FALLBACK_ENV_NAME}"; then
  conda activate "${FALLBACK_ENV_NAME}"
else
  echo "No ${ENV_NAME} or ${FALLBACK_ENV_NAME} Conda environment found." >&2
  echo "Run scripts/create_conda_env.sh or activate a compatible environment manually." >&2
  return 1 2>/dev/null || exit 1
fi

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WILOR_TRAINING_DATA="${WILOR_TRAINING_DATA:-${PROJECT_ROOT}/data/hamer_training_data}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
