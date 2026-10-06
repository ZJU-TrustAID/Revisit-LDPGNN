#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_ENV="${AEC_CONDA_ENV:-revisit-ldpgnn}"

if ! command -v "$CONDA_EXE" >/dev/null 2>&1; then
  echo "ERROR: conda is required. Load/install Conda before running this script." >&2
  exit 2
fi

eval "$("$CONDA_EXE" shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
  conda create -n "$CONDA_ENV" python=3.10 pip -y
fi

conda run -n "$CONDA_ENV" python -m pip install --upgrade pip
conda run -n "$CONDA_ENV" python -m pip install 'torch>=2.8,<2.9'
conda run -n "$CONDA_ENV" python -m pip install torch-geometric==2.8.0
TORCH_TAG="$(conda run -n "$CONDA_ENV" python -c 'import torch; print(torch.__version__)')"
if [[ "$TORCH_TAG" != *+* ]]; then TORCH_TAG="${TORCH_TAG}+cpu"; fi
PYG_WHL_URL="${PYG_WHL_URL:-https://data.pyg.org/whl/torch-${TORCH_TAG}.html}"
conda run -n "$CONDA_ENV" python -m pip install --only-binary=:all: --find-links "$PYG_WHL_URL" "torch-scatter>=2.1.2,<2.2" "torch-sparse>=0.6.18,<0.6.19"

TEMP_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$TEMP_REQUIREMENTS"' EXIT
grep -Ev '^(torch|torch-geometric|torch-scatter|torch-sparse)([<>=]|$)' "$REPO_ROOT/requirements.txt" > "$TEMP_REQUIREMENTS"
conda run -n "$CONDA_ENV" python -m pip install -r "$TEMP_REQUIREMENTS"
conda run -n "$CONDA_ENV" python -m ipykernel install --user --name "$CONDA_ENV" --display-name "revisit-ldpgnn"
echo "Environment ready: conda activate $CONDA_ENV"
