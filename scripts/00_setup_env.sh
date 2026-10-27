#!/usr/bin/env bash
# Create the project-local CPU-only environment.
#
# The machine has no CUDA device, so we install the "+cpu" wheels from the
# PyTorch CPU index.  That avoids the ~2.5 GB of nvidia-* dependencies that the
# default PyPI wheel would pull in, and keeps the install under ~1 GB.
#
# Usage:  bash scripts/00_setup_env.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PYTHON_TAG="${PYTHON_TAG:-3.14}"
TORCH_VER="${TORCH_VER:-2.14.1}"
TORCHVISION_VER="${TORCHVISION_VER:-0.29.1}"
CPU_INDEX="https://download.pytorch.org/whl/cpu"

# Keep uv's wheel cache inside the project: the shared ~/.cache/uv may not be
# writable (e.g. restricted sandboxes), and a local cache makes reruns fast.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HERE/.uv-cache}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required (https://docs.astral.sh/uv/)." >&2
  echo "       install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "==> creating .venv with Python ${PYTHON_TAG}"
uv venv --python "${PYTHON_TAG}" .venv

echo "==> installing torch ${TORCH_VER}+cpu / torchvision ${TORCHVISION_VER}+cpu"
uv pip install --python .venv/bin/python \
  --index-url "${CPU_INDEX}" \
  "torch==${TORCH_VER}+cpu" "torchvision==${TORCHVISION_VER}+cpu"

echo "==> installing analysis packages"
uv pip install --python .venv/bin/python \
  "numpy>=2.0" "pandas>=2.2" "scikit-learn>=1.4" "pillow>=10" "matplotlib>=3.8"

echo "==> verifying"
.venv/bin/python - <<'PY'
import platform
import torch, torchvision, numpy, pandas, sklearn, PIL

print("python      ", platform.python_version(), platform.machine())
print("torch       ", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("torchvision ", torchvision.__version__)
print("numpy/pandas/sklearn/pillow", numpy.__version__, pandas.__version__, sklearn.__version__, PIL.__version__)
print("threads     ", torch.get_num_threads())
PY

echo "==> done.  activate with: source .venv/bin/activate"
