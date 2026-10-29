#!/usr/bin/env bash
# Create the project-local CPU-only environment on macOS and Linux (arm64 + x86_64).
#
# Linux: `uv --torch-backend=cpu` reads torch/torchvision from the PyTorch CPU
# index (the "+cpu" wheels in requirements.txt). That avoids the ~2.5 GB of
# nvidia-* CUDA packages the PyPI Linux wheel would pull in, and it works on
# both x86_64 and aarch64. Mixing `--index-url` with `--extra-index-url` does
# not: uv's first-index strategy finds `torch` on PyPI and never looks at the
# CPU index.
#
# macOS Apple Silicon: PyPI already ships CPU/MPS wheels, and they have no
# "+cpu" local tag (PyTorch never published one for macOS). Intel Mac is not
# supported: torch 2.14.1 has no macosx x86_64 wheels.
#
# Usage:
#   source scripts/00_setup_env.sh   # create (if needed) and activate .venv
#   bash scripts/00_setup_env.sh     # install only if requirements changed (Make uses this)

_dsai_this_file() {
  if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    printf '%s\n' "${BASH_SOURCE[0]}"
  elif [[ -n "${ZSH_VERSION:-}" ]]; then
    # zsh: %x is the file being sourced or executed
    # shellcheck disable=SC2296
    printf '%s\n' "${(%):-%x}"
  else
    printf '%s\n' "$0"
  fi
}

_dsai_sourced() {
  if [[ -n "${ZSH_EVAL_CONTEXT:-}" ]]; then
    [[ "$ZSH_EVAL_CONTEXT" == *:file* ]]
  elif [[ -n "${BASH_VERSION:-}" ]]; then
    [[ "${BASH_SOURCE[0]}" != "$0" ]]
  else
    return 1
  fi
}

_dsai_fingerprint() {
  # requirements plus this script: a comment-only edit of either should reinstall.
  local src reqs sum
  reqs="$1"
  src="$2"
  if command -v shasum >/dev/null 2>&1; then
    sum=$(cat "$reqs" "$src" | shasum -a 256)
  else
    sum=$(cat "$reqs" "$src" | sha256sum)
  fi
  printf '%s\n' "${sum%% *}"
}

dsai_setup_env() {
  local HERE REQS OS ARCH PYTHON_TAG HASH_FILE NEW_HASH
  HERE="$(cd "$(dirname "$(_dsai_this_file)")/.." && pwd)"
  cd "$HERE" || return 1

  PYTHON_TAG="${PYTHON_TAG:-3.14}"
  REQS="$HERE/requirements.txt"
  HASH_FILE="$HERE/.venv/.requirements.sha256"
  NEW_HASH="$(_dsai_fingerprint "$REQS" "$(_dsai_this_file)")"

  # An existing venv with this fingerprint is already the cache. Recreating it
  # (uv venv --clear) would also make Make treat every downstream file as stale.
  if [[ -x "$HERE/.venv/bin/python" && -f "$HASH_FILE" && "$(cat "$HASH_FILE")" == "$NEW_HASH" ]]; then
    echo "==> .venv matches requirements.txt; leaving it in place"
    return 0
  fi
  if [[ -x "$HERE/.venv/bin/python" && ! -f "$HASH_FILE" ]]; then
    echo "==> .venv already installed; recording requirements fingerprint"
    printf '%s\n' "$NEW_HASH" > "$HASH_FILE" || return 1
    return 0
  fi

  # Keep uv's wheel cache inside the project: the shared ~/.cache/uv may not be
  # writable (e.g. restricted sandboxes), and a local cache makes reruns fast.
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$HERE/.uv-cache}"

  if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required (https://docs.astral.sh/uv/)." >&2
    echo "       install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    return 1
  fi

  OS="$(uname -s)"
  ARCH="$(uname -m)"
  case "${ARCH}" in
    x86_64|amd64) ARCH=x86_64 ;;
    aarch64|arm64) ARCH=arm64 ;;
  esac

  # Drop the Linux-only +cpu local tag so macOS can resolve torch from PyPI.
  strip_cpu_tag() {
    sed -E 's/^(torch(vision)?==[^[:space:]]+)\+cpu$/\1/' "$REQS"
  }

  echo "==> creating .venv with Python ${PYTHON_TAG}"
  uv venv --clear --python "${PYTHON_TAG}" .venv || return 1

  case "${OS}/${ARCH}" in
    Linux/x86_64|Linux/arm64)
      echo "==> ${OS}/${ARCH}: installing requirements.txt (torch from the PyTorch CPU index)"
      uv pip install --python .venv/bin/python \
        --torch-backend=cpu \
        -r "$REQS" || return 1
      ;;
    Darwin/arm64)
      echo "==> macOS/${ARCH}: installing requirements.txt from PyPI (no +cpu wheels on macOS)"
      uv pip install --python .venv/bin/python \
        -r <(strip_cpu_tag) || return 1
      ;;
    Darwin/x86_64)
      echo "error: PyTorch 2.14.1 has no Intel Mac (x86_64) wheels." >&2
      echo "       Use Apple Silicon, or Linux x86_64 / arm64." >&2
      return 1
      ;;
    *)
      echo "error: unsupported platform: ${OS}/${ARCH}" >&2
      echo "       supported: macOS arm64, Linux x86_64, Linux arm64" >&2
      echo "       (PyTorch 2.14.1 has no Intel Mac wheels)" >&2
      return 1
      ;;
  esac

  echo "==> verifying"
  .venv/bin/python - <<'PY' || return 1
import platform
import torch, torchvision, numpy, pandas, sklearn, PIL

print("python      ", platform.python_version(), platform.machine())
print("torch       ", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("torchvision ", torchvision.__version__)
print("numpy/pandas/sklearn/pillow", numpy.__version__, pandas.__version__, sklearn.__version__, PIL.__version__)
print("threads     ", torch.get_num_threads())
PY
  printf '%s\n' "$NEW_HASH" > "$HASH_FILE" || return 1
}

_DSAI_ROOT="$(cd "$(dirname "$(_dsai_this_file)")/.." && pwd)"

if _dsai_sourced; then
  if [[ ! -x "$_DSAI_ROOT/.venv/bin/python" ]]; then
    dsai_setup_env || return $?
  fi
  # shellcheck disable=SC1091
  source "$_DSAI_ROOT/.venv/bin/activate"
  echo "==> venv active. python3 -> $(command -v python3)"
else
  dsai_setup_env || exit $?
  echo "==> done.  activate with: source .venv/bin/activate"
  echo "           or re-run as:  source scripts/00_setup_env.sh"
fi
