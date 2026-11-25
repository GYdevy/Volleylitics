#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BACKEND="auto"

usage() {
    cat <<'EOF'
Usage: ./setup_linux.sh [--backend auto|amd|nvidia|cpu]

Environment overrides:
  PYTHON                         Python executable (default: python3)
  VOLLEYLITICS_ROCM_INDEX_URL   PyTorch ROCm wheel index
  VOLLEYLITICS_CUDA_INDEX_URL   PyTorch CUDA wheel index
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            [[ $# -ge 2 ]] || { echo "--backend requires a value" >&2; exit 2; }
            BACKEND="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$BACKEND" in
    auto|amd|nvidia|cpu) ;;
    *) echo "Unsupported backend: $BACKEND" >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Python was not found. Install Python 3.10 or newer." >&2
    exit 1
}

"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "Python 3.10 or newer is required." >&2
    exit 1
}

if [[ ! -x .venv-linux/bin/python ]]; then
    echo "Creating .venv-linux..."
    "$PYTHON_BIN" -m venv .venv-linux
fi

VENV_PYTHON="$SCRIPT_DIR/.venv-linux/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip

if [[ "$BACKEND" == "auto" ]]; then
    if [[ -e /dev/kfd ]] || command -v rocminfo >/dev/null 2>&1; then
        BACKEND="amd"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        BACKEND="nvidia"
    else
        BACKEND="cpu"
    fi
fi

echo "Installing PyTorch backend: $BACKEND"
case "$BACKEND" in
    amd)
        ROCM_INDEX="${VOLLEYLITICS_ROCM_INDEX_URL:-https://download.pytorch.org/whl/rocm7.1}"
        "$VENV_PYTHON" -m pip install --upgrade --force-reinstall \
            torch torchvision --index-url "$ROCM_INDEX"
        ;;
    nvidia)
        CUDA_INDEX="${VOLLEYLITICS_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
        "$VENV_PYTHON" -m pip install --upgrade --force-reinstall \
            torch torchvision --index-url "$CUDA_INDEX"
        ;;
    cpu)
        "$VENV_PYTHON" -m pip install --upgrade --force-reinstall \
            torch torchvision --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

"$VENV_PYTHON" -m pip install -r requirements.txt

echo "Verifying $BACKEND..."
"$VENV_PYTHON" - "$BACKEND" <<'PY'
import sys
import torch

backend = sys.argv[1]
hip = getattr(torch.version, "hip", None)
cuda = getattr(torch.version, "cuda", None)
gpu_expected = backend in {"amd", "nvidia"}

print("PyTorch:", torch.__version__)
print("HIP:", hip)
print("CUDA:", cuda)
print("GPU available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

if gpu_expected and not torch.cuda.is_available():
    raise SystemExit(f"{backend} was selected, but PyTorch cannot use a GPU")
if backend == "amd" and not hip:
    raise SystemExit("The installed PyTorch build does not include ROCm")
if backend == "nvidia" and (not cuda or hip):
    raise SystemExit("The installed PyTorch build does not include NVIDIA CUDA")

device = "cuda" if gpu_expected else "cpu"
values = torch.arange(16, device=device, dtype=torch.float32)
result = (values * values).sum().item()
if values.is_cuda:
    torch.cuda.synchronize()
assert result == 1240.0
PY

command -v ffmpeg >/dev/null 2>&1 || {
    echo "WARNING: FFmpeg is missing. Install it with your distribution package manager." >&2
}

missing=0
for model in \
    models/best_resnet_whistle_mix.pth \
    models/rally_model_best.pth \
    models/temp_best_ball.pt; do
    if [[ ! -f "$model" ]]; then
        echo "WARNING: Missing model: $model" >&2
        missing=1
    fi
done

echo
echo "Linux setup complete for backend: $BACKEND"
if [[ "$missing" -ne 0 ]]; then
    echo "Add the missing models before running the pipeline. See RUNNING.md."
else
    echo "Run ./run_linux.sh --help for pipeline options."
fi
