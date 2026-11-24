# Running Volleylitics

Volleylitics supports Windows 10/11 and modern Linux distributions. The Windows
desktop application provides the complete guided workflow. Linux uses the same
pipeline through a portable command-line launcher; the OpenCV calibration and
ambiguity-review windows still open on a Linux desktop.

## Requirements

- Python 3.10 or newer
- FFmpeg
- A current AMD or NVIDIA driver for GPU acceleration
- These model checkpoints:

```text
models/
  best_resnet_whistle_mix.pth
  rally_model_best.pth
  temp_best_ball.pt
```

## Windows

Install FFmpeg if necessary:

```powershell
winget install --id Gyan.FFmpeg -e
```

Double-click `run_windows.bat`. The first launch creates `.venv-windows`,
installs the dependencies, and then opens the desktop application. Setup can
also be run explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
.\run_windows.ps1
```

The setup detects AMD ROCm, NVIDIA CUDA, or CPU automatically. A backend can be
selected explicitly:

```powershell
.\setup_windows.ps1 -Backend amd
.\setup_windows.ps1 -Backend nvidia
.\setup_windows.ps1 -Backend cpu
```

For RDNA3 cards such as the RX 7800 XT, the AMD setup installs the matching
device wheel. PyTorch exposes ROCm devices through its `cuda` API, so seeing
`cuda:0` for an AMD card is expected.

### Desktop workflow

1. Choose the match video.
2. Choose **GPU required**, **Automatic**, or **CPU only**.
3. Use **Preview video and mark sets**, or enter the set ranges manually.
4. Use **Calibrate court** and select four court corners followed by both net
   edges.
5. Optionally use **Choose models** to select different compatible models.
6. Start the pipeline.
7. At the ambiguity prompt, review now, continue with automatic rallies, or
   pause safely and resume later with **Import analyzed**.
8. Use **Add to site** after processing to publish the match under
   `docs/data/matches`.

In the ambiguity reviewer, press `Y` for rally, `N` for not a rally, `S` to
skip, or `Q` to stop. Decisions are saved and integrated before clip creation
and ball analysis.

## Linux

On Ubuntu or Debian, install the system packages first:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip python3-tk ffmpeg \
  libgl1 libglib2.0-0 libportaudio2
```

Create `.venv-linux` and install the appropriate PyTorch build plus project
dependencies:

```bash
chmod +x setup_linux.sh run_linux.sh
./setup_linux.sh --backend auto
```

Valid backends are `auto`, `amd`, `nvidia`, and `cpu`. Automatic selection uses
ROCm when `/dev/kfd` or `rocminfo` is available, otherwise CUDA when
`nvidia-smi` is available, and otherwise CPU.

The default wheel indexes can be overridden when a particular driver requires
a different supported PyTorch build:

```bash
VOLLEYLITICS_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm7.1 \
  ./setup_linux.sh --backend amd

VOLLEYLITICS_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu130 \
  ./setup_linux.sh --backend nvidia
```

Use the [official PyTorch selector](https://pytorch.org/get-started/locally/)
to choose an index compatible with the installed driver. AMD also publishes
its supported Linux configurations in the
[ROCm PyTorch documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html).

### Calibrate and run

Create a calibration file once for the camera position:

```bash
./.venv-linux/bin/python heatmaps/point.py \
  --video /path/to/match.mp4 \
  --out output/match/calibration.json
```

Then run the complete pipeline:

```bash
./run_linux.sh \
  --video /path/to/match.mp4 \
  --match-id match \
  --calibration output/match/calibration.json \
  --device auto
```

Any `run_pipeline.py` option can be passed through the launcher, including
`--segments-json`, `--skip-heatmap`, and custom model paths. To use the manual
ambiguity gate outside the Windows GUI, pass `--review-mode prompt` together
with `--review-control`; another process must write `continue`, `reviewed`, or
`pause` to that control file. Without those arguments the command-line default
continues automatically and preserves uncertain intervals in `hitl.json`.

## Outputs and resuming

Results are written under `output/<match-name>/`. They include the rally JSON,
clips, ball-tracking results, court map, pipeline state, and `pipeline.log`.
The website feed includes every rally; a rally with no reliable ball landing
remains playable but does not add a misleading heatmap point.

The first GPU batch can take 10–20 seconds while ROCm or CUDA prepares kernels.
This one-time warm-up is expected.

The saved 960px ball-detector benchmark shows the main tradeoff:
`yolo26m-960-best.pt` reached 96.18% F1 at 47.82 FPS, while the default
`temp_best_ball.pt` reached 94.80% F1 at 196.77 FPS. These are model-only
figures; complete match processing also includes decoding and tracking.
