# Volleylitics

**Live Project Page:** [gydevy.github.io/Volleylitics](https://gydevy.github.io/Volleylitics/)



Volleylitics is a computer vision and audio analytics system that converts raw volleyball match recordings into structured gameplay data.

The system processes a full match video and reconstructs the structure of the game by detecting whistles, identifying rallies, and analyzing gameplay events. The long-term goal is to extract tactical insights such as ball landing heatmaps, serve patterns, and offensive tendencies from standard match recordings.

## Run on Windows

Requirements:

- Windows 10 or 11
- Python 3.10 or newer (with `python` on `PATH`)
- FFmpeg (install with `winget install --id Gyan.FFmpeg -e`)
- The three model files listed below
- For AMD GPU acceleration: the AMD Adrenalin driver and AMD HIP/ROCm runtime
- For NVIDIA GPU acceleration: a current NVIDIA driver

Place the model checkpoints in `models/` with these names:

```text
models/
  best_resnet_whistle_mix.pth
  rally_model_best.pth
  temp_best_ball.pt
```

Then double-click `run_windows.bat`. On first launch it creates a dedicated
`.venv-windows` environment and installs the dependencies. You can also run the
setup explicitly from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
.\run_windows.ps1
```

The setup detects AMD ROCm, NVIDIA CUDA, or CPU automatically. To force a
specific backend, use one of these commands:

```powershell
.\setup_windows.ps1 -Backend amd
.\setup_windows.ps1 -Backend nvidia
.\setup_windows.ps1 -Backend cpu
```

For the RX 7800 XT, the AMD option detects `gfx1101` and installs the matching
ROCm device wheel. PyTorch intentionally reports an AMD ROCm device as
`cuda:0`; the GUI will identify it as **AMD ROCm** during the runtime check.
Do not install `torch` or `torchvision` from ordinary PyPI afterward, because
that replaces the GPU build with a CPU build. If a driver was just installed
or updated and the GPU calculation fails, restart Windows and run the AMD
setup command again.

In the desktop app:

1. Choose a match video.
2. Choose **GPU required**, **Automatic**, or **CPU only**. GPU required stops
   with a visible diagnostic if ROCm/CUDA cannot execute a real calculation.
3. Click **Preview video and mark sets** to seek through the match and mark each
   set's start and end. You can also enter the ranges manually; blank means the
   full video.
4. Click **Calibrate court** and select the four lower-half court corners plus both net edges. Calibration is required because rally detection only counts ball-model detections above the measured net height.
5. Optionally click **Choose models** to select the whistle CNN, rally classifier,
   and ball detector. The selector shows the saved validation and speed benchmark
   results for each known compatible model.
6. Click **Run full pipeline**. Every stage and its live output stays visible in the app.
7. After rally detection, the pipeline stops before creating clips and asks how
   to handle ambiguous intervals. Choose **Yes** to review them now and then
   continue, **No** to continue with automatic rallies only, or **Cancel** to
   pause safely. A paused run can be restored with **Import analyzed** and the
   **Resume pipeline** button without repeating whistle or rally detection.
8. In the ambiguity reviewer, press `Y` for rally, `N` for not a rally, `S` to
   skip, or `Q` to stop. Saved decisions are integrated into
   `rallies_with_hitl.json` before clip creation and ball analysis.
9. Use **Open results** when processing finishes. You can also click **Review
   ambiguous** later to revise the saved decisions.
10. Click **Add to site** after ball analytics finishes. Enter the YouTube link,
   teams, set scores, and match metadata. The tool copies the deployable files
   into `docs/data/matches` and updates `index.json`; review those changes, then
   commit and push them to deploy the static site.

The saved 960px ball-detector benchmark shows the main tradeoff: the most
accurate `yolo26m-960-best.pt` reached 96.18% F1 at 47.82 FPS, while the default
`temp_best_ball.pt` reached 94.80% F1 at 196.77 FPS. These are model-only saved
benchmark figures; full-match speed also includes video decoding and tracking.

Use **Import analyzed** after restarting the app to select any existing
`output/<match-name>` folder. The app restores its original video path from
`pipeline.log` when available and enables results, review, and site-export
actions without rerunning the analysis.

Results are written to `output/<match-name>/`, including rally JSON, clips, the
court map, ball-tracking data, and `pipeline.log`. Uncertain rally intervals are
preserved in `hitl.json`; the automatic pipeline does not silently accept them.
The website rally feed contains every clipped rally, including rallies where the
ball model found no reliable landing point. Those rallies remain playable in the
list but do not create a misleading dot on the court heatmap.

The first GPU batch in a new environment can pause for roughly 10–20 seconds
while ROCm or CUDA prepares convolution kernels. This is a one-time warm-up,
not a hang. The live log now reports separate timings for audio loading, DSP
whistle detection, whistle CNN filtering, and the batched video timeline.

The same flow is available without the GUI:

```powershell
python run_pipeline.py --video "D:\matches\match.mp4" --match-id match `
  --calibration "output\match\calibration.json" --device gpu
```

---

# System Pipeline

```
Match Video
     │
     ▼
Whistle Detection
     │
     ▼
Rally Segmentation
     │
     ▼
Ball Tracking
     │
     ▼
Landing Detection
     │
     ▼
Tactical Analytics
     │
     ▼
Web Dashboard
```

---

# Development Roadmap

## 1. Whistle Detection

Detect candidate referee whistles from match audio using DSP.Filter through candidates using a CNN.

- [x] Detect whistle candidates
- [X] Filter out noise
- [X] Filter with CNN and create a list of real whistles.

---

## 2. Rally Classification

Determine whether adjacent whistles represent rally boundaries.

- [x] Pair adjacent whistles
- [x] Extract gameplay segments
- [x] Train rally classification CNN
- [x] Label segments as **in-play** or **dead time**
- [x] Export rally timeline

Example output

```json
{
    "start": 433.87065759637187,
    "end": 447.30920634920636,
    "duration": 13.438548752834492
  },
  {
    "start": 474.395283446712,
    "end": 485.4537868480726,
    "duration": 11.058503401360554
  },
```
The rally classifier is supplemented by the trained volleyball detector. A ball
detection only contributes to rally activity when its center is above the
calibrated net-height line, so this signal does not depend on the ball color.
---

## 3. Ball Tracking (Planned)

Track the volleyball during rally segments.

- [x] Detect volleyball using object detection
- [x] Track ball trajectory
- [x] Store ball coordinates over time

---

## 4. Ball Landing Detection (Planned)

Detect where the ball contacts the court floor.

- [x] Detect ball-floor contact
- [x] Estimate court coordinates
- [x] Store landing locations

---

## 5. Player Detection and Multi Camera Support

- [ ] Detect players during rallies.
- [ ] Add orthogonal camera support.
- [ ] Calibrate cameras to court coordinates.
- [ ] Track players across frames.
- [ ] Identify jerseys using OCR.
- [ ] Store player identity memory.
- [ ] Detect and report actions.
- [ ] Fuse detections across cameras.

## 6. Tactical Analytics (Planned)

Extract gameplay insights from rally and ball data.

- [x] Ball landing heatmaps
- [ ] Setting distribution
- [ ] Serve direction
- [ ] Approximate serve speed (maybe)
- [ ] Rally statistics

---

## 7. Top-Down Tactical View (Planned)
- Requires player detection,jersey number OCR, would increase compute time substantialy.
Normalize the court into a tactical view.

- [x] Estimate court geometry
- [ ] Transform ball positions to court space
- [ ] Generate rally visualizations

---

## 8. Web Dashboard

Present match analytics through a web interface.

- [x] Upload match recordings
- [x] Automatic processing pipeline
- [x] Rally timeline visualization
- [x] Heatmaps and match analytics

---

---

# Motivation

This project started as an attempt to analyze matches of my own volleyball team and extract insights that could help improve our performance.

Manually reviewing matches makes it difficult to consistently track patterns such as ball placement, serve tendencies, or rally structure across many games. Volleylitics aims to automate this process by converting match recordings into structured data that can later be visualized through heatmaps and dashboards.

---

