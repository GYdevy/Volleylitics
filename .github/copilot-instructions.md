# AI Agent Instructions for Volleyballey

## Project Overview
Volleyballey is a volleyball game analysis system focused on **automated whistle detection** in match videos using multi-stage ML pipelines and computer vision. The codebase contains parallel detection approaches: production detector (`WhistleDetector/`), experimental CNN-based detector (`detector_slop/`), and YOLO-based vision analysis (`Vision/`).

## Core Architecture

### 1. WhistleDetector Module (Production)
Primary whistle detection pipeline with modular architecture:

**Detection Flow:**
```
Video → Audio Extraction → Energy Detection → Frame Grouping → 
Feature Extraction → Multi-Model Classification → Human-in-the-Loop → Timeline Export
```

**Key Components:**
- `config.py`: Centralized configuration (audio params, thresholds, paths). **Always edit MATCH_NUM here first**
- `audio/features.py`: 18-feature extraction (RMS, spectral flatness, MFCCs, ridge analysis)
- `detection/energy.py`: Whistle-band (3700-4200 Hz) energy detection with hysteresis thresholding
- `detection/routing.py`: 4-tier classification routing:
  1. Auto-accept (p1 ≥ 0.72)
  2. Ambiguous model (0.10 ≤ p1 < 0.72) 
  3. Physics-based acceptance (high grad_f, ridge, low flatness)
  4. HITL for marginal cases
- `hitl/`: Tkinter GUI for manual labeling of ambiguous detections
- `scripts/run_detector.py`: Main entry point for running detection

**Critical Parameters:**
- `SR=22050, N_FFT=2048, HOP=128`: Fixed audio processing params
- `WHISTLE_LOW=3700, WHISTLE_HIGH=4200`: Whistle frequency band (Hz)
- `CLS_THRESHOLD=0.72`: Primary model confidence threshold
- Feature thresholds: `MAX_FLATNESS=0.025, MIN_GRAD_F=4.7, MIN_RIDGE=11`

### 2. detector_slop/ (Experimental CNN)
Research variants testing CNN-based detection with PyTorch:
- `train_cnn.py`: Dual-band (full + whistle) mel-spectrogram CNN with data augmentation
- `detector_cnn_runs.py`: Hybrid DSP+CNN inference pipeline
- `dsp_detector.py`: Pure signal processing baseline (no ML)
- Dataset split: `train: [match1,7,2,9,8,13,14,15], val: [11,4], test: [3,16]`

### 3. Vision/ (YOLO Object Detection)
YOLOv8-based volleyball detection for visual analysis:
- `train_model.py`: YOLO training script (uses `data.yaml` for dataset config)
- Frame extraction and annotation tools for ball/player tracking

### 4. Labeling Tools
Multiple Flask apps for human annotation:
- `WhistleDetector/app.py`: Main whistle review app with YouTube sync
- `WhistleDetector/app_viewer.py`: Detection review with spectrogram visualization
- `labeling_app/app.py`: Vision frame labeling for YOLO training
- All use `snippets/` for audio clips and spectrograms

## Critical Developer Workflows

### Running Whistle Detection
1. **Configure match:** Edit `WhistleDetector/config.py` → Set `MATCH_NUM`
2. **Run detector:**
   ```powershell
   cd WhistleDetector
   python scripts/run_detector.py
   ```
3. **Review ambiguous:** `python scripts/run_hitl.py` (Tkinter GUI)
4. **Output:** CSVs in `timelines/match{N}_whistles.csv`, clips in `clips/match{N}/`

### Training Models
**ML Models (scikit-learn):**
- Trained models saved as `best_model.pkl` and `ambiguous_best_model.pkl`
- Training data: `training_with_uid.csv` with 18 features + label
- Feature extraction: `big_labeler.py` for manual labeling → CSV

**CNN Models (PyTorch):**
```powershell
cd detector_slop
python train_cnn.py  # Outputs whistle_cnn.pth
```

**YOLO (Vision):**
```powershell
cd Vision
python train_model.py  # Requires data.yaml
```

### Data Annotation
- **Whistle Ground Truth:** `annotate_whistles.py` (PySide6 video player, saves to `whistles_match{N}.json`)
- **Re-anchoring:** `anchor_normalizer.py` → converts timestamps to precise attack-point anchors
- Combined GT: `detector_slop/whistles_all_anchored_attack.json`

## Project-Specific Conventions

### Path Management
- **Absolute Windows paths only** (`E:\Volleyballey\` or `D:\Volleyballey\`)
- Video files: `E:\Volleyballey\videos\match{N}.mp4`
- FFmpeg location: `C:\ffmpeg\bin\ffmpeg.exe`
- Dataset splits hardcoded in dicts (not config files)

### Feature Engineering
Features follow strict order for model compatibility:
```python
["rms", "flatness", "centroid", "band_energy", "peak_freq",
 "ridge_length", "grad_t", "grad_f", "rolloff", "bandwidth", 
 "zcr", "contrast", "tonnetz", "mfcc1"..."mfcc5"]
```
**Never reorder features** without retraining all models.

### Detection Philosophy
- **Physics-first:** Hard thresholds eliminate obvious non-whistles before ML
- **Cascade routing:** Models arranged by confidence, with HITL as safety net
- **UID format:** `match{N}_{start_ms:010d}_{end_ms:010d}` for tracking
- **Temporal merging:** MERGE_WINDOW_SEC=1.0 to deduplicate close detections

### Flask App Patterns
All labeling apps follow:
1. Load detections from CSV
2. Random sampling of unlabeled items
3. YouTube iframe sync with `?start={seconds}&autoplay=1`
4. Keyboard shortcuts (W=whistle, N=noise)
5. Save labels to separate CSV

## Integration Points

### External Dependencies
- **FFmpeg:** Audio extraction from video (`subprocess` calls)
- **librosa:** Audio processing (STFT, mel-spectrograms, features)
- **scikit-learn:** RandomForest/LightGBM models
- **PyTorch:** CNN training (CUDA optional)
- **ultralytics:** YOLO training
- **PySide6/Tkinter:** GUI annotation tools

### Cross-Module Communication
- JSON ground truth files passed between modules (`whistles_*.json`)
- Shared feature extraction in `WhistleDetector/audio/features.py`
- Models loaded via `joblib.load()` (scikit-learn) or `torch.load()` (PyTorch)

### Data Flow
```
Raw Video → annotate_whistles.py → whistles_match{N}.json
         → anchor_normalizer.py → whistles_all_anchored_attack.json
         → feat_extractor.py → whistle_dataset_attack/
         → train_cnn.py → whistle_cnn.pth
         → detector_cnn_runs.py → evaluation
```

## Common Pitfalls

1. **Match numbers mismatch:** Always sync `MATCH_NUM` in config.py with video filenames
2. **Feature order corruption:** Models fail silently if feature order changes
3. **Windows paths:** Use raw strings `r"E:\path"` to avoid escape issues
4. **Audio padding:** Short clips need padding before STFT (`len(y) < N_FFT`)
5. **Temporal duplicates:** Detection can fire multiple times on single whistle (use temporal merge)

## Quick Reference

**Run full detection pipeline:**
```powershell
cd WhistleDetector
# Edit config.py MATCH_NUM first!
python scripts/run_detector.py
python scripts/run_hitl.py  # If ambiguous detections exist
```

**Launch labeling app:**
```powershell
python app.py  # Default port 5000
```

**Evaluate with ground truth:**
```powershell
cd detector_slop
python eval.py  # Requires GT JSON
```

**Key files for understanding architecture:**
- [WhistleDetector/config.py](WhistleDetector/config.py): All thresholds and paths
- [WhistleDetector/detection/routing.py](WhistleDetector/detection/routing.py): Classification logic
- [WhistleDetector/ULTRA_DETECTOR.py](WhistleDetector/ULTRA_DETECTOR.py): Monolithic all-in-one script (reference impl)
- [detector_slop/train_cnn.py](detector_slop/train_cnn.py): CNN architecture and training
