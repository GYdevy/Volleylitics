# Volleylitics

**Live Project Page:** [gydevy.github.io/Volleylitics](https://gydevy.github.io/Volleylitics/)



Volleylitics is a computer vision and audio analytics system that converts raw volleyball match recordings into structured gameplay data.

The system processes a full match video and reconstructs the structure of the game by detecting whistles, identifying rallies, and analyzing gameplay events. The long-term goal is to extract tactical insights such as ball landing heatmaps, serve patterns, and offensive tendencies from standard match recordings.

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
Future notice - should use a light model to detect ball movement above net instead of yellow change.
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

