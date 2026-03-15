# Volleylitics

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
Whistle Classification
     │
     ▼
Whistle Timeline
     │
     ▼
Rally Classification
     │
     ▼
Rally Timeline
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

Detect candidate referee whistles from match audio using DSP.

- [x] Extract audio from match recordings
- [x] Detect whistle candidates using spectral analysis
- [x] Group whistle frames into events

---

## 2. Whistle Classification

Filter whistle candidates using a CNN classifier.

- [x] Generate log-mel spectrogram snippets
- [x] Train whistle classification CNN
- [x] Remove false positives
- [x] Export whistle timestamps

---

## 3. Rally Classification

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

---

## 4. Ball Tracking (Planned)

Track the volleyball during rally segments.

- [x] Detect volleyball using object detection
- [ ] Track ball trajectory
- [ ] Store ball coordinates over time

---

## 5. Ball Landing Detection (Planned)

Detect where the ball contacts the court floor.

- [ ] Detect ball-floor contact
- [ ] Estimate court coordinates
- [ ] Store landing locations

---

## 6. Tactical Analytics (Planned)

Extract gameplay insights from rally and ball data.

- [ ] Ball landing heatmaps
- [ ] Setting distribution
- [ ] Serve direction
- [ ] Approximate serve speed (maybe)
- [ ] Rally statistics

---

## 7. Top-Down Tactical View (Planned)

Normalize the court into a tactical view.

- [ ] Estimate court geometry
- [ ] Transform ball positions to court space
- [ ] Generate rally visualizations

---

## 8. Web Dashboard (Planned)

Present match analytics through a web interface.

- [ ] Upload match recordings
- [ ] Automatic processing pipeline
- [ ] Rally timeline visualization
- [ ] Heatmaps and match analytics

---

# Technologies

- Python
- PyTorch
- OpenCV
- Librosa
- NumPy
- FFmpeg

---

# Motivation

This project started as an attempt to analyze matches of my own volleyball team and extract insights that could help improve our performance.

Manually reviewing matches makes it difficult to consistently track patterns such as ball placement, serve tendencies, or rally structure across many games. Volleylitics aims to automate this process by converting match recordings into structured data that can later be visualized through heatmaps and dashboards.

---

