# Volleylitics – Technical System Writeup

This writeup summarizes the system design behind **Volleylitics**, a volleyball analytics pipeline built from raw match recordings.  
The structure is kept practical: **goal → what failed → what I learned → what worked → results**.

---

## Table of Contents

- [1. Project Goal](#1-project-goal)
- [2. Pipeline Overview](#2-pipeline-overview)
- [3. Whistle Detection](#3-whistle-detection)
  - [3.1 Goal](#31-goal)
  - [3.2 Problem](#32-problem)
  - [3.3 Annotation and Label Anchoring](#33-annotation-and-label-anchoring)
  - [3.4 DSP Candidate Detection](#34-dsp-candidate-detection)
  - [3.5 Lesson Learned](#35-lesson-learned)
  - [3.6 Final Solution](#36-final-solution)
  - [3.7 Results](#37-results)
- [4. Rally Segmentation](#4-rally-segmentation)
  - [4.1 Goal](#41-goal)
  - [4.2 Problem](#42-problem)
  - [4.3 Failed Attempts](#43-failed-attempts)
  - [4.4 CNN Frame Classifier](#44-cnn-frame-classifier)
  - [4.5 Final Decision Cascade](#45-final-decision-cascade)
  - [4.6 Results](#46-results)
- [5. Main Pipeline Weakness](#5-main-pipeline-weakness)
- [6. Ball Detection](#6-ball-detection)
- [7. Ground Contact Detection](#7-ground-contact-detection)
  - [7.1 Problem](#71-problem)
  - [7.2 Homography Solution](#72-homography-solution)
  - [7.3 Output Example](#73-output-example)
- [8. Ball Trajectory](#8-ball-trajectory)
- [9. Site and Output](#9-site-and-output)
- [10. Future Work](#10-future-work)

---

## 1. Project Goal

The goal was to turn raw volleyball match videos into structured analytics data.

The desired output was not just “detect a ball in a frame”, but a full pipeline that could:

- detect when rallies happen,
- isolate useful rally clips,
- detect ball movement,
- estimate landing / ground-contact points,
- map those points to court coordinates,
- generate data for heatmaps, trajectories, and match inspection.

The initial motivation came from recurring gameplay problems: weak serves, inaccurate freeball reception, poor defensive positioning, and unorganized blocking. Since the matches were already recorded, the project became a way to extract useful information from footage that already existed.

---

## 2. Pipeline Overview

The first naive plan was simple:

1. Load match.
2. Detect ball.
3. Find ground contact.
4. Report result.

The problem was compute cost. Running ball detection on a 25-second rally took about **3 minutes**. A full match is at least one hour, so running ball detection on the entire match would take roughly **7+ hours**.

That forced a pipeline change: instead of detecting the ball everywhere, first detect useful time windows.

```text
Raw Match Video
       │
       ├──────────────────────────────┐
       │                              │
       ▼                              ▼
Audio Track                    Video Frames
       │                              │
       ▼                              ▼
STFT + DSP Features           Rally Frame Classifier
       │                              │
       ▼                              ▼
Whistle Candidates            In-play Probabilities
       │                              │
       ▼                              ▼
Whistle CNN Classifier         Rally Decision Cascade
       │                              │
       └──────────────┬───────────────┘
                      ▼
               Rally Intervals
                      │
                      ▼
             Ball Detection (YOLO)
                      │
                      ▼
              Ball Position Track
                      │
                      ▼
      Ground Contact Estimation + Homography
                      │
                      ▼
       Court Coordinates + Trajectory JSON
                      │
                      ▼
                 Website / Heatmaps
```

This changed the project from “detect the ball in a whole match” to a staged system:

1. detect whistles,
2. segment rallies,
3. run ball detection only on relevant clips,
4. map important events to the court.

---

## 3. Whistle Detection

### 3.1 Goal

The first major need was reliable whistle detection.

A whistle gives useful temporal structure because points usually end with a referee whistle. If I can detect whistles, I can avoid processing the entire match and focus around likely rally endings.

---

### 3.2 Problem

Whistle detection looked easy at first because whistles are strong audio events. In practice, it was much harder because other sounds overlap with the same frequency band.

The main false positive was shoe squeaks.

Raw whistle example:

<audio controls src="/data/blog/images/match15_179.wav"></audio>

Band-filtered whistle:

<audio controls src="/data/blog/images/input_3700_4300.wav"></audio>

Shoe squeak:

<audio controls src="/data/blog/images/Shoe_squeak.wav"></audio>

Band-filtered shoe squeak:

<audio controls src="/data/blog/images/shoe_squeak_band.wav"></audio>

The bandpass filter did not solve the problem because shoe squeaks can live in the same 3700–4300 Hz range.

Shoe squeak spectrogram:

![Shoe Squeak Spectrogram](/data/blog/images/squeak.png)

Whistle spectrogram:

![Whistle Spectrogram](/data/blog/images/whistle.png)

The issue was not just “find high energy in whistle frequencies”. The detector had to separate very similar acoustic events.

---

### 3.3 Annotation and Label Anchoring

I annotated whistles manually across multiple matches. A whistle annotation looked like this:

```json
{
  "match_id": "match1",
  "whistle_id": 1,
  "time": 396.473,
  "type": "other",
  "t_raw": 396.473,
  "global_id": 1,
  "t_anchor": 396.303
}
```

The raw timestamp came from the manual key press during annotation. That was not accurate enough, because human clicking introduces anticipation and attention error.

So I added acoustic anchoring around each manual click:

```python
# around each manual whistle click:
# search ±0.60s in the 3700–4300 Hz whistle band

if energy crosses 0.25 and flux is above 0.35:
    if the rise holds for 3 of the next 5 frames:
        anchor = onset
elif peak_energy >= 0.20:
    anchor = local_peak
else:
    anchor = raw_click
```

This made the labels more consistent by moving the manual click toward an acoustic onset or peak.

A major mistake here was that some matches were downloaded from YouTube instead of using raw recordings. That introduced compression artifacts and reduced dataset quality.

---

### 3.4 DSP Candidate Detection

The first detector was based on STFT features.

```python
# Convert audio into overlapping short-time spectral frames
S = stft(audio, sr=22050, n_fft=2048, hop=128)
mag = abs(S)

# Keep only the whistle frequency region
band = mag[(freqs >= 3700) & (freqs <= 4300)]

# Represent each frame by whistle-band properties
band_energy = mean(band, axis=0)
band_peak = max(band, axis=0)
band_mean = mean(band, axis=0) + 1e-8
sharpness = band_peak / band_mean
flatness = spectral_flatness(mag)

# Compare frames using a whistle-likeness score
score = band_energy + sharpness - 1.2 * flatness
```

This was useful for candidate generation, but it was not reliable enough as a final detector.

It was too sensitive to shoe squeaks and other whistle-like sounds.

---

### 3.5 Lesson Learned

The most important lesson was that a classifier trained on perfectly centered whistle snippets does not necessarily work in the real pipeline.

At first, I trained a small CNN on centered positive and negative examples. Validation looked very good, but the full detector performed badly.

The reason was distribution mismatch:

- training samples were clean and centered,
- candidate detector outputs were noisy and not always centered.

So the real lesson was:

> The candidate detector defines the real input distribution.  
> Training data has to be generated the same way inference data is generated.

---

### 3.6 Final Solution

The final whistle detector became a multi-stage system:

1. compute STFT features,
2. score frames using DSP features,
3. group high-scoring regions into candidates,
4. refine candidate timing,
5. filter candidates with rule-based statistics,
6. classify 1-second log-mel snippets with a CNN,
7. apply temporal NMS to remove duplicates.

The learned classifier used a custom PyTorch ResNet18-style model with a 6-channel spectrogram representation:

- full-band log-mel,
- full-band delta,
- full-band delta-delta,
- whistle-band log-mel,
- whistle-band delta,
- whistle-band delta-delta.

In other words, whistle detection was not one model call. It was a sequence of signal processing, candidate generation, learned classification, and temporal cleanup.

---

### 3.7 Results

Final whistle detection results:

- **Precision:** 98.35%
- **Recall:** 99.05%

This was good enough to move on to rally segmentation.

---

## 4. Rally Segmentation

### 4.1 Goal

The next goal was to segment the match into actual rallies.

Whistles alone were not enough because volleyball has several whistle types:

- serve whistle,
- point-ending whistle,
- substitution whistle,
- timeout whistle,
- set / match administrative whistles.

The useful interval is not “any whistle minus 2 seconds”. The system needs to decide which whistle intervals contain real gameplay.

---

### 4.2 Problem

Volleyball rallies are discrete events, but they are not visually uniform.

A rally can be:

- a long chaotic exchange,
- a calm serve reception,
- a short missed serve,
- an ace,
- a noisy scene with players moving between points.

This makes simple motion or audio rules unreliable.

---

### 4.3 Failed Attempts

The first idea was motion difference between frames.

That worked in chaotic parts of a rally, but failed at the beginning of points because players are often standing still while waiting for the serve.

The second idea was audio response after a point, since players or spectators often react when the rally ends.

That was also unreliable:

- quiet games have weak reaction audio,
- loud games have constant crowd noise,
- crowd audio does not reliably indicate gameplay state.

---

### 4.4 CNN Frame Classifier

The working approach was a frame-level CNN classifier.

The model predicts whether a sampled frame is “in play” or “not in play”.

In-play example:

![In play frame](/data/blog/images/inplay.png)

Not-in-play example:

![Not in play frame](/data/blog/images/notinplay.png)

The model reached around **92% F1**, which was useful because a rally contains many frames. Even if single-frame classification is imperfect, the timeline can be smoothed and aggregated.

To reduce compute, frames were sampled every 8 frames instead of processing the full 60 FPS stream.

---

### 4.5 Final Decision Cascade

The CNN frame timeline worked well, but short rallies were still often missed.

To fix that, I combined the CNN timeline with a visual motion cue called `yellow_score`.

The `yellow_score` is a simple proxy for ball movement, based on how much yellow motion appears in the upper part of the image.

The final decision cascade looked like this:

```python
inplay_ratio = rally_cnn_vote(interval)
yellow_score = visual_motion_vote(interval)
score = inplay_ratio + 0.02 * yellow_score

if inplay_ratio > 0.35:
    rally = True
elif yellow_score > 12:
    rally = True
elif yellow_score > 10 and duration < 6:
    rally = True
elif inplay_ratio > 0.25 and yellow_score > 4:
    rally = True
elif inplay_ratio > 0.20 and yellow_score > 5:
    rally = True
elif inplay_ratio > 0.15 and yellow_score > 2:
    rally = True
elif 0.1 < score < 0.3:
    send_to_HITL()
else:
    reject()
```

Ambiguous intervals were sent to human-in-the-loop review. In practice, this meant around 30 short clips per match, with each one taking roughly 2–3 seconds to review.

One limitation is that `yellow_score` depends on the ball color. If the league switches to a non-yellow ball, this feature becomes weaker. A better future version would use the ball detector itself instead of color heuristics.

---

### 4.6 Results

<details>
<summary>Rally detection evaluation</summary>

```text
======== Rally Detection Evaluation ========
GT rallies: 116

===================================
MODEL: RAW
===================================
Detected rallies: 113
TP: 105 | FP: 8 | FN: 11
Precision: 0.929
Recall: 0.905
Mean start error: 0.052
Mean end error: 0.059

---- FALSE POSITIVES ----
FP: 00:10:57 → 00:11:04 (dur 6.12s)
FP: 00:11:04 → 00:11:26 (dur 21.98s)
FP: 00:21:15 → 00:21:25 (dur 10.28s)
FP: 00:21:25 → 00:21:29 (dur 4.03s)
FP: 00:33:54 → 00:33:58 (dur 4.16s)
FP: 00:37:21 → 00:37:31 (dur 10.77s)
FP: 00:42:21 → 00:42:27 (dur 5.46s)
FP: 00:42:27 → 00:42:29 (dur 2.68s)

---- FALSE NEGATIVES ----
FN: 00:10:57 → 00:11:25 (dur 28.84s)
FN: 00:11:44 → 00:11:53 (dur 8.34s)
FN: 00:14:23 → 00:14:32 (dur 8.66s)
FN: 00:21:15 → 00:21:29 (dur 14.24s)
FN: 00:28:01 → 00:28:11 (dur 10.1s)
FN: 00:37:19 → 00:37:31 (dur 12.65s)
FN: 00:42:21 → 00:42:29 (dur 8.04s)
FN: 00:43:44 → 00:43:50 (dur 6.15s)
FN: 01:05:26 → 01:05:34 (dur 8.13s)
FN: 01:05:48 → 01:05:58 (dur 9.91s)
FN: 01:07:45 → 01:07:51 (dur 5.96s)

===================================
MODEL: WITH_HITL
===================================
Detected rallies: 115
TP: 113 | FP: 2 | FN: 3
Precision: 0.983
Recall: 0.974
Mean start error: 0.059
Mean end error: 0.058

---- FALSE POSITIVES ----
FP: 00:33:54 → 00:33:58 (dur 4.16s)
FP: 00:37:21 → 00:37:31 (dur 10.77s)

---- FALSE NEGATIVES ----
FN: 00:14:23 → 00:14:32 (dur 8.66s)
FN: 00:37:19 → 00:37:31 (dur 12.65s)
FN: 01:05:26 → 01:05:34 (dur 8.13s)
```

</details>

With HITL:

- **Precision:** 0.983
- **Recall:** 0.974

Some “false positives” were practically acceptable because they referred to the same rally but started slightly later. Since the ground-contact use case only needs the rally ending, small start-time errors are usually not critical.

---

## 5. Main Pipeline Weakness

The biggest weakness is error propagation.

If whistle detection produces a false whistle near the end of a real rally, the rally can get cut too early. Then a short-rally filter may discard the ending segment, which is exactly the part needed for ground-contact detection.

![Probability Graph](/data/blog/images/graph.png)

The key lesson is:

> Early-stage temporal errors can remove the most important downstream data.

Planned fix:

- improve whistle detection further,
- use ball detection as an additional signal,
- evaluate short intervals before discarding them.

---

## 6. Ball Detection

### Goal

Detect the volleyball inside rally clips.

At this point, the earlier segmentation stages made the problem much smaller: ball detection no longer needed to run over full match videos, only over relevant clips.

---

### Solution

I annotated around 2000 images, initially using my own labeling tool and later moving to Roboflow for a cleaner workflow.

The model used was:

- **YOLOv11 Object Detection Nano**

---

### Results

- **mAP@50:** 97.7%
- **Precision:** 99.5%
- **Recall:** 92.3%
- **F1:** 95.8%

The model mostly misses far balls, which is acceptable because the ground-contact event of interest happens on the near side of the court.

A known failure case is very fast motion almost orthogonal to the camera axis, but this is relatively uncommon.

---

## 7. Ground Contact Detection

### 7.1 Problem

Ground contact is difficult because a single static camera cannot directly recover depth.

While the ball is airborne, its 2D image location does not uniquely determine its 3D court location. Also, actual ground contact can last only 1–2 frames.

So the goal was not full 3D reconstruction. The goal was to find a useful approximation of where the ball landed.

---

### 7.2 Homography Solution

The usable assumption is:

> When the ball contacts the ground, it is close enough to the court plane to project it using homography.

The court half is 9×9 meters. I manually selected the four corners of the near half and mapped them to real court coordinates.

![Court Coordinates](/data/blog/images/court.png)

```python
# The 4 corners
img_pts = np.array([
    [1, 1040],
    [1919, 1034],
    [1426, 779],
    [516, 780]
], dtype=np.float32)

# net line coords
net_line = np.array([
    [471, 530],
    [1441, 530]
], dtype=np.float32)

court_pts = np.array([
    [0, 0],
    [9, 0],
    [9, 9],
    [0, 9]
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, court_pts)
```

The landing frame is estimated by taking the largest image-space Y-coordinate from the recent ball positions:

```python
pt = np.array([[[985, 575]]], dtype=np.float32)
mapped = cv2.perspectiveTransform(pt, H)
```

This is a heuristic, but it fits the camera setup: in the near side view, larger image-space Y generally means closer to the ground / camera.

---

### 7.3 Output Example

Input frame:

![Match Frame](/data/blog/images/raw.png)

Projected output:

![Transformed Frame](/data/blog/images/image.png)

Example saved result:

<details>
<summary>Example rally output JSON</summary>

```json
[
  {
    "clip_name": "rally_006.mp4",
    "rally_id": 6,
    "start": 470.8136054421769,
    "end": 492.3675283446712,
    "set_id": 1,
    "positions": [
      [1240, 725, 374],
      [1241, 723, 384],
      [1242, 722, 396],
      [1292, 270, 796],
      [1293, 257, 800],
      [1294, 245, 788]
    ],
    "attack_point": [1.3600405679513186, 9.0],
    "landing_point": [-0.2877609431743622, 3.585444211959839],
    "debug_output_path": "/workspace/heatmaps/match14/debug_clips/rally_006_debug.mp4",
    "set": 1
  }
]
```

</details>

---

## 8. Ball Trajectory

### Goal

Estimate the direction of the attack, not just the landing point.

---

### Problem

The difficult part is deciding where the attack trajectory begins. With one camera, exact 3D trajectory reconstruction is not available.

---

### Solution

Use the net line as a proxy for the ball crossing moment.

```python
# net line coords
net_line = np.array([
    [471, 530],
    [1441, 530]
], dtype=np.float32)
```

Since the ball must cross the net before landing, the estimated net-crossing point can be connected to the ground-contact point.

![Ball trajectory and net crossing point](/data/blog/images/track.png)

This does not produce a perfect physical trajectory, but it gives a useful top-down approximation of attack direction.

---

## 9. Site and Output

The final outputs are saved as JSON and passed to the frontend.

The site is used to inspect:

- matches,
- rally clips,
- heatmaps,
- landing points,
- trajectories.

The frontend itself is mostly generated / assisted, but it provides a useful interface for reviewing results.

---

## 10. Future Work

The current system is infrastructure for a larger analytics project.

Possible next steps:

- add a camera on the opposite side of the court,
- add an orthogonal camera for depth,
- replace yellow-motion scoring with model-based ball activity,
- add pose and action detection,
- extract serve statistics,
- generate top-down tactical views,
- build a full match statistics generator.

The main challenge is that each additional feature increases compute cost and complexity, so the pipeline needs to remain staged and efficient.
