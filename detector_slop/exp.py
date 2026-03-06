import os
import json
import torch
import numpy as np
import cv2
# ==============================
# IMPORT DSP PIPELINE
# ==============================

from dsp_detector import (
    load_audio_from_video,
    detect_active_frames,
    group_frames,
    extract_candidates,
    refine_candidates,
    estimate_whistle_band,
    compute_match_stats,
    rule_based_sifter,
    evaluate_candidate_hits,
    cfg
)

# ==============================
# IMPORT CNN
# ==============================

from train_cnn import TinyCNN, wav_to_logmel_from_audio

# ==============================
# CONFIG
# ==============================

MATCH_ID = "match13"
VIDEO_PATH = rf"E:\Volleyballey\videos\{MATCH_ID}.mp4"
MODEL_PATH = "whistle_model_better_cent.pth"
GT_JSON = "whistles_all_reanchored.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SNIPPET_SEC = 1.0
HALF_SNIPPET = SNIPPET_SEC / 2
ANCHOR_TOLERANCE = 0.6

# ==============================
# LOAD MODEL
# ==============================

model = TinyCNN().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
THRESHOLD = 0.65
model.eval()

print("Model loaded.")
print("Using threshold:", THRESHOLD)

# ==============================
# LOAD AUDIO
# ==============================

print("Loading audio...")
y = load_audio_from_video(VIDEO_PATH, cfg.sr)

# ==============================
# DSP PIPELINE (IDENTICAL TO TRAINING)
# ==============================

# PASS 1
active = detect_active_frames(y)
groups = group_frames(active)
stage1 = extract_candidates(groups)

# Adaptive band
low, high = estimate_whistle_band(y, stage1)
original_low = cfg.whistle_low
original_high = cfg.whistle_high

cfg.whistle_low = low
cfg.whistle_high = high

print(f"Adaptive band: {int(low)}–{int(high)} Hz")

# PASS 2
active = detect_active_frames(y)
groups = group_frames(active)
stage1 = extract_candidates(groups)
refined = refine_candidates(y, stage1)

print("Refined DSP candidates:", len(refined))

# Rule-based sifter
stats = compute_match_stats(refined, y)
accepted = rule_based_sifter(refined, y, stats)

print("After rule-based sifter:", len(accepted))

# Restore band
cfg.whistle_low = original_low
cfg.whistle_high = original_high

# ==============================
# LOAD GT
# ==============================

with open(GT_JSON, "r", encoding="utf-8") as f:
    rows = json.load(f)

gt_list = [
    g for g in rows
    if g["match_id"] == MATCH_ID
]

gt_times = [g["t_anchor"] for g in gt_list]

print("GT whistles:", len(gt_list))

# ==============================
# DSP EVALUATION
# ==============================

dsp_recall, dsp_offsets, dsp_missed, dsp_tp, dsp_fp = evaluate_candidate_hits(
    accepted,
    gt_list
)

print("\n===== DSP ONLY =====")
print("TP:", dsp_tp)
print("FP:", dsp_fp)
print("FN:", len(gt_list) - dsp_tp)
print("Recall:", round(dsp_recall, 4))

# ==============================
# CNN STAGE
# ==============================

final_centers = []

for start, end in accepted:

    center = (start + end) / 2

    s0 = int((center - HALF_SNIPPET) * cfg.sr)
    s1 = int((center + HALF_SNIPPET) * cfg.sr)

    if s0 < 0 or s1 > len(y):
        continue

    snippet = y[s0:s1]

    mel = wav_to_logmel_from_audio(snippet)
    mel_tensor = torch.tensor(mel).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(mel_tensor)).item()

    if prob > THRESHOLD:
        final_centers.append(center)

print("\nFinal CNN predictions:", len(final_centers))

# Convert centers → windows (same 1s window geometry)
final_windows = [
    (c - HALF_SNIPPET, c + HALF_SNIPPET)
    for c in final_centers
]

# ==============================
# HYBRID EVALUATION (CONSISTENT)
# ==============================

recall, offsets, missed, tp, fp = evaluate_candidate_hits(
    final_windows,
    gt_list
)

fn = len(gt_list) - tp
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n===== HYBRID RESULTS =====")
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Explosion:", round(len(final_windows) / len(gt_list), 2))

if offsets:
    offsets = np.array(offsets)
    print("Median abs offset:", round(np.median(np.abs(offsets)), 4))

# ==============================
# DSP vs CNN MISS BREAKDOWN
# ==============================

# Which GT were missed by DSP?
dsp_missed_set = set(dsp_missed)

# Which GT were missed after CNN?
final_missed_set = set(missed)

missed_by_dsp = dsp_missed_set
missed_by_cnn = final_missed_set - dsp_missed_set

print("\nMissed by DSP:", len(missed_by_dsp))
print("Missed by CNN:", len(missed_by_cnn))



def compute_motion_features(video_path, t, before=2, after=3):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int((t - before) * fps)
    end_frame = int((t + after) * fps)

    motions = []
    prev = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

    for f in range(start_frame, end_frame):

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            motions.append(np.mean(diff))

        prev = gray

    cap.release()

    if len(motions) < 2:
        return 0,0

    mid = int(len(motions) * before / (before + after))

    motion_before = np.mean(motions[:mid])
    motion_after = np.mean(motions[mid:])

    return motion_before, motion_after

# ==============================
# STATE MACHINE CLASSIFICATION
# ==============================

print("\n===== STATE MACHINE CLASSIFICATION =====")

times = sorted(final_centers)

labels = []

for i, t in enumerate(times):

    prev_t = times[i-1] if i > 0 else None
    next_t = times[i+1] if i < len(times)-1 else None

    delta_prev = t - prev_t if prev_t else None
    delta_next = next_t - t if next_t else None
    motion_before, motion_after = compute_motion_features(VIDEO_PATH, t)

    motion_ratio = motion_after / (motion_before + 1e-6)
    motion_delta = motion_after - motion_before

    label = "unknown"

    # ---- ADMIN detection ----
    if delta_prev is not None and delta_prev > 40:
        label = "admin"

    elif delta_prev is not None and delta_prev < 1.5:
        label = "admin"

    # ---- Motion-based serve detection ----
    elif motion_ratio < 0.35:
        label = "serve"

    # ---- Rally end detection ----
    elif motion_before > motion_after and delta_prev is not None and delta_prev < 25:
        label = "rally_end"

    # ---- fallback temporal rules ----
    elif delta_next is not None and 5 < delta_next < 25:
        label = "serve"

    elif delta_prev is not None and 4 < delta_prev < 25:
        label = "rally_end"

    else:
        label = "admin"

    labels.append(label)

# Print first few results
for t, lab in list(zip(times, labels))[:20]:
    print(f"{t:.2f} -> {lab}")



# ==============================
# STATE MACHINE EVALUATION
# ==============================

gt_map = {
    g["t_anchor"]: g["type"]
    for g in gt_list
}

correct = 0
total = 0

for t, pred in zip(times, labels):

    # find closest GT whistle
    closest = None
    min_dist = 999

    for g in gt_list:
        d = abs(t - g["t_anchor"])
        if d < min_dist:
            min_dist = d
            closest = g

    if closest and min_dist < ANCHOR_TOLERANCE:
        total += 1
        if closest["type"] == pred:
            correct += 1

print("\nSTATE MACHINE ACCURACY:", correct / total if total else 0)