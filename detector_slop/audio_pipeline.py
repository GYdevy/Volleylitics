import json
import torch
import numpy as np
from tqdm import tqdm

from dsp_detector import (
    load_audio_from_video,
    detect_active_frames,
    group_frames,
    extract_candidates,
    refine_candidates,
    estimate_whistle_band,
    compute_match_stats,
    rule_based_sifter,
    cfg
)

from resnet_whistle import wav_to_logmel_from_audio

import torchvision.models as models
import torch.nn as nn

# =============================
# MODEL
# =============================

class ResNetWhistle(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            6, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.backbone.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.backbone(x).squeeze()


# =============================
# SETTINGS
# =============================

MATCH_ID = "match17"
VIDEO_PATH = f"/mnt/windows_share/{MATCH_ID}.mp4"

WHISTLE_MODEL = "best_resnet_whistle_mix.pth"
GT_FILE = "anchored_matches/whistles_all_with_serve_contacts.json"

print("\nLoading GT...")

with open(GT_FILE, "r") as f:
    gt_data = json.load(f)

# filter only this match
gt_times = [
    d["time"]
    for d in gt_data
    if d["match_id"] == MATCH_ID
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

SNIPPET_SEC = 1.0
HALF_SNIPPET = SNIPPET_SEC / 2

# thresholds for analysis
MAX_RALLY_DURATION = 35
MIN_RALLY_DURATION = 1.5   # avoid noise
TIMEOUT_GAP = 20
TOLERANCE = 0.7
# =============================
# LOAD MODEL
# =============================

print("Loading whistle CNN")

whistle_model = ResNetWhistle().to(DEVICE)

checkpoint = torch.load(WHISTLE_MODEL, map_location=DEVICE)
whistle_model.load_state_dict(checkpoint["model_state"])

whistle_model.eval()

# =============================
# LOAD AUDIO
# =============================

print("Loading audio")
y = load_audio_from_video(VIDEO_PATH, cfg.sr)

# =============================
# DSP DETECTION
# =============================

print("Running DSP detector")

active = detect_active_frames(y)
groups = group_frames(active)
stage1 = extract_candidates(groups)

low, high = estimate_whistle_band(y, stage1)

cfg.whistle_low = low
cfg.whistle_high = high

active = detect_active_frames(y)
groups = group_frames(active)
stage1 = extract_candidates(groups)

refined = refine_candidates(y, stage1)
stats = compute_match_stats(refined, y)
accepted = rule_based_sifter(refined, y, stats)

print("DSP candidates:", len(accepted))
# =============================
# DSP EVALUATION (NO CNN)
# =============================

print("\nEvaluating DSP alone...")

dsp_times = sorted([(s + e) / 2 for s, e in accepted])

matched_gt_dsp = set()
matched_dsp = set()

dsp_time_errors = []

for i, p in enumerate(dsp_times):
    best_j = None
    best_diff = float("inf")

    for j, g in enumerate(gt_times):
        if j in matched_gt_dsp:
            continue

        diff = abs(p - g)

        if diff < best_diff and diff <= TOLERANCE:
            best_diff = diff
            best_j = j

    if best_j is not None:
        matched_dsp.add(i)
        matched_gt_dsp.add(best_j)
        dsp_time_errors.append(best_diff)

# metrics
TP_dsp = len(matched_dsp)
FP_dsp = len(dsp_times) - TP_dsp
FN_dsp = len(gt_times) - TP_dsp

precision_dsp = TP_dsp / (TP_dsp + FP_dsp) if TP_dsp + FP_dsp > 0 else 0
recall_dsp = TP_dsp / (TP_dsp + FN_dsp) if TP_dsp + FN_dsp > 0 else 0

if precision_dsp + recall_dsp > 0:
    f1_dsp = 2 * precision_dsp * recall_dsp / (precision_dsp + recall_dsp)
else:
    f1_dsp = 0

print("\n===== DSP METRICS =====")
print(f"TP: {TP_dsp}")
print(f"FP: {FP_dsp}")
print(f"FN: {FN_dsp}")

print(f"Precision: {precision_dsp:.4f}")
print(f"Recall:    {recall_dsp:.4f}")
print(f"F1:        {f1_dsp:.4f}")

if dsp_time_errors:
    print(f"\nAvg timing error: {np.mean(dsp_time_errors):.3f}s")
# =============================
# CNN FILTER
# =============================

print("Running whistle CNN")

cnn_centers = []

for start, end in tqdm(accepted):

    center = (start + end) / 2

    s0 = int((center - HALF_SNIPPET) * cfg.sr)
    s1 = int((center + HALF_SNIPPET) * cfg.sr)

    if s0 < 0 or s1 > len(y):
        continue

    snippet = y[s0:s1]
    mel = wav_to_logmel_from_audio(snippet)

    mel_tensor = torch.tensor(mel).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(whistle_model(mel_tensor)).item()

    if prob > 0.05:
        cnn_centers.append(center)

cnn_centers = sorted(cnn_centers)

print("Final whistles:", len(cnn_centers))

# =============================
# EVALUATION AGAINST GT
# =============================



gt_times = sorted(gt_times)
pred_times = sorted(cnn_centers)

print("GT whistles:", len(gt_times))
print("Pred whistles:", len(pred_times))

# =============================
# MATCHING
# =============================

matched_gt = set()
matched_pred = set()

time_errors = []

for i, p in enumerate(pred_times):
    best_j = None
    best_diff = float("inf")

    for j, g in enumerate(gt_times):
        if j in matched_gt:
            continue

        diff = abs(p - g)

        if diff < best_diff and diff <= TOLERANCE:
            best_diff = diff
            best_j = j

    if best_j is not None:
        matched_pred.add(i)
        matched_gt.add(best_j)
        time_errors.append(best_diff)

# =============================
# METRICS
# =============================

TP = len(matched_pred)
FP = len(pred_times) - TP
FN = len(gt_times) - TP

precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0

if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0

print("\n===== WHISTLE METRICS =====")
print(f"TP: {TP}")
print(f"FP: {FP}")
print(f"FN: {FN}")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")

if time_errors:
    print(f"\nAvg timing error: {np.mean(time_errors):.3f}s")
    print(f"Median error:     {np.median(time_errors):.3f}s")

# =============================
# DEBUG OUTPUT
# =============================

# False positives (pred not matched)
fp_times = [pred_times[i] for i in range(len(pred_times)) if i not in matched_pred]

# False negatives (gt not matched)
fn_times = [gt_times[i] for i in range(len(gt_times)) if i not in matched_gt]

print("\nSample FP:", fp_times[:10])
print("Sample FN:", fn_times[:10])