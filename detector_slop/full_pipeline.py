import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

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

from train_cnn import TinyCNN, wav_to_logmel_from_audio
from rse import build_model


# =============================
# SETTINGS
# =============================

MATCH_ID = "match13"

VIDEO_PATH = rf"E:\Volleyballey\videos\{MATCH_ID}.mp4"

WHISTLE_MODEL = "whistle_model_better_cent.pth"
RALLY_MODEL = "rally_model_best_corrected.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SNIPPET_SEC = 1.0
HALF_SNIPPET = SNIPPET_SEC / 2

RALLY_THRESHOLD = 0.90
MIN_RALLY_RATIO = 0.35
FP_RATIO = 0.1

MAX_RALLY_DURATION = 35
TIMEOUT_GAP = 20

TIMELINE_STEP = 0.2


SEGMENTS = [
    (7*60 + 10, 27*60 + 8),
    (30*60 + 20, 53*60 + 40),
    (56*60 + 30, 1*3600 + 12*60 + 10)
]


# =============================
# LOAD MODELS
# =============================

print("Loading whistle CNN")

whistle_model = TinyCNN().to(DEVICE)

checkpoint = torch.load(WHISTLE_MODEL, map_location=DEVICE)
whistle_model.load_state_dict(checkpoint["model_state"])

whistle_model.eval()


print("Loading rally CNN")

rally_model = build_model().to(DEVICE)
rally_model.load_state_dict(torch.load(RALLY_MODEL, map_location=DEVICE))
rally_model.eval()


# =============================
# LOAD AUDIO
# =============================

print("Loading audio")

y = load_audio_from_video(VIDEO_PATH, cfg.sr)


# =============================
# DSP WHISTLE DETECTION
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


# =============================
# CNN WHISTLE FILTER
# =============================

print("Running whistle CNN filter")

cnn_centers = []

for start, end in tqdm(accepted, desc="Whistle CNN"):

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

    if prob > 0.5:
        cnn_centers.append(center)

print("Detected whistles:", len(cnn_centers))


# =============================
# BUILD RALLY TIMELINE
# =============================

print("Building rally probability timeline")

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

timeline_t = []
timeline_p = []

total_steps = int(duration / TIMELINE_STEP)

pbar = tqdm(total=total_steps, desc="Rally timeline")

t = 0

while t < duration:

    frame_idx = int(t * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = rally_model(x)

        probs = F.softmax(logits, dim=1)

        p = probs[0,1].item()

    timeline_t.append(t)
    timeline_p.append(p)

    t += TIMELINE_STEP
    pbar.update(1)

pbar.close()
cap.release()

timeline_t = np.array(timeline_t)
timeline_p = np.array(timeline_p)


# =============================
# SEGMENT FILTER
# =============================

def in_segments(t):

    for s,e in SEGMENTS:
        if s <= t <= e:
            return True

    return False


# =============================
# ANALYZE WHISTLE INTERVALS
# =============================

print("Analyzing whistle intervals")

rallies = []

for i in tqdm(range(len(cnn_centers)-1), desc="Rally detection"):

    w0 = cnn_centers[i]
    w1 = cnn_centers[i+1]

    if not in_segments(w0):
        continue

    mask = (timeline_t >= w0) & (timeline_t <= w1)

    probs = timeline_p[mask]

    if len(probs) == 0:
        continue

    inplay_ratio = np.mean(probs > RALLY_THRESHOLD)

    interval_len = w1 - w0


    # =============================
    # REAL RALLY
    # =============================

    if inplay_ratio > MIN_RALLY_RATIO:

        rally_indices = np.where(probs > RALLY_THRESHOLD)[0]

        start_t = timeline_t[mask][rally_indices[0]]
        end_t = timeline_t[mask][rally_indices[-1]]

        rallies.append({
            "start": float(start_t),
            "end": float(end_t),
            "duration": float(end_t - start_t)
        })

        if end_t - start_t > MAX_RALLY_DURATION:
            print("Possible missed whistle near", start_t)


    # =============================
    # FALSE POSITIVE WHISTLE
    # =============================

    elif inplay_ratio < FP_RATIO:

        print("Possible FP whistle between", w0, w1)


# =============================
# TIMEOUT DETECTION
# =============================

print("Checking for timeouts")

for i in range(1,len(rallies)):

    gap = rallies[i]["start"] - rallies[i-1]["end"]

    if gap > TIMEOUT_GAP:
        print("Possible timeout or set break at", rallies[i]["start"])


# =============================
# SAVE
# =============================

print("Detected rallies:", len(rallies))

output_path = f"{MATCH_ID}_rallies_behavior.json"

with open(output_path,"w") as f:
    json.dump(rallies,f,indent=2)

print("Saved:",output_path)