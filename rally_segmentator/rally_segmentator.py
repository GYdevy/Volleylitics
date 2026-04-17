import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from config import (WHISTLE_MODEL,
                    RALLY_MODEL,
                    VIDEO_DIR,
                    MODEL_DIR,
                    OUTPUT_DIR,
)

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



from models_dir.resnet_whistle import wav_to_logmel, wav_to_logmel_from_audio

import torchvision.models as models
import torch.nn as nn

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
from models_dir.rse import build_model

def smooth_signal(x, k=7):
    out = np.zeros_like(x)
    half = k // 2

    for i in range(len(x)):
        s = max(0, i - half)
        e = min(len(x), i + half + 1)
        out[i] = np.mean(x[s:e])

    return out


def window_nms(candidates, window=1.5):
    candidates = sorted(candidates, key=lambda x: x[0])

    result = []
    i = 0

    while i < len(candidates):
        t0 = candidates[i][0]

        cluster = []
        j = i

        while j < len(candidates) and candidates[j][0] - t0 <= window:
            cluster.append(candidates[j])
            j += 1

        center_mean = np.mean([c[0] for c in cluster])

        def score_fn(x):
            t, p = x
            return p - 0.15 * abs(t - center_mean)

        best = max(cluster, key=score_fn)

        result.append(best[0])
        i = j

    return result

def hysteresis(probs):
    state = 0
    out = []

    for p in probs:
        if state == 0 and p > ENTER:
            state = 1
        elif state == 1 and p < EXIT:
            state = 0

        out.append(state)

    return np.array(out)


def remove_short_runs(states, min_len=3):
    out = states.copy()
    i = 0

    while i < len(states):
        j = i
        while j < len(states) and states[j] == states[i]:
            j += 1

        if (j - i) < min_len:
            out[i:j] = 1 - states[i]

        i = j

    return out

# =============================
# SETTINGS
# =============================

MATCH_ID = "match17"
VIDEO_PATH = f"/videos/{MATCH_ID}.mp4"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

SNIPPET_SEC = 1.0
HALF_SNIPPET = SNIPPET_SEC / 2

ENTER = 0.65
EXIT  = 0.4
MIN_RALLY_RATIO = 0.35
FP_RATIO = 0.1

MAX_RALLY_DURATION = 35
TIMEOUT_GAP = 20

TIMELINE_STEP = 0.2

SEGMENTS = [
    (5*60 + 10, 29*60 + 8),
    (30*60 + 20, 55*60 + 40),
    (57*60 + 30, 1*3600 + 9*60 + 4)
]

print("Timeline Step:", TIMELINE_STEP)
# =============================
# LOAD MODELS
# =============================

print("Loading whistle CNN")

import torchvision.models as models
import torch.nn as nn

whistle_model = ResNetWhistle().to(DEVICE)

checkpoint = torch.load(MODEL_DIR / WHISTLE_MODEL, map_location=DEVICE)
whistle_model.load_state_dict(checkpoint["model_state"])

whistle_model.eval()


print("Loading rally CNN")
torch.backends.cudnn.benchmark = True

rally_model = build_model().to(DEVICE)
rally_model.load_state_dict(torch.load(MODEL_DIR / RALLY_MODEL, map_location=DEVICE))
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


cnn_candidates = [] 

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

    if prob > 0.08:
        cnn_candidates.append((center, prob))


cnn_centers = window_nms(cnn_candidates, window=1.5)

print("Detected whistles after NMS:", len(cnn_centers))



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

frame_idx = 0
step_frames = int(fps * TIMELINE_STEP)

BATCH_SIZE = 64
batch_frames = []
batch_times = []
import time

decode_time = 0
preprocess_time = 0
inference_time = 0
loop_count = 0


while True:

    t0 = time.time()
    ret, frame = cap.read()
    decode_time += time.time() - t0

    if not ret:
        break

    if frame_idx % step_frames != 0:
        frame_idx += 1
        continue

    t1 = time.time()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    tensor = transform(img)

    preprocess_time += time.time() - t1

    batch_frames.append(tensor)
    batch_times.append(frame_idx / fps)

    if len(batch_frames) == BATCH_SIZE:

        t2 = time.time()

        x = torch.stack(batch_frames).to(DEVICE)

        with torch.inference_mode():
            logits = rally_model(x)
            probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()

        inference_time += time.time() - t2

        timeline_t.extend(batch_times)
        timeline_p.extend(probs)

        batch_frames.clear()
        batch_times.clear()

        pbar.update(BATCH_SIZE)

        loop_count += BATCH_SIZE

    frame_idx += 1


if batch_frames:

    x = torch.stack(batch_frames).to(DEVICE)

    with torch.inference_mode():
        logits = rally_model(x)
        probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()

    timeline_t.extend(batch_times)
    timeline_p.extend(probs)

pbar.close()

timeline_t = np.array(timeline_t)
timeline_p = np.array(timeline_p)
timeline_p_smooth = smooth_signal(timeline_p, k=9)

print("\n===== PIPELINE METRICS =====")
print("Frames processed:", loop_count)
print("Decode time:", decode_time)
print("Preprocess time:", preprocess_time)
print("Inference time:", inference_time)

total = decode_time + preprocess_time + inference_time
print("\nPercent breakdown:")
print("Decode:", decode_time / total * 100)
print("Preprocess:", preprocess_time / total * 100)
print("Inference:", inference_time / total * 100)
# =============================
# SEGMENT FILTER
# =============================

def in_segments(t):
    for s,e in SEGMENTS:
        if s <= t <= e:
            return True
    return False

def is_accept(label):
        return label != "REJECT"
def merge_consecutive_accepts(rallies, max_gap=2.0):
    """
    Merge consecutive non-reject intervals (CNN / HITL / WEAK / etc.)
    if they are separated by a small gap (likely FP whistle).

    Args:
        rallies: list of dicts with keys:
            - start
            - end
            - label (e.g. 'CNN', 'REJECT', 'HITL', etc.)
        max_gap: max allowed gap (seconds) to merge

    Returns:
        merged list of rallies
    """

    

    merged = []

    for r in rallies:
        if not merged:
            merged.append(r.copy())
            continue

        prev = merged[-1]

        gap = r["start"] - prev["end"]

        if is_accept(prev["label"]) and is_accept(r["label"]) and gap <= max_gap:
            #  merge
            prev["end"] = r["end"]
            prev["duration"] = prev["end"] - prev["start"]

            # optional: keep strongest label
            if prev["label"] != "CNN":
                prev["label"] = r["label"]

            # optional: accumulate metadata
            prev.setdefault("merged_count", 1)
            prev["merged_count"] += 1

        else:
            merged.append(r.copy())

    return merged
def merge_small_gaps(rallies, max_small=5.0):
    if not rallies:
        return []

    merged = []
    i = 0

    while i < len(rallies):
        curr = rallies[i]

        # look ahead for A + B + C pattern
        if i + 2 < len(rallies):
            mid = rallies[i + 1]
            nxt = rallies[i + 2]

            if (
                mid["duration"] <= max_small
                and is_accept(curr["label"])
                and is_accept(nxt["label"])
            ):
                #  merge all three
                merged_rally = {
                    "start": curr["start"],
                    "end": nxt["end"],
                    "label":curr["label"]
                }
                merged_rally["duration"] = merged_rally["end"] - merged_rally["start"]

                merged.append(merged_rally)

                i += 3
                continue

        # normal case
        merged.append(curr)
        i += 1

    return merged

def compute_moving_yellow(cap, fps, start, end):
    FRAME_SKIP = 8
    RESIZE = (320, 180)
    CROP_TOP = 0.5
    MOTION_THRESHOLD = 25

    frame_start = int(start * fps)
    frame_end   = int(end * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    if frame_end - frame_start <5:
        return 0.0
    prev_gray = None
    frame_id = frame_start

    scores = []

    while frame_id <= frame_end:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        # preprocess
        frame = cv2.resize(frame, RESIZE)
        h, w = frame.shape[:2]
        frame = frame[:int(h * CROP_TOP), :]

        # yellow mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([15, 80, 80])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, motion = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)

            moving_yellow = cv2.bitwise_and(motion, mask)

            score = np.sum(moving_yellow > 0)
            scores.append(score)

        prev_gray = gray

        frame_id += 1

 
    if not scores:   
        return 0.0

    scores = np.array(scores)

    # remove NaNs just in case
    scores = scores[~np.isnan(scores)]

    if scores.size == 0:
        return 0.0

    try:
        return float(np.percentile(scores, 90))
    except Exception as e:
        print("[WARN] percentile failed:", e)
        return 0.0
# =============================
# ANALYZE WHISTLE INTERVALS
# =============================

print("Analyzing whistle intervals")

rallies = []
hitl_candidates = []
for i in range(len(cnn_centers) - 1):

    w0 = cnn_centers[i]
    w1 = cnn_centers[i + 1]

    if not in_segments(w0):
        continue

    mask = (timeline_t >= w0) & (timeline_t <= w1)
    probs = timeline_p_smooth[mask]

    if len(probs) == 0:
        continue

    # CNN-based signal
    states = hysteresis(probs)
    states = remove_short_runs(states, min_len=3)
    inplay_ratio = np.mean(states)

    # Yellow motion signal
    yellow_score = compute_moving_yellow(cap, fps, w0, w1)

    
    score = inplay_ratio + 0.02 * yellow_score
    duration = w1 - w0

    if inplay_ratio > MIN_RALLY_RATIO:
        reason = "CNN"
        is_rally = True

    elif yellow_score > 12:
        reason = "YELLOW_STRONG"
        is_rally = True

#  FRAGMENTS
    elif yellow_score > 10 and duration < 6.0:
        reason = "FRAGMENT"
        is_rally = True

#  MEDIUM CONFIDENCE
    elif inplay_ratio > 0.25 and yellow_score > 4:
        reason = "WEAK_RALLY"
        is_rally = True

    elif inplay_ratio > 0.20 and yellow_score > 5:
        reason = "WEAK_RALLY_2"
        is_rally = True

#  LOW CONFIDENCE (your key FN zone)
    elif inplay_ratio > 0.15 and yellow_score > 2.0:
        reason = "WEAK_RALLY_3"
        is_rally = True

#  VERY WEAK 
    elif 0.1 < score < 0.3:
        reason = "HITL"
        is_rally = False
        hitl_candidates.append({
        "start": float(w0),
        "end": float(w1),
        "duration": float(duration),
        "ratio": float(inplay_ratio),
        "yellow": float(yellow_score),
        "score": float(score),
        "index": i
    })

# REJECT
    else:
        reason = "REJECT"
        is_rally = False 

    print(f"{i}: start:{w0}, end:{w1} inplay_ratio={inplay_ratio:.2f}, yellow={yellow_score:.2f}, dur={duration:.2f}, score={score:.2f} → {reason}")

    # =========================
    # APPLY DECISION
    # =========================

    if is_rally:

        rallies.append({
            "start": float(w0),
            "end": float(w1),
            "duration": float(duration),
            "label": reason
        })

        if duration > MAX_RALLY_DURATION:
            print("Possible missed whistle near", w0)

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
# =============================
# POST-PROCESSING FIXES
# =============================

print("\nApplying post-processing fixes...")

#  1. Remove tiny garbage intervals
MIN_DURATION = 2.0


print("After min-duration filter:", len(rallies))



rallies = merge_consecutive_accepts(rallies)
#rallies = merge_small_gaps(rallies)    
print("After Merging small gaps:",len(rallies))# fix fragmentation first



rallies = [
    r for r in rallies
    if (r["end"] - r["start"]) > MIN_DURATION
]

print("After final filter:", len(rallies))

print("HITL Count:",len(hitl_candidates))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # ensure folder exists

output_path = OUTPUT_DIR / f"{MATCH_ID}_rallies_behavior.json"

with open(output_path,"w") as f:
    json.dump(rallies,f,indent=2)

hitl_path = OUTPUT_DIR / f"{MATCH_ID}_hitl_candidates.json"

with open(hitl_path, "w") as f:
    json.dump(hitl_candidates, f, indent=2)

print("Saved HITLs:", hitl_path)


print("Saved:",output_path)
