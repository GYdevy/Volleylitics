import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
import os
from config import (
    WHISTLE_MODEL,
    RALLY_MODEL,
    VIDEO_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    MATCH_ID,
    RALLY_BASE,
    SEGMENTS,
)

from rally_segmentator.dsp_detector import (
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

from rally_segmentator.models_dir.resnet_whistle import ResNetWhistle, wav_to_logmel_from_audio
from rally_segmentator.models_dir.rse import build_model


# =============================
# SETTINGS
# =============================


VIDEO_PATH = f"/videos/{MATCH_ID}.mp4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SNIPPET_SEC = 1.0
HALF_SNIPPET = SNIPPET_SEC / 2

ENTER = 0.65
EXIT = 0.4
MIN_RALLY_RATIO = 0.35
FP_RATIO = 0.1

TIMELINE_STEP = 0.2


MAX_RALLY_DURATION = 35

# =============================
# LOAD MODELS
# =============================

def load_models():
    print("Loading models...")

    whistle_model = ResNetWhistle().to(DEVICE)
    checkpoint = torch.load(MODEL_DIR / WHISTLE_MODEL, map_location=DEVICE)
    whistle_model.load_state_dict(checkpoint["model_state"])
    whistle_model.eval()

    rally_model = build_model().to(DEVICE)
    rally_model.load_state_dict(torch.load(MODEL_DIR / RALLY_MODEL, map_location=DEVICE))
    rally_model.eval()

    return whistle_model, rally_model




# =============================
# HELPERS
# =============================

def compute_moving_yellow(video_path, start, end):
    FRAME_SKIP = 8
    RESIZE = (320, 180)
    CROP_TOP = 0.5
    MOTION_THRESHOLD = 25
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

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

    cap.release() 
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

def in_segments(t):
    for s,e in SEGMENTS:
        if s <= t <= e:
            return True
    return False

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

# =============================
# DSP WHISTLES
# =============================

def detect_whistles_dsp(y):
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

    return rule_based_sifter(refined, y, stats)


# =============================
# CNN FILTER
# =============================

def filter_whistles_cnn(accepted, y, model):
    cnn_candidates = []

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

        if prob > 0.08:
            cnn_candidates.append((center, prob))

    return cnn_candidates


# =============================
# TIMELINE
# =============================

def build_timeline(model):
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = int(fps * TIMELINE_STEP)

    total_steps = frame_count // step_frames
    pbar = tqdm(total=total_steps, desc="Timeline")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    timeline_t = []
    timeline_p = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step_frames != 0:
            frame_idx += 1
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = F.softmax(model(x), dim=1)[0, 1].item()

        timeline_t.append(frame_idx / fps)
        timeline_p.append(prob)

        pbar.update(1)

        frame_idx += 1

    pbar.close()

    return np.array(timeline_t), np.array(timeline_p), cap, fps

# =============================
# ANALYSIS
# =============================

def analyze_intervals(cnn_centers, timeline_t, timeline_p,cap,fps):
    rallies = []
    hitl_candidates = []
    timeline_p_smooth = smooth_signal(timeline_p, k=9)
    for i in tqdm(range(len(cnn_centers) - 1), desc="Analyzing intervals"):

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
        yellow_score = compute_moving_yellow(VIDEO_PATH, w0, w1)

        
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
    return rallies, hitl_candidates


    
# =============================
# SAVE
# =============================
def save_results(rallies, hitl):
    match_output_dir = RALLY_BASE / "output" / MATCH_ID
    match_output_dir.mkdir(parents=True, exist_ok=True)

    rallies_path = match_output_dir / "rallies.json"
    hitl_path = match_output_dir / "hitl.json"

    with open(rallies_path, "w") as f:
        json.dump(rallies, f, indent=2)

    with open(hitl_path, "w") as f:
        json.dump(hitl, f, indent=2)

    print(f"Rallies saved to: {rallies_path}")
    print(f"HITL saved to: {hitl_path}")

# =============================
# HITL DIALOG
# =============================
def maybe_run_hitl():
    ans = input("\nReview HITLs now? [y/n]: ").strip().lower()

    if ans == "y":
       # from hitl_reviewer import review
        #decisions = review()
        print()
       # print(f"Reviewed {len(decisions)} HITLs")

# =============================
# MAIN
# =============================

def main():
    print("Running in:", os.getcwd())
    print(f"Analyzing {MATCH_ID}")
    whistle_model, rally_model = load_models()

    print("Loading audio...")
    y = load_audio_from_video(VIDEO_PATH, cfg.sr)

    print("DSP detection...")
    accepted = detect_whistles_dsp(y)
    print("Accepted candidates:",len(accepted))

    print("CNN filtering...")
    cnn_centers = filter_whistles_cnn(accepted, y, whistle_model)
    print("Total whistles: ",len(cnn_centers))
    print("Applying NMS for whistles...")
    cnn_centers = window_nms(cnn_centers,window=1.5)
    print("Total whistles after NMS: ",len(cnn_centers))
    print("Building timeline...")
    timeline_t, timeline_p,cap,fps = build_timeline(rally_model)

    print("Analyzing...")
    rallies, hitl = analyze_intervals(cnn_centers, timeline_t, timeline_p,cap,fps)

    save_results(rallies, hitl)
    print("Rallies Count: ", len(rallies))
    print("HITL Count:",len(hitl))
    maybe_run_hitl()
    

if __name__ == "__main__":
    main()
