import json
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

import rse
from config import SPLIT, FHD_MATCHES

# =========================
# CONFIG
# =========================

WHISTLE_JSON = "anchored_matches/whistles_deduped.json"

FRAME_SKIP = 6
MIN_RALLY = 2.0
MAX_RALLY = 50.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_LEN = 100

# =========================
# VIDEO PATH
# =========================

import os

def get_video_path(match_id):
    path = f"/videos/{match_id}.mp4"

    if not os.path.exists(path):
        print(f"❌ Missing file: {path}")
        return None

    return path
   

# =========================
# LOAD WHISTLES
# =========================

with open(WHISTLE_JSON, "r") as f:
    all_data = json.load(f)

# =========================
# LOAD MODEL
# =========================

model = rse.build_model()
model.load_state_dict(torch.load("rally_model_best.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# CLEANING LOGIC (🔥 CORE)
# =========================

def clean_seq(probs, label):
    probs = np.array(probs)

    if len(probs) < 20:
        return None

    # 🔥 1. HARD TRIM (remove edges)
    trim = int(0.2 * len(probs))
    core = probs[trim:-trim]

    if len(core) < 10:
        return None

    mean = np.mean(core)
    std  = np.std(core)

    # 🔥 2. REMOVE AMBIGUOUS
    if 0.4 < mean < 0.6:
        return None

    # 🔥 3. REMOVE DEAD SIGNAL
    if std < 0.05:
        return None

    # 🔥 4. CLASS-SPECIFIC FILTER
    if label == 1 and mean < 0.6:
        return None

    if label == 0 and mean > 0.4:
        return None

    # 🔥 5. RESAMPLE (ONLY ONCE)
    core = np.interp(
        np.linspace(0, len(core)-1, TARGET_LEN),
        np.arange(len(core)),
        core
    )

    return core.tolist()

# =========================
# FEATURE EXTRACTION (probs)
# =========================

def extract_probs(cap, fps, start, end):

    frame_start = int(start * fps)
    frame_end   = int(end * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    probs = []
    frame_idx = frame_start
    batch = []

    while frame_idx <= frame_end:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:

            h, w = frame.shape[:2]
            crop = frame[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
            crop = cv2.resize(crop, (224, 224))

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0

            batch.append(x)

            if len(batch) == 64:
                x_batch = torch.stack(batch).to(DEVICE)

                with torch.no_grad():
                    logits = model(x_batch)
                    p = F.softmax(logits, dim=1)[:,1].cpu().numpy()

                probs.extend(p)
                batch.clear()

        frame_idx += 1

    if batch:
        x_batch = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            logits = model(x_batch)
            p = F.softmax(logits, dim=1)[:,1].cpu().numpy()
        probs.extend(p)

    return probs

# =========================
# BUILD DATASET
# =========================

for split_name, match_list in SPLIT.items():

    print(f"\n===== {split_name.upper()} =====")

    dataset = []

    for MATCH_ID in match_list:

        print(f"\n--- {MATCH_ID} ---")

        VIDEO_PATH = get_video_path(MATCH_ID)
        cap = cv2.VideoCapture(VIDEO_PATH)

        if not cap.isOpened():
            print("Failed:", VIDEO_PATH)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = total_frames / fps

        data = [d for d in all_data if d["match_id"] == MATCH_ID]
        data = sorted(data, key=lambda x: x["t_anchor"])

        # =========================
        # BUILD RALLIES
        # =========================

        rallies = []
        for i in range(len(data)):
            if data[i]["type"] != "serve":
                continue

            start = data[i]["t_anchor"]

            for j in range(i+1, len(data)):
                if data[j]["type"] == "rally_end":
                    end = data[j]["t_anchor"]
                    duration = end - start

                    if MIN_RALLY < duration < MAX_RALLY:
                        rallies.append((start, end))
                    break

        # =========================
        # BUILD BREAKS
        # =========================

        breaks = []

        for i in range(len(rallies)-1):
            prev_end = rallies[i][1]
            next_start = rallies[i+1][0]

            if next_start > prev_end:
                breaks.append((prev_end, next_start))

        if rallies:
            last_end = rallies[-1][1]
            if last_end < video_duration - 1.0:
                breaks.append((last_end, video_duration))

        # =========================
        # EXTRACT + CLEAN
        # =========================

        print("Rallies...")
        for start, end in tqdm(rallies):
            probs = extract_probs(cap, fps, start, end)
            seq = clean_seq(probs, label=1)

            if seq:
                dataset.append({"seq": seq, "label": 1})

        print("Breaks...")
        for start, end in tqdm(breaks):
            probs = extract_probs(cap, fps, start, end)
            seq = clean_seq(probs, label=0)

            if seq:
                dataset.append({"seq": seq, "label": 0})

        cap.release()

    # =========================
    # SAVE
    # =========================

    out_path = f"dataset_{split_name}.json"

    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved {out_path} with {len(dataset)} samples")
