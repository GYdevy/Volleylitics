import json
import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
WHISTLES_JSON = "whistles_all.json"

VIDEO_AUDIO_PATHS = {
    "match3": r"D:\Volleyballey\videos\match3.mp4",
    "match4": r"D:\Volleyballey\videos\match4.mp4",
    "match11": r"D:\Volleyballey\videos\match11.mp4",
}

OUTPUT_DIR = "../dataset"
WINDOW = 0.5
SR = 22050

SPLIT = {
    "train": ["match4"],
    "val":   ["match11"],
    "test":  ["match3"],
}

# ===============================
# HELPERS
# ===============================

def load_audio(path):
    return librosa.load(path, sr=SR)[0]

def extract_snippet(y, center_time):
    half = WINDOW / 2
    start = int((center_time - half) * SR)
    end   = int((center_time + half) * SR)

    if start < 0 or end > len(y):
        return None

    return y[start:end]

def is_safe_negative(candidate_time, whistle_times, margin=1.0):
    return all(abs(candidate_time - w) > margin for w in whistle_times)

# ===============================
# MAIN
# ===============================

with open(WHISTLES_JSON, "r") as f:
    whistles = json.load(f)

# Group whistles by match
matches = {}
for w in whistles:
    matches.setdefault(w["match_id"], []).append(w)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in tqdm(SPLIT.keys(), desc="Splits"):

    match_ids = SPLIT[split]
    split_dir = os.path.join(OUTPUT_DIR, split)
    pos_dir = os.path.join(split_dir, "positive")
    neg_dir = os.path.join(split_dir, "negative")

    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    for match_id in tqdm(match_ids, desc=f"{split} matches", leave=False):

        print(f"\nLoading audio for {match_id}...")
        y = load_audio(VIDEO_AUDIO_PATHS[match_id])
        whistle_times = sorted([w["time"] for w in matches[match_id]])

        # --------------------
        # POSITIVES
        # --------------------
        for i, t in enumerate(tqdm(whistle_times, desc="Positives", leave=False)):
            snippet = extract_snippet(y, t)
            if snippet is None:
                continue

            sf.write(
                os.path.join(pos_dir, f"{match_id}_{i}.wav"),
                snippet,
                SR
            )

        # --------------------
        # HARD NEGATIVES
        # --------------------
        neg_count = 0

        for t in tqdm(whistle_times, desc="Negatives", leave=False):
            for offset in [-2.0, 2.0]:
                candidate = t + offset

                if not is_safe_negative(candidate, whistle_times):
                    continue

                snippet = extract_snippet(y, candidate)
                if snippet is None:
                    continue

                sf.write(
                    os.path.join(neg_dir, f"{match_id}_{neg_count}.wav"),
                    snippet,
                    SR
                )
                neg_count += 1

print("\nDataset built.")
