# WhistleDetector/hitl/labeling.py

import csv
import pandas as pd
from pathlib import Path
import joblib

from WhistleDetector.config import BASE_OUTPUT_DIR


def append_training_row(uid, X, label, csv_path):
    header = ["uid","rms","flatness","centroid","band_energy","peak_freq","ridge_length","grad_t","grad_f","rolloff","bandwidth","zcr","contrast","tonnetz","mfcc1","mfcc2","mfcc3","mfcc4","mfcc5","label"]


    if csv_path.exists():
        existing = set(pd.read_csv(csv_path)["uid"])
        if uid in existing:
            return

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(header)
        writer.writerow([uid, *X.tolist(), label])


def load_ambiguous(match_num: int):
    ambig_dir = Path(BASE_OUTPUT_DIR) / "ambiguous" / f"match{match_num}"
    meta_path = ambig_dir / "ambiguous_meta.pkl"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"No ambiguous metadata found for match {match_num}\n"
            f"Expected: {meta_path}"
        )

    detections = joblib.load(meta_path)

    items = []
    for d in detections:
        clip = ambig_dir / d["clip_name"]
        if clip.exists():
            items.append((clip, d))

    print(f"[HITL] Loaded {len(items)} ambiguous items")
    return items
