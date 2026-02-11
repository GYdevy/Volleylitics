import joblib
import warnings

import librosa
import numpy as np
from WhistleDetector.config import *
from WhistleDetector.audio.utils import save_clip
from WhistleDetector.detection.energy import detect_active_frames
from WhistleDetector.detection.grouping import group_frames
from WhistleDetector.detection.candidates import extract_candidates
from WhistleDetector.detection.routing import route_detection

# ===============================
# SILENCE KNOWN HARMLESS WARNINGS
# ===============================
warnings.filterwarnings(
    "ignore",
    message="n_fft=.* is too large for input signal"
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)
def main():
    clf = joblib.load(MODEL_PATH)
    ambig_clf = joblib.load(AMBIG_MODEL_PATH)

    # 1️⃣ Load audio ONCE
    y, sr = librosa.load(VIDEO_PATH, sr=SR)

    # 2️⃣ Energy detection
    active_frames, S_w, freqs_w = detect_active_frames(y)

    # 3️⃣ Group frames
    groups = group_frames(active_frames)

    # 4️⃣ Extract candidates
    detections = extract_candidates(
        groups,
        y,
        S_w,
        freqs_w
    )

    # 5️⃣ Routing
    accepted, ambiguous = route_detection(
        detections,
        clf,
        ambig_clf
    )
    # --------------------------------------------------
    # SAVE ACCEPTED CLIPS
    # --------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in accepted:
        save_clip(d["start"], d["end"], OUTPUT_DIR)

    # --------------------------------------------------
    # SAVE AMBIGUOUS CLIPS + METADATA
    # --------------------------------------------------
    AMBIG_DIR.mkdir(parents=True, exist_ok=True)

    ambig_meta = []

    for d in ambiguous:
        clip_path = save_clip(d["start"], d["end"], AMBIG_DIR)

        ambig_meta.append({
            "start": d["start"],
            "end": d["end"],
            "core_score": d.get("core_score"),
            "grad_f": d.get("grad_f"),
            "ridge": d.get("ridge"),
            "peak_std": d.get("peak_std"),
            "bandwidth_hz": d.get("bandwidth_hz"),
            "clip_name": clip_path.name,
        })

    joblib.dump(
        ambig_meta,
        AMBIG_DIR / "ambiguous_meta.pkl"
    )

    print(f"[DETECTOR] Saved {len(ambig_meta)} ambiguous items")







if __name__ == "__main__":
    main()
