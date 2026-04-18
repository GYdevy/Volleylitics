import json
import numpy as np
from config import MATCH_ID
GT_FILE = f"rally_segmentator/output/{MATCH_ID}/{MATCH_ID}_rallies_gt.json"

DET_FILES = {
    
    "RAW": f"rally_segmentator/output/{MATCH_ID}/rallies.json",
    "WITH_HITL": f"rally_segmentator/output/{MATCH_ID}/rallies_with_hitl.json",
}

TOLERANCE = 1.0  # seconds


# -----------------------------
# UTILS
# -----------------------------

def sec_to_hms(sec):
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load(path):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# EVALUATION FUNCTION
# -----------------------------

def evaluate(gt, det, name):
    matched_gt = set()
    matched_det = set()

    start_errors = []
    end_errors = []

    for i, d in enumerate(det):
        for j, g in enumerate(gt):

            start_diff = abs(d["start"] - g["start"])
            end_diff = abs(d["end"] - g["end"])

            if start_diff < TOLERANCE and end_diff < TOLERANCE:
                matched_det.add(i)
                matched_gt.add(j)

                start_errors.append(start_diff)
                end_errors.append(end_diff)
                break

    TP = len(matched_det)
    FP = len(det) - TP
    FN = len(gt) - TP

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    print("\n===================================")
    print(f"MODEL: {name}")
    print("===================================")

    print("Detected rallies:", len(det))
    print("TP:", TP, "| FP:", FP, "| FN:", FN)
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))

    if start_errors:
        print("Mean start error:", round(np.mean(start_errors), 3))
        print("Mean end error:", round(np.mean(end_errors), 3))

    # -----------------------------
    # FALSE POSITIVES
    # -----------------------------
    print("\n---- FALSE POSITIVES ----")

    for i, d in enumerate(det):
        if i not in matched_det:
            start = d["start"]
            end = d["end"]

            print(
                f"FP: {sec_to_hms(start)} → {sec_to_hms(end)} "
                f"(dur {round(end - start, 2)}s)"
            )

    # -----------------------------
    # FALSE NEGATIVES
    # -----------------------------
    print("\n---- FALSE NEGATIVES ----")

    for j, g in enumerate(gt):
        if j not in matched_gt:
            start = g["start"]
            end = g["end"]

            print(
                f"FN: {sec_to_hms(start)} → {sec_to_hms(end)} "
                f"(dur {round(end - start, 2)}s)"
            )


# -----------------------------
# MAIN
# -----------------------------

gt = load(GT_FILE)

print("======== Rally Detection Evaluation ========")
print("GT rallies:", len(gt))

for name, path in DET_FILES.items():
    det = load(path)
    evaluate(gt, det, name)
