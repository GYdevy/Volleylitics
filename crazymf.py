import cv2
import joblib
import warnings
import librosa
import numpy as np

# ===============================
# WHISTLE DETECTOR IMPORTS
# ===============================
from WhistleDetector.config import *
from WhistleDetector.detection.energy import detect_active_frames
from WhistleDetector.detection.grouping import group_frames
from WhistleDetector.detection.candidates import extract_candidates
from WhistleDetector.detection.routing import route_detection

# ===============================
# CONFIG
# ===============================
VIDEO_PATH = r"D:\Volleyballey\videos\match4.mp4"

TARGET_FPS = 5
RESIZE_W, RESIZE_H = 160, 90

ROI_TIME_SEC = 7 * 60
QUIET_THRESH = 1.2
QUIET_MIN_SEC = 1.5

PRE_DELTA  = 3.0
POST_DELTA = 1.3

MOTION_RATIO = 1.25   # motion drop ratio (pre > post)

OUT_PKL = "serve_rally_structure.pkl"

warnings.filterwarnings("ignore")

# ============================================================
# ROI SELECTION
# ============================================================
def select_polygon_roi(frame, title, n_points=4):
    pts = []

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n_points:
            pts.append((x, y))

    clone = frame.copy()
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, cb)

    while True:
        img = clone.copy()
        for p in pts:
            cv2.circle(img, p, 5, (0, 255, 0), -1)
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (0, 255, 0), 2)
        if len(pts) == n_points:
            cv2.line(img, pts[-1], pts[0], (0, 255, 0), 2)

        cv2.imshow(title, img)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(pts) == n_points:
            break
        if key == ord('r'):
            pts.clear()

    cv2.destroyAllWindows()
    return np.array(pts, dtype=np.int32)

# ============================================================
# ROI FRAME
# ============================================================
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, ROI_TIME_SEC * 1000)

frame = None
for _ in range(10):
    ret, frame = cap.read()
    if ret:
        break
cap.release()

if frame is None:
    raise RuntimeError("Failed to grab frame")

court_poly = select_polygon_roi(frame, "COURT ROI (4 clicks)", 4)

court_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
cv2.fillPoly(court_mask, [court_poly], 255)

# ============================================================
# 1️⃣ WHISTLE DETECTION
# ============================================================
print("[1] Detecting whistles...")

clf = joblib.load(MODEL_PATH)
ambig_clf = joblib.load(AMBIG_MODEL_PATH)

y, sr = librosa.load(VIDEO_PATH, sr=SR)

active_frames, S_w, freqs_w = detect_active_frames(y)
groups = group_frames(active_frames)
detections = extract_candidates(groups, y, S_w, freqs_w)
accepted, ambiguous = route_detection(detections, clf, ambig_clf)

whistles = accepted + ambiguous
print(f"    Found {len(whistles)} whistles")

# ============================================================
# 2️⃣ COURT MOTION
# ============================================================
print("[2] Computing court motion...")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_step = max(1, int(fps / TARGET_FPS))

prev = None
times, motion = [], []
idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if idx % frame_step != 0:
        idx += 1
        continue

    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=court_mask)
    gray = cv2.resize(gray, (RESIZE_W, RESIZE_H))

    if prev is not None:
        diff = np.abs(gray.astype(np.float32) - prev.astype(np.float32))
        motion.append(diff.mean())
        times.append(t)

    prev = gray
    idx += 1

cap.release()
motion = np.convolve(motion, np.ones(5) / 5, mode="same")

# ============================================================
# 3️⃣ QUIET WINDOWS
# ============================================================
quiet = []
in_q = False

for i, m in enumerate(motion):
    if m < QUIET_THRESH and not in_q:
        start = times[i]
        in_q = True
    elif m >= QUIET_THRESH and in_q:
        end = times[i]
        if end - start >= QUIET_MIN_SEC:
            quiet.append((start, end))
        in_q = False

print(f"[3] Found {len(quiet)} quiet windows")

# ============================================================
# 4️⃣ TIER-1 SERVES (QUIET + WHISTLE)
# ============================================================
tier1_anchors = []

for qs, qe in quiet:
    aligned = [
        w for w in whistles
        if qs - PRE_DELTA <= w["start"] <= qs + POST_DELTA
    ]
    if not aligned:
        continue

    w = min(aligned, key=lambda x: abs(x["start"] - qs))
    tier1_anchors.append(w["start"])

tier1_anchors.sort()
print(f"[4] Tier-1 anchors: {len(tier1_anchors)}")

# ============================================================
# 5️⃣ TIER-2 SERVES (MOTION DROP)
# ============================================================
tier2_anchors = []

for w in whistles:
    t = w["start"]

    if any(abs(t - s) < 2.0 for s in tier1_anchors):
        continue

    idxs = [i for i, tt in enumerate(times) if t - 2.0 < tt < t + 2.0]
    if len(idxs) < 4:
        continue

    mid = idxs[len(idxs) // 2]

    pre  = motion[max(0, mid - 4):mid]
    post = motion[mid:mid + 4]

    if len(pre) < 2 or len(post) < 2:
        continue

    if np.mean(pre) > MOTION_RATIO * np.mean(post):
        tier2_anchors.append(t)

print(f"[5] Tier-2 anchors: {len(tier2_anchors)}")

# ============================================================
# MERGE + DEDUP
# ============================================================
serve_anchors = sorted(set(tier1_anchors + tier2_anchors))
print(f"[SERVES] Total anchors: {len(serve_anchors)}")

# ============================================================
# 6️⃣ RALLY SEGMENTS (MANUAL REVIEW)
# ============================================================
rally_segments = []

for i in range(len(serve_anchors) - 1):
    s0 = serve_anchors[i]
    s1 = serve_anchors[i + 1]

    candidates = [
        {
            "time": w["start"],
            "core_score": w.get("core_score"),
            "grad_f": w.get("grad_f"),
            "noisy": bool(w.get("noisy", False))
        }
        for w in whistles
        if s0 + 0.5 < w["start"] < s1 - 0.5
    ]

    rally_segments.append({
        "serve_start": s0,
        "serve_next": s1,
        "duration": s1 - s0,
        "candidate_whistles": candidates
    })

# ============================================================
# SAVE
# ============================================================
joblib.dump({
    "serve_anchors": serve_anchors,
    "tier1_anchors": tier1_anchors,
    "tier2_anchors": tier2_anchors,
    "quiet_windows": quiet,
    "rally_segments": rally_segments,
    "court_roi": court_poly.tolist()
}, OUT_PKL)

print(f"\nSaved → {OUT_PKL}")
print("Ready for manual rally-end annotation.")