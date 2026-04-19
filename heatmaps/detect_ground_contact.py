from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
#from config import MATCH_ID
from draw_court import draw_court

# =========================
# CONFIG
# =========================

MATCH_ID = "match17"
CLIPS_DIR = f"/workspace/rally_segmentator/output/{MATCH_ID}/rally_clips"
RALLIES_JSON = f"/workspace/rally_segmentator/output/{MATCH_ID}/rallies_with_clips_with_sets.json"

OUTPUT_JSON = "/workspace/heatmaps/landings.json"
OUTPUT_IMG = "/workspace/heatmaps/all_landings.png"

# =========================
# LOAD CLIPS
# =========================
clips = sorted([
    f for f in os.listdir(CLIPS_DIR)
    if f.endswith(".mp4")
])

# =========================
# LOAD RALLIES (for sets)
# =========================
with open(RALLIES_JSON, "r") as f:
    rallies = json.load(f)

# map: rally_000.mp4 → set
clip_to_set = {
    r["clip_path"].split("/")[-1]: r.get("set")
    for r in rallies
}

# =========================
# COLORS PER SET
# =========================
SET_COLORS = {
    1: (0, 0, 255),     # red
    2: (0, 255, 0),     # green
    3: (255, 0, 0),     # blue
    None: (120, 120, 120)  # fallback (gray)
}

# =========================
# BALL OBJECT
# =========================
class Ball:
    def __init__(self):
        self.positions = []

    def add(self, frame_idx, x, y):
        self.positions.append((frame_idx, x, y))

    def get_landing_point(self):
        if not self.positions:
            return None
        return max(self.positions, key=lambda p: p[2])  # max y


# =========================
# LOAD MODEL
# =========================
model = YOLO("/workspace/model/best.pt")

# =========================
# HOMOGRAPHY
# =========================
img_pts = np.array([
    [1, 1040],
    [1919, 1034],
    [1426, 779],
    [516, 780]
], dtype=np.float32)

court_pts = np.array([
    [0, 0],
    [9, 0],
    [9, 9],
    [0, 9]
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, court_pts)


# =========================
# PROCESS ONE CLIP
# =========================
def process_clip(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # last few seconds
    start_frame = max(0, total_frames - int(3 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ball = Ball()
    frame_idx = start_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)[0]

        if len(results.boxes) > 0:
            best_box = max(results.boxes, key=lambda b: float(b.conf[0]))

            x1, y1, x2, y2 = map(int, best_box.xyxy[0])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            ball.add(frame_idx, cx, cy)

        frame_idx += 1

    cap.release()

    landing = ball.get_landing_point()
    if not landing:
        return None

    _, x, y = landing

    pt = np.array([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)

    court_x, court_y = mapped[0][0]
    if  court_y > 9 or court_y < 0:
        return None
    # clamp
    EXT = 3  # meters outside court

    court_x = max(-EXT, min(9 + EXT, court_x))
    court_y = max(0, min(9, court_y))
    return float(court_x), float(court_y)


# =========================
# MAIN LOOP
# =========================
results = []

for clip in clips:
    path = os.path.join(CLIPS_DIR, clip)

    print(f"\n▶ Processing {clip}")

    landing = process_clip(path)

    if landing:
        x, y = landing
        set_id = clip_to_set.get(clip)

        print(f"✔ Landing: ({x:.2f}, {y:.2f}) | set={set_id}")

        results.append({
            "clip": clip,
            "x": x,
            "y": y,
            "set": set_id
        })
    else:
        print("No landing")


# =========================
# SAVE JSON
# =========================
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to {OUTPUT_JSON}")


# =========================
# DRAW COURT
# =========================
court_img = draw_court(0, 0)

COURT_H = 18
FREE = 3.5
SCALE = 250
offset = int(FREE * SCALE)

# =========================
# DRAW LANDINGS
# =========================
for r in results:
    x, y = r["x"], r["y"]
    set_id = r.get("set")

    px = offset + int(x * SCALE)
    py = offset + int((COURT_H - y) * SCALE)

    color = SET_COLORS.get(set_id, (120,120,120))

    

    # inner colored dot
    cv2.circle(court_img, (px, py), 9, color, -1, lineType=cv2.LINE_AA)


# =========================
# DRAW LEGEND
# =========================
legend_x = 50
legend_y = 50

for i, (set_id, color) in enumerate(SET_COLORS.items()):
    if set_id is None:
        continue

    y = legend_y + i * 40

    cv2.circle(court_img, (legend_x, y), 10, color, -1, lineType=cv2.LINE_AA)

    cv2.putText(
        court_img,
        f"Set {set_id}",
        (legend_x + 25, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2,
        lineType=cv2.LINE_AA
    )


# =========================
# SAVE IMAGE
# =========================
cv2.imwrite(OUTPUT_IMG, court_img)

print(f"Saved visualization to {OUTPUT_IMG}")
