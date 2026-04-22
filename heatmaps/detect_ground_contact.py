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

OUTPUT_JSON = f"/workspace/heatmaps/{MATCH_ID}/landings.json"
OUTPUT_IMG = f"/workspace/heatmaps/{MATCH_ID}/all_landings.png"


output_dir = os.path.dirname(OUTPUT_JSON)
os.makedirs(output_dir, exist_ok=True)
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


def get_net_point(ball_positions):
    if len(ball_positions) < 5:
        return None

    net_y = 9

    # take last N points
    N = 15
    pts = ball_positions[-N:]

    # convert to court space
    court_pts = []
    for (_, x, y) in pts:
        pt = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)
        cx, cy = mapped[0][0]
        court_pts.append((cx, cy))

    # compute average direction
    dx = 0
    dy = 0

    for i in range(1, len(court_pts)):
        dx += court_pts[i][0] - court_pts[i-1][0]
        dy += court_pts[i][1] - court_pts[i-1][1]

    dx /= (len(court_pts) - 1)
    dy /= (len(court_pts) - 1)

    # landing point
    lx, ly = court_pts[-1]

    if abs(dy) < 1e-6:
        nx = lx
    else:
        t = (net_y - ly) / dy
        nx = lx + t * dx

    # 🔥 clamp to extended court bounds
    nx = max(-1, min(10, nx))
    ny = net_y

    return float(nx), float(ny)

def get_attack_point(ball_positions):
    if len(ball_positions) < 5:
        return None

    NET_Y_IMG = 530

    # go forward to detect crossing
    for i in range(1, len(ball_positions)):
        _, x1, y1 = ball_positions[i - 1]
        _, x2, y2 = ball_positions[i]

        # detect crossing (one side → other)
        if (y1 > NET_Y_IMG and y2 <= NET_Y_IMG):
            # 🔥 TAKE THE POINT BEFORE CROSSING
            hx, hy = x1, y1

            pt = np.array([[[hx, hy]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(pt, H)

            ax, ay = mapped[0][0]

            ay = 9
            ax = max(-1, min(10, ax))

            return float(ax), float(ay)

    return None
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


net_line = np.array([
    [471, 530],
    [1441, 530]
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
    start_frame = max(0, total_frames - int(2 * fps))
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
    return float(court_x), float(court_y),ball.positions



# =========================
# MAIN LOOP
# =========================
results = []

for r in rallies:
    clip = r["clip_path"].split("/")[-1]          
    rid = r["id"]             

    path = os.path.join(CLIPS_DIR, clip)

    print(f"\n▶ Processing {clip}")

    data = process_clip(path)

    if data:
        x, y, positions = data
        #net_pt = get_net_point(positions)
        attack_pt = get_attack_point(positions)
        set_id = clip_to_set.get(clip)

        print(f"✔ Landing: ({x:.2f}, {y:.2f}) | set={set_id}")

        results.append({
        "id": rid,
        "clip": clip,
        "x": x,
        "y": y,
        "set": set_id,
        "attack_point": attack_pt

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
    rid = r["id"]
    px = offset + int(x * SCALE)
    py = offset + int((COURT_H - y) * SCALE)

    color = SET_COLORS.get(set_id, (120,120,120))

    

    # inner colored dot
    cv2.circle(court_img, (px, py), 9, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(court_img, str(rid), (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255,255,255), 3, cv2.LINE_AA)

    cv2.putText(court_img, str(rid), (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,0), 1, cv2.LINE_AA)


    attack_pt = r.get("attack_point")

    if attack_pt:
        ax, ay = attack_pt
        lx, ly = r["x"], r["y"]

        px1 = offset + int(ax * SCALE)
        py1 = offset + int((COURT_H - ay) * SCALE)

        px2 = offset + int(lx * SCALE)
        py2 = offset + int((COURT_H - ly) * SCALE)

        cv2.line(
            court_img,
            (px1, py1),
            (px2, py2),
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
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
