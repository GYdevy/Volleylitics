import cv2
import numpy as np
from ultralytics import YOLO
from draw_court import draw_court
VIDEO_PATH = "/workspace/rally_segmentator/output/match17/rally_clips/rally_017.mp4"
MODEL_PATH = "/workspace/model/best.pt"

NET_Y_IMG = 530
NET_LEFT_X = 471
NET_RIGHT_X = 1441
# =========================
# HOMOGRAPHY (image → court)
# =========================
img_pts = np.array([
    [1, 1040],     # bottom-left corner (image)
    [1919, 1034],  # bottom-right
    [1426, 779],   # top-right (near net)
    [516, 780]     # top-left
], dtype=np.float32)

court_pts = np.array([
    [0, 0],   # bottom-left (court)
    [9, 0],   # bottom-right
    [9, 9],   # top-right (net)
    [0, 9]    # top-left
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, court_pts)


model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 🔥 only last 2 seconds
start_frame = max(0, total_frames - int(2 * fps))
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

positions = []
frames = []

frame_idx = start_frame





def img_to_court(pt, H):
    pt = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)
    return mapped[0][0]
# =========================
# TRACK BALL
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(frame.copy())

    results = model(frame, conf=0.3)[0]

    if len(results.boxes) > 0:
        best = max(results.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best.xyxy[0])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        positions.append((frame_idx, cx, cy))

    frame_idx += 1

cap.release()

print("Total points:", len(positions))

# =========================
# FIND CROSSING (robust)
# =========================
cross = None

for i in range(1, len(positions)):
    _, x1, y1 = positions[i - 1]
    _, x2, y2 = positions[i]

    if (y1 > NET_Y_IMG and y2 <= NET_Y_IMG):
        cross = (x1, y1)
        print("CROSSING detected")
        break

# fallback: closest point to net
if cross is None:
    best = None
    best_dist = float("inf")

    for _, x, y in positions:
        dist = abs(y - NET_Y_IMG)
        if dist < best_dist:
            best_dist = dist
            best = (x, y)

    cross = best
    print("Using closest-to-net fallback")

hx, hy = cross

# =========================
# LANDING (max Y)
# =========================
landing = max(positions, key=lambda p: p[2])
_, lx_img, ly_img = landing

# =========================
# 🔥 YOUR ASSUMPTION MAPPING
# =========================
t = (hx - NET_LEFT_X) / (NET_RIGHT_X - NET_LEFT_X)
t = max(0, min(1, t))  # clamp

ax = t * 9
ay = 9

# =========================
# DEBUG VIDEO OUTPUT
# =========================
h, w = frames[0].shape[:2]
out = cv2.VideoWriter(
    "debug_rally_017.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (w, h)
)
landing = max(positions, key=lambda p: p[2])
landing_idx = positions.index(landing)

_, lx_img, ly_img = landing

for i, frame in enumerate(frames):

    # net line
    cv2.line(frame, (0, NET_Y_IMG), (w, NET_Y_IMG), (255,255,255), 2)

    # trajectory (STOP at landing, no frame dependency)
    for j in range(1, landing_idx + 1):
        _, x1, y1 = positions[j - 1]
        _, x2, y2 = positions[j]

        # optional: skip big jumps (recommended)
        if positions[j][0] - positions[j-1][0] > 3:
            continue

        cv2.line(frame, (x1, y1), (x2, y2), (0,255,255), 2)

    # crossing point
    cv2.circle(frame, (int(hx), int(hy)), 10, (255,0,0), -1)
    cv2.putText(frame, "NET", (int(hx)+10, int(hy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    out.write(frame)

out.release()
print("Attack X (court):", ax)
# =========================
# COURT MAP DRAW
# =========================
COURT_W = 9
COURT_H = 18
FREE = 3.5
SCALE = 250

offset = int(FREE * SCALE)

img_w = int((COURT_W + 2 * FREE) * SCALE)
img_h = int((COURT_H + 2 * FREE) * SCALE)

court_img = draw_court(0,0)

# =========================
# MAP FUNCTIONS
# =========================
def img_to_court_x(x):
    t = (x - NET_LEFT_X) / (NET_RIGHT_X - NET_LEFT_X)
    t = max(0, min(1, t))
    return t * 9

# =========================
# COMPUTE LANDING + ATTACK
# =========================
landing = max(positions, key=lambda p: p[2])
landing_idx = positions.index(landing)

_, lx_img, ly_img = landing

lx, ly = img_to_court((lx_img, ly_img), H)

# attack (net point)
ax = img_to_court_x(hx)
ay = 9

# =========================
# DRAW TRAJECTORY (NET PROJECTION)
# =========================
for j in range(1, landing_idx + 1):
    x1_img = positions[j-1][1]
    x2_img = positions[j][1]

    # skip jumps
    if positions[j][0] - positions[j-1][0] > 3:
        continue

    cx1 = img_to_court_x(x1_img)
    cx2 = img_to_court_x(x2_img)

    cy = 9  # projected to net

    px1 = offset + int(cx1 * SCALE)
    py1 = offset + int((COURT_H - cy) * SCALE)

    px2 = offset + int(cx2 * SCALE)
    py2 = offset + int((COURT_H - cy) * SCALE)

    cv2.line(court_img, (px1, py1), (px2, py2), (0,255,255), 2)

# =========================
# DRAW ATTACK → LANDING LINE
# =========================
px1 = offset + int(ax * SCALE)
py1 = offset + int((COURT_H - ay) * SCALE)

px2 = offset + int(lx * SCALE)
py2 = offset + int((COURT_H - ly) * SCALE)

cv2.line(court_img, (px1, py1), (px2, py2), (0,0,255), 4)

# draw points
cv2.circle(court_img, (px1, py1), 8, (255,0,0), -1)  # attack
cv2.circle(court_img, (px2, py2), 8, (0,255,0), -1)  # landing

# =========================
# SAVE
# =========================
cv2.imwrite("court_map_017.png", court_img)

print("Saved court_map_017.png")
