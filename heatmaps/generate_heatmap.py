import os
import cv2
import numpy as np
from ultralytics import YOLO
from heatmaps.draw_court import draw_court
import json
import argparse

from config import BALL_MODEL










COURT_W = 9
COURT_H = 18
FREE = 3.5
SCALE = 250

SET_COLORS = {
    1: (0, 0, 255),      # red
    2: (0, 255, 0),      # green
    3: (255, 0, 0),      # blue
    4: (0, 255, 255),    # yellow
    5: (255, 0, 255),    # magenta
    None: (180, 180, 180)
}

def load_calibration(path):
    with open(path, "r") as f:
        data = json.load(f)

    img_pts = np.array(data["img_pts"], dtype=np.float32)
    net_left_x = int(data["net_left_x"])
    net_right_x = int(data["net_right_x"])
    net_y_img = int(data["net_y_img"])

    return img_pts, net_left_x, net_right_x, net_y_img


court_pts = np.array([
    [0, 0],   # bottom-left (court)
    [9, 0],   # bottom-right
    [9, 9],   # top-right (net)
    [0, 9]    # top-left
], dtype=np.float32)

def load_rally_metadata(rallies_json_path):
    with open(rallies_json_path, "r") as f:
        rallies = json.load(f)

    clip_to_meta = {}

    for r in rallies:
        clip_name = os.path.basename(r["clip_path"])
        clip_to_meta[clip_name] = {
            "id": r.get("id"),
            "start":r.get("start"),
            "end":r.get("end"),
            "set": r.get("set")
        }

    return clip_to_meta

def ensure_dirs(output_dir,debug_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)


def img_to_court(pt, H):
    pt = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)
    return mapped[0][0]


def img_to_court_x(x):
    t = (x - NET_LEFT_X) / (NET_RIGHT_X - NET_LEFT_X)
    t = max(0, min(1, t))
    return t * 9


def get_clip_paths(clips_dir):
    clips = []
    for name in sorted(os.listdir(clips_dir)):
        if name.lower().endswith(".mp4"):
            clips.append(os.path.join(clips_dir, name))
    return clips


def open_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, total_frames - int(2 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    return cap, fps, total_frames, start_frame


def track_ball(cap, model, start_frame):
    positions = []
    frames = []
    frame_idx = start_frame

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

    return positions, frames


def find_crossing(positions):
    cross = None

    MAX_DX = 120
    MAX_DY = 120

    for i in range(1, len(positions)):
        _, x1, y1 = positions[i - 1]
        _, x2, y2 = positions[i]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # reject suspicious jumps
        if dx > MAX_DX or dy > MAX_DY:
            continue

        if y1 > NET_Y_IMG and y2 <= NET_Y_IMG:
            if y2 != y1:
                t = (NET_Y_IMG - y1) / (y2 - y1)
                cx = x1 + t * (x2 - x1)
            else:
                cx = x1

            cross = (int(cx), NET_Y_IMG)
            print("CROSSING detected")
            break

    if cross is None:
        best = None
        best_dist = float("inf")

        for _, x, y in positions:
            dist = abs(y - NET_Y_IMG)

            
            if x < 300 or x > 1650:
                continue

            if dist < best_dist:
                best_dist = dist
                best = (x, y)

        if best is not None:
            cross = (best[0], NET_Y_IMG)
            print("Using closest-to-net fallback")
        else:
            return None

    return cross


def find_landing(positions):
    return max(positions, key=lambda p: p[2])


def compute_attack_point(hx):
    t = (hx - NET_LEFT_X) / (NET_RIGHT_X - NET_LEFT_X)
    t = max(0, min(1, t))
    ax = t * 9
    ay = 9
    return ax, ay


def write_debug_video(frames, positions, hx, hy, output_path):
    if not frames:
        return

    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (w, h)
    )

    landing = max(positions, key=lambda p: p[2])
    landing_idx = positions.index(landing)

    for frame in frames:
        cv2.line(frame, (0, NET_Y_IMG), (w, NET_Y_IMG), (255, 255, 255), 2)

        for j in range(1, landing_idx + 1):
            _, x1, y1 = positions[j - 1]
            _, x2, y2 = positions[j]

            if positions[j][0] - positions[j - 1][0] > 3:
                continue

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.circle(frame, (int(hx), int(hy)), 10, (255, 0, 0), -1)
        cv2.putText(
            frame,
            "NET",
            (int(hx) + 10, int(hy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        out.write(frame)

    out.release()


def create_court_image():
    offset = int(FREE * SCALE)
    img_w = int((COURT_W + 2 * FREE) * SCALE)
    img_h = int((COURT_H + 2 * FREE) * SCALE)

    court_img = draw_court(0, 0)
    return court_img, offset, img_w, img_h


def draw_projected_trajectory(court_img, positions, offset,color):
    landing = max(positions, key=lambda p: p[2])
    landing_idx = positions.index(landing)

    for j in range(1, landing_idx + 1):
        x1_img = positions[j - 1][1]
        x2_img = positions[j][1]

        if positions[j][0] - positions[j - 1][0] > 3:
            continue

        cx1 = img_to_court_x(x1_img)
        cx2 = img_to_court_x(x2_img)

        cy = 9  # projected to net

        px1 = offset + int(cx1 * SCALE)
        py1 = offset + int((COURT_H - cy) * SCALE)

        px2 = offset + int(cx2 * SCALE)
        py2 = offset + int((COURT_H - cy) * SCALE)

        cv2.line(court_img, (px1, py1), (px2, py2), color, 2)


def draw_attack_to_landing(court_img, ax, ay, lx, ly, offset, color, rally_id=None):
    px1 = offset + int(ax * SCALE)
    py1 = offset + int((COURT_H - ay) * SCALE)

    px2 = offset + int(lx * SCALE)
    py2 = offset + int((COURT_H - ly) * SCALE)

    cv2.line(court_img, (px1, py1), (px2, py2), color, 3)

    cv2.circle(court_img, (px1, py1), 8, (255, 0, 0), -1)   # attack
    cv2.circle(court_img, (px2, py2), 8, color, -1)         # landing

    if rally_id is not None:
        text = str(rally_id)
        cv2.putText(
            court_img,
            text,
            (px2 + 10, py2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )
        cv2.putText(
            court_img,
            text,
            (px2 + 10, py2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )


def save_court_map(court_img, output_path):
    cv2.imwrite(output_path, court_img)


def process_single_clip(video_path, model,clip_meta,H,debug_dir):
    clip_name = os.path.basename(video_path)

    cap, fps, total_frames, start_frame = open_video(video_path)
    meta = clip_meta.get(clip_name, {})
    rally_id = meta.get("id")
    set_id = meta.get("set")
    start = meta.get("start")
    end = meta.get("end")
    positions, frames = track_ball(cap, model, start_frame)
    cap.release()

    print(f"{clip_name}: Total points: {len(positions)}")

    if len(positions) == 0:
        print(f"{clip_name}: No detections found.")
        return None

    cross = find_crossing(positions)
    if cross is None:
        print(f"{clip_name}: skipped (no valid crossing found)")
        return None

    hx, hy = cross

    landing = find_landing(positions)
    _, lx_img, ly_img = landing

    ax, ay = compute_attack_point(hx)
    lx, ly = img_to_court((lx_img, ly_img), H)
    
    if ly > 9 or ly < 0:
        print(f"{clip_name}: skipped (landing outside court: y={ly:.2f})")
        return None

    debug_output_path = os.path.join(
    debug_dir,
    os.path.splitext(clip_name)[0] + "_debug.mp4"
) 

    #write_debug_video(
        #frames=frames,
        #positions=positions,
        #hx=hx,
        #hy=hy,
        #output_path=debug_output_path
    #)

    return {
    "clip_name": clip_name,
    "rally_id": rally_id,
    "start": start,
    "end": end,
    "set_id": set_id,
    "positions": [[int(f), int(x), int(y)] for f, x, y in positions],
    "attack_point": [float(ax), float(ay)],
    "landing_point": [float(lx), float(ly)],
    "debug_output_path": debug_output_path,
} 


def draw_clip_on_court(court_img, clip_result, offset):
    positions = clip_result["positions"]
    ax, ay = clip_result["attack_point"]
    lx, ly = clip_result["landing_point"]
    rally_id = clip_result.get("rally_id")
    set_id = clip_result.get("set_id")

    color = SET_COLORS.get(set_id, SET_COLORS[None])

    draw_projected_trajectory(court_img, positions, offset, color)
    draw_attack_to_landing(court_img, ax, ay, lx, ly, offset, color, rally_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--calibration", required=True)
    args = parser.parse_args()

    match_id = args.match_id
    calibration_path = args.calibration

    clips_dir = f"/workspace/rally_segmentator/output/{match_id}/rally_clips"
    rallies_json = f"/workspace/rally_segmentator/output/{match_id}/rallies_with_clips.json"
    output_dir = f"/workspace/heatmaps/{match_id}"
    debug_dir = os.path.join(output_dir, "debug_clips")
    output_json = os.path.join(output_dir, "rally_results.json")
    print("=== generate_heatmap started ===")
    print("match_id:", match_id)
    print("clips_dir:", clips_dir)
    print("rallies_json:", rallies_json)
    ensure_dirs(output_dir, debug_dir)

    model = YOLO(str(BALL_MODEL))
    clip_paths = get_clip_paths(clips_dir)
    clip_meta = load_rally_metadata(rallies_json)

    if not clip_paths:
        print("No clips found.")
        return

    court_img, offset, _, _ = create_court_image()

    global NET_LEFT_X, NET_RIGHT_X, NET_Y_IMG
    img_pts, NET_LEFT_X, NET_RIGHT_X, NET_Y_IMG = load_calibration(calibration_path)

    H, _ = cv2.findHomography(img_pts, court_pts)

    processed = 0
    all_results = []

    for video_path in clip_paths:
        print(f"\nProcessing {os.path.basename(video_path)}")

        clip_result = process_single_clip(video_path, model, clip_meta, H, debug_dir)
        if clip_result is None:
            continue

        draw_clip_on_court(court_img, clip_result, offset)
        all_results.append(clip_result)
        processed += 1

    output_court_path = os.path.join(output_dir, "court_map_all.png")
    save_court_map(court_img, output_court_path)

    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nProcessed {processed} clips.")
    print(f"Saved combined court map to {output_court_path}")
    print(f"Saved debug clips to {debug_dir}")
    print(f"Saved results to {output_json}")



if __name__ == "__main__":
    main()
