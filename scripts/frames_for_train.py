import os
import cv2
import json
import random
from pathlib import Path

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

VIDEO_DIR = Path(r"F:\Volleyballey\videos")

WHISTLE_JSON = "anchored_matches/whistles_all_with_contacts.json"

OUTPUT_DIR = "rally_dataset_v7"

RESIZE = 256

TARGET_RALLY = 13000
TARGET_NON = 15000

PRE_SERVE = 4.0
POST_RALLY = 4.0
SAFE_MARGIN = 0.8

FRAME_SKIP = 4

TEMPORAL_OFFSET = 3  # frames for t-3, t+3

# ------------------------------------------------
# DIRS
# ------------------------------------------------

os.makedirs(f"{OUTPUT_DIR}/rally", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/non_rally", exist_ok=True)

# ------------------------------------------------
# LOAD WHISTLES
# ------------------------------------------------

with open(WHISTLE_JSON) as f:
    raw = json.load(f)

whistles = {}

for w in raw:

    match = w["match_id"]

    if match not in whistles:
        whistles[match] = []

    whistles[match].append(w)

for match in whistles:
    whistles[match].sort(key=lambda x: x["t_anchor"])

# ------------------------------------------------
# LOAD FRAME
# ------------------------------------------------

def read_frame(cap, frame_idx):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret:
        return None

    frame = cv2.resize(frame, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)

    return frame

# ------------------------------------------------
# SAVE TEMPORAL STACK
# ------------------------------------------------

def save_frame(cap, frame_idx, label, counter, match):

    f1 = read_frame(cap, frame_idx - TEMPORAL_OFFSET)
    f2 = read_frame(cap, frame_idx)
    f3 = read_frame(cap, frame_idx + TEMPORAL_OFFSET)

    if f1 is None or f2 is None or f3 is None:
        return

    stacked = cv2.hconcat([f1, f2, f3])

    path = f"{OUTPUT_DIR}/{label}/{match}_{label}_{counter[0]:06d}.jpg"

    cv2.imwrite(path, stacked, [cv2.IMWRITE_JPEG_QUALITY, 90])

    counter[0] += 1

# ------------------------------------------------
# SAMPLE WINDOW
# ------------------------------------------------

def sample_window(cap, start, end, label, counter, fps, density, match):

    if start >= end:
        return

    start_f = int(start * fps)
    end_f = int(end * fps)

    frames = list(range(start_f, end_f, FRAME_SKIP))

    if len(frames) == 0:
        return

    random.shuffle(frames)

    frames = frames[:density]

    for f in frames:
        save_frame(cap, f, label, counter, match)

# ------------------------------------------------
# EXTRACTION
# ------------------------------------------------

rally_counter = [0]
non_counter = [0]

for match, events in whistles.items():

    video_path = None

    for f in VIDEO_DIR.glob(f"{match}*.mp4"):
        video_path = f
        break

    if video_path is None:
        print("Video not found for", match)
        continue

    print("Using video:", video_path)

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)

    for i, e in enumerate(events):

        if e["type"] != "serve":
            continue

        if "serve_contact" not in e:
            continue

        whistle_time = e["t_anchor"]

        serve_contact = e["serve_contact"]

        rally_end = None
        next_serve = None

        for j in range(i + 1, len(events)):
            if events[j]["type"] == "rally_end":
                rally_end = events[j]["t_anchor"]
                break

        for j in range(i + 1, len(events)):
            if events[j]["type"] == "serve":
                next_serve = events[j]["t_anchor"]
                break

        if rally_end is None:
            continue

        rally_len = rally_end - serve_contact

        if rally_len < 2:
            continue

        # ------------------------------------------------
        # RALLY SAMPLING
        # ------------------------------------------------

        if rally_counter[0] < TARGET_RALLY:

            rally_start = serve_contact + 0.5
            rally_stop = rally_end - SAFE_MARGIN

            if rally_stop > rally_start:

                sample_window(
                    cap,
                    rally_start,
                    rally_stop,
                    "rally",
                    rally_counter,
                    fps,
                    density=8,
                    match=match
                )

        # ------------------------------------------------
        # PRE SERVE NEGATIVE
        # ------------------------------------------------

        if non_counter[0] < TARGET_NON:

            pre_start = whistle_time - PRE_SERVE
            pre_end = whistle_time - SAFE_MARGIN

            if pre_start > 0:

                sample_window(
                    cap,
                    pre_start,
                    pre_end,
                    "non_rally",
                    non_counter,
                    fps,
                    6,
                    match
                )

        # ------------------------------------------------
        # POST RALLY NEGATIVE
        # ------------------------------------------------

        if non_counter[0] < TARGET_NON:

            post_start = rally_end + SAFE_MARGIN
            post_end = rally_end + POST_RALLY

            if next_serve:
                post_end = min(post_end, next_serve - SAFE_MARGIN)

            if post_end > post_start:

                sample_window(
                    cap,
                    post_start,
                    post_end,
                    "non_rally",
                    non_counter,
                    fps,
                    6,
                    match
                )

        # ------------------------------------------------
        # IDLE NEGATIVE
        # ------------------------------------------------

        if next_serve and non_counter[0] < TARGET_NON:

            idle_start = rally_end + POST_RALLY + 1
            idle_end = next_serve - PRE_SERVE

            if idle_end > idle_start:

                sample_window(
                    cap,
                    idle_start,
                    idle_end,
                    "non_rally",
                    non_counter,
                    fps,
                    3,
                    match
                )

        if rally_counter[0] >= TARGET_RALLY and non_counter[0] >= TARGET_NON:
            break

    cap.release()

print("\nDataset complete")
print("Rally frames:", rally_counter[0])
print("Non rally frames:", non_counter[0])