import json
import cv2
import os
import random

VIDEO_DIR = r"E:\Volleyballey\videos"
GT_JSON = r"anchored_matches\whistles_all_reanchored.json"

OUT_DIR = r"E:\Volleyballey\rally_dataset"

os.makedirs(os.path.join(OUT_DIR, "rally"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "non_rally"), exist_ok=True)

with open(GT_JSON) as f:
    rows = json.load(f)

matches = set(r["match_id"] for r in rows)

for match in matches:

    video_path = os.path.join(VIDEO_DIR, f"{match}.mp4")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    events = [r for r in rows if r["match_id"] == match]

    for e in events:

        t = e["t_anchor"]

        if e["type"] == "rally_end":

            # rally frames BEFORE whistle
            for dt in [1.5,2.0,2.5]:

                frame = int((t - dt) * fps)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

                ret,img = cap.read()

                if ret:
                    name = f"{match}_{frame}.jpg"
                    cv2.imwrite(os.path.join(OUT_DIR,"rally",name), img)

            # non-rally AFTER whistle
            for dt in [1.0,2.0,3.0]:

                frame = int((t + dt) * fps)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

                ret,img = cap.read()

                if ret:
                    name = f"{match}_{frame}.jpg"
                    cv2.imwrite(os.path.join(OUT_DIR,"non_rally",name), img)

    cap.release()
