from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import csv
import random
from datetime import datetime
from pathlib import Path
app = Flask(__name__)

# Folder of matches
FRAMES_DIR = Path(r"/Vision/data\matches")
LABELS_FILE = "../Vision/data/labels.csv"

os.makedirs("../Vision/data", exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

if not os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "match", "frame", "has_ball", "x", "y", "user"])

def get_all_frames():
    frames = []
    for match_dir in FRAMES_DIR.iterdir():
        if not match_dir.is_dir():
            continue

        match_name = match_dir.name
        for f in match_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".png"):
                # ✅ Return both match name and full frame path
                frames.append((match_name, str(f)))
    return frames

@app.route("/")
def index():
    frames = get_all_frames()
    print("DEBUG: Found frames:", len(frames))
    if not frames:
        return "No frames found in data/matches/"
    match, frame = random.choice(frames)
    print("DEBUG: Selected frame:", frame)
    return render_template("index.html", match=match, frame=frame)

@app.route("/frame/<match>/<filename>")
def serve_frame(match, filename):
    return send_from_directory(os.path.join(FRAMES_DIR, match), filename)

@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json()

    # Safely extract all possible fields
    has_ball = data.get("has_ball", False)
    x = data.get("x", "")
    y = data.get("y", "")
    img_w = data.get("img_w", "")
    img_h = data.get("img_h", "")
    user = data.get("user", "anon")

    with open(LABELS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),  # timestamp
            data.get("match"),
            data.get("frame"),
            has_ball,
            x,
            y,
            img_w,
            img_h,
            user
        ])

    return jsonify(success=True)


@app.route("/next")
def next_frame():
    frames = get_all_frames()
    match, frame = random.choice(frames)
    return jsonify({"match": match, "frame": frame})

if __name__ == "__main__":
    app.run(host="localhost", port=80)
