from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
from pathlib import Path
import csv, random

# === CONFIG ===
DETECTIONS_CSV = Path(r"D:\Volleyballey\WhistleDetector\detections_tree_model\detections_tree.csv")
LABELS_CSV = Path(r"D:\Volleyballey\WhistleDetector\labels_tree.csv")
SNIPPETS_DIR = Path(r"D:\Volleyballey\WhistleDetector\snippets")
YOUTUBE_LINKS = {
    #"match1": "https://www.youtube.com/embed/T_qRa2Lspc8",
    #"match3": "https://www.youtube.com/embed/lUtJ-kOxCLU",
   # "match4": "https://www.youtube.com/embed/v_6xbN60XH8",
    #"match5": "https://www.youtube.com/embed/Wb0wOW0LMV0",
   # "match6": "https://www.youtube.com/embed/LZT9RmtBm1c",
  #  "match7": "https://www.youtube.com/embed/iiB-bSHHOwk",
    "match8": "https://www.youtube.com/embed/PNNOog-IoFc",
}
LABEL_OPTIONS = ["whistle", "crowd", "squeak", "speech", "other", "skip"]
# ==================================

app = Flask(__name__)

# --- Ensure labels.csv exists ---
LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
if not LABELS_CSV.exists():
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["match", "timestamp", "label"])

# --- Load detections ---
df = pd.read_csv(DETECTIONS_CSV)
if "match" not in df or "timestamp" not in df:
    raise ValueError("CSV must include columns 'match' and 'timestamp'")

# --- Load labeled data ---
if LABELS_CSV.exists():
    labeled = pd.read_csv(LABELS_CSV)
else:
    labeled = pd.DataFrame(columns=["match", "timestamp", "label"])

# --- Timestamp parser ---
def parse_timestamp(ts):
    ts = str(ts).strip()
    parts = ts.replace(".", ":").split(":")
    parts = [float(p) for p in parts if p]
    if len(parts) == 4:
        h, m, s, ms = parts
        return h * 3600 + m * 60 + s + ms / 1000
    elif len(parts) == 3:
        m, s, ms = parts
        return m * 60 + s + ms / 1000
    elif len(parts) == 2:
        s, ms = parts
        return s + ms / 1000
    return float(parts[0]) if parts else 0.0

# --- Save label ---
def save_label(match, timestamp, label):
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([match, timestamp, label])

# --- Serve snippet files ---
@app.route("/snippets/<match>/<filename>")
def serve_snippet(match, filename):
    return send_from_directory(SNIPPETS_DIR / match, filename)

# --- Main route ---
@app.route("/")
def index():
    global df, labeled

    labeled_set = set(zip(labeled["match"], labeled["timestamp"]))
    remaining = df[~df.apply(lambda r: (r["match"], r["timestamp"]) in labeled_set, axis=1)]

    if remaining.empty:
        return "<h2>✅ All detections labeled!</h2>"

    row = remaining.sample(1).iloc[0]
    match = row["match"]
    timestamp = str(row["timestamp"])
    seconds = parse_timestamp(timestamp)
    start_time = max(0, int(seconds) - 1)

    video_url = YOUTUBE_LINKS.get(match, "")
    if video_url:
        video_url += f"?start={start_time}"

    wav_name = f"{match}_{seconds:.2f}.wav"
    png_name = f"{match}_{seconds:.2f}.png"
    wav_path = f"/snippets/{match}/{wav_name}"
    png_path = f"/snippets/{match}/{png_name}"

    return render_template(
        "label_viewer.html",
        match=match,
        timestamp=timestamp,
        video_url=video_url,
        wav_path=wav_path,
        png_path=png_path,
        label_options=LABEL_OPTIONS,
    )

@app.route("/label", methods=["POST"])
def label():
    match = request.form["match"]
    timestamp = request.form["timestamp"]
    label = request.form["label"]

    save_label(match, timestamp, label)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
