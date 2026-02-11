from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
from pathlib import Path
import csv, os, math

# === CONFIG ===
DETECTIONS_CSV = Path(r"D:\Volleyballey\WhistleDetector\match8_scan\match8_whistle_peaks.csv")
LABELS_CSV = Path(r"D:\Volleyballey\WhistleDetector\labels.csv")
SNIPPETS_DIR = Path(r"D:\Volleyballey\WhistleDetector\snippets")  # 👈 parent folder only!
YOUTUBE_LINKS = {"match8": "https://www.youtube.com/embed/PNNOog-IoFc"}
# =============

app = Flask(__name__)

# --- Ensure labels.csv exists ---
LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
if not LABELS_CSV.exists():
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["match", "timestamp", "label"])

# --- Load detections ---
detections = pd.read_csv(DETECTIONS_CSV)

# If the CSV has "time" column, rename it to "timestamp"
if "timestamp" not in detections.columns and "time" in detections.columns:
    detections.rename(columns={"time": "timestamp"}, inplace=True)

# Force a "match" column if missing
if "match" not in detections.columns:
    detections["match"] = "match8"

# --- Load existing labels ---
labeled = pd.read_csv(LABELS_CSV) if LABELS_CSV.exists() else pd.DataFrame(columns=["match", "timestamp", "label"])

# --- Mark which detections are already labeled ---
detections["labeled"] = detections.apply(
    lambda row: any(
        (labeled["match"] == row["match"]) &
        (labeled["timestamp"].astype(str) == str(row["timestamp"]))
    ),
    axis=1,
)

# --- Timestamp parser ---
# --- Timestamp parser (robust) ---
def parse_timestamp(ts):
    """Safely parse numeric or string timestamps into seconds."""
    try:
        # Case 1: already a float (your case)
        return float(ts)
    except ValueError:
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
        return float(parts[0])

# --- Save label ---
def save_label(match, timestamp, label):
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([match, timestamp, label])

# --- Serve snippets and spectrograms ---
@app.route("/snippets/<match>/<filename>")
def serve_snippet(match, filename):
    # 👇 now correctly serves from snippets/match8/
    folder = SNIPPETS_DIR / match
    return send_from_directory(folder, filename)

# --- Main route ---
@app.route("/")
def index():
    remaining = detections[~detections["labeled"]]
    if remaining.empty:
        return "<h2>✅ All detections reviewed!</h2>"

    row = remaining.sample(1).iloc[0]
    match = row["match"]
    timestamp = row["timestamp"]
    seconds = parse_timestamp(timestamp)
    start_time = max(0, int(math.floor(seconds - 1)))

    # ✅ YouTube embed with correct seek + autoplay
    video_url = YOUTUBE_LINKS.get(match, "")
    if video_url:
        sep = "&" if "?" in video_url else "?"
        video_url = f"{video_url}{sep}start={start_time}&autoplay=1"

    wav_name = f"{match}_{seconds:.2f}.wav"
    png_name = f"{match}_{seconds:.2f}.png"
    wav_path = f"/snippets/{match}/{wav_name}"
    png_path = f"/snippets/{match}/{png_name}"

    print(f"🎬 Serving {wav_name} — YouTube start={start_time}s")

    return render_template(
        "review.html",
        match=match,
        timestamp=timestamp,
        video_url=video_url,
        wav_path=wav_path,
        png_path=png_path,
    )

@app.route("/label", methods=["POST"])
def label():
    match = request.form["match"]
    timestamp = request.form["timestamp"]
    label_value = request.form["label"]
    save_label(match, timestamp, label_value)
    detections.loc[
        (detections["match"] == match) & (detections["timestamp"] == timestamp),
        "labeled",
    ] = True
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
