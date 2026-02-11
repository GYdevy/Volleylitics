import os
import joblib
import subprocess

# =========================
# CONFIG
# =========================
VIDEO_PATH = r"D:\Volleyballey\videos\match4.mp4"
DATA_PATH  = "serve_rally_structure.pkl"
OUT_DIR = r"D:\Volleyballey\megatest"

PRE_PAD  = 1.0   # seconds before serve
POST_PAD = 1.0   # seconds after next serve

FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"

# =========================
# LOAD DATA
# =========================
data = joblib.load(DATA_PATH)

rallies = data["rally_segments"]

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Generating {len(rallies)} rally clips…")

# =========================
# CLIP LOOP
# =========================
for idx, r in enumerate(rallies):
    t0 = max(0.0, r["serve_start"] - PRE_PAD)
    t1 = r["serve_next"] + POST_PAD
    dur = t1 - t0

    out_name = f"rally_{idx:03d}_{r['serve_start']:.1f}_{r['serve_next']:.1f}.mp4"
    out_path = os.path.join(OUT_DIR, out_name)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{t0:.3f}",
        "-i", VIDEO_PATH,
        "-to", f"{t1:.3f}",
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c", "copy",
        out_path
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("FFMPEG FAILED:", out_name)

    if idx % 10 == 0:
        print(f"  [{idx}/{len(rallies)}] {out_name}")

print("\nDone.")
print(f"Clips saved to → {OUT_DIR}/")