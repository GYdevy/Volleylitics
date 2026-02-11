import subprocess
from pathlib import Path

# ===============================
# CONFIG
# ===============================
MATCH_NUM = 4

BASE_VIDEO_DIR = r"D:\Volleyballey\videos"
BASE_OUTPUT_DIR = r"D:\Volleyballey\WhistleDetector\debug"

VIDEO_PATH = Path(f"{BASE_VIDEO_DIR}\\match{MATCH_NUM}.mp4")
OUT_VIDEO  = Path(f"{BASE_OUTPUT_DIR}\\whistle_band_match{MATCH_NUM}.mp4")

FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"

LOW_HZ  = 3900
HIGH_HZ = 4200

BASE_OUTPUT_DIR = Path(BASE_OUTPUT_DIR)
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# FFMPEG COMMAND
# ===============================
# - bandpass filter
# - keep video stream untouched
# - re-encode audio only
cmd = [
    FFMPEG, "-y",
    "-i", str(VIDEO_PATH),
    "-map", "0:v:0",
    "-map", "0:a:0",
    "-af", f"bandpass=f={int((LOW_HZ+HIGH_HZ)/2)}:width_type=h:width={HIGH_HZ-LOW_HZ}",
    "-c:v", "copy",
    "-c:a", "aac",
    "-b:a", "192k",
    str(OUT_VIDEO)
]

print("Running FFmpeg band-pass filter...")
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("DONE:", OUT_VIDEO)
