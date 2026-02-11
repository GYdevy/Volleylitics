import os
import subprocess

# ---------- CONFIG ----------
VIDEO_PATH = r"../videos/match7.mp4"
OUTPUT_DIR = r"/data/matches/match7"
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
FFPROBE_PATH = r"C:\ffmpeg\bin\ffprobe.exe"
N_FRAMES = 1000
START_MINUTES = 10          # start after 10 minutes
END_OFFSET_MINUTES = 5      # stop 5 minutes before end
# ----------------------------

def get_video_duration(video_path):
    """Return duration in seconds using ffprobe."""
    cmd = [
        FFPROBE_PATH,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def extract_frames(video_path, output_dir, start_sec, end_sec, n_frames):
    os.makedirs(output_dir, exist_ok=True)
    duration = end_sec - start_sec
    step = duration / n_frames

    # Get match name from folder name (e.g. "match5")
    match_name = os.path.basename(output_dir.rstrip("\\/"))

    print(f"🎥 Extracting {n_frames} frames from {start_sec:.1f}s to {end_sec:.1f}s "
          f"(every {step:.2f}s)...")

    for i in range(n_frames):
        t = start_sec + i * step
        out_file = os.path.join(output_dir, f"{match_name}_frame_{i:04d}.jpg")

        cmd = [
            FFMPEG_PATH,
            "-ss", str(t),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",  # quality (1=best, 31=worst)
            "-y",
            out_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if i % 100 == 0:
            print(f"  → Extracted {i}/{n_frames}")

    print(f"✅ Done! Saved {n_frames} frames in {output_dir}")

if __name__ == "__main__":
    total_duration = get_video_duration(VIDEO_PATH)
    print(f"Total duration: {total_duration/60:.1f} minutes")

    start_sec = START_MINUTES * 60
    end_sec = max(0, total_duration - END_OFFSET_MINUTES * 60)

    if start_sec >= end_sec:
        raise ValueError("⚠️ Invalid time range — video too short for given offsets.")

    extract_frames(VIDEO_PATH, OUTPUT_DIR, start_sec, end_sec, N_FRAMES)
