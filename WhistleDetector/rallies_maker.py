import re
import subprocess
from pathlib import Path

# ===============================
# CONFIG
# ===============================
MATCH_NUM = 4

BASE_VIDEO_DIR = Path(r"D:\Volleyballey\videos")
CLIPS_DIR = Path(r"D:\Volleyballey\WhistleDetector\clips\match4")
OUT_DIR = Path(r"D:\Volleyballey\WhistleDetector\rallies\match4")

VIDEO_PATH = BASE_VIDEO_DIR / f"match{MATCH_NUM}.mp4"
FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# HELPERS
# ===============================
def ms_to_sec(ms):
    return ms / 1000.0

def fmt_time(sec):
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"

def ffmpeg_escape_text(s: str) -> str:
    """
    Escape text for FFmpeg drawtext.
    """
    return (
        s.replace("\\", "\\\\")
         .replace(":", "\\:")
         .replace("'", "\\'")
         .replace("→", "->")
    )

def parse_time_to_sec(t: str) -> float:
    """
    Accepts:
      - seconds (e.g. "4224" or "12.5")
      - mm:ss (e.g. "10:24")
      - hh:mm:ss (e.g. "1:10:24")
    Returns seconds as float.
    """
    t = t.strip()

    # plain seconds
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return float(t)

    parts = t.split(":")
    if len(parts) == 2:  # mm:ss
        m, s = parts
        return int(m) * 60 + float(s)

    if len(parts) == 3:  # hh:mm:ss
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)

    raise ValueError(f"Invalid time format: {t}")

def parse_uid_times(path: Path):
    """
    Extract start/end ms from filename.
    """
    m = re.search(r"_(\d{6,})_(\d{6,})", path.stem)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

# ===============================
# LOAD + SORT WHISTLES
# ===============================
whistles = []

for p in CLIPS_DIR.glob("*.mp4"):
    times = parse_uid_times(p)
    if times:
        start_ms, end_ms = times
        whistles.append({
            "path": p,
            "start": ms_to_sec(start_ms),
            "end": ms_to_sec(end_ms)
        })

whistles.sort(key=lambda x: x["start"])

print(f"Loaded {len(whistles)} whistle clips")

# ===============================
# PROMPT FOR SET STRUCTURE
# ===============================
num_sets = int(input("How many sets in this match? "))

sets = []

for i in range(num_sets):
    print(f"\n--- Set {i+1} ---")
    s_start = parse_time_to_sec(
        input("  Time of FIRST serve whistle (hh:mm:ss | mm:ss | sec): ")
    )
    s_end = parse_time_to_sec(
        input("  Time of LAST rally-end whistle (hh:mm:ss | mm:ss | sec): ")
    )
    sets.append((s_start, s_end))

# ===============================
# BUILD RALLIES
# ===============================
rallies = []
rally_idx = 0

for set_idx, (set_start, set_end) in enumerate(sets, start=1):
    # restrict whistles to this set
    ws = [w for w in whistles if set_start <= w["start"] <= set_end]

    print(f"\nSet {set_idx}: using {len(ws)} whistles")

    # assume alternating: serve -> end -> serve -> end
    for i in range(0, len(ws) - 1, 2):
        start_whistle = ws[i]
        end_whistle   = ws[i + 1]

        rally_start = start_whistle["end"]
        rally_end   = end_whistle["start"]

        if rally_end <= rally_start:
            continue

        rallies.append({
            "set": set_idx,
            "start": rally_start,
            "end": rally_end
        })

# ===============================
# CUT RALLIES
# ===============================
cut_files = []

for i, r in enumerate(rallies):
    out = OUT_DIR / f"rally_{i:04d}.mp4"
    raw_label = f"Set {r['set']}  {fmt_time(r['start'])} -> {fmt_time(r['end'])}"
    label = ffmpeg_escape_text(raw_label)

    duration = r["end"] - r["start"]

    cmd = [
        FFMPEG, "-y",
        "-ss", str(r["start"]),
        "-i", str(VIDEO_PATH),
        "-t", str(duration),
        "-vf",
        f"drawtext=text='{label}':x=20:y=20:fontsize=24:"
        f"fontcolor=white:box=1:boxcolor=black@0.6",
        "-c:v", "mpeg4",
        "-qscale:v", "3",
        "-c:a", "aac",
        str(out)
    ]

    subprocess.run(cmd, check=True)

    cut_files.append(out)

print(f"\nCut {len(cut_files)} rallies")

# ===============================
# CONCAT ALL RALLIES
# ===============================
concat_file = OUT_DIR / "concat.txt"
with open(concat_file, "w", encoding="utf-8") as f:
    for c in cut_files:
        f.write(f"file {c.resolve().as_posix()}\n")

final_out = OUT_DIR / f"match{MATCH_NUM}_rallies.mp4"

subprocess.run([
    FFMPEG, "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", str(concat_file),
    "-c", "copy",
    str(final_out)
], check=True)



print("\nDONE:")
print(final_out)
