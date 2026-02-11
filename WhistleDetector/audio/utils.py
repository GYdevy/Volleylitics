import subprocess
from WhistleDetector.config import VIDEO_PATH, HOP,PAD_BEFORE,PAD_AFTER
import numpy as np
FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"

def fmt_time(sec):
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"

def whistle_salience(spec):
    peak = spec.max()
    if peak < -60:
        return -1e9
    active = (spec > peak - 6).sum()
    return peak - active * 2.5

def save_clip(start, end, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"whistle_{int(start*1000):010d}_{int(end*1000):010d}.mp4"
    clip_start = max(0, start - PAD_BEFORE)
    clip_end = end + PAD_AFTER
    duration = clip_end - clip_start
    subprocess.run([
        FFMPEG, "-y",
        "-ss", str(start),
        "-i", VIDEO_PATH,
        "-t", str(duration),
        "-c:v", "mpeg4",
        "-qscale:v", "3",
        "-c:a", "aac",
        str(out)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out

def band_width_hz(S_w, freqs_w):
    """
    Width (Hz) of concentrated energy inside whistle band.
    """
    if S_w.size == 0:
        return np.inf

    mean_spec = np.mean(S_w, axis=1)

    if mean_spec.max() <= 0:
        return np.inf

    active = mean_spec > (mean_spec.max() - 6)  # within 6 dB of peak

    if not np.any(active):
        return np.inf

    active_freqs = freqs_w[active]
    return active_freqs.max() - active_freqs.min()