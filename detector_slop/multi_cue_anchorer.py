"""
Re-anchor whistles using multi-cue whistle center detection
"""

import json
import os
import tempfile
import subprocess
import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d


# ============================================================
# CONFIG
# ============================================================

VIDEO_DIR = r"E:\Volleyballey\videos"
INPUT_JSON = "whistles_all.json"
OUTPUT_JSON = "whistles_all_reanchored.json"

FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"

SR = 22050
N_FFT = 2048
HOP = 128

WHISTLE_LOW = 3700
WHISTLE_HIGH = 4300

SEARCH_RADIUS = 0.35


# ============================================================
# AUDIO LOADER
# ============================================================

def load_audio_from_video(video_path):

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-i", video_path,
            "-ac", "1",
            "-ar", str(SR),
            "-vn",
            tmp_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    y, _ = librosa.load(tmp_path, sr=SR)
    os.remove(tmp_path)

    return y


# ============================================================
# MULTI-CUE CENTER DETECTOR
# ============================================================
def ridge_center_frame(peak_freqs):
    """
    Bird-call style ridge tracking.
    Finds the longest stable frequency segment.
    """

    freq_diff = np.abs(np.diff(peak_freqs, prepend=peak_freqs[0]))

    ridge_frames = np.where(freq_diff < 40)[0]

    if len(ridge_frames) == 0:
        return None

    groups = []
    cur = [ridge_frames[0]]

    for f in ridge_frames[1:]:
        if f == cur[-1] + 1:
            cur.append(f)
        else:
            groups.append(cur)
            cur = [f]

    groups.append(cur)

    longest = max(groups, key=len)

    if len(longest) < 4:
        return None

    return int(np.mean(longest))

def compute_center_frame(band_mag, freqs_w):

    band_energy = band_mag.mean(axis=0)

    band_peak = np.max(band_mag, axis=0)
    band_mean = np.mean(band_mag, axis=0) + 1e-8
    sharpness = band_peak / band_mean

    peak_bins = np.argmax(band_mag, axis=0)
    peak_freqs = freqs_w[peak_bins]

    freq_diff = np.abs(np.diff(peak_freqs, prepend=peak_freqs[0]))

    narrow_ratio = band_peak / (band_energy + 1e-8)

    band_energy = uniform_filter1d(band_energy, size=5)
    sharpness = uniform_filter1d(sharpness, size=5)
    freq_diff = uniform_filter1d(freq_diff, size=5)
    narrow_ratio = uniform_filter1d(narrow_ratio, size=5)

    def norm(x):
        return (x - np.median(x)) / (np.std(x) + 1e-8)

    band_energy = norm(band_energy)
    sharpness = norm(sharpness)
    freq_diff = norm(freq_diff)
    narrow_ratio = norm(narrow_ratio)

    score = (
        1.0 * band_energy +
        1.2 * sharpness +
        0.8 * narrow_ratio -
        1.0 * freq_diff
    )

    score = uniform_filter1d(score, size=5)

    max_score = np.max(score)

    plateau = np.where(score > 0.7 * max_score)[0]

    if len(plateau) > 0:
        center_frame = int(np.mean(plateau))
    else:
        center_frame = int(np.argmax(score))

    return center_frame


# ============================================================
# ANCHOR COMPUTATION
# ============================================================

def compute_anchor(y, t_raw):

    start_t = max(0.0, t_raw - SEARCH_RADIUS)
    end_t = min(len(y)/SR, t_raw + SEARCH_RADIUS)

    s0 = int(start_t * SR)
    s1 = int(end_t * SR)

    segment = y[s0:s1]

    if len(segment) < N_FFT:
        return t_raw

    S = librosa.stft(segment, n_fft=N_FFT, hop_length=HOP)
    mag = np.abs(S)

    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

    mask = (freqs >= WHISTLE_LOW) & (freqs <= WHISTLE_HIGH)

    if not np.any(mask):
        return t_raw

    band_mag = mag[mask]
    freqs_w = freqs[mask]

    center_frame = compute_center_frame(band_mag, freqs_w)

    t_anchor = (s0 + center_frame * HOP) / SR

    return float(max(0.0, min(t_anchor, len(y)/SR)))


# ============================================================
# MAIN
# ============================================================

def main():

    with open(INPUT_JSON, "r") as f:
        rows = json.load(f)

    audio_cache = {}
    out_rows = []

    for row in rows:

        match_id = row["match_id"]

        if match_id not in audio_cache:

            video_path = os.path.join(VIDEO_DIR, f"{match_id}.mp4")

            print("Loading audio:", match_id)

            audio_cache[match_id] = load_audio_from_video(video_path)

        y = audio_cache[match_id]

        t_raw = row.get("time") or row.get("t_raw")

        if t_raw is None:
            continue

        t_anchor = compute_anchor(y, float(t_raw))

        updated = dict(row)

        updated["t_raw"] = round(float(t_raw), 3)
        updated["t_anchor"] = round(float(t_anchor), 3)

        out_rows.append(updated)

    out_rows.sort(key=lambda x: (x["match_id"], x["t_anchor"]))

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out_rows, f, indent=4)

    print("\nSaved anchored whistles:", OUTPUT_JSON)
    print("Total:", len(out_rows))


if __name__ == "__main__":
    main()