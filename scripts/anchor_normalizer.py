"""
Canonical whistle anchoring (onset-first, peak fallback).

This script converts manual click times into a stable acoustic anchor field:
`t_anchor` = estimated whistle-band attack onset.
"""

import json
import os
from pathlib import Path

import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"E:\Volleyballey\detector_slop")
VIDEO_DIR = Path(r"E:\Volleyballey\videos")
INPUT_JSON = BASE_DIR / "whistles_all.json"
OUTPUT_JSON = BASE_DIR / "whistles_all_anchored_attack.json"

SR = 22050
N_FFT = 2048
HOP = 128

BAND_LOW = 3700
BAND_HIGH = 4300

SEARCH_RADIUS = 0.60
BACKSEARCH_SEC = 0.25
SMOOTH_SIZE = 5

E_ONSET_TH = 0.25
F_ONSET_TH = 0.35
E_MIN_FLOOR = 0.20
PERSISTENCE_WINDOW = 5
PERSISTENCE_REQUIRED = 3


# ============================================================
# AUDIO LOADER
# ============================================================

def load_audio(path: Path):
    y, _ = librosa.load(str(path), sr=SR)
    return y


# ============================================================
# ATTACK-BASED ANCHOR
# ============================================================

def _robust_norm(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    z = (x - med) / (1.4826 * mad)
    z_min = np.min(z)
    z_max = np.max(z)
    if z_max - z_min < 1e-8:
        return np.zeros_like(z)
    return (z - z_min) / (z_max - z_min)


def _find_onset_frame(energy_norm: np.ndarray, flux_norm: np.ndarray, peak_idx: int) -> int | None:
    back_frames = int((BACKSEARCH_SEC * SR) / HOP)
    start = max(1, peak_idx - back_frames)

    for i in range(start, peak_idx + 1):
        crossed = energy_norm[i - 1] < E_ONSET_TH <= energy_norm[i]
        flux_ok = flux_norm[i] >= F_ONSET_TH if i < len(flux_norm) else False
        if not (crossed and flux_ok):
            continue

        j0 = i
        j1 = min(len(energy_norm), i + PERSISTENCE_WINDOW)
        hold = np.sum(energy_norm[j0:j1] >= E_ONSET_TH)
        if hold >= PERSISTENCE_REQUIRED:
            return i

    return None


def compute_anchor(y: np.ndarray, t_raw: float) -> dict:

    start_t = max(0.0, t_raw - SEARCH_RADIUS)
    end_t   = min(len(y)/SR, t_raw + SEARCH_RADIUS)

    s0 = int(start_t * SR)
    s1 = int(end_t * SR)

    segment = y[s0:s1]

    if len(segment) < N_FFT:
        return {
            "t_anchor": float(t_raw),
            "t_peak": float(t_raw),
            "anchor_method": "raw_fallback_short_segment",
            "anchor_confidence": 0.0,
        }

    # STFT
    S = librosa.stft(segment, n_fft=N_FFT, hop_length=HOP)
    mag = np.abs(S)

    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    mask = (freqs >= BAND_LOW) & (freqs <= BAND_HIGH)

    if not np.any(mask):
        return {
            "t_anchor": float(t_raw),
            "t_peak": float(t_raw),
            "anchor_method": "raw_fallback_band_empty",
            "anchor_confidence": 0.0,
        }

    S_w = mag[mask]
    freqs_w = freqs[mask]

    # 1) Band energy
    band_energy = S_w.mean(axis=0)
    band_energy = uniform_filter1d(band_energy, size=SMOOTH_SIZE)

    if band_energy.max() <= 0:
        return {
            "t_anchor": float(t_raw),
            "t_peak": float(t_raw),
            "anchor_method": "raw_fallback_zero_energy",
            "anchor_confidence": 0.0,
        }

    band_energy_norm = band_energy / (band_energy.max() + 1e-8)

    # 2) Positive derivative / flux
    derivative = np.diff(band_energy_norm, prepend=band_energy_norm[0])
    derivative[derivative < 0] = 0
    derivative = uniform_filter1d(derivative, size=SMOOTH_SIZE)

    # 3) Prominence (tonal clarity)
    peak_vals = S_w.max(axis=0)
    mean_vals = S_w.mean(axis=0) + 1e-8
    prominence = peak_vals / mean_vals
    prominence = uniform_filter1d(prominence, size=SMOOTH_SIZE)

    # 4) Narrowband concentration around dominant frequency
    mean_spectrum = S_w.mean(axis=1)
    peak_idx = np.argmax(mean_spectrum)
    center_freq = freqs_w[peak_idx]

    narrow_mask = (freqs >= center_freq - 150) & (freqs <= center_freq + 150)

    if np.any(narrow_mask):
        narrow_energy = mag[narrow_mask].mean(axis=0)
        full_band_energy = S_w.mean(axis=0) + 1e-8
        narrow_ratio = narrow_energy / full_band_energy
        narrow_ratio = uniform_filter1d(narrow_ratio, size=SMOOTH_SIZE)
    else:
        narrow_ratio = np.zeros_like(band_energy_norm)

    # 5) Peak score (for t_peak)
    e_n = _robust_norm(band_energy)
    p_n = _robust_norm(prominence)
    r_n = _robust_norm(narrow_ratio)
    f_n = _robust_norm(derivative)

    peak_score = 0.50 * e_n + 0.25 * p_n + 0.25 * r_n

    center_frame = int((t_raw - start_t) * SR / HOP)
    center_frame = max(0, min(center_frame, len(peak_score) - 1))
    local_rad_frames = int(0.35 * SR / HOP)
    p0 = max(0, center_frame - local_rad_frames)
    p1 = min(len(peak_score), center_frame + local_rad_frames + 1)

    local_scores = peak_score[p0:p1]
    if len(local_scores) == 0:
        best_frame = center_frame
    else:
        rel_idx = int(np.argmax(local_scores))
        best_frame = p0 + rel_idx

    onset_frame = _find_onset_frame(e_n, f_n, best_frame)
    if onset_frame is not None:
        anchor_frame = onset_frame
        method = "onset"
    elif e_n[best_frame] >= E_MIN_FLOOR:
        anchor_frame = best_frame
        method = "peak_fallback"
    else:
        raw_frame = center_frame
        anchor_frame = raw_frame
        method = "raw_fallback_low_energy"

    t_anchor = (s0 + anchor_frame * HOP) / SR
    t_peak = (s0 + best_frame * HOP) / SR

    t_anchor = float(max(0.0, min(t_anchor, len(y)/SR)))
    t_peak = float(max(0.0, min(t_peak, len(y)/SR)))

    confidence = float(peak_score[best_frame]) if len(peak_score) else 0.0
    return {
        "t_anchor": t_anchor,
        "t_peak": t_peak,
        "anchor_method": method,
        "anchor_confidence": confidence,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        rows = json.load(f)

    out_rows = []
    audio_cache = {}

    for row in rows:

        match_id = row["match_id"]

        if match_id not in audio_cache:
            audio_path = VIDEO_DIR / f"{match_id}.mp4"
            print(f"Loading {match_id} audio...")
            if not audio_path.exists():
                print(f"  ! missing video: {audio_path}, skipping match")
                audio_cache[match_id] = None
                continue
            audio_cache[match_id] = load_audio(audio_path)

        y = audio_cache[match_id]
        if y is None:
            continue

        t_raw = row.get("t_raw", row.get("time"))
        if t_raw is None:
            continue

        anchor = compute_anchor(y, float(t_raw))

        updated = dict(row)
        updated["t_raw"] = round(float(t_raw), 3)
        updated["t_anchor"] = round(float(anchor["t_anchor"]), 3)
        updated["t_peak"] = round(float(anchor["t_peak"]), 3)
        updated["anchor_method"] = anchor["anchor_method"]
        updated["anchor_confidence"] = round(float(anchor["anchor_confidence"]), 4)
        updated["anchor_offset_ms"] = round((updated["t_anchor"] - updated["t_raw"]) * 1000.0, 1)

        out_rows.append(updated)

    out_rows.sort(key=lambda x: (x["match_id"], x["t_anchor"]))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=4)

    print(f"\nSaved reanchored file → {OUTPUT_JSON}")
    print("Total whistles:", len(out_rows))


if __name__ == "__main__":
    main()