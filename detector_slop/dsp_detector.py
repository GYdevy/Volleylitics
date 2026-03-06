"""
Pure DSP + Rule-Based Whistle Detector
No ML
Multi-match evaluation
"""

import json
import tempfile
import subprocess
import os
from dataclasses import dataclass

import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d


# ============================================================
# CONFIG
# ============================================================

VIDEO_DIR = r"E:\Volleyballey\videos"
GT_PATH = r"E:\Volleyballey\detector_slop\whistles_all_reanchored.json"
FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"

ANCHOR_TOLERANCE = 0.6
MIN_DURATION_SEC = 0.18

@dataclass
class Config:
    sr: int = 22050
    n_fft: int = 2048
    hop: int = 128
    whistle_low: float = 3700
    whistle_high: float = 4300
    min_duration_sec: float = 0.13
    max_gap_frames: int = 8


cfg = Config()


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio_from_video(video_path, target_sr):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-i", video_path,
            "-ac", "1",
            "-ar", str(target_sr),
            "-vn",
            tmp_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    y, _ = librosa.load(tmp_path, sr=target_sr)
    os.remove(tmp_path)
    return y
def estimate_whistle_band(y, stage1_detections):
    peak_freqs = []

    for start, end in stage1_detections[:150]:  # limit for speed
        s0 = int(start * cfg.sr)
        s1 = int(end * cfg.sr)
        seg = y[s0:s1]

        if len(seg) < cfg.n_fft:
            continue

        S = librosa.stft(seg, n_fft=cfg.n_fft, hop_length=cfg.hop)
        mag = np.abs(S)

        freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)

        # Wide search band
        mask = (freqs >= 3000) & (freqs <= 5000)
        if not np.any(mask):
            continue

        S_w = mag[mask]
        freqs_w = freqs[mask]

        mean_spec = S_w.mean(axis=1)
        peak_idx = np.argmax(mean_spec)

        peak_freqs.append(freqs_w[peak_idx])

    if not peak_freqs:
        return 3500, 4300  # fallback

    dominant = np.median(peak_freqs)

    low = dominant - 250
    high = dominant + 250

    return low, high

# ============================================================
# STAGE A — ROI DETECTOR
# ============================================================

def detect_active_frames(y):
    S = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop)
    mag = np.abs(S)

    freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)
    band_mask = (freqs >= cfg.whistle_low) & (freqs <= cfg.whistle_high)

    band_mag = mag[band_mask]
    band_energy = band_mag.mean(axis=0)

    flatness = librosa.feature.spectral_flatness(S=mag)[0]

    band_peak = np.max(band_mag, axis=0)
    band_mean = np.mean(band_mag, axis=0)
    sharpness = band_peak / (band_mean + 1e-8)

    # Normalize
    def normalize(x):
        return (x - np.median(x)) / (np.std(x) + 1e-8)

    band_energy = normalize(band_energy)
    sharpness = normalize(sharpness)
    flatness = normalize(flatness)

    band_energy = uniform_filter1d(band_energy, size=5)
    sharpness = uniform_filter1d(sharpness, size=5)
    flatness = uniform_filter1d(flatness, size=5)

    score = 1.0 * band_energy + 1.0 * sharpness - 1.2 * flatness

    START_TH = 0.8
    CONTINUE_TH = 0.25

    active = []
    in_whistle = False

    for i, s in enumerate(score):
        if not in_whistle:
            if s > START_TH:
                in_whistle = True
                active.append(i)
        else:
            if s > CONTINUE_TH:
                active.append(i)
            else:
                in_whistle = False

    return active


# ============================================================
# GROUPING
# ============================================================

def group_frames(active):
    if not active:
        return []

    groups = []
    cur = [active[0]]

    for f in active[1:]:
        if f - cur[-1] <= cfg.max_gap_frames:
            cur.append(f)
        else:
            groups.append(cur)
            cur = [f]

    groups.append(cur)
    return groups


def extract_candidates(groups):
    detections = []
    pad_sec = 0.35
    pad_frames = int((pad_sec * cfg.sr) / cfg.hop)

    for g in groups:
        duration_sec = (g[-1] - g[0]) * cfg.hop / cfg.sr

        if duration_sec < cfg.min_duration_sec:
            continue

        start_frame = max(0, g[0] - pad_frames)
        end_frame = g[-1] + pad_frames

        start_sec = start_frame * cfg.hop / cfg.sr
        end_sec = end_frame * cfg.hop / cfg.sr

        detections.append((start_sec, end_sec))

    return detections


# ============================================================
# REFINEMENT
# ============================================================

def compute_center_frame(band_mag, freqs_w):
    """
    Multi-cue whistle center detection.
    """

    # -----------------------
    # Band energy
    # -----------------------
    band_energy = band_mag.mean(axis=0)

    # -----------------------
    # Tonal sharpness
    # -----------------------
    band_peak = np.max(band_mag, axis=0)
    band_mean = np.mean(band_mag, axis=0) + 1e-8
    sharpness = band_peak / band_mean

    # -----------------------
    # Frequency stability
    # -----------------------
    peak_bins = np.argmax(band_mag, axis=0)
    peak_freqs = freqs_w[peak_bins]

    freq_diff = np.abs(np.diff(peak_freqs, prepend=peak_freqs[0]))

    # -----------------------
    # Narrowband ratio
    # -----------------------
    narrow_ratio = band_peak / (band_energy + 1e-8)

    # Smooth signals
    band_energy = uniform_filter1d(band_energy, size=5)
    sharpness = uniform_filter1d(sharpness, size=5)
    freq_diff = uniform_filter1d(freq_diff, size=5)
    narrow_ratio = uniform_filter1d(narrow_ratio, size=5)

    # Normalize
    def norm(x):
        return (x - np.median(x)) / (np.std(x) + 1e-8)

    band_energy = norm(band_energy)
    sharpness = norm(sharpness)
    freq_diff = norm(freq_diff)
    narrow_ratio = norm(narrow_ratio)

    # Combined whistle score
    score = (
        1.0 * band_energy +
        1.2 * sharpness +
        0.8 * narrow_ratio -
        1.0 * freq_diff
    )

    # Smooth final score
    score = uniform_filter1d(score, size=5)

    center_frame = int(np.argmax(score))

    return center_frame

def refine_candidates(y, detections):
    refined = []

    for start, end in detections:

        s0 = int(start * cfg.sr)
        s1 = int(end * cfg.sr)
        seg = y[s0:s1]

        if len(seg) < cfg.n_fft:
            continue

        S = librosa.stft(seg, n_fft=cfg.n_fft, hop_length=cfg.hop)
        mag = np.abs(S)

        freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)
        mask = (freqs >= cfg.whistle_low) & (freqs <= cfg.whistle_high)

        if not np.any(mask):
            continue

        band_mag = mag[mask]
        freqs_w = freqs[mask]

        # ------------------------------------------------
        # FEATURE 1 — Band energy
        # ------------------------------------------------
        band_energy = band_mag.mean(axis=0)

        # ------------------------------------------------
        # FEATURE 2 — Tonal sharpness
        # ------------------------------------------------
        band_peak = np.max(band_mag, axis=0)
        band_mean = np.mean(band_mag, axis=0) + 1e-8
        sharpness = band_peak / band_mean

        # ------------------------------------------------
        # FEATURE 3 — Frequency stability
        # ------------------------------------------------
        peak_bins = np.argmax(band_mag, axis=0)
        peak_freqs = freqs_w[peak_bins]
        freq_diff = np.abs(np.diff(peak_freqs, prepend=peak_freqs[0]))

        # ------------------------------------------------
        # FEATURE 4 — Narrowband ratio
        # ------------------------------------------------
        narrow_ratio = band_peak / (band_energy + 1e-8)

        # ------------------------------------------------
        # Smooth signals
        # ------------------------------------------------
        band_energy = uniform_filter1d(band_energy, size=5)
        sharpness = uniform_filter1d(sharpness, size=5)
        freq_diff = uniform_filter1d(freq_diff, size=5)
        narrow_ratio = uniform_filter1d(narrow_ratio, size=5)

        # ------------------------------------------------
        # Normalize locally
        # ------------------------------------------------
        def norm(x):
            return (x - np.median(x)) / (np.std(x) + 1e-8)

        band_energy = norm(band_energy)
        sharpness = norm(sharpness)
        freq_diff = norm(freq_diff)
        narrow_ratio = norm(narrow_ratio)

        # ------------------------------------------------
        # Multi-cue whistle score
        # ------------------------------------------------
        score = (
            1.0 * band_energy +
            1.2 * sharpness +
            0.8 * narrow_ratio -
            1.0 * freq_diff
        )

        score = uniform_filter1d(score, size=5)

        if len(score) == 0:
            continue

        # ------------------------------------------------
        # Plateau center detection
        # ------------------------------------------------
        max_score = np.max(score)

        plateau = np.where(score > 0.7 * max_score)[0]

        if len(plateau) > 0:
            center_frame = int(np.mean(plateau))
        else:
            center_frame = int(np.argmax(score))

        # ------------------------------------------------
        # Fallback safety
        # ------------------------------------------------
        if not np.isfinite(score[center_frame]):
            center_frame = int(np.argmax(band_energy))

        t_peak = (s0 + center_frame * cfg.hop) / cfg.sr

        pre_sec = 0.5
        post_sec = 0.5

        new_start = max(0.0, t_peak - pre_sec)
        new_end = min(len(y) / cfg.sr, t_peak + post_sec)

        refined.append((new_start, new_end))

    return refined


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_window_features(y, start, end):
    s0 = int(start * cfg.sr)
    s1 = int(end * cfg.sr)
    seg = y[s0:s1]

    if len(seg) < cfg.n_fft:
        return None

    S = librosa.stft(seg, n_fft=cfg.n_fft, hop_length=cfg.hop)
    mag = np.abs(S)

    freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)
    band_mask = (freqs >= cfg.whistle_low) & (freqs <= cfg.whistle_high)

    if not np.any(band_mask):
        return None

    S_w = mag[band_mask]
    freqs_w = freqs[band_mask]

    # Narrow ratio
    mean_spectrum = S_w.mean(axis=1)
    peak_idx = np.argmax(mean_spectrum)
    center_freq = freqs_w[peak_idx]

    narrow_mask = (freqs >= center_freq - 150) & (freqs <= center_freq + 150)

    if np.any(narrow_mask):
        narrow_energy = mag[narrow_mask].mean()
        band_energy_full = mag[band_mask].mean() + 1e-8
        narrow_ratio = narrow_energy / band_energy_full
    else:
        narrow_ratio = 0.0

    band_energy = S_w.mean()
    total_energy = mag.mean() + 1e-8
    band_ratio = band_energy / total_energy

    if S_w.shape[1] < 3:
        freq_std = np.inf
    else:
        peak_bins = np.argmax(S_w, axis=0)
        peak_freqs = freqs_w[peak_bins]
        freq_std = np.std(peak_freqs)

    flatness_band = librosa.feature.spectral_flatness(S=S_w)[0].mean()

    band_peak = S_w.max(axis=0)
    band_mean = S_w.mean(axis=0) + 1e-8
    peak_prominence = np.mean(band_peak / band_mean)

    return {
        "band_ratio": band_ratio,
        "freq_std": freq_std,
        "flatness_band": flatness_band,
        "peak_prominence": peak_prominence,
        "band_energy": band_energy,
        "narrow_ratio": narrow_ratio,
    }


# ============================================================
# RULE-BASED SIFTER
# ============================================================

def rule_based_sifter(detections, y, stats):
    if stats is None:
        return []

    accepted = []

    for start, end in detections:
        feats = extract_window_features(y, start, end)
        if feats is None:
            continue

        z_band_ratio = (feats["band_ratio"] - stats["band_ratio"]["median"]) / stats["band_ratio"]["std"]
        z_flat = (feats["flatness_band"] - stats["flatness_band"]["median"]) / stats["flatness_band"]["std"]
        z_prom = (feats["peak_prominence"] - stats["peak_prominence"]["median"]) / stats["peak_prominence"]["std"]
        z_narrow = (feats["narrow_ratio"] - stats["narrow_ratio"]["median"]) / stats["narrow_ratio"]["std"]

        score = (
            1.0 * z_band_ratio +
            1.3 * z_prom +
            1.0 * z_narrow -
            0.8 * z_flat
        )

        if score > -0.1:
            accepted.append((start, end))
            continue

        # rescue
        if (
            feats["band_ratio"] > stats["band_ratio"]["median"] * 0.6 and
            feats["peak_prominence"] > stats["peak_prominence"]["median"] * 0.6
        ):
            accepted.append((start, end))

    return suppress_close_centers(accepted, 0.95)


# ============================================================
# UTIL
# ============================================================

def suppress_close_centers(detections, min_gap_sec):
    detections = sorted(detections, key=lambda x: (x[0] + x[1]) / 2)
    filtered = []

    for d in detections:
        center = (d[0] + d[1]) / 2
        if not filtered:
            filtered.append(d)
            continue

        prev_center = (filtered[-1][0] + filtered[-1][1]) / 2
        if abs(center - prev_center) > min_gap_sec:
            filtered.append(d)

    return filtered


# ============================================================
# EVALUATION
# ============================================================

def evaluate_candidate_hits(detections, gt):
    detections = sorted(detections, key=lambda x: (x[0] + x[1]) / 2)
    gt = sorted(gt, key=lambda g: g["t_anchor"])

    used_det = set()
    matched = 0
    offsets = []
    missed = []

    for g in gt:
        anchor = g["t_anchor"]

        best_idx = None
        best_offset = None

        for i, (start, end) in enumerate(detections):
            if i in used_det:
                continue

            if (start - ANCHOR_TOLERANCE) <= anchor <= (end + ANCHOR_TOLERANCE):
                center = (start + end) / 2
                offset = center - anchor

                # pick closest match
                if best_idx is None or abs(offset) < abs(best_offset):
                    best_idx = i
                    best_offset = offset

        if best_idx is not None:
            used_det.add(best_idx)
            matched += 1
            offsets.append(best_offset)
        else:
            missed.append(anchor)

    recall = matched / len(gt) if gt else 0

    true_positives = matched
    false_positives = len(detections) - len(used_det)

    return recall, offsets, missed, true_positives, false_positives


# ============================================================
# MATCH EVAL
# ============================================================
SNIPPET_SEC = 1.0
HALF_SNIPPET = SNIPPET_SEC / 2
DATASET_ROOT = r"E:\Volleyballey\cnn_dataset_by_match_best_cent"

def save_snippets(match_id, detections, gt_list, y):

    import soundfile as sf

    match_dir = os.path.join(DATASET_ROOT, match_id)
    pos_dir = os.path.join(match_dir, "pos")
    neg_dir = os.path.join(match_dir, "neg")

    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    snippet_id = 0

    for start, end in detections:

        center = (start + end) / 2
        s0 = int((center - HALF_SNIPPET) * cfg.sr)
        s1 = int((center + HALF_SNIPPET) * cfg.sr)

        if s0 < 0 or s1 > len(y):
            continue

        segment = y[s0:s1]

        # label using same tolerance
        min_dist = min(abs(center - g["t_anchor"]) for g in gt_list) if gt_list else np.inf

        if min_dist <= ANCHOR_TOLERANCE:
            label_dir = pos_dir

        elif min_dist > 1.0:
            label_dir = neg_dir

        else:
            # ambiguous zone → skip
            continue
        out_path = os.path.join(label_dir, f"{match_id}_{snippet_id}.wav")

        sf.write(out_path, segment, cfg.sr)
        snippet_id += 1

    print(f"Saved {snippet_id} snippets for {match_id}")



def evaluate_match(match_id, all_gt):
    print(f"\n==================== {match_id} ====================")

    video_path = os.path.join(VIDEO_DIR, f"{match_id}.mp4")
    y = load_audio_from_video(video_path, cfg.sr)

    # ---------- PASS 1 (rough band) ----------
    active = detect_active_frames(y)
    groups = group_frames(active)
    stage1 = extract_candidates(groups)

    # Estimate adaptive band
    low, high = estimate_whistle_band(y, stage1)
    print(f"Adaptive band: {int(low)}–{int(high)} Hz")

    # Update config
    original_low = cfg.whistle_low
    original_high = cfg.whistle_high

    cfg.whistle_low = low
    cfg.whistle_high = high

    # ---------- PASS 2 (adaptive band) ----------
    active = detect_active_frames(y)
    groups = group_frames(active)
    stage1 = extract_candidates(groups)
    refined = refine_candidates(y, stage1)

    gt_filtered = [g for g in all_gt if g["match_id"] == match_id]

    stats = compute_match_stats(refined, y)
    accepted = rule_based_sifter(refined, y, stats)
    save_snippets(match_id, accepted, gt_filtered, y)
    recall, offsets, missed, tp, fp = evaluate_candidate_hits(accepted, gt_filtered)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    explosion = len(accepted) / len(gt_filtered) if gt_filtered else 0

    print("GT:", len(gt_filtered))
    print("Stage1:", len(stage1))
    print("Accepted:", len(accepted))
    print("Recall:", round(recall, 3))
    print("Precision:", round(precision, 3))
    print("TP:", tp, "FP:", fp)
    print("Explosion:", round(explosion, 2))
    print("Missed:", len(missed))

    if offsets:
        offsets = np.array(offsets)
        print("Median abs offset:", round(np.median(np.abs(offsets)), 3))
        print("90th percentile:", round(np.percentile(np.abs(offsets), 90), 3))
        print("Mean offset:", round(np.mean(offsets), 3))

    # Restore original band (important for next match)
    cfg.whistle_low = original_low
    cfg.whistle_high = original_high

    return {
        "match": match_id,
        "recall": recall,
        "explosion": explosion
    }


def compute_match_stats(detections, y):
    all_feats = []

    for start, end in detections:
        feats = extract_window_features(y, start, end)
        if feats:
            all_feats.append(feats)

    if not all_feats:
        return None

    stats = {}
    keys = all_feats[0].keys()

    for k in keys:
        arr = np.array([f[k] for f in all_feats])
        stats[k] = {
            "median": np.median(arr),
            "std": np.std(arr) + 1e-8
        }

    return stats


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    with open(GT_PATH) as f:
        all_gt = json.load(f)

    match_ids = sorted(set(g["match_id"] for g in all_gt))
    results = []

    for m in match_ids:
        r = evaluate_match(m, all_gt)
        results.append(r)

    print("\n================ SUMMARY ================")
    for r in results:
        print(r)