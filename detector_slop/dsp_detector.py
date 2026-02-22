"""
Multi-Match DSP Whistle Detector Evaluation
Evaluates recall + explosion + centering across ALL matches in GT
"""

import json
import tempfile
import subprocess
import os
from dataclasses import dataclass
import random
import numpy as np
import librosa
from scipy.signal import find_peaks
import soundfile as sf
import os
# ============================================================
# CONFIG
# ============================================================

VIDEO_DIR = r"E:\Volleyballey\videos"
GT_PATH = r"E:\Volleyballey\detector_slop\whistles_all_anchored.json"
FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"

ANCHOR_TOLERANCE = 0.6  # seconds


@dataclass
class Config:
    sr: int = 22050
    n_fft: int = 2048
    hop: int = 128
    whistle_low: float = 3700
    whistle_high: float = 4300
    min_frames: int = 30
    max_gap_frames: int = 2


cfg = Config()


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio_from_video(video_path, target_sr):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    subprocess.run([
        FFMPEG, "-y", "-i", video_path,
        "-ac", "1",
        "-ar", str(target_sr),
        "-vn", tmp_path
    ], stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL,
       check=True)

    y, _ = librosa.load(tmp_path, sr=target_sr)
    os.remove(tmp_path)
    return y


# ============================================================
# DSP DETECTION
# ============================================================



from scipy.ndimage import uniform_filter1d

def detect_active_frames(y):

    # -----------------------------
    # STFT
    # -----------------------------
    S = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop)
    mag = np.abs(S)

    freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)
    band_mask = (freqs >= cfg.whistle_low) & (freqs <= cfg.whistle_high)

    # -----------------------------
    # Whistle-band features
    # -----------------------------
    band_mag = mag[band_mask]
    band_energy = band_mag.mean(axis=0)

    flatness = librosa.feature.spectral_flatness(S=mag)[0]

    band_peak = np.max(band_mag, axis=0)
    band_mean = np.mean(band_mag, axis=0)
    sharpness = band_peak / (band_mean + 1e-8)

    # -----------------------------
    # Robust normalization
    # -----------------------------
    band_energy_norm = (
        band_energy - np.median(band_energy)
    ) / (np.std(band_energy) + 1e-8)

    sharpness_norm = (
        sharpness - np.median(sharpness)
    ) / (np.std(sharpness) + 1e-8)

    flatness_norm = (
        flatness - np.median(flatness)
    ) / (np.std(flatness) + 1e-8)

    # -----------------------------
    # Temporal smoothing
    # -----------------------------
    band_energy_norm = uniform_filter1d(band_energy_norm, size=5)
    sharpness_norm   = uniform_filter1d(sharpness_norm, size=5)
    flatness_norm    = uniform_filter1d(flatness_norm, size=5)

    # -----------------------------
    # Continuous whistle score
    # -----------------------------
    score = (
        1.0 * band_energy_norm +
        1.0 * sharpness_norm -
        1.2 * flatness_norm
    )

    # -----------------------------
    # Hysteresis thresholds
    # -----------------------------
    START_TH = 0.82      # must be strong to start whistle
    CONTINUE_TH = 0.3   # relaxed to continue whistle

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

    return active, band_energy, score

def refine_candidates(
    y,
    detections,
    sr,
    n_fft,
    hop,
    whistle_low,
    whistle_high,
    out_window_sec=1.2,
    peak_prom=0.25,
    min_peak_dist_sec=0.30,
):
    refined = []

    min_dist_frames = int(min_peak_dist_sec * sr / hop)
    half_out = out_window_sec / 2.0

    for start, end in detections:

        s0 = int(start * sr)
        s1 = int(end * sr)

        if s1 - s0 < n_fft:
            continue

        seg = y[s0:s1]

        S = librosa.stft(seg, n_fft=n_fft, hop_length=hop)
        mag = np.abs(S)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mask = (freqs >= whistle_low) & (freqs <= whistle_high)

        if not np.any(mask):
            continue

        # -----------------------------
        # Whistle-band energy (max bin)
        # -----------------------------
        band_peak = np.max(mag[mask], axis=0)
        band_peak = uniform_filter1d(band_peak, size=5)

        # Robust normalization
        band_peak = (
            band_peak - np.median(band_peak)
        ) / (np.std(band_peak) + 1e-8)

        # -----------------------------
        # Peak detection
        # -----------------------------
        peaks, props = find_peaks(
            band_peak,
            prominence=peak_prom,
            distance=min_dist_frames
        )

        # Handle fallback safely
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(band_peak))])
            prominences = np.array([0.0])
        else:
            prominences = props.get("prominences", np.zeros(len(peaks)))

        flat = librosa.feature.spectral_flatness(S=mag)[0]
        flat = uniform_filter1d(flat, size=5)

        valid_peaks = []

        for i, p in enumerate(peaks):

            energy_score = band_peak[p]
            prom_score = prominences[i]
            tonal_score = 1.0 - flat[p]

            # Strong filtering
            if (
                    prom_score > 0.4 and
                    energy_score > 0.3 and
                    tonal_score > 0.3
            ):
                valid_peaks.append(p)

        # If nothing valid, fallback to strongest peak
        if len(valid_peaks) == 0:
            valid_peaks = [int(np.argmax(band_peak))]

        # Create window per peak
        for p in valid_peaks:
            t_peak = (s0 + p * hop) / sr

            new_start = max(0.0, t_peak - half_out)
            new_end = min(len(y) / sr, t_peak + half_out)

            refined.append((new_start, new_end))

    return refined

def group_frames(active):
    if not active:
        return []

    groups = []
    cur = [active[0]]

    for f in active[1:]:
        # allow larger gap directly here
        if f - cur[-1] <= 6:   # ~35ms
            cur.append(f)
        else:
            groups.append(cur)
            cur = [f]

    groups.append(cur)
    MAX_WHISTLE_SEC = 1.2
    max_frames = int(MAX_WHISTLE_SEC * cfg.sr / cfg.hop)

    filtered = []
    for g in groups:
        if len(g) <= max_frames:
            filtered.append(g)
        else:
            # split into chunks
            for i in range(0, len(g), max_frames):
                filtered.append(g[i:i + max_frames])

    return filtered

def merge_close_groups(groups, max_gap_frames=8):
    """
    Merge groups that are separated by small gaps.
    """

    if not groups:
        return []

    merged = [groups[0]]

    for g in groups[1:]:
        prev = merged[-1]

        # If start of current group is close to end of previous group
        if g[0] - prev[-1] <= max_gap_frames:
            merged[-1] = prev + g
        else:
            merged.append(g)

    return merged

def extract_candidates(groups, S_w, freqs_w, band_energy):

    detections = []

    pad_sec = 0.35
    pad_frames = int((pad_sec * cfg.sr) / cfg.hop)

    for g in groups:

        if len(g) < cfg.min_frames:
            continue

        start_frame = max(0, g[0] - pad_frames)
        end_frame   = g[-1] + pad_frames

        start_sec = start_frame * cfg.hop / cfg.sr
        end_sec   = end_frame   * cfg.hop / cfg.sr

        detections.append((start_sec, end_sec))

    return detections


# ============================================================
# EVALUATION
# ============================================================

def evaluate_frame_hits(active_frames, gt):
    frame_times = np.array(active_frames) * cfg.hop / cfg.sr
    matched = 0

    for g in gt:
        anchor = g["t_anchor"]
        if np.any(np.abs(frame_times - anchor) < ANCHOR_TOLERANCE):
            matched += 1

    return matched / len(gt)


def evaluate_group_hits(groups, gt):
    matched = 0

    for g in gt:
        anchor = g["t_anchor"]
        for group in groups:
            start = group[0] * cfg.hop / cfg.sr
            end   = group[-1] * cfg.hop / cfg.sr
            if (start - ANCHOR_TOLERANCE) <= anchor <= (end + ANCHOR_TOLERANCE):
                matched += 1
                break

    return matched / len(gt)


def evaluate_candidate_hits(detections, gt):
    matched = 0
    offsets = []
    missed = []

    for g in gt:
        anchor = g["t_anchor"]
        found = False

        for start, end in detections:
            if (start - ANCHOR_TOLERANCE) <= anchor <= (end + ANCHOR_TOLERANCE):
                center = (start + end) / 2
                offsets.append(center - anchor)
                matched += 1
                found = True
                break

        if not found:
            missed.append(anchor)

    recall = matched / len(gt)
    return recall, offsets, missed

def temporal_nms(detections, iou_threshold=0.2):
    """
    detections: list of (start, end)
    returns: filtered detections
    """

    if len(detections) == 0:
        return []

    dets = np.array(detections)

    # Sort by window length descending (or just keep original order)
    lengths = dets[:, 1] - dets[:, 0]
    order = np.argsort(-lengths)

    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        start_i, end_i = dets[i]

        rest = order[1:]
        if len(rest) == 0:
            break

        start_rest = dets[rest, 0]
        end_rest = dets[rest, 1]

        inter_start = np.maximum(start_i, start_rest)
        inter_end = np.minimum(end_i, end_rest)
        inter = np.maximum(0, inter_end - inter_start)

        union = (end_i - start_i) + (end_rest - start_rest) - inter
        iou = inter / (union + 1e-8)

        order = rest[iou < iou_threshold]

    return [detections[i] for i in keep]

def suppress_close_centers(detections, min_gap_sec=0.7):

    detections = sorted(detections, key=lambda x: (x[0]+x[1])/2)
    filtered = []

    for d in detections:
        center = (d[0]+d[1])/2

        if not filtered:
            filtered.append(d)
            continue

        prev_center = (filtered[-1][0]+filtered[-1][1])/2

        if abs(center - prev_center) > min_gap_sec:
            filtered.append(d)

    return filtered


def extract_cnn_dataset_clean(match_id, y, refined_detections, gt, out_root="cnn_dataset"):

    os.makedirs(out_root, exist_ok=True)
    pos_dir = os.path.join(out_root, "pos")
    neg_dir = os.path.join(out_root, "neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    metadata_path = os.path.join(out_root, "metadata.csv")

    WINDOW_SEC = 1.2
    MARGIN_SEC = 1.0
    SAFE_NEG_PER_POS = 3   # ratio of negatives to positives

    total_len_sec = len(y) / cfg.sr

    # =========================================================
    # 1️⃣ BUILD EXCLUSION INTERVALS
    # =========================================================

    exclusion_intervals = []

    # GT anchors
    for g in gt:
        center = g["t_anchor"]
        exclusion_intervals.append((
            center - (WINDOW_SEC/2 + MARGIN_SEC),
            center + (WINDOW_SEC/2 + MARGIN_SEC)
        ))

    # Also exclude refined whistle detections (double whistles etc.)
    for start, end in refined_detections:
        center = (start + end) / 2
        exclusion_intervals.append((
            center - (WINDOW_SEC/2 + MARGIN_SEC),
            center + (WINDOW_SEC/2 + MARGIN_SEC)
        ))

    # Clamp to audio bounds
    exclusion_intervals = [
        (max(0, s), min(total_len_sec, e))
        for s, e in exclusion_intervals
    ]

    # Sort & merge intervals
    exclusion_intervals.sort()
    merged = []

    for s, e in exclusion_intervals:
        if not merged:
            merged.append([s, e])
        else:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

    exclusion_intervals = merged

    # =========================================================
    # 2️⃣ SAVE POSITIVES (CENTERED ON GT)
    # =========================================================

    with open(metadata_path, "a") as meta:

        pos_count = 0

        for i, g in enumerate(gt):
            center = g["t_anchor"]
            start = center - WINDOW_SEC/2
            end   = center + WINDOW_SEC/2

            if start < 0 or end > total_len_sec:
                continue

            s0 = int(start * cfg.sr)
            s1 = int(end   * cfg.sr)

            clip = y[s0:s1]
            filename = f"{match_id}_pos_{i}_{round(center,3)}.wav"
            out_path = os.path.join(pos_dir, filename)

            sf.write(out_path, clip, cfg.sr)
            meta.write(f"{match_id},{filename},{start},{end},1\n")

            pos_count += 1

        # =========================================================
        # 3️⃣ SAMPLE NEGATIVES FROM SAFE GAPS
        # =========================================================

        neg_target = pos_count * SAFE_NEG_PER_POS
        neg_count = 0
        attempts = 0
        MAX_ATTEMPTS = neg_target * 20

        while neg_count < neg_target and attempts < MAX_ATTEMPTS:
            attempts += 1

            center = random.uniform(WINDOW_SEC/2, total_len_sec - WINDOW_SEC/2)
            start = center - WINDOW_SEC/2
            end   = center + WINDOW_SEC/2

            # Check overlap with exclusion zones
            overlaps = False
            for s, e in exclusion_intervals:
                if start < e and end > s:
                    overlaps = True
                    break

            if overlaps:
                continue

            s0 = int(start * cfg.sr)
            s1 = int(end   * cfg.sr)

            clip = y[s0:s1]
            filename = f"{match_id}_neg_{neg_count}_{round(center,3)}.wav"
            out_path = os.path.join(neg_dir, filename)

            sf.write(out_path, clip, cfg.sr)
            meta.write(f"{match_id},{filename},{start},{end},0\n")

            neg_count += 1
# ============================================================
# MAIN
# ============================================================

def extract_debug_snippet(y, sr, t, window=2.0):
    start = max(0, int((t - window/2) * sr))
    end   = min(len(y), int((t + window/2) * sr))
    return y[start:end]

def evaluate_match(match_id, all_gt):

    print(f"\n==================== {match_id} ====================")

    video_path = os.path.join(VIDEO_DIR, f"{match_id}.mp4")
    if not os.path.exists(video_path):
        print("Video not found. Skipping.")
        return None

    y = load_audio_from_video(video_path, cfg.sr)

    active, band_energy, ratio = detect_active_frames(y)

    groups = group_frames(active)
    groups = merge_close_groups(groups, max_gap_frames=8)
    S = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop)
    mag = np.abs(S)

    freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)
    band_mask = (freqs >= cfg.whistle_low) & (freqs <= cfg.whistle_high)

    S_w = mag[band_mask]
    freqs_w = freqs[band_mask]
    stage1_detections = extract_candidates(groups, S_w, freqs_w, band_energy)
    refined= refine_candidates(
        y=y,
        detections=stage1_detections,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop=cfg.hop,
        whistle_low=cfg.whistle_low,
        whistle_high=cfg.whistle_high,
        out_window_sec=1.2,
        peak_prom=0.25
    )
    #refined = temporal_nms(refined, iou_threshold=0.2)
    refined = suppress_close_centers(refined, min_gap_sec=0.9)
    gt_filtered = [g for g in all_gt if g["match_id"] == match_id]
    if not gt_filtered:
        print("No GT found.")
        return None

    frame_recall = evaluate_frame_hits(active, gt_filtered)
    group_recall = evaluate_group_hits(groups, gt_filtered)
    candidate_recall, offsets, missed = evaluate_candidate_hits(refined, gt_filtered)
    extract_cnn_dataset_clean(
        match_id=match_id,
        y=y,
        refined_detections=refined,
        gt=gt_filtered,
        out_root="cnn_dataset"
    )
    explosion = len(refined) / len(gt_filtered)

    print("GT whistles:", len(gt_filtered))
    print("Stage1 Candidates:", len(stage1_detections))
    print("Refined Candidates:", len(refined))
    print("Frame recall:", round(frame_recall, 3))
    print("Group recall:", round(group_recall, 3))
    print("Candidate recall:", round(candidate_recall, 3))
    print("Explosion ratio:", round(explosion, 2))
    print("Missed count:", len(missed))
    if len(missed) > 0:
        debug_dir = os.path.join("debug_stage1", match_id)
        os.makedirs(debug_dir, exist_ok=True)

        print("Saving Stage-1 windows for first 10 misses...")

        for i, anchor in enumerate(missed[:10]):
            s0 = int(max(0, (anchor - 2.0)) * cfg.sr)
            s1 = int(min(len(y) / cfg.sr, (anchor + 2.0)) * cfg.sr)

            snippet = y[s0:s1]

            out_path = os.path.join(
                debug_dir,
                f"missed_anchor_{i}_{round(anchor, 2)}.wav"
            )

            sf.write(out_path, snippet, cfg.sr)

    if offsets:
        offsets = np.array(offsets)
        print("Median abs offset:", round(np.median(np.abs(offsets)), 3))
        print("90th percentile:", round(np.percentile(np.abs(offsets), 90), 3))

    return {
        "match": match_id,
        "recall": candidate_recall,
        "explosion": explosion
    }


if __name__ == "__main__":

    print("Loading GT...")
    with open(GT_PATH) as f:
        all_gt = json.load(f)

    match_ids = sorted(set(g["match_id"] for g in all_gt))

    results = []

    for m in match_ids:
        r = evaluate_match(m, all_gt)
        if r:
            results.append(r)

    print("\n==================== SUMMARY ====================")
    for r in results:
        print(r)


