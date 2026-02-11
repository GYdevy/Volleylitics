import numpy as np
import librosa
from WhistleDetector.config import (
    MIN_FRAMES,
    PAD_BEFORE,
    PAD_AFTER,
    MAX_FLATNESS,
    MAX_CENTROID,
    HOP,
    SR
)

from WhistleDetector.audio.tonality import split_by_tonality
from WhistleDetector.audio.features import peak_freq_std
from WhistleDetector.audio.utils import band_width_hz
from WhistleDetector.audio.utils import fmt_time


def extract_candidates(groups, y, S_w, freqs_w):
    detections = []

    for g in groups:
        if len(g) < MIN_FRAMES:
            continue

        tonal_groups = split_by_tonality(g, S_w, MIN_FRAMES)
        group_candidates = []

        for tg in tonal_groups:
            best_score = -np.inf
            core_frame = None

            for frame in tg:
                spec = S_w[:, frame]
                peak = spec.max()
                if peak < -60:
                    continue

                active_bins = np.sum(spec > (peak - 6))
                score = peak - active_bins * 2.0

                if score > best_score:
                    best_score = score
                    core_frame = frame

            if core_frame is None:
                continue

            half = MIN_FRAMES // 2
            s = max(tg[0], core_frame - half)
            e = min(tg[-1], core_frame + half)

            if (e - s + 1) < MIN_FRAMES:
                continue

            start_sec = s * HOP / SR
            end_sec   = e * HOP / SR

            segment = S_w[:, s:e+1]

            ridge = np.sum(
                np.percentile(segment, 75, axis=0)
                > np.mean(segment) + np.std(segment)
            )
            grad_f = np.abs(np.diff(segment, axis=0)).mean()

            peak_std = peak_freq_std(segment, freqs_w)
            bw_hz = band_width_hz(segment, freqs_w)

            # ✅ FIXED AUDIO SNIPPET
            start_s = max(0, int((start_sec - PAD_BEFORE) * SR))
            end_s   = min(len(y), int((end_sec + PAD_AFTER) * SR))
            snippet = y[start_s:end_s]

            if len(snippet) > 0:
                flat = librosa.feature.spectral_flatness(y=snippet).mean()
                cent = librosa.feature.spectral_centroid(y=snippet, sr=SR).mean()
                noisy = (flat > MAX_FLATNESS) or (cent > MAX_CENTROID)
            else:
                noisy = True

            group_candidates.append({
                "start": start_sec,
                "end": end_sec,
                "audio": snippet,
                "core_score": best_score,
                "grad_f": grad_f,
                "ridge": ridge,
                "peak_std": peak_std,
                "bandwidth_hz": bw_hz,
                "noisy": noisy,
            })

        if not group_candidates:
            continue

        best = max(group_candidates, key=lambda d: d["core_score"])
        detections.append(best)

    return detections
