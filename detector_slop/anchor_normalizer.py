import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


DEFAULT_VIDEO_AUDIO_PATHS = {
    "match3": r"D:\Volleyballey\videos\match3.mp4",
    "match4": r"D:\Volleyballey\videos\match4.mp4",
    "match11": r"D:\Volleyballey\videos\match11.mp4",
}


def load_audio(path: str, sr: int):
    y, _ = librosa.load(path, sr=sr)
    return y


def peak_band_anchor(
    y: np.ndarray,
    t_raw: float,
    sr: int,
    band_low: float,
    band_high: float,
    search_radius: float,
    n_fft: int,
    hop_length: int,
):
    start_t = max(0.0, t_raw - search_radius)
    end_t = max(start_t, t_raw + search_radius)

    start_s = int(start_t * sr)
    end_s = int(end_t * sr)

    if end_s - start_s < n_fft:
        pad = n_fft - (end_s - start_s)
        end_s = min(len(y), end_s + pad)

    segment = y[start_s:end_s]
    if len(segment) < n_fft:
        return t_raw

    stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs >= band_low) & (freqs <= band_high)

    if not np.any(mask):
        return t_raw

    band_energy = mag[mask].mean(axis=0)
    if band_energy.size == 0:
        return t_raw

    peak_frame = int(np.argmax(band_energy))
    # librosa.stft uses center=True by default, so frame index already maps to
    # center time at t * hop_length relative to this segment.
    frame_center_samples = peak_frame * hop_length
    t_anchor = (start_s + frame_center_samples) / sr

    return float(max(0.0, min(t_anchor, len(y) / sr)))


def normalize_annotations(
    input_json: Path,
    output_json: Path,
    video_audio_paths: dict,
    sr: int,
    band_low: float,
    band_high: float,
    search_radius: float,
    n_fft: int,
    hop_length: int,
):
    with open(input_json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    by_match = {}
    for row in rows:
        by_match.setdefault(row["match_id"], []).append(row)

    out_rows = []

    for match_id, items in by_match.items():
        if match_id not in video_audio_paths:
            raise KeyError(f"No audio path configured for match_id={match_id}")

        y = load_audio(video_audio_paths[match_id], sr=sr)

        for row in tqdm(items, desc=f"Anchoring {match_id}"):
            t_raw = row.get("t_raw", row.get("time"))
            if t_raw is None:
                raise KeyError(
                    f"Row missing both 't_raw' and 'time': whistle_id={row.get('whistle_id')}"
                )

            t_anchor = peak_band_anchor(
                y=y,
                t_raw=float(t_raw),
                sr=sr,
                band_low=band_low,
                band_high=band_high,
                search_radius=search_radius,
                n_fft=n_fft,
                hop_length=hop_length,
            )

            updated = dict(row)
            updated["t_raw"] = round(float(t_raw), 3)
            updated["t_anchor"] = round(float(t_anchor), 3)
            out_rows.append(updated)

    out_rows.sort(key=lambda x: (x["match_id"], x["t_anchor"]))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=4)

    print(f"Saved anchored labels → {output_json} ({len(out_rows)} rows)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Create canonical whistle anchors as peak energy in a target frequency band."
    )
    p.add_argument("--input", default="whistles_all.json", help="Input annotations JSON")
    p.add_argument(
        "--output",
        default="whistles_all_anchored.json",
        help="Output JSON with both t_raw and t_anchor",
    )
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--band-low", type=float, default=3700.0)
    p.add_argument("--band-high", type=float, default=4300.0)
    p.add_argument("--search-radius", type=float, default=0.25)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()

    normalize_annotations(
        input_json=Path(args.input),
        output_json=Path(args.output),
        video_audio_paths=DEFAULT_VIDEO_AUDIO_PATHS,
        sr=args.sr,
        band_low=args.band_low,
        band_high=args.band_high,
        search_radius=args.search_radius,
        n_fft=args.n_fft,
        hop_length=args.hop,
    )


if __name__ == "__main__":
    main()
