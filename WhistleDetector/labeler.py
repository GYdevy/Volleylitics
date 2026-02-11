import os
import csv
import subprocess
from pathlib import Path
import librosa
import numpy as np

CLIPS_DIR = Path(r"D:\Volleyballey\WhistleDetector\clips_match6")
OUT_CSV = Path(r"D:\Volleyballey\WhistleDetector\csv_For_match6.csv")

SR = 22050
N_FFT = 2048
HOP = 128

WHISTLE_LOW = 3600
WHISTLE_HIGH = 4400


# ============================================================
# FEATURE EXTRACTION (same as model!)
# ============================================================
def extract_features(snippet_path):
    y, sr = librosa.load(snippet_path, sr=SR)
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr)
    mask = (freqs >= WHISTLE_LOW) & (freqs <= WHISTLE_HIGH)
    S_w = S_db[mask, :]

    rms = librosa.feature.rms(y=y).mean()
    flatness = librosa.feature.spectral_flatness(y=y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

    band_energy = S_w.mean()
    peak_freq = freqs[mask][np.argmax(S_w.mean(axis=1))]

    frame_energy = np.percentile(S_w, 75, axis=0)
    ridge_length = np.sum(frame_energy > (frame_energy.mean() + frame_energy.std()))

    grad_t = np.abs(np.diff(S_w, axis=1)).mean()
    grad_f = np.abs(np.diff(S_w, axis=0)).mean()

    return [
        rms, flatness, centroid, band_energy,
        peak_freq, ridge_length, grad_t, grad_f
    ]


# ============================================================
# LOAD EXISTING LABELED DATA
# ============================================================
existing = set()

if OUT_CSV.exists():
    with open(OUT_CSV, "r", newline="", encoding="utf8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            existing.add(row[0])

# ============================================================
# MAIN LABELING LOOP
# ============================================================
clips = sorted(CLIPS_DIR.glob("*.mp4"))
print(f"Found {len(clips)} clips, already labeled {len(existing)}.\n")

write_header = not OUT_CSV.exists()

with open(OUT_CSV, "a", newline="", encoding="utf8") as f:
    writer = csv.writer(f)

    if write_header:
        writer.writerow([
            "filename", "rms", "flatness", "centroid", "band_energy",
            "peak_freq", "ridge_length", "grad_t", "grad_f", "label"
        ])

    for clip in clips:
        fname = clip.name
        if fname in existing:
            continue

        print(f"\n=== {fname} ===")
        subprocess.Popen(["start", "", clip.as_posix()], shell=True)

        while True:
            ans = input("Whistle? (1=yes, 0=no, s=skip, q=quit): ").strip().lower()
            if ans in ["1", "0", "s"]:
                break
            if ans == "q":
                print("Exiting.")
                exit()
            print("Invalid input.")

        if ans == "s":
            continue

        label = int(ans)

        print("Extracting features...")
        feats = extract_features(clip)

        writer.writerow([fname] + feats + [label])
        print("Saved.")


print("\nDONE! Appended new labeled rows to:")
print(OUT_CSV)
