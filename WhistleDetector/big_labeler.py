import os
import csv
import subprocess
from pathlib import Path
import librosa
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

CLIPS_DIR = Path(r"D:\Volleyballey\WhistleDetector\ambiguous_matches\ambiguous_match12")
OUT_CSV = Path(r"D:\Volleyballey\WhistleDetector\ambiguous_match12_full_features.csv")

SR = 22050
N_FFT = 2048
HOP = 128

WHISTLE_LOW = 3600
WHISTLE_HIGH = 4400


# ============================================================
# FEATURE EXTRACTION (18 FEATURES, same as master CSV)
# ============================================================
def extract_all_features(path):
    y, sr = librosa.load(path, sr=SR)

    # STFT
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr)
    mask = (freqs >= WHISTLE_LOW) & (freqs <= WHISTLE_HIGH)

    if S_db.shape[1] > 0:
        S_w = S_db[mask, :]
    else:
        S_w = np.zeros((len(freqs[mask]), 1))

    # Original features
    rms = float(librosa.feature.rms(y=y).mean())
    flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    band_energy = float(S_w.mean()) if S_w.size > 0 else 0.0
    peak_freq = float(freqs[mask][np.argmax(S_w.mean(axis=1))]) if S_w.size > 0 else 0.0

    if S_w.size > 0:
        frame_energy = np.percentile(S_w, 75, axis=0)
        ridge_length = int(np.sum(frame_energy > (frame_energy.mean() + frame_energy.std())))
    else:
        ridge_length = 0

    grad_t = float(np.abs(np.diff(S_w, axis=1)).mean()) if S_w.shape[1] > 1 else 0.0
    grad_f = float(np.abs(np.diff(S_w, axis=0)).mean()) if S_w.shape[0] > 1 else 0.0

    # New extra features
    rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    contrast = float(librosa.feature.spectral_contrast(S=S_mag, sr=sr).mean())

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz_mean = float(tonnetz.mean())

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    mfcc_means = [float(v) for v in mfcc.mean(axis=1)]

    return [
        rms, flatness, centroid, band_energy, peak_freq,
        ridge_length, grad_t, grad_f,
        rolloff, bandwidth, zcr, contrast, tonnetz_mean,
        *mfcc_means
    ]


# ============================================================
# CHECK ALREADY LABELED
# ============================================================
existing = set()

if OUT_CSV.exists():
    with open(OUT_CSV, "r", newline="", encoding="utf8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            existing.add(row[0])


# ============================================================
# START LABELING
# ============================================================
clips = sorted(CLIPS_DIR.glob("*.mp4"))
print(f"Found {len(clips)} clips, already labeled {len(existing)}.\n")

write_header = not OUT_CSV.exists()

with open(OUT_CSV, "a", newline="", encoding="utf8") as f:
    writer = csv.writer(f)

    if write_header:
        writer.writerow([
            "filename",
            "rms","flatness","centroid","band_energy","peak_freq",
            "ridge_length","grad_t","grad_f",
            "rolloff","bandwidth","zcr","contrast","tonnetz",
            "mfcc1","mfcc2","mfcc3","mfcc4","mfcc5",
            "label"
        ])

    for clip in clips:
        fname = clip.name
        if fname in existing:
            continue

        print(f"\n=== {fname} ===")
        subprocess.Popen(["start", "", clip.as_posix()], shell=True)

        while True:
            ans = input("Whistle? (1=yes, 0=no, s=skip, q=quit): ").strip().lower()
            if ans in ("1","0","s"):
                break
            if ans == "q":
                print("Exiting.")
                exit()
            print("Invalid input.")

        if ans == "s":
            continue

        label = int(ans)

        print("Extracting ALL 18 features...")
        feats = extract_all_features(clip)

        # FULL VALUE PRECISION — EXACT MATCH TO MASTER CSV
        row = [fname] + [repr(v) for v in feats] + [label]

        writer.writerow(row)
        print("Saved!")


print("\n===================================================")
print("DONE! New labeled match6 rows saved to:")
print(OUT_CSV)
print("===================================================")
