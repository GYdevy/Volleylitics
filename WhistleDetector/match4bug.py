import numpy as np
import librosa
import subprocess
import tempfile
import os
from pathlib import Path

# ===============================
# CONFIG — CHANGE THESE
# ===============================
VIDEO_PATH = r"D:\Volleyballey\videos\match4.mp4"

# whistle timestamp (seconds)
START_SEC = 8 * 60 + 37.96
END_SEC   = 8 * 60 + 39.00

# audio
SR = 22050
N_FFT = 2048
HOP = 128

WHISTLE_LOW = 3700
WHISTLE_HIGH = 4200

PAD_BEFORE = 0.1
PAD_AFTER  = 0.1

FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"

# optional CSV output
CSV_OUT = r"D:\Volleyballey\WhistleDetector\debug_whistle_features.csv"

# ===============================
# FEATURE NAMES (MUST MATCH TRAINING ORDER)
# ===============================
FEATURE_NAMES = [
    "rms",
    "flatness",
    "centroid",
    "band_energy",
    "peak_freq",
    "ridge",
    "grad_t",
    "grad_f",
    "rolloff",
    "bandwidth",
    "zcr",
    "contrast",
    "tonnetz",
    "mfcc1",
    "mfcc2",
    "mfcc3",
    "mfcc4",
    "mfcc5",
]

# ===============================
# LOAD AUDIO SNIPPET
# ===============================
def load_snippet(start, end):
    duration = end - start
    safe_start = max(0, start - PAD_BEFORE)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    subprocess.run([
        FFMPEG, "-y",
        "-ss", str(safe_start),
        "-i", VIDEO_PATH,
        "-t", str(duration + PAD_BEFORE + PAD_AFTER),
        "-vn",
        "-ac", "1",
        "-ar", str(SR),
        tmp_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    y, _ = librosa.load(tmp_path, sr=SR)
    os.remove(tmp_path)
    return y

# ===============================
# FEATURE EXTRACTION (18 FEATURES)
# ===============================
def extract_features(y):
    if len(y) < 4096:
        y = np.pad(y, (0, 4096 - len(y)))

    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    mask = (freqs >= WHISTLE_LOW) & (freqs <= WHISTLE_HIGH)
    S_w = S_db[mask]

    # core features
    rms = librosa.feature.rms(y=y).mean()
    flat = librosa.feature.spectral_flatness(y=y).mean()
    cent = librosa.feature.spectral_centroid(y=y, sr=SR).mean()
    band_E = S_w.mean()
    peak_f = freqs[mask][np.argmax(S_w.mean(axis=1))]

    fe = np.percentile(S_w, 75, axis=0)
    ridge = np.sum(fe > (fe.mean() + fe.std()))

    grad_t = np.abs(np.diff(S_w, axis=1)).mean()
    grad_f = np.abs(np.diff(S_w, axis=0)).mean()

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR).mean()
    bw = librosa.feature.spectral_bandwidth(y=y, sr=SR).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    contrast = librosa.feature.spectral_contrast(S=S_mag, sr=SR).mean()

    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(y),
        sr=SR
    ).mean()

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=5).mean(axis=1)

    return np.array([
        rms, flat, cent, band_E, peak_f,
        ridge, grad_t, grad_f,
        rolloff, bw, zcr, contrast, tonnetz,
        *mfcc
    ])

# ===============================
# MAIN
# ===============================
print("\n=== INSPECTING WHISTLE ===")
print(f"Video: {VIDEO_PATH}")
print(f"Time:  {START_SEC:.2f} → {END_SEC:.2f}\n")

y = load_snippet(START_SEC, END_SEC)
features = extract_features(y)

# ---- PRINT CSV STYLE ----
print(",".join(FEATURE_NAMES))
print(",".join(f"{v:.6f}" for v in features))

# ---- APPEND TO CSV FILE ----
write_header = not Path(CSV_OUT).exists()

with open(CSV_OUT, "a") as f:
    if write_header:
        f.write(",".join(FEATURE_NAMES) + "\n")
    f.write(",".join(repr(float(v)) for v in features) + "\n")

print(f"\nSaved to: {CSV_OUT}")
print("================================\n")
