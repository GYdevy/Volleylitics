import librosa
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# ===============================
# CONFIG
# ===============================
SR = 22050
N_FFT = 2048
HOP = 128
WHISTLE_LOW = 3600
WHISTLE_HIGH = 4400

# 🔥 CHANGE THIS to any clip you want:
CLIP_PATH = Path(r"/WhistleDetector/clips/clips_match8_SUPER\match8_whistle_0059.mp4")


def extract_features(path):
    y, sr = librosa.load(path, sr=SR)

    # === STFT ===
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr)
    mask = (freqs >= WHISTLE_LOW) & (freqs <= WHISTLE_HIGH)
    S_w = S_db[mask, :] if S_db.shape[1] > 0 else np.zeros((len(freqs[mask]), 1))

    # === Original features ===
    rms = float(librosa.feature.rms(y=y).mean())
    flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    band_energy = float(S_w.mean()) if S_w.size > 0 else 0.0
    peak_freq = float(freqs[mask][np.argmax(S_w.mean(axis=1))]) if S_w.size > 0 else 0.0

    # Ridge length
    if S_w.size > 0:
        frame_energy = np.percentile(S_w, 75, axis=0)
        ridge_length = int(np.sum(frame_energy > (frame_energy.mean() + frame_energy.std())))
    else:
        ridge_length = 0

    grad_t = float(np.abs(np.diff(S_w, axis=1)).mean()) if S_w.shape[1] > 1 else 0.0
    grad_f = float(np.abs(np.diff(S_w, axis=0)).mean()) if S_w.shape[0] > 1 else 0.0

    # === Extra features ===
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


# ===============================
# MAIN — HARDCODED
# ===============================
if not CLIP_PATH.exists():
    print(f"❌ File not found: {CLIP_PATH}")
else:
    feats = extract_features(CLIP_PATH)
    values = [repr(v) for v in feats]
    csv_row = ",".join([CLIP_PATH.name] + values)

    print("\n=== CSV ROW OUTPUT ===")
    print(csv_row)
    print("======================")
