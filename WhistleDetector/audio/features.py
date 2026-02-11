import numpy as np
import librosa
from WhistleDetector.config import SR, N_FFT, HOP, WHISTLE_LOW, WHISTLE_HIGH

def extract_features(y):
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))

    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    freqs = librosa.fft_frequencies(sr=SR)
    mask = (freqs >= WHISTLE_LOW) & (freqs <= WHISTLE_HIGH)
    S_w = S_db[mask]

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
        y=librosa.effects.harmonic(y), sr=SR
    ).mean()

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=5).mean(axis=1)

    physics = {
        "flatness": flat,
        "ridge": ridge,
        "grad_f": grad_f,
        "centroid": cent,
        "peak_freq": peak_f,
    }

    X = np.array([
        rms, flat, cent, band_E, peak_f,
        ridge, grad_t, grad_f,
        rolloff, bw, zcr, contrast, tonnetz,
        *mfcc
    ])

    return physics, X


def physics_suspect(feats):
    score = 0
    if feats["flatness"] > 0.030:
        score += 1
    if feats["ridge"] < 4:
        score += 1
    if feats["grad_f"] < 4.3:
        score += 1
    return score >= 2

def peak_freq_std(S_w, freqs):
    """
    Std of peak frequency over time (Hz).
    Low for whistles, high for voices/chants.
    """
    if S_w.shape[1] < 3:
        return np.inf

    peak_bins = np.argmax(S_w, axis=0)
    peak_freqs = freqs[peak_bins]
    return np.std(peak_freqs)