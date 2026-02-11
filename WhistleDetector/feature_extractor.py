import numpy as np
import librosa

SR = 22050
N_FFT = 2048
HOP = 128
WHISTLE_LOW = 3600
WHISTLE_HIGH = 4400

def extract_whistle_features(y):
    # RMS, flatness, centroid
    rms = librosa.feature.rms(y=y).mean()
    flatness = librosa.feature.spectral_flatness(y=y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR).mean()

    # STFT
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    mask = (freqs >= WHISTLE_LOW) & (freqs <= WHISTLE_HIGH)
    S_w = S_db[mask, :]

    # Band features
    band_energy = S_w.mean()

    peak_idx = np.argmax(S_w.mean(axis=1))
    peak_freq = freqs[mask][peak_idx]

    # Ridge length
    frame_energy = np.percentile(S_w, 75, axis=0)
    ridge_length = np.sum(frame_energy > (frame_energy.mean() + frame_energy.std()))

    # Gradients
    grad_t = np.abs(np.diff(S_w, axis=1)).mean()
    grad_f = np.abs(np.diff(S_w, axis=0)).mean()

    return np.array([
        rms, flatness, centroid, band_energy,
        peak_freq, ridge_length, grad_t, grad_f
    ])
