# WhistleDetector/detection/energy.py

import numpy as np
import librosa

from WhistleDetector.config import (
    SR, N_FFT, HOP,
    WHISTLE_LOW, WHISTLE_HIGH,
    HOLD_TIME_SEC
)

def detect_active_frames(y):
    """
    Returns:
      active_frames : list[int]
      S_w           : whistle-band spectrogram (freq x time)
      freqs_w       : frequencies for S_w
    """

    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    freqs = librosa.fft_frequencies(sr=SR)
    mask = (freqs >= WHISTLE_LOW) & (freqs <= WHISTLE_HIGH)

    S_w = S_db[mask]
    freqs_w = freqs[mask]

    # ---- energy ----
    energy = np.sqrt(np.mean((10 ** (S_w / 20)) ** 2, axis=0))
    energy_db = 20 * np.log10(energy + 1e-12)

    HIGH_THR = np.percentile(energy_db, 70)
    LOW_THR  = np.percentile(energy_db, 45)

    hold_frames = int(HOLD_TIME_SEC / (HOP / SR))

    active = []
    in_event = False
    hold = 0

    for i, e in enumerate(energy_db):
        if not in_event:
            if e > HIGH_THR:
                in_event = True
                hold = hold_frames
                active.append(i)
        else:
            if e > LOW_THR:
                hold = hold_frames
                active.append(i)
            else:
                hold -= 1
                if hold > 0:
                    active.append(i)
                else:
                    in_event = False

    return active, S_w, freqs_w
