"""Runtime-only model definitions used by the Volleylitics pipeline.

Training code and checkpoints are intentionally kept separate.  Keeping the
small inference definitions here makes a fresh checkout runnable even though
large local model/training folders are ignored by Git.
"""

import librosa
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18


SAMPLE_RATE = 22_050
N_MELS = 96
N_FFT = 2_048
HOP_LENGTH = 128


def wav_to_logmel_from_audio(audio: np.ndarray) -> np.ndarray:
    """Convert a one-second mono waveform to the six-channel model input."""

    mel_full = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=SAMPLE_RATE // 2,
    )
    logmel_full = librosa.power_to_db(mel_full)
    logmel_full = (logmel_full - np.mean(logmel_full)) / (
        np.std(logmel_full) + 1e-8
    )

    mel_band = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=2_500,
        fmax=6_000,
    )
    logmel_band = librosa.power_to_db(mel_band)
    logmel_band = (logmel_band - np.mean(logmel_band)) / (
        np.std(logmel_band) + 1e-8
    )

    return np.stack(
        [
            logmel_full,
            librosa.feature.delta(logmel_full),
            librosa.feature.delta(logmel_full, order=2),
            logmel_band,
            librosa.feature.delta(logmel_band),
            librosa.feature.delta(logmel_band, order=2),
        ],
        axis=0,
    ).astype(np.float32)


class ResNetWhistle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            6, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone.fc = nn.Linear(512, 1)

    def forward(self, inputs):
        return self.backbone(inputs).squeeze()


def build_rally_model():
    """Build the exact ResNet layout stored in rally_model_best.pth."""

    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2),
    )
    return model
