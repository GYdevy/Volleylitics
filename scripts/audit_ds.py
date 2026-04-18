import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
from tqdm import tqdm
import shutil
import re
# ==============================
# CONFIG
# ==============================

DATASET_DIR = r"E:\Volleyballey\cnn_dataset_by_match"
MODEL_PATH = "whistle_cnn.pth"

SR = 22050
N_MELS = 96
N_FFT = 2048
HOP = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIGH_NEG_THRESHOLD = 0.80   # suspicious negatives
LOW_POS_THRESHOLD  = 0.30   # suspicious positives

# ==============================
# MODEL DEFINITION
# ==============================

def get_match_id(path):
    fname = os.path.basename(path)
    match = re.search(r"(match\d+)", fname)
    return match.group(1) if match else None

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),   # shrink time only

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze()

# ==============================
# FEATURE EXTRACTION
# ==============================

def wav_to_logmel(path):
    y, _ = librosa.load(path, sr=SR)

    # FULL BAND
    mel_full = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=0,
        fmax=SR // 2
    )

    logmel_full = librosa.power_to_db(mel_full)
    logmel_full = (logmel_full - np.mean(logmel_full)) / (np.std(logmel_full) + 1e-8)

    delta_full = librosa.feature.delta(logmel_full)
    delta2_full = librosa.feature.delta(logmel_full, order=2)

    # WHISTLE BAND
    mel_band = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=2500,
        fmax=6000
    )

    logmel_band = librosa.power_to_db(mel_band)
    logmel_band = (logmel_band - np.mean(logmel_band)) / (np.std(logmel_band) + 1e-8)

    delta_band = librosa.feature.delta(logmel_band)
    delta2_band = librosa.feature.delta(logmel_band, order=2)

    stacked = np.stack([
        logmel_full,
        delta_full,
        delta2_full,
        logmel_band,
        delta_band,
        delta2_band
    ], axis=0)

    return stacked.astype(np.float32)

# ==============================
# LOAD MODEL
# ==============================

model = TinyCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==============================
# LOAD FILES
# ==============================

pos_files = glob.glob(os.path.join(DATASET_DIR, "match*", "pos", "*.wav"))
neg_files = glob.glob(os.path.join(DATASET_DIR, "match*", "neg", "*.wav"))

print("Total POS:", len(pos_files))
print("Total NEG:", len(neg_files))

# ==============================
# INFERENCE
# ==============================

suspicious_neg = []
weak_pos = []

print("\nScanning NEG files...")
for f in tqdm(neg_files):
    mel = torch.tensor(wav_to_logmel(f)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(mel)).item()

    if prob > HIGH_NEG_THRESHOLD:
        suspicious_neg.append((f, prob))

print("\nScanning POS files...")
for f in tqdm(pos_files):
    mel = torch.tensor(wav_to_logmel(f)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(mel)).item()

    if prob < LOW_POS_THRESHOLD:
        weak_pos.append((f, prob))

# ==============================
# RESULTS
# ==============================

print("\n============================")
print("High-prob NEG:", len(suspicious_neg))
print("Low-prob POS:", len(weak_pos))

# Sort for inspection
suspicious_neg.sort(key=lambda x: -x[1])
weak_pos.sort(key=lambda x: x[1])

print("\nTop 10 suspicious NEG:")
for f, p in suspicious_neg[:50]:
    print(f"{p:.3f}  {f}")

print("\nTop 10 weak POS:")
for f, p in weak_pos[:50]:
    print(f"{p:.3f}  {f}")


AUDIT_DIR = os.path.join(DATASET_DIR, "audit")
HIGH_NEG_DIR = os.path.join(AUDIT_DIR, "high_prob_neg")
LOW_POS_DIR = os.path.join(AUDIT_DIR, "low_prob_pos")

os.makedirs(HIGH_NEG_DIR, exist_ok=True)
os.makedirs(LOW_POS_DIR, exist_ok=True)

TOP_K = 50

print("\nCopying top suspicious files for manual review...")

# Copy top high-prob negatives
for f, p in suspicious_neg[:TOP_K]:
    new_name = f"{p:.3f}_" + os.path.basename(f)
    dst = os.path.join(HIGH_NEG_DIR, new_name)
    shutil.move(f, dst)

# Copy top low-prob positives
for f, p in weak_pos[:TOP_K]:
    new_name = f"{p:.3f}_" + os.path.basename(f)
    dst = os.path.join(LOW_POS_DIR, new_name)
    shutil.move(f, dst)

print("Audit files saved to:", AUDIT_DIR)