import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
# ==============================
# CONFIG
# ==============================


DATASET_DIR = r"E:\Volleyballey\cnn_dataset_by_match_best_cent"
SR = 22050
N_MELS = 96
N_FFT = 2048
HOP = 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT = {
    "train": ["match1","match7","match2","match9","match8","match14","match15","match16"],
    "val":   ["match11","match4","match10"],
    "test":  ["match3","match13"],
}
# ==============================
# DATASET
# ==============================

def wav_to_logmel(path, augment=False):
    y, _ = librosa.load(path, sr=SR)

    if augment:

        # ------------------------------------------------
        # 1. Background noise (crowd / commentary)
        # ------------------------------------------------
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.003, len(y))
            y = y + noise

        # ------------------------------------------------
        # 2. Random gain (broadcast volume variation)
        # ------------------------------------------------
        if np.random.rand() < 0.4:
            gain = np.random.uniform(0.85, 1.15)
            y = y * gain

        # ------------------------------------------------
        # 3. Mis-centering (DSP detector offset)
        # ------------------------------------------------
        if np.random.rand() < 0.35:
            shift = np.random.randint(-2000, 2000)  # ~90ms
            y = np.roll(y, shift)

        # ------------------------------------------------
        # 4. Truncated whistle (missing onset or tail)
        # ------------------------------------------------
        if np.random.rand() < 0.3:

            cut_ratio = np.random.uniform(0.2, 0.6)
            cut_samples = int(len(y) * cut_ratio)

            if np.random.rand() < 0.5:
                # remove onset
                y[:cut_samples] = 0
            else:
                # remove tail
                y[-cut_samples:] = 0

        # ------------------------------------------------
        # 5. Fade tail (real whistle decay)
        # ------------------------------------------------
        if np.random.rand() < 0.3:
            fade = np.linspace(1.0, 0.0, len(y))
            y = y * fade

        # ------------------------------------------------
        # 6. Loud crowd burst masking whistle
        # ------------------------------------------------
        if np.random.rand() < 0.25:
            burst_len = int(len(y) * np.random.uniform(0.1, 0.3))
            start = np.random.randint(0, len(y) - burst_len)

            burst = np.random.normal(0, 0.02, burst_len)
            y[start:start + burst_len] += burst

        # ------------------------------------------------
        # 7. Small time stretch (ref whistle length variation)
        # ------------------------------------------------
        if np.random.rand() < 0.2:

            stretch = np.random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y, rate=stretch)

            # keep same length
            if len(y) > SR:
                y = y[:SR]
            else:
                y = np.pad(y, (0, SR - len(y)))


    # ------------------------
    # FULL BAND MEL
    # ------------------------
    mel_full = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=0,
        fmax=SR//2
    )

    logmel_full = librosa.power_to_db(mel_full)
    logmel_full = (logmel_full - np.mean(logmel_full)) / (np.std(logmel_full) + 1e-8)

    delta_full = librosa.feature.delta(logmel_full)
    delta2_full = librosa.feature.delta(logmel_full, order=2)

    # ------------------------
    # WHISTLE BAND MEL
    # ------------------------
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

    # ------------------------
    # STACK ALL 6 CHANNELS
    # ------------------------
    stacked = np.stack([
        logmel_full,
        delta_full,
        delta2_full,
        logmel_band,
        delta_band,
        delta2_band
    ], axis=0)

    return stacked.astype(np.float32)

def wav_to_logmel_from_audio(y):
    # ------------------------
    # FULL BAND MEL
    # ------------------------
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

    # ------------------------
    # WHISTLE BAND MEL
    # ------------------------
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

    # ------------------------
    # STACK ALL 6 CHANNELS
    # ------------------------
    stacked = np.stack([
        logmel_full,
        delta_full,
        delta2_full,
        logmel_band,
        delta_band,
        delta2_band
    ], axis=0)

    return stacked.astype(np.float32)

    logmel = librosa.power_to_db(mel)

    # Normalize
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)

    # Δ and ΔΔ
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)

    # Stack as channels
    stacked = np.stack([logmel, delta, delta2], axis=0)

    return stacked.astype(np.float32)

class WhistleDataset(Dataset):
    def __init__(self, filepaths, labels,augment=False):
        self.filepaths = filepaths
        self.labels = labels
        self.augment = augment
    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        mel = wav_to_logmel(self.filepaths[idx], augment=self.augment)
        return torch.tensor(mel), torch.tensor(self.labels[idx]).float(), self.filepaths[idx]

# ==============================
# LOAD FILES (MATCH-BASED SPLIT)
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

            # 🔥 ADD THIS BLOCK HERE
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

if __name__ == "__main__":
    model = TinyCNN().to(DEVICE)
    pos_files = glob.glob(os.path.join(DATASET_DIR, "match*", "pos", "*.wav"))
    neg_files = glob.glob(os.path.join(DATASET_DIR, "match*", "neg", "*.wav"))

    all_files = pos_files + neg_files
    all_labels = [1] * len(pos_files) + [0] * len(neg_files)

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for f, label in zip(all_files, all_labels):

        match_id = get_match_id(f)

        if match_id in SPLIT["train"]:
            X_train.append(f)
            y_train.append(label)

        elif match_id in SPLIT["val"]:
            X_val.append(f)
            y_val.append(label)

        elif match_id in SPLIT["test"]:
            X_test.append(f)
            y_test.append(label)

        else:
            print("Warning: unknown match:", f)

    print("Train samples:", len(X_train))
    print("Val samples:", len(X_val))
    print("Test samples:", len(X_test))

    pos_count = sum(y_train)
    neg_count = len(y_train) - pos_count

    print("Pos:", pos_count)
    print("Neg:", neg_count)

    pos_weight = torch.tensor([(neg_count / pos_count)]).to(DEVICE)

    train_ds = WhistleDataset(X_train, y_train,augment=True)
    val_ds   = WhistleDataset(X_val, y_val,augment=False)
    test_ds  = WhistleDataset(X_test, y_test,augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # ==============================
    # TINY CNN
    # ==============================





    # ==============================
    # LOSS + OPTIMIZER
    # ==============================

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        verbose = True
    )
    # ==============================
    # TRAIN LOOP
    # ==============================

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0

        for x, y, _ in tqdm(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        with torch.no_grad():
            for x, y, _ in tqdm(val_loader):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                out = torch.sigmoid(model(x))
                preds = (out > 0.35).float()

                tp += ((preds == 1) & (y == 1)).sum().item()
                tn += ((preds == 0) & (y == 0)).sum().item()
                fp += ((preds == 1) & (y == 0)).sum().item()
                fn += ((preds == 0) & (y == 1)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        print(f"\nEpoch {epoch + 1}")
        print("Train loss:", train_loss / len(train_loader))
        print(f"Val Accuracy:  {accuracy:.4f}")
        print(f"Val Precision: {precision:.4f}")
        print(f"Val Recall:    {recall:.4f}")
        print(f"Val F1:        {f1:.4f}")

        scheduler.step(f1)
    # ==============================
    # SAVE MODEL
    # ==============================


    print("Model saved.")

    # ==============================
    # TEST EVALUATION
    # ==============================

    model.eval()

    all_probs = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for x, y, paths in test_loader:

            x = x.to(DEVICE)
            y_np = y.numpy()

            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(y_np)
            all_paths.extend(paths)

    # ==============================
    # NOW run sweep AFTER loop
    # ==============================

    import numpy as np

    probs = np.array(all_probs)
    labels = np.array(all_labels)

    print("\n===== THRESHOLD SWEEP =====")

    best_f1 = 0
    best_thresh = 0

    for thresh in np.linspace(0.05, 0.95, 19):

        preds = (probs > thresh).astype(int)

        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        print(f"Thresh={thresh:.2f} | Acc={accuracy:.4f} | F1={f1:.4f} | Rec={recall:.4f} | Prec={precision:.4f} | FN={fn} | FP={fp}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print("\nBest threshold:", best_thresh)
    print("Best F1:", best_f1)
    print("Model saved.")
    torch.save({
        "model_state": model.state_dict(),
        "best_thresh": best_thresh
    }, "whistle_model_better_cent.pth")