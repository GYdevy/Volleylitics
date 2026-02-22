import os
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
DATA_ROOT = r"E:\Volleyballey\dataset_8matches"
SR = 22050
N_FFT = 2048
HOP = 256
WHISTLE_LOW = 3500
WHISTLE_HIGH = 4500


from scipy.stats import entropy

def extract_features(path):
    y, sr = librosa.load(path, sr=SR)



    S_complex = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S = np.abs(S_complex)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # -----------------------------
    # Whistle band
    # -----------------------------
    band_mask = (freqs >= WHISTLE_LOW) & (freqs <= WHISTLE_HIGH)
    band = S[band_mask]

    band_energy = band.mean(axis=0)  # per-frame whistle band energy

    band_energy_mean = np.mean(band_energy)
    band_energy_max = np.max(band_energy)
    band_energy_std = np.std(band_energy)
    band_energy_p95 = np.percentile(band_energy, 95)

    peak_bins = np.argmax(band, axis=0)
    peak_freqs = freqs[band_mask][peak_bins]

    # -----------------------------
    # Neighbor band contrast
    # -----------------------------
    lower_mask = (freqs >= 2000) & (freqs < 3000)
    upper_mask = (freqs > 4500) & (freqs <= 5500)

    lower_energy = S[lower_mask].mean()
    upper_energy = S[upper_mask].mean()
    whistle_energy = band.mean()

    band_contrast = whistle_energy / (lower_energy + upper_energy + 1e-8)

    # -----------------------------
    # Harmonic / Percussive ratio
    # -----------------------------
    H, P = librosa.decompose.hpss(S_complex)
    harmonic_ratio = np.mean(np.abs(H)) / (np.mean(S) + 1e-8)
    percussive_ratio = np.mean(np.abs(P)) / (np.mean(S) + 1e-8)

    # -----------------------------
    # Spectral Flux
    # -----------------------------
    flux = librosa.onset.onset_strength(S=S, sr=sr)
    flux_mean = np.mean(flux)
    flux_std = np.std(flux)
    flux_max = np.max(flux)

    # -----------------------------
    # Stability ratio (peak continuity)
    # -----------------------------
    if len(peak_freqs) > 1:
        stable = np.abs(np.diff(peak_freqs)) < 100
        stability_ratio = np.sum(stable) / len(stable)
        slope = np.polyfit(range(len(peak_freqs)), peak_freqs, 1)[0]
    else:
        stability_ratio = 0
        slope = 0

    # -----------------------------
    # Band entropy
    # -----------------------------
    norm_band = band / (band.sum(axis=0, keepdims=True) + 1e-8)
    entropy_vals = entropy(norm_band)
    band_entropy_mean = np.mean(entropy_vals)

    rolloff = librosa.feature.spectral_rolloff(
        S=S,
        sr=sr,
        roll_percent=0.85

    )
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)
    # -----------------------------
    # Core features (your originals)
    # -----------------------------
    features = {

        "band_energy_mean": band_energy_mean,
        "band_energy_max": band_energy_max,
        "band_energy_p95": band_energy_p95,
        "band_energy_std": band_energy_std,

        #"peak_freq_mean": np.mean(peak_freqs),
        "peak_freq_std": np.std(peak_freqs),

        "flatness_mean": np.mean(librosa.feature.spectral_flatness(S=S)),
        "bandwidth_mean": np.mean(librosa.feature.spectral_bandwidth(S=S)),

        "rms": np.mean(librosa.feature.rms(y=y)),
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y)),

        # 🔥 NEW FEATURES
        "harmonic_ratio": harmonic_ratio,
        #"percussive_ratio": percussive_ratio,

        "flux_mean": flux_mean,
        "flux_std": flux_std,
        #"flux_max": flux_max,

        "stability_ratio": stability_ratio,
        "peak_freq_slope": slope,

        "band_contrast": band_contrast,
        "band_entropy": band_entropy_mean,
        "rolloff_mean": rolloff_mean,
        "rolloff_std": rolloff_std,
    }

    return features


def build_dataframe():
    rows = []

    for split in ["train", "val", "test"]:
        for label in ["positive", "negative"]:
            folder = os.path.join(DATA_ROOT, split, label)
            if not os.path.exists(folder):
                continue

            for f in tqdm(os.listdir(folder), desc=f"{split}/{label}"):
                if not f.endswith(".wav"):
                    continue

                path = os.path.join(folder, f)
                feats = extract_features(path)

                feats["label"] = 1 if label == "positive" else 0
                feats["split"] = split
                feats["filename"] = f
                feats["path"] = path  # optional but very useful

                rows.append(feats)

    df = pd.DataFrame(rows)
    return df


def plot_histograms(df):
    features = df.select_dtypes(include=[np.number]).columns
    features = [c for c in features if c != "label"]

    for feat in features:
        plt.figure()
        sns.histplot(df[df.label == 1][feat], color="blue", label="POS", kde=True, stat="density")
        sns.histplot(df[df.label == 0][feat], color="red", label="NEG", kde=True, stat="density")
        plt.title(feat)
        plt.legend()
        plt.show()


def plot_scatter(df, feat1, feat2):
    plt.figure()
    plt.scatter(df[df.label == 0][feat1],
                df[df.label == 0][feat2],
                alpha=0.3,
                label="NEG")
    plt.scatter(df[df.label == 1][feat1],
                df[df.label == 1][feat2],
                alpha=0.3,
                label="POS")
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.legend()
    plt.show()


def plot_pca(df):
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Remove label from feature list
    features = [c for c in numeric_cols if c != "label"]

    X = df[features].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure()
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], alpha=0.3, label="NEG")
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], alpha=0.3, label="POS")
    plt.title("PCA Projection")
    plt.legend()
    plt.show()

    for i, feature in enumerate(features):
        print(feature, pca.components_[0][i])


if __name__ == "__main__":
    df = build_dataframe()
    print(df.describe())
    # Save once
    df.to_csv(
        r"E:\Volleyballey\dataset_8matches\full_dataframe.csv",
        index=False
    )

    print("Full dataframe saved.")
    # Histograms
    plot_histograms(df)

    # Example scatter
    plot_scatter(df, "peak_freq_std", "bandwidth_mean")

    # PCA cluster view
    plot_pca(df)
    plot_scatter(df, "stability_ratio", "band_contrast")
    plot_scatter(df, "harmonic_ratio", "flux_std")
    plot_scatter(df, "peak_freq_slope", "stability_ratio")

    # -------------------------
    # SPLIT DATA
    # -------------------------
    train_df = df[df.split == "train"]

    val_df = df[df.split == "val"]
    test_df = df[df.split == "test"]

    X_train = train_df.select_dtypes(include=[np.number]).drop(columns=["label"])
    y_train = train_df["label"]

    X_val = val_df.select_dtypes(include=[np.number]).drop(columns=["label"])
    y_val = val_df["label"]

    X_test = test_df.select_dtypes(include=[np.number]).drop(columns=["label"])
    y_test = test_df["label"]

    # -------------------------
    # MODEL
    # -------------------------
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000)
    )

    model.fit(X_train, y_train)
    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------
    logreg = model.named_steps["logisticregression"]
    scaler = model.named_steps["standardscaler"]

    feature_names = X_train.columns

    importance = pd.Series(
        np.abs(logreg.coef_[0]),
        index=feature_names
    ).sort_values(ascending=False)

    print("\n=== FEATURE IMPORTANCE (ABS COEFF) ===")
    print(importance)
    # -------------------------
    # EVALUATION
    # -------------------------
    print("\n=== TRAIN ===")
    print(classification_report(y_train, model.predict(X_train)))

    print("\n=== VAL ===")
    print(classification_report(y_val, model.predict(X_val)))

    print("\n=== TEST ===")
    print(classification_report(y_test, model.predict(X_test)))

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, model.predict(X_test)))

    # FALSE NEGATIVES ANALYSIS
    # -------------------------
    probs_test = model.predict_proba(X_test)[:, 1]
    pred_test = (probs_test >= 0.4).astype(int)

    test_results = test_df.copy()
    test_results["prob"] = probs_test
    test_results["pred"] = pred_test

    false_negatives = test_results[
        (test_results.label == 1) & (test_results.pred == 0)
        ]
    print(
        false_negatives
        .sort_values("prob")
        [["path", "prob"]]
        .head(20)
    )
    print("\nNumber of False Negatives:", len(false_negatives))



    print("\nNumber of False Negatives:", len(false_negatives))
    print(false_negatives.sort_values("prob").head(10))

    import joblib

    MODEL_PATH = r"E:\Volleyballey\detector_slop\whistle_logreg.pkl"

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")