import os
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# ==========================
# CONFIG
# ==========================
DATASET_DIR = "../dataset"
SR = 22050
SAVE_FALSE_POSITIVES = False
# ==========================
# FEATURE EXTRACTION
# ==========================


from scipy.signal import butter, filtfilt


def extract_features(path):
    y, sr = librosa.load(path, sr=SR)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc_delta.mean(axis=1),
        mfcc_delta.std(axis=1),
        centroid.mean(axis=1),
        bandwidth.mean(axis=1),
        zcr.mean(axis=1)
    ])

    return features

# ==========================
# LOAD DATA
# ==========================

def load_split(split):
    X = []
    y = []
    paths = []

    for label, folder in enumerate(["negative", "positive"]):
        path = os.path.join(DATASET_DIR, split, folder)
        files = os.listdir(path)

        for f in tqdm(files, desc=f"{split}-{folder}"):
            full_path = os.path.join(path, f)
            features = extract_features(full_path)

            X.append(features)
            y.append(label)
            paths.append(full_path)

    return np.array(X), np.array(y), paths

X_train, y_train, _ = load_split("train")
X_val, y_val, _ = load_split("val")
X_test, y_test, test_paths = load_split("test")


# ==========================
# MODEL
# ==========================
MODEL_TYPE = "rf"


def build_model(model_type):

    if model_type == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        )

    elif model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42
        )

    elif model_type == "svm":
        clf = SVC(
            kernel="rbf",
            probability=True,   # IMPORTANT for threshold tuning
            class_weight="balanced",
            C=5,
            gamma="scale",
            random_state=42
        )

    elif model_type == "lgb":
        clf = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42
        )

    else:
        raise ValueError("Unknown model type")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

print("\nTraining...")

model = build_model(MODEL_TYPE)

model.fit(X_train, y_train)

# ==========================
# VALIDATION (Threshold Tuning)
# ==========================

from sklearn.metrics import precision_recall_curve

print("\nValidation (Threshold Search)...")

y_proba_val = model.predict_proba(X_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba_val)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best threshold from VAL: {best_threshold:.3f}")
print(f"Best validation F1: {f1_scores[best_idx]:.3f}")

y_pred_val = (y_proba_val >= best_threshold).astype(int)

print("\nValidation Results:")
print(confusion_matrix(y_val, y_pred_val))
print(classification_report(y_val, y_pred_val))


# ==========================
# TEST (Using tuned threshold)
# ==========================

print("\nTest Results (Using tuned threshold)...")

y_proba_test = model.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= best_threshold).astype(int)

print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))



# ==========================
# FALSE POSITIVE ANALYSIS
# ==========================

if SAVE_FALSE_POSITIVES:

    import shutil

    FP_DIR = "false_positives"
    os.makedirs(FP_DIR, exist_ok=True)

    fp_count = 0

    for i in range(len(y_test)):
        # False Positive = predicted whistle (1) but actually negative (0)
        if y_test[i] == 0 and y_pred_test[i] == 1:
            src_path = test_paths[i]
            dst_path = os.path.join(FP_DIR, f"fp_{fp_count}.wav")
            shutil.copy(src_path, dst_path)
            fp_count += 1

    print(f"\nSaved {fp_count} false positives to '{FP_DIR}'")

else:
    print("\nFalse positive saving disabled.")


# ==========================
# SAVE MODEL + METRICS
# ==========================

from datetime import datetime
import json
import os

MODEL_NAME = "rf"

os.makedirs("models", exist_ok=True)

results = {
    "model_name": MODEL_NAME,
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "best_threshold": float(best_threshold),
    "val_confusion_matrix": confusion_matrix(y_val, y_pred_val).tolist(),
    "val_report": classification_report(y_val, y_pred_val, output_dict=True),
    "test_confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
    "test_report": classification_report(y_test, y_pred_test, output_dict=True)
}

joblib.dump(model, f"detector_slop/models/{MODEL_NAME}.pkl")

with open(f"detector_slop/models/{MODEL_NAME}_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"\nModel + metrics saved → models/{MODEL_NAME}")
