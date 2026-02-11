import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib
import os

# === CONFIG ===
LABELS = Path(r"D:\Volleyballey\WhistleDetector\labels.csv")
FEATURES = Path(r"D:\Volleyballey\WhistleDetector\match8_scan\match8_whistle_peaks.csv")
# ==============

# --- Load data ---
labels = pd.read_csv(LABELS)
features = pd.read_csv(FEATURES)

# Normalize columns
if "timestamp" not in features.columns and "time" in features.columns:
    features.rename(columns={"time": "timestamp"}, inplace=True)
if "match" not in features.columns:
    features["match"] = "match8"

labels["timestamp"] = labels["timestamp"].astype(str)
features["timestamp"] = features["timestamp"].astype(str)

# --- Merge ---
merged = pd.merge(
    features,
    labels,
    how="inner",
    on=["timestamp", "match"] if "match" in labels.columns else ["timestamp"]
)

print(f"✅ Merged {len(merged)} labeled samples")
print(merged.groupby("label").size())
print(merged.describe())

# --- Keep only whistle/noise ---
df = merged[merged["label"].isin(["whistle", "noise"])].copy()

# === Features ===
X = df[["rms", "flatness", "centroid", "energy"]].values
y = (df["label"] == "whistle").astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 1️⃣ Rule-based threshold test ===
pred_rule = (df["rms"] >= 0.090) & (df["flatness"] <= 0.0011)
acc_rule = (pred_rule == (df["label"] == "whistle")).mean()
cm_rule = confusion_matrix(y, pred_rule)
print(f"\n⚙️ Rule-based accuracy ≈ {acc_rule*100:.1f}%")
print("Confusion matrix:\n", cm_rule)

# === 2️⃣ EllipticEnvelope ===
model_ee = EllipticEnvelope(contamination=0.1, random_state=42)
model_ee.fit(X_scaled[y == 1])
pred_ee = (model_ee.predict(X_scaled) == 1).astype(int)
acc_ee = accuracy_score(y, pred_ee)
cm_ee = confusion_matrix(y, pred_ee)
print(f"\n🔵 EllipticEnvelope accuracy ≈ {acc_ee*100:.1f}%")
print("Confusion matrix:\n", cm_ee)

# === 3️⃣ IsolationForest ===
model_iso = IsolationForest(contamination=0.1, random_state=42)
model_iso.fit(X_scaled[y == 1])
pred_iso = (model_iso.predict(X_scaled) == 1).astype(int)
acc_iso = accuracy_score(y, pred_iso)
cm_iso = confusion_matrix(y, pred_iso)
print(f"\n🌲 IsolationForest accuracy ≈ {acc_iso*100:.1f}%")
print("Confusion matrix:\n", cm_iso)

# === 4️⃣ DecisionTree ===
clf_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_tree.fit(X_scaled, y)
pred_tree = clf_tree.predict(X_scaled)
acc_tree = accuracy_score(y, pred_tree)
cm_tree = confusion_matrix(y, pred_tree)
# Create the directory if it doesn't exist
MODEL_DIR = r"D:\Volleyballey\WhistleDetector\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model and scaler
joblib.dump(clf_tree, os.path.join(MODEL_DIR, "whistle_tree.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

print("✅ Saved model and scaler for reuse at", MODEL_DIR)
# Save the trained model and scaler
joblib.dump(clf_tree, r"D:\Volleyballey\WhistleDetector\models\whistle_tree.joblib")
joblib.dump(scaler,   r"D:\Volleyballey\WhistleDetector\models\scaler.joblib")
print("✅ Saved model and scaler for reuse.")
print(f"\n🌳 DecisionTree accuracy ≈ {acc_tree*100:.1f}%")
print("Confusion matrix:\n", cm_tree)

# === SUMMARY TABLE ===
summary = pd.DataFrame({
    "Model": ["Rule", "EllipticEnvelope", "IsolationForest", "DecisionTree"],
    "Accuracy": [acc_rule, acc_ee, acc_iso, acc_tree],
})
print("\n📊 Summary:")
print(summary.to_string(index=False))

# === Visualizations ===
sns.set(style="whitegrid", context="talk")

# 2D boundary visualization (RMS vs Flatness)
rms_vals = np.linspace(df["rms"].min(), df["rms"].max(), 200)
flat_vals = np.linspace(df["flatness"].min(), df["flatness"].max(), 200)
xx, yy = np.meshgrid(rms_vals, flat_vals)
grid = np.c_[xx.ravel(), yy.ravel(),
             np.full_like(xx.ravel(), df["centroid"].mean()),
             np.full_like(xx.ravel(), df["energy"].mean())]
grid_scaled = scaler.transform(grid)

# Plot EllipticEnvelope boundary
Z_ee = model_ee.decision_function(grid_scaled).reshape(xx.shape)
plt.figure(figsize=(7,5))
plt.contour(xx, yy, Z_ee, levels=[0], linewidths=2, colors="red")
sns.scatterplot(data=df, x="rms", y="flatness", hue="label", palette="Set2")
plt.title("Elliptic Boundary (RMS vs Flatness)")
plt.show()

# Decision Tree visualization
plt.figure(figsize=(12,6))
plot_tree(clf_tree, feature_names=["rms", "flatness", "centroid", "energy"],
          class_names=["noise", "whistle"], filled=True)
plt.title("Decision Tree Rules")
plt.show()

# Pairplot
sns.pairplot(
    df,
    hue="label",
    vars=["rms", "flatness", "centroid", "energy"],
    diag_kind="kde",
    palette="Set2"
)
plt.suptitle("Feature Distributions by Label", y=1.02)
plt.show()
