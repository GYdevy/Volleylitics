import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ===============================
# CONFIG
# ===============================
CSV_PATH = r"D:\Volleyballey\WhistleDetector\whistle_features_all_features.csv"
OUT_DIR = Path("feature_plots")
OUT_DIR.mkdir(exist_ok=True)

sns.set(style="whitegrid")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(CSV_PATH)

print(f"Loaded dataset:")
print(f"- {sum(df.label == 1)} whistles")
print(f"- {sum(df.label == 0)} noise samples\n")

features = [c for c in df.columns if c not in ("filename", "label")]

whistles = df[df.label == 1]
noise = df[df.label == 0]

# ===============================
# STAT PRINTER
# ===============================
def print_stats(feature):
    print(f"\n===== {feature.upper()} =====")

    for name, data in [("Whistles", whistles), ("Noise", noise)]:
        print(f"{name}:")
        print(f"  min={data[feature].min():.4f}  max={data[feature].max():.4f}")
        print(f"  p5={data[feature].quantile(0.05):.4f}  p95={data[feature].quantile(0.95):.4f}")

# ===============================
# PLOTTING LOOP
# ===============================
for feature in features:
    print_stats(feature)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(feature, fontsize=14)

    # -------- HISTOGRAM --------
    sns.histplot(
        whistles[feature],
        bins=40,
        color="green",
        label="Whistle (1)",
        stat="count",
        alpha=0.5,
        ax=axes[0]
    )

    sns.histplot(
        noise[feature],
        bins=40,
        color="red",
        label="Noise (0)",
        stat="count",
        alpha=0.5,
        ax=axes[0]
    )

    # Min / max lines
    axes[0].axvline(whistles[feature].min(), color="green", linestyle="--", alpha=0.7)
    axes[0].axvline(whistles[feature].max(), color="green", linestyle="--", alpha=0.7)

    axes[0].axvline(noise[feature].min(), color="red", linestyle="--", alpha=0.7)
    axes[0].axvline(noise[feature].max(), color="red", linestyle="--", alpha=0.7)

    axes[0].set_title("Histogram")
    axes[0].legend()

    # -------- BOXPLOT --------
    sns.boxplot(
        x="label",
        y=feature,
        data=df,
        ax=axes[1],
        showfliers=True
    )

    axes[1].set_xticklabels(["Noise (0)", "Whistle (1)"])
    axes[1].set_title("Boxplot")

    # -------- SAVE --------
    out_path = OUT_DIR / f"{feature}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

print("\n✅ All plots saved to:", OUT_DIR.resolve())
