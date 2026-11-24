#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path(
    "/home/gyank/projects/Volleylitics/"
    "model_evaluation/model_ranking.csv"
)

OUTPUT_PATH = Path(
    "/home/gyank/projects/Volleylitics/"
    "model_evaluation/model_comparison.png"
)


def shorten_model_name(model_path: str) -> str:
    path = Path(model_path)

    # For paths such as:
    # runs/detect/train12/weights/best.pt
    if path.parent.name == "weights":
        return f"{path.parent.parent.name}/{path.stem}"

    return path.stem


def add_bar_labels(axis, decimals=3):
    for container in axis.containers:
        axis.bar_label(
            container,
            fmt=f"%.{decimals}f",
            fontsize=7,
            padding=2,
            rotation=90,
        )


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    dataframe = pd.read_csv(CSV_PATH)

    numeric_columns = [
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
        "inference_ms",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    # Keep only models that passed the evaluation threshold.
    dataframe = dataframe[
        dataframe["status"] == "success"
    ].copy()

    dataframe = dataframe.dropna(
        subset=["map50_95"]
    )

    if dataframe.empty:
        raise SystemExit("No successful models found in the CSV.")

    dataframe["model_name"] = dataframe["model"].apply(
        shorten_model_name
    )

    # Order all plots by mAP50-95.
    dataframe = dataframe.sort_values(
        "map50_95",
        ascending=False,
    ).reset_index(drop=True)

    # Rank 1 means best.
    accuracy_metrics = [
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
    ]

    rank_columns = []

    for metric in accuracy_metrics:
        rank_column = f"{metric}_rank"

        dataframe[rank_column] = dataframe[metric].rank(
            ascending=False,
            method="min",
        )

        rank_columns.append(rank_column)

    # Lower inference time is better.
    dataframe["speed_rank"] = dataframe[
        "inference_ms"
    ].rank(
        ascending=True,
        method="min",
    )

    rank_columns.append("speed_rank")

    # Overall accuracy ranking does not include speed.
    dataframe["average_accuracy_rank"] = dataframe[
        [
            "precision_rank",
            "recall_rank",
            "f1_rank",
            "map50_rank",
            "map50_95_rank",
        ]
    ].mean(axis=1)

    model_names = dataframe["model_name"].tolist()
    x = np.arange(len(dataframe))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(18, 12),
        constrained_layout=True,
    )

    figure.suptitle(
        "Volleylitics Ball Detector Comparison",
        fontsize=20,
        fontweight="bold",
    )

    # ========================================================
    # Plot 1: precision, recall and F1
    # ========================================================

    axis = axes[0, 0]
    width = 0.25

    axis.bar(
        x - width,
        dataframe["precision"],
        width,
        label="Precision",
    )

    axis.bar(
        x,
        dataframe["recall"],
        width,
        label="Recall",
    )

    axis.bar(
        x + width,
        dataframe["f1"],
        width,
        label="F1",
    )

    axis.set_title("Precision, Recall and F1")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.08)
    axis.set_xticks(x)
    axis.set_xticklabels(
        model_names,
        rotation=25,
        ha="right",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    add_bar_labels(axis)

    # ========================================================
    # Plot 2: mAP
    # ========================================================

    axis = axes[0, 1]
    width = 0.35

    axis.bar(
        x - width / 2,
        dataframe["map50"],
        width,
        label="mAP50",
    )

    axis.bar(
        x + width / 2,
        dataframe["map50_95"],
        width,
        label="mAP50-95",
    )

    axis.set_title("Detection Accuracy")
    axis.set_ylabel("mAP")
    axis.set_ylim(0, 1.08)
    axis.set_xticks(x)
    axis.set_xticklabels(
        model_names,
        rotation=25,
        ha="right",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    add_bar_labels(axis)

    # ========================================================
    # Plot 3: inference speed
    # ========================================================

    axis = axes[1, 0]

    speed_bars = axis.bar(
        x,
        dataframe["inference_ms"],
    )

    axis.set_title("Inference Time — Lower Is Better")
    axis.set_ylabel("Milliseconds per image")
    axis.set_xticks(x)
    axis.set_xticklabels(
        model_names,
        rotation=25,
        ha="right",
    )
    axis.grid(axis="y", alpha=0.25)

    for bar, value in zip(
        speed_bars,
        dataframe["inference_ms"],
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # ========================================================
    # Plot 4: rank heatmap
    # ========================================================

    axis = axes[1, 1]

    rank_labels = [
        "Precision",
        "Recall",
        "F1",
        "mAP50",
        "mAP50-95",
        "Speed",
    ]

    rank_matrix = dataframe[
        rank_columns
    ].to_numpy(dtype=float)

    heatmap = axis.imshow(
        rank_matrix,
        cmap="RdYlGn_r",
        aspect="auto",
        vmin=1,
        vmax=max(len(dataframe), 2),
    )

    axis.set_title("Metric Ranking — 1 Is Best")
    axis.set_xticks(
        np.arange(len(rank_labels))
    )
    axis.set_xticklabels(
        rank_labels,
        rotation=30,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(model_names))
    )
    axis.set_yticklabels(model_names)

    for row_index in range(rank_matrix.shape[0]):
        for column_index in range(rank_matrix.shape[1]):
            rank_value = rank_matrix[
                row_index,
                column_index,
            ]

            axis.text(
                column_index,
                row_index,
                f"{rank_value:.0f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    colorbar = figure.colorbar(
        heatmap,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label("Rank")

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"Chart saved to: {OUTPUT_PATH}")

    print()
    print("Accuracy ranking:")

    accuracy_ranking = dataframe.sort_values(
        "average_accuracy_rank"
    )

    for position, row in enumerate(
        accuracy_ranking.itertuples(),
        start=1,
    ):
        print(
            f"{position}. {row.model_name}: "
            f"average rank={row.average_accuracy_rank:.2f}, "
            f"mAP50-95={row.map50_95:.4f}"
        )


if __name__ == "__main__":
    main()
