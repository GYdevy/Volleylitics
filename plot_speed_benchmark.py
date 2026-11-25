#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path(
    "/home/gyank/projects/Volleylitics/"
    "model_evaluation/speed_benchmark.csv"
)

OUTPUT_PATH = Path(
    "/home/gyank/projects/Volleylitics/"
    "model_evaluation/speed_benchmark.png"
)


def shorten_model_name(model_path: str) -> str:
    path = Path(model_path)

    if path.parent.name == "weights":
        return f"{path.parent.parent.name}/{path.stem}"

    return path.stem


def add_horizontal_labels(axis, bars, decimals=1):
    for bar in bars:
        value = bar.get_width()

        axis.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.{decimals}f}",
            va="center",
            ha="left",
            fontsize=8,
        )


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    dataframe = pd.read_csv(CSV_PATH)

    numeric_columns = [
        "mean_ms",
        "median_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "fps_from_mean",
        "fps_from_median",
        "peak_gpu_memory_mb",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe[
        dataframe["status"] == "success"
    ].copy()

    dataframe = dataframe.dropna(
        subset=["median_ms"]
    )

    if dataframe.empty:
        raise SystemExit(
            "No successful benchmark results found."
        )

    dataframe["model_name"] = dataframe["model"].apply(
        shorten_model_name
    )

    # Fastest model first.
    dataframe = dataframe.sort_values(
        "median_ms",
        ascending=True,
    ).reset_index(drop=True)

    model_names = dataframe["model_name"].tolist()
    positions = np.arange(len(dataframe))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(18, 12),
        constrained_layout=True,
    )

    figure.suptitle(
        "Volleylitics Ball Detector Speed Benchmark",
        fontsize=20,
        fontweight="bold",
    )

    # ========================================================
    # Plot 1: mean, median and P95 latency
    # ========================================================

    axis = axes[0, 0]
    height = 0.24

    mean_bars = axis.barh(
        positions - height,
        dataframe["mean_ms"],
        height,
        label="Mean",
    )

    median_bars = axis.barh(
        positions,
        dataframe["median_ms"],
        height,
        label="Median",
    )

    p95_bars = axis.barh(
        positions + height,
        dataframe["p95_ms"],
        height,
        label="P95",
    )

    axis.set_title("End-to-End Latency — Lower Is Better")
    axis.set_xlabel("Milliseconds per image")
    axis.set_yticks(positions)
    axis.set_yticklabels(model_names)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)
    axis.legend()

    add_horizontal_labels(axis, mean_bars)
    add_horizontal_labels(axis, median_bars)
    add_horizontal_labels(axis, p95_bars)

    # ========================================================
    # Plot 2: FPS
    # ========================================================

    axis = axes[0, 1]

    fps_bars = axis.barh(
        positions,
        dataframe["fps_from_median"],
    )

    axis.set_title("Throughput from Median Latency")
    axis.set_xlabel("Frames per second")
    axis.set_yticks(positions)
    axis.set_yticklabels(model_names)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)

    add_horizontal_labels(
        axis,
        fps_bars,
        decimals=1,
    )

    # ========================================================
    # Plot 3: latency range and stability
    # ========================================================

    axis = axes[1, 0]

    medians = dataframe["median_ms"].to_numpy()
    minimums = dataframe["min_ms"].to_numpy()
    maximums = dataframe["max_ms"].to_numpy()

    lower_errors = medians - minimums
    upper_errors = maximums - medians

    axis.errorbar(
        medians,
        positions,
        xerr=np.vstack(
            [lower_errors, upper_errors]
        ),
        fmt="o",
        capsize=5,
        markersize=7,
    )

    for position, median, std in zip(
        positions,
        dataframe["median_ms"],
        dataframe["std_ms"],
    ):
        axis.text(
            median,
            position - 0.18,
            f"median {median:.1f} ms | σ {std:.1f}",
            fontsize=8,
            ha="center",
        )

    axis.set_title("Latency Range and Stability")
    axis.set_xlabel(
        "Milliseconds per image — minimum to maximum"
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(model_names)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)

    # ========================================================
    # Plot 4: peak GPU memory
    # ========================================================

    axis = axes[1, 1]

    memory_bars = axis.barh(
        positions,
        dataframe["peak_gpu_memory_mb"],
    )

    axis.set_title("Peak GPU Memory")
    axis.set_xlabel("MiB")
    axis.set_yticks(positions)
    axis.set_yticklabels(model_names)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)

    add_horizontal_labels(
        axis,
        memory_bars,
        decimals=0,
    )

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

    print(f"Plot saved to: {OUTPUT_PATH}")

    print()
    print("Speed ranking:")

    for rank, row in enumerate(
        dataframe.itertuples(),
        start=1,
    ):
        print(
            f"{rank}. {row.model_name}: "
            f"median={row.median_ms:.2f} ms, "
            f"P95={row.p95_ms:.2f} ms, "
            f"FPS={row.fps_from_median:.2f}"
        )


if __name__ == "__main__":
    main()
