#!/usr/bin/env python3

import csv
import gc
import hashlib
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODELS_ROOT = Path("/home/gyank/projects/Volleylitics")

DATASET_ROOT = Path("/run/media/gyank/ssd/dataset")
TEST_IMAGES_DIR = DATASET_ROOT / "test" / "images"

EVALUATION_CSV = (
    MODELS_ROOT
    / "model_evaluation"
    / "model_ranking.csv"
)

OUTPUT_CSV = (
    MODELS_ROOT
    / "model_evaluation"
    / "speed_benchmark.csv"
)

IMAGE_SIZE = 960
DEVICE = 0
CONFIDENCE = 0.25

# Images are loaded into RAM before benchmarking.
BENCHMARK_IMAGES = 20

# Warm-up is important, especially with ROCm.
WARMUP_RUNS = 15

# Number of measured single-image predictions per model.
MEASURED_RUNS = 100

# True: benchmark only models marked successful in model_ranking.csv.
# False: benchmark every unique valid YOLO detector.
ONLY_SUCCESSFUL_MODELS = True


CSV_FIELDS = [
    "speed_rank",
    "model",
    "sha256",
    "image_size",
    "warmup_runs",
    "measured_runs",
    "mean_ms",
    "median_ms",
    "p95_ms",
    "min_ms",
    "max_ms",
    "std_ms",
    "fps_from_mean",
    "fps_from_median",
    "peak_gpu_memory_mb",
    "status",
    "error",
]


# ============================================================
# MODEL DISCOVERY
# ============================================================

def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def successful_model_paths() -> set[Path]:
    if not EVALUATION_CSV.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {EVALUATION_CSV}"
        )

    paths = set()

    with EVALUATION_CSV.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("status") == "success":
                paths.add(
                    Path(row["model"]).resolve()
                )

    return paths


def find_unique_models() -> list[tuple[Path, str]]:
    if ONLY_SUCCESSFUL_MODELS:
        candidate_paths = sorted(
            successful_model_paths()
        )
    else:
        candidate_paths = sorted(
            MODELS_ROOT.rglob("*.pt")
        )

    hashes_seen = {}
    unique_models = []

    for model_path in candidate_paths:
        if not model_path.is_file():
            print(f"Missing model, skipping: {model_path}")
            continue

        file_hash = calculate_sha256(model_path)

        if file_hash in hashes_seen:
            print("Skipping duplicate:")
            print(f"  Duplicate: {model_path}")
            print(f"  Original:  {hashes_seen[file_hash]}")
            print()
            continue

        hashes_seen[file_hash] = model_path
        unique_models.append((model_path, file_hash))

    return unique_models


# ============================================================
# IMAGE LOADING
# ============================================================

def find_test_images() -> list[Path]:
    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    images = sorted(
        path
        for path in TEST_IMAGES_DIR.iterdir()
        if path.suffix.lower() in supported_extensions
    )

    if not images:
        raise FileNotFoundError(
            f"No test images found in {TEST_IMAGES_DIR}"
        )

    return images[:BENCHMARK_IMAGES]


def load_images_into_ram(
    image_paths: list[Path],
) -> list[np.ndarray]:
    images = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"Could not read image, skipping: {image_path}"
            )
            continue

        images.append(image)

    if not images:
        raise RuntimeError(
            "None of the benchmark images could be loaded."
        )

    return images


# ============================================================
# BENCHMARKING
# ============================================================

def synchronize_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clean_gpu_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        synchronize_gpu()


def percentile_95(values: list[float]) -> float:
    return float(np.percentile(values, 95))


def benchmark_model(
    model_path: Path,
    file_hash: str,
    images: list[np.ndarray],
) -> dict:
    row = {
        "speed_rank": "",
        "model": str(model_path),
        "sha256": file_hash,
        "image_size": IMAGE_SIZE,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "mean_ms": "",
        "median_ms": "",
        "p95_ms": "",
        "min_ms": "",
        "max_ms": "",
        "std_ms": "",
        "fps_from_mean": "",
        "fps_from_median": "",
        "peak_gpu_memory_mb": "",
        "status": "",
        "error": "",
    }

    clean_gpu_memory()

    model = YOLO(str(model_path))

    if getattr(model, "task", None) != "detect":
        row["status"] = "skipped_non_detection"
        row["error"] = (
            f"Model task is {model.task!r}, not 'detect'"
        )
        return row

    print(f"  Warm-up: {WARMUP_RUNS} runs")

    # Warm-up covers predictor setup, model fusion and ROCm kernels.
    for index in range(WARMUP_RUNS):
        image = images[index % len(images)]

        model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            conf=CONFIDENCE,
            batch=1,
            verbose=False,
            save=False,
        )

    synchronize_gpu()

    # Reset peak memory after warm-up.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"  Measuring: {MEASURED_RUNS} runs")

    latencies_ms = []

    for index in range(MEASURED_RUNS):
        image = images[index % len(images)]

        # Synchronization prevents asynchronous GPU execution from
        # producing artificially low timing measurements.
        synchronize_gpu()
        start_time = time.perf_counter_ns()

        model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            conf=CONFIDENCE,
            batch=1,
            verbose=False,
            save=False,
        )

        synchronize_gpu()
        end_time = time.perf_counter_ns()

        elapsed_ms = (
            end_time - start_time
        ) / 1_000_000

        latencies_ms.append(elapsed_ms)

        if (index + 1) % 20 == 0:
            print(
                f"    Completed {index + 1}/"
                f"{MEASURED_RUNS}"
            )

    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    minimum_ms = min(latencies_ms)
    maximum_ms = max(latencies_ms)

    if len(latencies_ms) > 1:
        std_ms = statistics.stdev(latencies_ms)
    else:
        std_ms = 0.0

    p95_ms = percentile_95(latencies_ms)

    peak_memory_mb = 0.0

    if torch.cuda.is_available():
        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

    row.update(
        {
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "min_ms": minimum_ms,
            "max_ms": maximum_ms,
            "std_ms": std_ms,
            "fps_from_mean": 1000.0 / mean_ms,
            "fps_from_median": 1000.0 / median_ms,
            "peak_gpu_memory_mb": peak_memory_mb,
            "status": "success",
        }
    )

    del model
    clean_gpu_memory()

    return row


# ============================================================
# OUTPUT
# ============================================================

def save_results(results: list[dict]) -> None:
    successful = [
        row
        for row in results
        if row["status"] == "success"
    ]

    successful.sort(
        key=lambda row: float(row["median_ms"])
    )

    for row in results:
        row["speed_rank"] = ""

    for rank, row in enumerate(successful, start=1):
        row["speed_rank"] = rank

    successful_ids = {id(row) for row in successful}

    remaining = [
        row
        for row in results
        if id(row) not in successful_ids
    ]

    ordered_results = successful + remaining

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerows(ordered_results)


def print_result(result: dict) -> None:
    if result["status"] != "success":
        print(f"  Status: {result['status']}")
        print(f"  Error:  {result['error']}")
        return

    print(
        f"  Mean:        {result['mean_ms']:.2f} ms\n"
        f"  Median:      {result['median_ms']:.2f} ms\n"
        f"  P95:         {result['p95_ms']:.2f} ms\n"
        f"  Min:         {result['min_ms']:.2f} ms\n"
        f"  Max:         {result['max_ms']:.2f} ms\n"
        f"  Std:         {result['std_ms']:.2f} ms\n"
        f"  Mean FPS:    {result['fps_from_mean']:.2f}\n"
        f"  Median FPS:  {result['fps_from_median']:.2f}\n"
        f"  Peak memory: "
        f"{result['peak_gpu_memory_mb']:.1f} MB"
    )


def print_final_ranking(results: list[dict]) -> None:
    successful = [
        row
        for row in results
        if row["status"] == "success"
    ]

    successful.sort(
        key=lambda row: float(row["median_ms"])
    )

    print()
    print("=" * 90)
    print("FINAL SPEED RANKING — MEDIAN SINGLE-IMAGE LATENCY")
    print("=" * 90)

    for rank, row in enumerate(successful, start=1):
        print(
            f"{rank}. {row['model']}\n"
            f"   Median={row['median_ms']:.2f} ms, "
            f"Mean={row['mean_ms']:.2f} ms, "
            f"P95={row['p95_ms']:.2f} ms, "
            f"FPS={row['fps_from_median']:.2f}, "
            f"Memory={row['peak_gpu_memory_mb']:.1f} MB"
        )

    print()
    print(f"Results saved to: {OUTPUT_CSV}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "ROCm/CUDA GPU is not available inside the container."
        )

    print("=" * 90)
    print("VOLLEYLITICS UNIQUE-MODEL SPEED BENCHMARK")
    print("=" * 90)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: 1")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Measured runs: {MEASURED_RUNS}")
    print()

    unique_models = find_unique_models()

    if not unique_models:
        raise SystemExit(
            "No unique models were found."
        )

    image_paths = find_test_images()
    images = load_images_into_ram(image_paths)

    print(f"Unique models: {len(unique_models)}")
    print(f"Images loaded into RAM: {len(images)}")
    print()

    for model_path, _ in unique_models:
        print(f"  {model_path}")

    print()

    results = []

    for index, (model_path, file_hash) in enumerate(
        unique_models,
        start=1,
    ):
        print("-" * 90)
        print(
            f"[{index}/{len(unique_models)}] "
            f"{model_path}"
        )

        try:
            result = benchmark_model(
                model_path,
                file_hash,
                images,
            )

        except KeyboardInterrupt:
            print()
            print(
                "Interrupted. Saving completed results..."
            )
            save_results(results)
            raise

        except Exception as exception:
            result = {
                "speed_rank": "",
                "model": str(model_path),
                "sha256": file_hash,
                "image_size": IMAGE_SIZE,
                "warmup_runs": WARMUP_RUNS,
                "measured_runs": MEASURED_RUNS,
                "mean_ms": "",
                "median_ms": "",
                "p95_ms": "",
                "min_ms": "",
                "max_ms": "",
                "std_ms": "",
                "fps_from_mean": "",
                "fps_from_median": "",
                "peak_gpu_memory_mb": "",
                "status": "failed",
                "error": (
                    f"{type(exception).__name__}: "
                    f"{exception}"
                ),
            }

            print(f"  Failed: {result['error']}")
            clean_gpu_memory()

        results.append(result)
        save_results(results)
        print_result(result)
        print()

    print_final_ranking(results)


if __name__ == "__main__":
    main()
