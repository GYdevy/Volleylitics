#!/usr/bin/env python3

import csv
import hashlib
import json
import traceback
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODELS_ROOT = Path("/home/gyank/projects/Volleylitics")
DATASET_ROOT = Path("/run/media/gyank/ssd/dataset")

OUTPUT_DIR = MODELS_ROOT / "model_evaluation"
CSV_PATH = OUTPUT_DIR / "model_ranking.csv"
JSON_PATH = OUTPUT_DIR / "model_ranking.json"

DEVICE = 0
IMAGE_SIZE = 960
BATCH_SIZE = 4
WORKERS = 2

# Models below this mAP50 are recorded but excluded from ranking.
MIN_MAP50 = 0.50

# True: test best.pt and last.pt.
# False: test only best.pt when both exist in the same weights folder.
TEST_LAST_CHECKPOINTS = True


# ============================================================
# HELPERS
# ============================================================

CSV_FIELDS = [
    "rank",
    "model",
    "status",
    "precision",
    "recall",
    "f1",
    "map50",
    "map50_95",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "error",
]


def find_data_yaml() -> Path:
    direct_path = DATASET_ROOT / "data.yaml"

    if direct_path.is_file():
        return direct_path

    candidates = sorted(DATASET_ROOT.rglob("data.yaml"))

    if not candidates:
        raise FileNotFoundError(
            f"No data.yaml found anywhere under {DATASET_ROOT}"
        )

    if len(candidates) > 1:
        print("Multiple data.yaml files found:")
        for candidate in candidates:
            print(f"  {candidate}")
        print(f"Using: {candidates[0]}")
        print()

    return candidates[0]


def find_models() -> list[Path]:
    models = []

    for model_path in MODELS_ROOT.rglob("*.pt"):
        # Avoid accidentally evaluating anything inside our output folder.
        if OUTPUT_DIR in model_path.parents:
            continue

        if not TEST_LAST_CHECKPOINTS and model_path.name == "last.pt":
            best_path = model_path.with_name("best.pt")

            if best_path.exists():
                continue

        models.append(model_path)

    return sorted(models)


def make_run_name(model_path: Path) -> str:
    relative_path = model_path.relative_to(MODELS_ROOT)
    readable_name = "_".join(relative_path.with_suffix("").parts)

    # Prevent collisions when paths produce similar names.
    path_hash = hashlib.sha1(
        str(relative_path).encode("utf-8")
    ).hexdigest()[:8]

    safe_name = "".join(
        character if character.isalnum() or character in "-_"
        else "_"
        for character in readable_name
    )

    return f"{safe_name[:120]}_{path_hash}"


def empty_result(model_path: Path) -> dict:
    return {
        "rank": "",
        "model": str(model_path),
        "status": "",
        "precision": "",
        "recall": "",
        "f1": "",
        "map50": "",
        "map50_95": "",
        "preprocess_ms": "",
        "inference_ms": "",
        "postprocess_ms": "",
        "error": "",
    }


def evaluate_model(model_path: Path, data_yaml: Path) -> dict:
    row = empty_result(model_path)

    model = YOLO(str(model_path))

    # Skip models that are not object-detection models.
    task = getattr(model, "task", None)

    if task != "detect":
        row["status"] = "skipped_non_detection"
        row["error"] = f"Model task is {task!r}, not 'detect'"
        return row

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        plots=False,
        save=False,
        save_json=False,
        verbose=False,
        project=str(OUTPUT_DIR / "validation_runs"),
        name=make_run_name(model_path),
        exist_ok=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    speed = metrics.speed or {}

    row.update(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": map50,
            "map50_95": map50_95,
            "preprocess_ms": float(speed.get("preprocess", 0.0)),
            "inference_ms": float(speed.get("inference", 0.0)),
            "postprocess_ms": float(speed.get("postprocess", 0.0)),
        }
    )

    if map50 < MIN_MAP50:
        row["status"] = "below_threshold"
        row["error"] = (
            f"mAP50 {map50:.4f} is below minimum {MIN_MAP50:.4f}"
        )
    else:
        row["status"] = "success"

    return row


def assign_ranks(results: list[dict]) -> list[dict]:
    successful = [
        row for row in results
        if row["status"] == "success"
    ]

    # Main ranking: mAP50-95.
    # Tie breakers: F1, recall, then precision.
    successful.sort(
        key=lambda row: (
            float(row["map50_95"]),
            float(row["f1"]),
            float(row["recall"]),
            float(row["precision"]),
        ),
        reverse=True,
    )

    for row in results:
        row["rank"] = ""

    for rank, row in enumerate(successful, start=1):
        row["rank"] = rank

    ranked_ids = {id(row) for row in successful}
    remaining = [
        row for row in results
        if id(row) not in ranked_ids
    ]

    return successful + remaining


def save_results(results: list[dict]) -> None:
    ordered_results = assign_ranks(results)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(ordered_results)

    with JSON_PATH.open("w", encoding="utf-8") as json_file:
        json.dump(ordered_results, json_file, indent=2)


def print_result(row: dict) -> None:
    status = row["status"]

    if status in {"success", "below_threshold"}:
        print(
            f"  Status:       {status}\n"
            f"  Precision:    {row['precision']:.4f}\n"
            f"  Recall:       {row['recall']:.4f}\n"
            f"  F1:           {row['f1']:.4f}\n"
            f"  mAP50:        {row['map50']:.4f}\n"
            f"  mAP50-95:     {row['map50_95']:.4f}\n"
            f"  Inference:    {row['inference_ms']:.2f} ms/image"
        )
    else:
        print(f"  Status: {status}")
        print(f"  Reason: {row['error']}")


def print_final_ranking(results: list[dict]) -> None:
    ranked = assign_ranks(results)

    successful = [
        row for row in ranked
        if row["status"] == "success"
    ]

    print()
    print("=" * 90)
    print("FINAL MODEL RANKING")
    print("Primary metric: mAP50-95")
    print("=" * 90)

    if not successful:
        print("No model passed the minimum mAP50 threshold.")
    else:
        for row in successful:
            print(
                f"{row['rank']:>2}. {row['model']}\n"
                f"    P={row['precision']:.4f} "
                f"R={row['recall']:.4f} "
                f"F1={row['f1']:.4f} "
                f"mAP50={row['map50']:.4f} "
                f"mAP50-95={row['map50_95']:.4f} "
                f"inference={row['inference_ms']:.2f} ms"
            )

    below_threshold = sum(
        row["status"] == "below_threshold" for row in results
    )
    failed = sum(
        row["status"] == "failed" for row in results
    )
    non_detection = sum(
        row["status"] == "skipped_non_detection" for row in results
    )

    print()
    print(f"Passed threshold:      {len(successful)}")
    print(f"Below threshold:       {below_threshold}")
    print(f"Failed to evaluate:    {failed}")
    print(f"Non-detection models:  {non_detection}")
    print(f"CSV results:           {CSV_PATH}")
    print(f"JSON results:          {JSON_PATH}")

    if successful:
        print()
        print(f"Best model: {successful[0]['model']}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODELS_ROOT.is_dir():
        raise SystemExit(
            f"Volleylitics folder does not exist: {MODELS_ROOT}"
        )

    if not DATASET_ROOT.is_dir():
        raise SystemExit(
            f"Dataset folder does not exist: {DATASET_ROOT}"
        )

    data_yaml = find_data_yaml()
    model_paths = find_models()

    if not model_paths:
        raise SystemExit(
            f"No .pt files found under {MODELS_ROOT}"
        )

    print("=" * 90)
    print("VOLLEYLITICS MODEL EVALUATION")
    print("=" * 90)
    print(f"Models folder:     {MODELS_ROOT}")
    print(f"Dataset YAML:      {data_yaml}")
    print(f"Models found:      {len(model_paths)}")
    print(f"Dataset split:     test")
    print(f"Image size:        {IMAGE_SIZE}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Minimum mAP50:     {MIN_MAP50}")
    print()

    for index, model_path in enumerate(model_paths, start=1):
        print(f"[{index}/{len(model_paths)}] {model_path}")

    print()
    print("Starting evaluation...")
    print()

    results = []

    for index, model_path in enumerate(model_paths, start=1):
        print("-" * 90)
        print(f"[{index}/{len(model_paths)}] Testing:")
        print(f"  {model_path}")

        try:
            result = evaluate_model(model_path, data_yaml)

        except KeyboardInterrupt:
            print()
            print("Interrupted by user. Saving completed results...")
            save_results(results)
            raise

        except Exception as exception:
            result = empty_result(model_path)
            result["status"] = "failed"
            result["error"] = (
                f"{type(exception).__name__}: {exception}"
            )

            print("  Evaluation failed; continuing to next model.")
            print(f"  {result['error']}")

            # Full error is useful for debugging without cluttering CSV.
            error_log = OUTPUT_DIR / "errors.log"

            with error_log.open("a", encoding="utf-8") as error_file:
                error_file.write("=" * 90 + "\n")
                error_file.write(f"Model: {model_path}\n")
                error_file.write(traceback.format_exc())
                error_file.write("\n")

        results.append(result)
        save_results(results)
        print_result(result)
        print(f"  Progress saved to: {CSV_PATH}")
        print()

    print_final_ranking(results)


if __name__ == "__main__":
    main()
