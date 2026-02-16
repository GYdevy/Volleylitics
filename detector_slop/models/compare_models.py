import os
import json
import pandas as pd

MODELS_DIR = "/detector_slop/models"

rows = []

for f in os.listdir(MODELS_DIR):
    if not f.endswith(".json"):
        continue

    path = os.path.join(MODELS_DIR, f)

    with open(path, "r") as file:
        data = json.load(file)

    model_name = data["model_name"]

    test_report = data["test_report"]
    cm = data["test_confusion_matrix"]

    TN, FP = cm[0]
    FN, TP = cm[1]

    rows.append({
        "model": model_name,
        "accuracy": test_report["accuracy"],
        "precision_whistle": test_report["1"]["precision"],
        "recall_whistle": test_report["1"]["recall"],
        "f1_whistle": test_report["1"]["f1-score"],
        "FP": FP,
        "FN": FN
    })

df = pd.DataFrame(rows)

print("\n=== MODEL COMPARISON (Test Set) ===\n")
print(df.sort_values("f1_whistle", ascending=False))
