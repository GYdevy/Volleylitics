from ultralytics import YOLO
import torch
import os

# --- CONFIG ---
DATA_YAML = r"D:\Volleyballey\dataset\data.yaml"
MODEL_WEIGHTS = "yolov8s.pt"   # you can later change to yolov8m.pt if GPU strong enough
EPOCHS = 100
IMG_SIZE = 960
BATCH_SIZE = 8                 # adjust if out-of-memory
PROJECT_NAME = "volleyball_detector"
# ---------------


def main():
    # --- GPU Check ---
    print("\n🔍 Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected — training will run on CPU (very slow!)")

    # --- Load YOLO model ---
    print(f"\n📦 Loading model: {MODEL_WEIGHTS}")
    model = YOLO(MODEL_WEIGHTS)

    # --- Start training ---
    print(f"🚀 Starting training for {EPOCHS} epochs...")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=PROJECT_NAME,
        workers=2,  # safe for Windows multiprocessing
        device=0 if torch.cuda.is_available() else "cpu",
    )

    # --- Locate best weights ---
    best_weights = os.path.join("runs", "detect", PROJECT_NAME, "weights", "best.pt")
    if not os.path.exists(best_weights):
        print("⚠️ Could not find best weights, using last.pt instead")
        best_weights = os.path.join("runs", "detect", PROJECT_NAME, "weights", "last.pt")

    # --- Validation ---
    print("\n🎯 Running validation on best weights...")
    model = YOLO(best_weights)
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        split="val",
        save_txt=True,    # saves predicted boxes as txt files
        save_conf=True,   # include confidence values
        save=True         # saves visualized images under runs/val/
    )

    # --- Print summary ---
    print("\n📊 Validation complete:")
    print(val_results)

    print("\n✅ All done. Check runs/detect/ and runs/val/ for outputs.")


if __name__ == "__main__":
    main()
