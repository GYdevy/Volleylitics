from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = r"D:\Volleyballey\Vision\runs\detect\volleyball_detector11\weights\best.pt"
VIDEO_PATH = r"D:\Volleyballey\WhistleDetector\rallies\match4\rally_0017.mp4"
OUTPUT_DIR = r"/runs/detect/inference_test"
# ---------------

model = YOLO(MODEL_PATH)

# Run detection on the video
results = model.predict(
    source=VIDEO_PATH,
    save=True,             # saves annotated video to runs/detect/...
    save_txt=False,        # set True if you want per-frame .txt outputs
    project=OUTPUT_DIR,
    name="pred",
    conf=0.4,              # confidence threshold (try 0.25–0.5)
    imgsz=960,             # match your training resolution
    show=True             # set True if you want a pop-up window
)

print(f"✅ Done! Check the output video at:\n{results[0].save_dir}")
