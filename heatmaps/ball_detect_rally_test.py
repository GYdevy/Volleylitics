from ultralytics import YOLO
import cv2

model = YOLO("/workspace/model/best.pt")

cap = cv2.VideoCapture("/workspace/rally_segmentator/output/match17/rally_clips/rally_017.mp4")

# --- setup output video ---
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("/workspace/heatmaps/output.mp4", fourcc, fps, (w, h))

# --- main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.2)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    out.write(frame)   # ✅ write frame instead of showing

# --- cleanup ---
cap.release()
out.release()

print("done")
