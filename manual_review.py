import cv2
import joblib
VIDEO_PATH = "D:\Volleyballey\videos\match4.mp4"
OUT_REVIEW_PKL = "rally_end_manual.pkl"

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)


DATA_PATH = "serve_rally_structure.pkl"

data = joblib.load(DATA_PATH)

rally_segments = data["rally_segments"]
serve_anchors  = data["serve_anchors"]
court_roi      = data["court_roi"]
results = []

def seek(sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)

def draw_overlay(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(
        frame, text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )

print("\n=== MANUAL RALLY REVIEW ===")
print("SPACE=play | N/P=next/prev whistle | E=select | X=none | Q=quit\n")

for idx, r in enumerate(rally_segments):
    serve_t = r["serve_start"]
    next_t  = r["serve_next"]
    cands   = r["candidate_whistles"]

    # skip easy auto cases if you want
    print(
        f"Rally {idx}: duration={r['duration']:.1f}s, "
        f"candidates={len(cands)}"
    )
    if 6 <= r["duration"] <= 35 and len(cands) == 1:
        results.append({
            "serve_start": serve_t,
            "serve_next": next_t,
            "rally_end": cands[0]["time"],
            "confidence": "auto"
        })
        continue

    print(f"[{idx}] Reviewing rally {serve_t:.1f}s → {next_t:.1f}s")

    cand_idx = 0
    playing = False
    chosen = None

    # start at serve
    seek(serve_t)

    while True:
        if playing:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            seek(pos)
            ret, frame = cap.read()
            if not ret:
                break

        info = (
            f"Rally {idx} | "
            f"Serve {serve_t:.1f}s → {next_t:.1f}s | "
            f"Cand {cand_idx+1}/{len(cands)}"
        )
        draw_overlay(frame, info)
        cv2.imshow("Rally Review", frame)

        key = cv2.waitKey(30 if playing else 0) & 0xFF

        if key == ord(' '):  # play/pause
            playing = not playing

        elif key == ord('n') and cands:
            cand_idx = min(cand_idx + 1, len(cands) - 1)
            seek(cands[cand_idx]["time"])
            playing = False

        elif key == ord('p') and cands:
            cand_idx = max(cand_idx - 1, 0)
            seek(cands[cand_idx]["time"])
            playing = False

        elif key == ord('e'):
            chosen = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            print(f"  ✔ Selected rally end @ {chosen:.2f}s")
            results.append({
                "serve_start": serve_t,
                "serve_next": next_t,
                "rally_end": chosen,
                "confidence": "manual"
            })
            break

        elif key == ord('x'):
            print("  ✖ Marked as uncertain / no whistle")
            results.append({
                "serve_start": serve_t,
                "serve_next": next_t,
                "rally_end": None,
                "confidence": "uncertain"
            })
            break

        elif key == ord('q'):
            print("Saving & exiting...")
            joblib.dump(results, OUT_REVIEW_PKL)
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    cv2.destroyAllWindows()

cap.release()
joblib.dump(results, OUT_REVIEW_PKL)
print(f"\nSaved manual review → {OUT_REVIEW_PKL}")