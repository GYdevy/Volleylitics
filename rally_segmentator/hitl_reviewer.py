import cv2
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RALLY_DIR = Path(__file__).resolve().parent

RALLY_OUTPUT_DIR = RALLY_DIR / "output"
VIDEO_DIR = PROJECT_ROOT / "videos"

PADDING = 0.0  # seconds


# =========================
# LOAD HITL
# =========================
def load_hitl(hitl_json):
    with open(hitl_json, "r") as f:
        return json.load(f)


# =========================
# PLAY CLIP + DECIDE
# =========================
def play_clip(start, end, h, i, decisions, video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Failed to open video")
        return "quit"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    start_frame = int(start * fps)
    end_frame = int(end * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current = start_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # =========================
        # OVERLAYS
        # =========================
        elapsed = (current - start_frame) / fps
        total = (end_frame - start_frame) / fps

        cv2.putText(frame, f"HITL {i}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2)

        cv2.putText(frame, f"{elapsed:.1f}s / {total:.1f}s",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

        cv2.putText(frame, "y=rally n=not s=skip q=quit",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        cv2.imshow("HITL Review", frame)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF

        # =========================
        # LIVE INPUT
        # =========================
        if key == ord('q'):
            cap.release()
            return "quit"

        elif key == ord('y'):
            decisions.append({
                "index": i,
                "decision": "rally",
                "start": h["start"],
                "end": h["end"]
            })
            break

        elif key == ord('n'):
            decisions.append({
                "index": i,
                "decision": "not_rally",
                "start": h["start"],
                "end": h["end"]
            })
            break

        elif key == ord('s'):
            break

        current += 1

        # =========================
        # END OF CLIP → FREEZE
        # =========================
        if current > end_frame:
            while True:
                cv2.imshow("HITL Review", frame)
                key = cv2.waitKey(50) & 0xFF

                if key == ord('y'):
                    decisions.append({
                        "index": i,
                        "decision": "rally",
                        "start": h["start"],
                        "end": h["end"]
                    })
                    break

                elif key == ord('n'):
                    decisions.append({
                        "index": i,
                        "decision": "not_rally",
                        "start": h["start"],
                        "end": h["end"]
                    })
                    break

                elif key == ord('s'):
                    break

                elif key == ord('q'):
                    cap.release()
                    return "quit"
            break

    cap.release()
    return None


# =========================
# REVIEW LOOP
# =========================
def review(video_path, hitl_json, previous_decisions=None):
    hitls = load_hitl(hitl_json)
    decisions_by_index = {
        int(decision["index"]): decision
        for decision in (previous_decisions or [])
        if "index" in decision
        and "decision" in decision
        and 0 <= int(decision["index"]) < len(hitls)
    }
    decisions = list(decisions_by_index.values())

    for i, h in enumerate(hitls):
        if i in decisions_by_index:
            continue
        start = max(0, h["start"] - PADDING)
        end = h["end"] + PADDING

        print("\n==========================")
        print(f"HITL {i}")
        print(f"{h['start']:.2f} → {h['end']:.2f}")
        if "ball_ratio" in h:
            visual_summary = (
                f"ball_ratio={h['ball_ratio']:.2f} | "
                f"ball_hits={h.get('ball_hits', 0)} | "
                f"ball_conf={h.get('ball_confidence', 0.0):.2f}"
            )
        else:
            visual_summary = f"yellow={h.get('yellow', 0.0):.2f}"
        print(
            f"dur={h['duration']:.2f} | ratio={h['ratio']:.2f} | "
            f"{visual_summary} | score={h['score']:.2f}"
        )
        
        result = play_clip(start, end, h, i, decisions,video_path)


        if result == "quit":
            break

    cv2.destroyAllWindows()
    return sorted(decisions, key=lambda item: item["index"]), hitls

def overlaps(a, b):
    return not (a["end"] < b["start"] or b["end"] < a["start"])

def merge_intervals(r1, r2):
    return {
        "start": min(r1["start"], r2["start"]),
        "end": max(r1["end"], r2["end"]),
        "duration": max(r1["end"], r2["end"]) - min(r1["start"], r2["start"]),
        "label": "MERGED_HITL",
        "set": r1.get("set") or r2.get("set")
    }


def add_hitl_to_rallies(rallies, hitl_decisions,hitls):
   


    hitl_rallies = []

    for d in hitl_decisions:
        if d["decision"] != "rally":
            continue

        original = hitls[d["index"]]

        hitl_rallies.append({
            "start": d["start"],
            "end": d["end"],
            "duration": d["end"] - d["start"],
            "label": "HITL",
            "set": original.get("set")   # ✅ THIS FIXES IT
        })
    all_rallies = rallies.copy()

    for h in hitl_rallies:
        merged = False

        for i in range(len(all_rallies)):
            r = all_rallies[i]

            if overlaps(r, h):
                all_rallies[i] = merge_intervals(r, h)
                merged = True
                break

        if not merged:
            all_rallies.append(h)

    # final merge pass
    all_rallies = sorted(all_rallies, key=lambda x: x["start"])
    merged_final = []

    for r in all_rallies:
        if not merged_final:
            merged_final.append(r)
            continue

        prev = merged_final[-1]

        if overlaps(prev, r) or r["start"] <= prev["end"] + 0.5:
            merged_final[-1] = merge_intervals(prev, r)
        else:
            merged_final.append(r)

    for r in merged_final:
        r["duration"] = r["end"] - r["start"]

    return merged_final
# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", help="Original full-match video")
    parser.add_argument("--match-dir", help="Analyzed match output directory")
    parser.add_argument("--reset-decisions", action="store_true")
    args = parser.parse_args()

    match_id = args.match_id

    video_path = Path(args.video).expanduser().resolve() if args.video else VIDEO_DIR / f"{match_id}.mp4"

    match_dir = Path(args.match_dir).expanduser().resolve() if args.match_dir else RALLY_OUTPUT_DIR / match_id
    hitl_json = match_dir / "hitl.json"
    rallies_json = match_dir / "rallies.json"
    final_output_json = match_dir / "rallies_with_hitl.json"
    decisions_json = match_dir / "hitl_decisions.json"
    if not video_path.is_file():
        raise FileNotFoundError(f"Original video not found: {video_path}")
    if not hitl_json.is_file():
        raise FileNotFoundError(f"Uncertain intervals not found: {hitl_json}")
    if not rallies_json.is_file():
        raise FileNotFoundError(f"Rally timeline not found: {rallies_json}")

    previous_decisions = []
    if decisions_json.is_file() and not args.reset_decisions:
        with decisions_json.open("r", encoding="utf-8") as decision_file:
            previous_decisions = json.load(decision_file)

    decisions, hitls = review(str(video_path), hitl_json, previous_decisions)
    with decisions_json.open("w", encoding="utf-8") as decision_file:
        json.dump(decisions, decision_file, indent=2)

    with open(rallies_json, "r") as f:
        rallies = json.load(f)

    print("Original rallies:", len(rallies))

    merged = add_hitl_to_rallies(rallies, decisions, hitls)

    print("After HITL merge:", len(merged))

    match_dir.mkdir(parents=True, exist_ok=True)

    with open(final_output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print("Review decisions:", len(decisions), "/", len(hitls))
    print("FINAL saved to:", final_output_json)
