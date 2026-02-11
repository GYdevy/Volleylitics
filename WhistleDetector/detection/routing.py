from sympy.codegen.cnodes import sizeof

from WhistleDetector.config import (
    CLS_THRESHOLD,
    AMBIG_MODEL_LOW,
    SCORE_OK,
    PHYSICS_MIN_PROBA,
    PEAK_STD_MAX,
    BANDWIDTH_MAX,
    TIMELINE_CSV,
    MATCH_NUM
)

from WhistleDetector.audio.features import extract_features, physics_suspect
from WhistleDetector.audio.utils import fmt_time
from WhistleDetector.detection.writer import append_timeline_row


def route_detection(detections, clf, ambig_clf):
    accepted = []
    ambiguous = []

    for d in detections:
        uid = (
            f"match{MATCH_NUM}_"
            f"{int(d['start'] * 1000):010d}_"
            f"{int(d['end'] * 1000):010d}"
        )
        feats, X = extract_features(d["audio"])

        p1 = clf.predict_proba(X.reshape(1, -1))[0, 1]

        ts = f"{fmt_time(d['start'])} → {fmt_time(d['end'])}"
        print(f"[ROUTE] {ts} p1={p1:.3f}")

        if p1 >= CLS_THRESHOLD:
            accepted.append(d)

            append_timeline_row(
                TIMELINE_CSV,
                uid,
                d["start"],
                d["end"],
                label="whistle",
                confidence=p1,
                source="auto_strong",
            )

            print("  → ACCEPT-1")
            continue

        if AMBIG_MODEL_LOW <= p1 < CLS_THRESHOLD:
            p2 = ambig_clf.predict_proba(X.reshape(1, -1))[0, 1]

            if (
                p2 >= 0.60
                and d["peak_std"] < PEAK_STD_MAX
                and d["bandwidth_hz"] < BANDWIDTH_MAX
            ):
                accepted.append(d)

                append_timeline_row(
                    TIMELINE_CSV,
                    uid,
                    d["start"],
                    d["end"],
                    label="whistle",
                    confidence=p2,
                    source="auto_ambig",
                )

                print("  → ACCEPT-2")
                continue
            else:
                ambiguous.append(d)
                append_timeline_row(
                    TIMELINE_CSV,
                    uid,
                    d["start"],
                    d["end"],
                    label="whistle",
                    confidence=p1,
                    source="needs_hitl",
                )
                print("  → AMBIG-2")
            continue

        if (
                d["core_score"] >= SCORE_OK
                and d["ridge"] >= 48  # from analysis
                and feats["grad_f"] >= 4.85  # from analysis
        ):
            ambiguous.append(d)
            append_timeline_row(
                TIMELINE_CSV,
                uid,
                d["start"],
                d["end"],
                label="whistle",
                confidence=p1,
                source="needs_hitl",
            )
            print("  → AMBIG-CORE")
            continue
        # 3.5️⃣ Physics-confirmed whistle
        if (
                p1 >= 0.20
                and feats["grad_f"] >= 4.85
                and d["ridge"] >= 48
                and feats["flatness"] <= 0.016
                and d["bandwidth_hz"] <= 2200
        ):
            accepted.append(d)

            append_timeline_row(
                TIMELINE_CSV,
                uid,
                d["start"],
                d["end"],
                label="whistle",
                confidence=p1,
                source="auto_phys",
            )

            print("  → ACCEPT-PHYS")
            continue

        # 4️⃣ Physics-based ambiguity
        if (
            PHYSICS_MIN_PROBA <= p1 < CLS_THRESHOLD
            and (d["noisy"] or physics_suspect(feats))
        ):
            ambiguous.append(d)
            append_timeline_row(
                TIMELINE_CSV,
                uid,
                d["start"],
                d["end"],
                label="whistle",
                confidence=p1,
                source="needs_hitl",
            )
            print("  → AMBIG-PHYS")
            continue

        print("  → REJECT")
    print(
        f"\n[RESULTS]"
        f"\n  Total Detections : {len(detections)}"
        f"\n  Accepted : {len(accepted)}"
        f"\n  Ambiguous: {len(ambiguous)}"
        f"\n  Rejected : {len(detections) - (len(accepted) + len(ambiguous))}"
    )
    return accepted, ambiguous
