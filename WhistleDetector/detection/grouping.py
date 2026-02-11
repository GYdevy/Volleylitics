from WhistleDetector.config import MAX_GAP_FRAMES

def group_frames(active):
    groups = []
    cur = [active[0]] if active else []

    for f in active[1:]:
        if f - cur[-1] <= MAX_GAP_FRAMES:
            cur.append(f)
        else:
            groups.append(cur)
            cur = [f]

    if cur:
        groups.append(cur)

    return groups
