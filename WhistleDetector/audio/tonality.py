import numpy as np

def is_tonal_frame(spec, max_bins=6):
    peak = spec.max()
    if peak < -60:
        return False
    active_bins = np.sum(spec > (peak - 6))
    return active_bins <= max_bins

def split_by_tonality(group_frames, S_w, min_frames, max_gap=3):
    tonal = []
    for f in group_frames:
        tonal.append(f if is_tonal_frame(S_w[:, f]) else None)

    subgroups, cur = [], []
    for f in tonal:
        if f is not None:
            if not cur or f - cur[-1] <= max_gap:
                cur.append(f)
            else:
                if len(cur) >= min_frames:
                    subgroups.append(cur)
                cur = [f]
        else:
            if cur and len(cur) >= min_frames:
                subgroups.append(cur)
            cur = []

    if cur and len(cur) >= min_frames:
        subgroups.append(cur)

    return subgroups
