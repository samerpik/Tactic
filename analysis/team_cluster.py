"""Match-agnostic team discovery (adapted from ai-football-analytics, MIT).

Instead of hardcoded kit rules, cluster the stable outfield tracks' shirt
colors in Lab space with k=2 k-means (k-means++ restarts). Tracks far from
both centroids are outliers = keeper/referee candidates. Also builds a
per-observation team-label timeline per track to detect mid-track identity
swaps (label flips), which motion-only stitching cannot see.
"""
import json
import cv2
import numpy as np

SEED = 7
OUTLIER_DE = 46.0        # Lab distance beyond which a track is no team's

def to_lab(bgr):
    return cv2.cvtColor(np.uint8([[np.clip(bgr, 0, 255)]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)

def kmeans2(X, seed=SEED, n_init=6, iters=60):
    best = None
    for i in range(n_init):
        rng = np.random.RandomState(seed + i)
        c0 = X[rng.randint(len(X))]
        d = np.linalg.norm(X - c0, axis=1) ** 2
        c1 = X[rng.choice(len(X), p=d / d.sum())]
        C = np.stack([c0, c1])
        for _ in range(iters):
            lab = np.argmin(np.linalg.norm(X[:, None] - C[None], axis=2), axis=1)
            newC = np.stack([X[lab == j].mean(axis=0) if (lab == j).any() else C[j]
                             for j in (0, 1)])
            if np.allclose(newC, C): break
            C = newC
        inertia = sum(np.linalg.norm(X[lab == j] - C[j], axis=1).sum() for j in (0, 1))
        if best is None or inertia < best[0]:
            best = (inertia, C, lab)
    return best[1], best[2]

def discover_teams(obs, min_len_s=2.0):
    """obs: id -> {"ts": [(t, img)], "cls": {...}, "colors": [...]}
    Returns (centroids, team_of, outliers, timelines)."""
    stable, vecs = [], []
    for tid, o in obs.items():
        if not o["colors"]: continue
        tot = sum(o["cls"].values())
        if o["cls"].get(3, 0) > tot / 2: continue          # referee class
        if o["cls"].get(1, 0) >= 0.3 * tot: continue        # keeper class
        ts = sorted(t for t, _ in o["ts"])
        if ts[-1] - ts[0] < min_len_s: continue
        stable.append(tid)
        vecs.append(to_lab(np.median(o["colors"], axis=0)))
    X = np.stack(vecs)
    C, _ = kmeans2(X)
    # refit on inliers so outliers don't drag the centroids
    d = np.linalg.norm(X[:, None] - C[None], axis=2)
    near = d.argmin(axis=1); neard = d.min(axis=1)
    inl = neard <= OUTLIER_DE
    if inl.sum() >= 2:
        C = np.stack([X[inl & (near == j)].mean(axis=0) if (inl & (near == j)).any()
                      else C[j] for j in (0, 1)])
    team_of, outliers = {}, []
    for tid, v in zip(stable, X):
        dd = np.linalg.norm(v - C, axis=1)
        j = int(dd.argmin())
        if dd[j] <= OUTLIER_DE: team_of[tid] = j
        else: outliers.append(tid)
    return C, team_of, outliers

def label_timeline(o, C, sample_every=3):
    """Per-observation nearest-team labels (None = outlier) for swap checks."""
    out = []
    for k, ((t, _), col) in enumerate(zip(o["ts"], o["colors"])):
        if k % sample_every: continue
        v = to_lab(col)
        dd = np.linalg.norm(v - C, axis=1)
        j = int(dd.argmin())
        out.append((t, j if dd[j] <= OUTLIER_DE else None))
    return out

def find_flips(timeline, min_run=8):
    """Runs where the label departs from the dominant team (swap/drift)."""
    lab = [(t, l) for t, l in timeline if l is not None]
    if len(lab) < 2 * min_run: return None, []
    teams = [l for _, l in lab]
    dom = max(set(teams), key=teams.count)
    runs, cur = [], []
    for t, l in lab:
        if l != dom: cur.append((t, l))
        else:
            if len(cur) >= min_run: runs.append((cur[0][0], cur[-1][0], len(cur)))
            cur = []
    if len(cur) >= min_run: runs.append((cur[0][0], cur[-1][0], len(cur)))
    return dom, runs

if __name__ == "__main__":
    tracks_raw = json.load(open("tracks_bt.json"))
    obs = {}
    for kf in tracks_raw:
        for dd in kf["dets"]:
            o = obs.setdefault(dd["id"], {"ts": [], "cls": {}, "colors": []})
            o["ts"].append((kf["t"], dd["img"]))
            o["cls"][dd["cls"]] = o["cls"].get(dd["cls"], 0) + 1
            if "color" in dd: o["colors"].append(dd["color"])

    C, team_of, outliers = discover_teams(obs)
    for j, c in enumerate(C):
        bgr = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_LAB2BGR)[0, 0]
        print(f"team {j} centroid Lab={np.round(c,1)} ~BGR={bgr}")
    print(f"assigned {len(team_of)} tracks, outliers: {outliers}")

    # compare with the hardcoded-rule assignment used in v6
    def kit_rule(colors):
        b = np.uint8([[np.median(colors, axis=0)]])
        h, s, v = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[0, 0].astype(int)
        if h < 28 and s >= 90: return "orange"
        if 30 <= h < 78 and s >= 50: return "green"
        cy = (44 <= s) * max(0, 34 - abs(int(h) - 95)); wh = max(0, 44 - int(s))
        return "cyan" if cy > wh else "white"
    diffs = 0
    for tid, j in sorted(team_of.items()):
        rule = kit_rule(obs[tid]["colors"])
        tag = {0: None, 1: None}
        # figure out which cluster is which by centroid hue
        hsv0 = cv2.cvtColor(np.uint8([[cv2.cvtColor(np.uint8([[C[0]]]), cv2.COLOR_LAB2BGR)[0,0]]]), cv2.COLOR_BGR2HSV)[0,0]
        cyan_cluster = 0 if (78 <= hsv0[0] <= 112 and hsv0[1] >= 44) else 1
        want = "cyan" if j == cyan_cluster else "white"
        if rule != want:
            diffs += 1
            print(f"  DIFF id={tid}: cluster says {want}, rule said {rule}")
    print(f"agreement with hardcoded rules: {len(team_of)-diffs}/{len(team_of)}")

    # swap / drift detection over per-observation labels
    n_flips = 0
    for tid, o in sorted(obs.items()):
        if not o["colors"] or len(o["ts"]) < 50: continue
        dom, runs = find_flips(label_timeline(o, C))
        for a, b, n in runs:
            n_flips += 1
            print(f"  FLIP id={tid} (dom team {dom}): looks like other team "
                  f"{a:.2f}-{b:.2f}s ({n} samples)")
    print(f"mid-track team-label flips: {n_flips}")
