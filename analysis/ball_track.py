"""Unattended ball extraction — scalable replacement for hand-audited anchors.

Three stages, each fixing a real failure mode of ball tracking:

1. DETECT + ASSOCIATE (adapted from ai-football-analytics' ball tracker, MIT):
   dedicated ball detector every 2nd frame; association by inflated-box IoU
   with center-distance fallback against a velocity-predicted position
   (the ball out-runs its own box size per frame, so plain IoU fails);
   tracks confirm after 2 consecutive hits. Raw detections are cached to
   ball_dets_raw.json so selection can be iterated without re-detection.

2. PRUNE STATICS: a "ball" that doesn't move more than 3m over its life is
   furniture — a spare ball behind the goal, a keeper's white gloves, a
   line intersection. Real match balls travel.

3. GLOBAL PATH SELECTION (dynamic programming): among all remaining
   confirmed observations, pick the time-ordered path that maximizes
   coverage + confidence subject to ball physics (<= 45 m/s between
   points). Isolated false-positive islands lose to the long consistent
   trajectory instead of stealing time slots from it. A final pass removes
   physically impossible zig-zags (>120 degree reversal with both legs
   > 3m inside 1s).

The goal mouth is playable volume: x may go a few meters beyond the goal
line so a scored ball ends IN the net, not clipped at the line.

Usage: python3 ball_track.py <video> <calibration.json> [out_anchors.json]
"""
import json
import os
import sys
import cv2
import numpy as np

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "clip1080.mp4"
CALIB = sys.argv[2] if len(sys.argv) > 2 else "pnlH1080.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else "ball_anchors_auto.json"
CACHE = os.path.splitext(VIDEO)[0] + ".ball_dets_raw.json"
STRIDE = 2
CONF = 0.12
INFLATE, MIN_BOX = 2.5, 22.0
DIST_GATE = 70.0
MAX_GAP = 20
MIN_HITS = 2
X_MARGIN, Y_MARGIN = 4.0, 2.0        # meters beyond the lines (goal interior)
STATIC_EXTENT_M = 3.0                # tracks that move less than this...
STATIC_MIN_DUR = 1.0                 # ...over at least this long are furniture
V_MAX = 45.0                         # m/s hard physics cap between path points
DP_LOOKBACK = 1.8                    # s

anchors_H = {float(k): np.array(v) for k, v in json.load(open(CALIB)).items()}
def H_at(t): return anchors_H[min(anchors_H, key=lambda u: abs(u - t))]
def to_pitch(H, px, py):
    v = H @ np.array([px, py, 1.0]); return float(v[0]/v[2]), float(v[1]/v[2])

# ---------------- stage 1: detect (cached) + associate ----------------
def detect_all():
    if os.path.exists(CACHE):
        print(f"using cached detections {CACHE}")
        return {float(k): v for k, v in json.load(open(CACHE)).items()}
    from ultralytics import YOLO
    model = YOLO("rfmodels/football-ball-detection.pt")
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dets = {}
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if fi % STRIDE == 0:
            res = model.predict(frame, verbose=False, conf=CONF, imgsz=1280)[0]
            d = [[*map(float, b[:4]), float(c)]
                 for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())]
            if d: dets[round(fi / fps, 4)] = d
            if fi % 100 == 0:
                print(f"f={fi:4d} t={fi/fps:5.2f}s dets={len(d)}", flush=True)
        fi += 1
    cap.release()
    json.dump(dets, open(CACHE, "w"))
    return dets

def inflate(b):
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    w = max((b[2]-b[0])*INFLATE, MIN_BOX); h = max((b[3]-b[1])*INFLATE, MIN_BOX)
    return (cx-w/2, cy-h/2, cx+w/2, cy+h/2)

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2-x1, 0) * max(y2-y1, 0)
    if inter <= 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua

class BTrack:
    _next = 1
    def __init__(s, box, conf, fi):
        s.id = BTrack._next; BTrack._next += 1
        s.box, s.conf, s.last = box, conf, fi
        s.center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        s.vel = (0.0, 0.0); s.hits = 1; s.n = 1; s.confirmed = False
        s.obs = [(fi, s.center, conf)]
    def pred_center(s, fi):
        dt = fi - s.last
        return (s.center[0] + s.vel[0]*dt, s.center[1] + s.vel[1]*dt)
    def pred_box(s, fi):
        dt = fi - s.last; dx, dy = s.vel[0]*dt, s.vel[1]*dt
        return (s.box[0]+dx, s.box[1]+dy, s.box[2]+dx, s.box[3]+dy)
    def match(s, box, conf, fi):
        c = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        dt = max(fi - s.last, 1)
        mv = ((c[0]-s.center[0])/dt, (c[1]-s.center[1])/dt)
        s.vel = mv if s.n == 1 else (0.5*s.vel[0]+0.5*mv[0], 0.5*s.vel[1]+0.5*mv[1])
        s.box, s.center, s.conf, s.last = box, c, conf, fi
        s.hits += 1; s.n += 1
        if s.hits >= MIN_HITS: s.confirmed = True
        s.obs.append((fi, c, conf))

def associate(dets_by_t, fps):
    tracks, dead = [], []
    for t in sorted(dets_by_t):
        fi = round(t * fps)
        dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in dets_by_t[t]]
        cands = []
        for ti, tr in enumerate(tracks):
            pb = inflate(tr.pred_box(fi)); pc = tr.pred_center(fi)
            for di, (box, conf) in enumerate(dets):
                i = iou(pb, inflate(box))
                c = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                d = np.hypot(pc[0]-c[0], pc[1]-c[1])
                gate = DIST_GATE + 12.0 * (fi - tr.last)
                if i > 0: cands.append((1-i, ti, di))
                elif d <= gate: cands.append((1 + d/gate, ti, di))
        cands.sort()
        ut, ud = set(), set()
        for cost, ti, di in cands:
            if ti in ut or di in ud: continue
            ut.add(ti); ud.add(di)
            tracks[ti].match(*dets[di], fi)
        for ti, tr in enumerate(tracks):
            if ti not in ut and tr.last < fi: tr.hits = 0
        for di, (box, conf) in enumerate(dets):
            if di not in ud: tracks.append(BTrack(box, conf, fi))
        alive = []
        for tr in tracks:
            if fi - tr.last > MAX_GAP:
                if tr.confirmed and tr.n >= 2: dead.append(tr)
            else: alive.append(tr)
        tracks = alive
    dead.extend(tr for tr in tracks if tr.confirmed and tr.n >= 2)
    return dead

# ---------------- stage 2: project + prune statics ----------------
def candidate_points(tracks, fps):
    pts = []
    n_static = 0
    for tr in tracks:
        proj = []
        for f, c, conf in tr.obs:
            t = f / fps
            X, Y = to_pitch(H_at(t), *c)
            if -X_MARGIN <= X <= 105 + X_MARGIN and -Y_MARGIN <= Y <= 68 + Y_MARGIN:
                proj.append((round(t, 3), X, Y, conf))
        if len(proj) < 2: continue
        xs = [p[1] for p in proj]; ys = [p[2] for p in proj]
        extent = np.hypot(max(xs)-min(xs), max(ys)-min(ys))
        dur = proj[-1][0] - proj[0][0]
        if extent < STATIC_EXTENT_M and dur >= STATIC_MIN_DUR:
            n_static += 1
            continue                             # furniture, not the match ball
        pts.extend(proj)
    pts.sort()
    # persistent-spot suppression: track fragmentation can smuggle a static
    # object past the per-track prune, so also drop any point whose SPOT is
    # occupied by other "ball" detections seconds apart — a real ball moves on
    keep = []
    for i, (t, x, y, c) in enumerate(pts):
        support = sum(1 for (t2, x2, y2, _) in pts
                      if abs(t2 - t) > 2.0 and np.hypot(x2-x, y2-y) < 1.5)
        if support < 3:
            keep.append(pts[i])
    print(f"candidates: {len(keep)} points ({n_static} static tracks pruned, "
          f"{len(pts)-len(keep)} persistent-spot points dropped)")
    return keep

# ---------------- stage 3: global DP path + physics filter ----------------
def best_path(pts):
    n = len(pts)
    dp = [1.0 + 0.5*p[3] for p in pts]
    prev = [-1] * n
    for i in range(n):
        ti, xi, yi, ci = pts[i]
        j = i - 1
        while j >= 0 and ti - pts[j][0] <= DP_LOOKBACK:
            tj, xj, yj, cj = pts[j]
            dt = ti - tj
            if dt > 0:
                d = np.hypot(xi-xj, yi-yj)
                if d / dt <= V_MAX:
                    cand = dp[j] + 1.0 + 0.5*ci
                    if cand > dp[i]:
                        dp[i] = cand; prev[i] = j
            j -= 1
    i = int(np.argmax(dp))
    path = []
    while i != -1:
        path.append(pts[i]); i = prev[i]
    path.reverse()
    # physics filter: remove impossible zig-zags (sharp reversal, long legs)
    changed = True
    while changed and len(path) > 2:
        changed = False
        for k in range(1, len(path)-1):
            a, b, c = path[k-1], path[k], path[k+1]
            v1 = np.array([b[1]-a[1], b[2]-a[2]]); v2 = np.array([c[1]-b[1], c[2]-b[2]])
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if l1 > 3 and l2 > 3 and c[0]-a[0] < 1.2:
                cos = float(v1 @ v2) / (l1*l2)
                if cos < -0.5:                    # > 120 degree reversal
                    path.pop(k); changed = True; break
    return path

if __name__ == "__main__":
    fps = cv2.VideoCapture(VIDEO).get(cv2.CAP_PROP_FPS)
    dets = detect_all()
    tracks = associate(dets, fps)
    print(f"confirmed tracks: {len(tracks)}")
    pts = candidate_points(tracks, fps)
    path = best_path(pts)
    print(f"selected path: {len(path)} points, "
          f"{path[0][0] if path else '-'}s .. {path[-1][0] if path else '-'}s")

    anchors = {}
    tmax = max(dets) if dets else 0
    t = 0.0
    while t <= tmax + 0.5:
        near = [p for p in path if abs(p[0] - t) <= 0.26]
        if near:
            p = min(near, key=lambda p: (abs(p[0]-t), -p[3]))
            anchors[round(t, 2)] = {"pitch": [round(p[1], 2), round(p[2], 2)],
                                    "conf": round(p[3], 3), "src": "auto"}
        t = round(t + 0.5, 2)
    if path:                                     # ball's final resting point
        p = path[-1]
        anchors[round(p[0], 2)] = {"pitch": [round(p[1], 2), round(p[2], 2)],
                                   "conf": round(p[3], 3), "src": "auto"}

    # anchor-level physics filter, by DETOUR COST: an anchor that forces the
    # ball far off the line between its neighbors and straight back is a
    # spike, and the spike itself (largest detour) is the point to remove —
    # a plain reversal check can't tell which of the three points is lying.
    while True:
        ts = sorted(anchors)
        worst_t, worst_detour = None, 10.0       # meters; real motion stays below
        for k in range(1, len(ts)-1):
            ta, tb, tc = ts[k-1], ts[k], ts[k+1]
            if tc - ta > 2.4: continue
            pa, pb, pc = (np.array(anchors[x]["pitch"]) for x in (ta, tb, tc))
            detour = (np.linalg.norm(pb-pa) + np.linalg.norm(pc-pb)
                      - np.linalg.norm(pc-pa))
            if detour > worst_detour:
                worst_detour, worst_t = detour, tb
        if worst_t is None: break
        print(f"  dropped spike anchor t={worst_t} (detour {worst_detour:.1f} m)")
        del anchors[worst_t]
    json.dump(anchors, open(OUT, "w"), indent=1)
    print(f"after anchor physics filter: {len(anchors)} anchors")
    json.dump(anchors, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(anchors)} anchors, span "
          f"{min(anchors) if anchors else '-'}..{max(anchors) if anchors else '-'}")
