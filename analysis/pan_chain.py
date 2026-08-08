"""Chain frame-to-frame homographies (for panning/zooming cameras) so one calibrated reference frame
covers the whole panning clip. Valid because a pilot camera rotates/zooms
from a fixed point, which relates any two frames by a global homography."""
import json
import cv2
import numpy as np

import argparse
_ap = argparse.ArgumentParser(description="Track camera pan/zoom so one calibrated frame covers a whole clip. Outputs per-time pitch homographies for match_to_tactic.py --chain.")
_ap.add_argument("video")
_ap.add_argument("--calib", required=True, help="calib.json for the reference frame (from calibrate.html)")
_ap.add_argument("--ref-time", type=float, required=True, help="seconds: the frame the calibration was made on")
_ap.add_argument("--step", type=float, default=0.25, help="chaining step in seconds")
_ap.add_argument("--out", default="chainH.json")
_args = _ap.parse_args()
VIDEO = _args.video
REF_T = _args.ref_time
STEP = _args.step
CALIB = _args.calib

def static_mask(shape):
    m = np.full(shape[:2], 255, np.uint8)
    h, w = shape[:2]
    m[: int(h * 0.10), :] = 0            # broadcast/recording chrome
    m[int(h * 0.95):, :] = 0
    m[int(h*0.17):int(h*0.26), int(w*0.09):int(w*0.28)] = 0   # scoreboard area
    return m

def load_ref_H():
    pairs = json.load(open(CALIB))["points"]
    src = np.array([p["px"] for p in pairs], np.float64)
    dst = np.array([p["pitch"] for p in pairs], np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    return H

def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    times = [round(i * STEP, 4) for i in range(int((n / fps) / STEP) + 1)]
    # read all sampled frames (gray)
    frames = {}
    for t in times:
        idx = round(t * fps)
        if idx >= n: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok: break
        frames[t] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    cap.release()
    ts = sorted(frames)
    ref_t = min(ts, key=lambda t: abs(t - REF_T))
    print(f"{len(ts)} frames, reference at t={ref_t}")

    sift = cv2.SIFT_create(nfeatures=4000)
    mask = static_mask(next(iter(frames.values())).shape)
    feats = {}
    def get_feats(t):
        if t not in feats:
            feats[t] = sift.detectAndCompute(frames[t], mask)
        return feats[t]

    matcher = cv2.BFMatcher()
    def pair_H(t1, t2):
        """homography mapping frame t1 -> frame t2"""
        k1, d1 = get_feats(t1)
        k2, d2 = get_feats(t2)
        if d1 is None or d2 is None: return None, 0
        m = matcher.knnMatch(d1, d2, k=2)
        good = [a for a, b in m if a.distance < 0.75 * b.distance]
        if len(good) < 12: return None, len(good)
        p1 = np.float32([k1[g.queryIdx].pt for g in good])
        p2 = np.float32([k2[g.trainIdx].pt for g in good])
        H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
        return H, 0 if inl is None else int(inl.sum())

    # chain outward from the reference
    Hs = {ref_t: np.eye(3)}          # H mapping frame t -> reference frame
    order = sorted(ts, key=lambda t: abs(t - ref_t))
    for t in order:
        if t in Hs: continue
        prev = min((u for u in Hs), key=lambda u: abs(u - t))
        H, inliers = pair_H(t, prev)
        if H is None:
            print(f"  t={t}: chain broken (inliers={inliers})")
            continue
        Hs[t] = Hs[prev] @ H
        if abs((t * 4) % 8) < 1e-6:
            print(f"  t={t:5.2f}s chained via {prev:5.2f}s  inliers={inliers}")

    H_ref = load_ref_H()             # reference image -> pitch meters
    out = {str(t): (H_ref @ Hs[t]).tolist() for t in Hs}
    json.dump(out, open(_args.out, "w"))
    print(f"wrote {_args.out} with {len(out)} frame homographies")

if __name__ == "__main__":
    main()
