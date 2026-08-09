"""Refine per-frame homographies by snapping the projected pitch model onto
detected white lines (Levenberg-Marquardt on a distance-transform cost).
Initialized from the pan-chain homographies; corrects both chain drift and
reference-calibration error."""
import json
import sys
import cv2
import numpy as np

PITCH_W, PITCH_H = 105.0, 68.0


def model_samples(step=0.75):
    """Dense sample points along every pitch line/arc, in meters."""
    pts = []
    midY = PITCH_H / 2
    def seg(a, b):
        n = max(2, int(np.hypot(b[0]-a[0], b[1]-a[1]) / step))
        for i in range(n + 1):
            t = i / n
            pts.append((a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t))
    seg((0,0),(PITCH_W,0)); seg((PITCH_W,0),(PITCH_W,PITCH_H))
    seg((PITCH_W,PITCH_H),(0,PITCH_H)); seg((0,PITCH_H),(0,0))
    seg((PITCH_W/2,0),(PITCH_W/2,PITCH_H))
    for x, d in ((0,1),(PITCH_W,-1)):
        seg((x, midY-20.16),(x+d*16.5, midY-20.16)); seg((x+d*16.5, midY-20.16),(x+d*16.5, midY+20.16)); seg((x+d*16.5, midY+20.16),(x, midY+20.16))
        seg((x, midY-9.16),(x+d*5.5, midY-9.16)); seg((x+d*5.5, midY-9.16),(x+d*5.5, midY+9.16)); seg((x+d*5.5, midY+9.16),(x, midY+9.16))
    for cx, a0, a1 in [(PITCH_W/2, 0, 2*np.pi),
                       (11, -np.arccos(5.5/9.15), np.arccos(5.5/9.15)),
                       (94, np.pi-np.arccos(5.5/9.15), np.pi+np.arccos(5.5/9.15))]:
        for a in np.arange(a0, a1, step / 9.15):
            pts.append((cx + 9.15*np.cos(a), midY + 9.15*np.sin(a)))
    return np.array(pts)


def line_mask(frame):
    """White pitch-line pixels: thin bright elongated structures on grass."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
    m = (th > 28).astype(np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = ((hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 60)).astype(np.uint8) * 255
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((31,31), np.uint8))
    cnts, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = np.zeros(m.shape, np.uint8)
    if cnts:
        cv2.drawContours(roi, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
        roi = cv2.erode(roi, np.ones((7,7), np.uint8))
    m &= roi
    # keep elongated structures only (drop players/blobs)
    keep = np.zeros_like(m)
    for ang in range(0, 180, 30):
        k = np.zeros((25, 25), np.uint8)
        c = 12
        dx, dy = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        cv2.line(k, (int(c-11*dx), int(c-11*dy)), (int(c+11*dx), int(c+11*dy)), 1, 1)
        keep |= cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return keep


def dt_of(mask):
    inv = (mask == 0).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def bilinear(dt, xs, ys):
    h, w = dt.shape
    xs = np.clip(xs, 0, w - 2); ys = np.clip(ys, 0, h - 2)
    x0 = xs.astype(int); y0 = ys.astype(int)
    fx = xs - x0; fy = ys - y0
    return (dt[y0, x0]*(1-fx)*(1-fy) + dt[y0, x0+1]*fx*(1-fy)
            + dt[y0+1, x0]*(1-fx)*fy + dt[y0+1, x0+1]*fx*fy)


def residuals(G, pts, dt, cap):
    P = np.c_[pts, np.ones(len(pts))] @ G.T
    w = P[:, 2]
    ok = w > 1e-6
    xs = np.full(len(pts), -1e4); ys = np.full(len(pts), -1e4)
    xs[ok] = P[ok, 0] / w[ok]; ys[ok] = P[ok, 1] / w[ok]
    h, wd = dt.shape
    vis = ok & (xs > 2) & (xs < wd-3) & (ys > 2) & (ys < h-3)
    r = np.zeros(len(pts))
    r[vis] = np.minimum(bilinear(dt, xs[vis], ys[vis]), cap)
    r[~vis] = 0.0
    return r, vis


def refine_frame(frame, H_img2pitch, iters=30):
    """Return refined image->pitch homography."""
    mask = line_mask(frame)
    dt = dt_of(mask)
    pts = model_samples()
    G = np.linalg.inv(H_img2pitch)
    G = G / G[2, 2]
    p = G.flatten()[:8].copy()
    lam = 1e-3
    def unpack(p):
        return np.array([*p, 1.0]).reshape(3, 3)
    for it in range(iters):
        cap = 25.0 if it < iters // 2 else 12.0
        r, vis = residuals(unpack(p), pts, dt, cap)
        n_vis = int(vis.sum())
        if n_vis < 50:
            return H_img2pitch, -1.0
        cost = (r[vis]**2).sum() / n_vis
        # numeric Jacobian
        J = np.zeros((len(pts), 8))
        for j in range(8):
            eps = max(1e-6, abs(p[j]) * 1e-4)
            p2 = p.copy(); p2[j] += eps
            r2, _ = residuals(unpack(p2), pts, dt, cap)
            J[:, j] = (r2 - r) / eps
        Jv = J[vis]; rv = r[vis]
        A = Jv.T @ Jv + lam * np.eye(8)
        try:
            dp = np.linalg.solve(A, -Jv.T @ rv)
        except np.linalg.LinAlgError:
            break
        p_new = p + dp
        r_new, vis_new = residuals(unpack(p_new), pts, dt, cap)
        if vis_new.sum() > 50 and (r_new[vis_new]**2).sum() / vis_new.sum() < cost:
            p = p_new
            lam = max(lam * 0.5, 1e-6)
        else:
            lam = min(lam * 4, 1e3)
    r, vis = residuals(unpack(p), pts, dt, 12.0)
    mean_px = float(r[vis].mean()) if vis.sum() else -1
    return np.linalg.inv(unpack(p)), mean_px


def main():
    chain_file, video, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
    only = [float(t) for t in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
    chain = {float(k): np.array(v) for k, v in json.load(open(chain_file)).items()}
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = {}
    for t in sorted(chain):
        if only is not None and not any(abs(t - u) < 1e-6 for u in only):
            out[str(t)] = chain[t].tolist()
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
        ok, frame = cap.read()
        if not ok:
            out[str(t)] = chain[t].tolist()
            continue
        H, err = refine_frame(frame, chain[t])
        out[str(t)] = H.tolist()
        print(f"t={t:6.2f}s  mean line residual: {err:5.2f}px")
    cap.release()
    json.dump(out, open(out_file, "w"))
    print("wrote", out_file)


if __name__ == "__main__":
    main()
