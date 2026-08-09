"""Sequential line-locking: walk outward from the verified reference frame,
initializing each frame from its refined neighbour composed with the
feature-tracked inter-frame camera motion. Guards against mislocks by
rejecting solutions that jump too far from their initialization."""
import json
import cv2
import numpy as np
import refine as rf

VIDEO = "TestGoal.mov"
REF_T = 14.25

chain_pitch = {float(k): np.array(v) for k, v in json.load(open("chainH.json")).items()}
# recover frame->ref-frame homographies: chainH stored H_refcalib @ Hs[t]
# so Hs[t] = H_refcalib^-1 @ chainH[t]; we only need relative motions, and
# H_refcalib cancels: Hs[t1]^-1 @ Hs[t2] = chainH[t1]^-1 @ chainH[t2]
ts = sorted(chain_pitch)

# start from the current refined reference (visually verified)
refined_all = {float(k): np.array(v) for k, v in json.load(open("refinedH-all.json")).items()}

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
def read_frame(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
    ok, f = cap.read()
    return f if ok else None

score_pts = rf.model_samples(step=1.0)
def align_score(frame_mask_dt, H):
    """median distance of visible model samples to nearest line pixel -
    computed WITHOUT trimming, so a mislocked fit cannot hide."""
    G = np.linalg.inv(H); G /= G[2,2]
    r, vis = rf.residuals(G, score_pts, frame_mask_dt, 60.0)
    return float(np.median(r[vis])) if vis.sum() > 50 else 1e9

out = {}
# refine the reference itself first (tight, init from existing refined)
f = read_frame(REF_T)
H0, err0 = rf.refine_frame(f, refined_all[REF_T])
out[REF_T] = H0
print(f"ref t={REF_T}: residual {err0:.2f}px")

order = sorted([t for t in ts if t != REF_T], key=lambda t: abs(t - REF_T))
for t in order:
    prev_t = min(out, key=lambda u: abs(u - t))
    M_rel = np.linalg.inv(chain_pitch[prev_t]) @ chain_pitch[t]   # frame_t px -> frame_prev px
    H_init = out[prev_t] @ M_rel                                   # frame_t px -> pitch
    frame = read_frame(t)
    if frame is None:
        out[t] = H_init
        continue
    H_ref, err = rf.refine_frame(frame, H_init)
    dt = rf.dt_of(rf.line_mask(frame))
    s_init, s_ref = align_score(dt, H_init), align_score(dt, H_ref)
    if err < 0 or s_ref > s_init:
        out[t] = H_init
        tag = "init"
        s = s_init
    else:
        out[t] = H_ref
        tag = "refined"
        s = s_ref
    if abs(t*2 % 4) < 1e-9 or s > 5:
        print(f"t={t:6.2f}s kept {tag:7s} score {s:5.2f}px (init {s_init:.2f} / refined {s_ref:.2f})")
cap.release()

json.dump({str(t): out[t].tolist() for t in out}, open("refinedH-seq.json", "w"))
print("wrote refinedH-seq.json")
