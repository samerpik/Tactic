"""Build extraction v6 from the full-rate ByteTrack pass.

Track identity comes from ByteTrack (image space, 25fps); positions are
projected through the per-frame neural calibration only at the calibrated
0.5s marks; the audited ball trajectory is reused unchanged.
Outputs fine_positions_bt.json (renderer schema) + bt_page_data.json (PoC page).
"""
import json
import cv2
import numpy as np

FINE = 0.5
tracks_raw = json.load(open("tracks_bt.json"))
anchors = {float(k): np.array(v) for k, v in json.load(open("pnlH1080.json")).items()}
old = json.load(open("fine_positions.json"))
OUT_T = old["times"]

def H_at(t):
    return anchors[min(anchors, key=lambda u: abs(u - t))]

def to_pitch(H, px, py):
    v = H @ np.array([px, py, 1.0])
    return float(v[0]/v[2]), float(v[1]/v[2])

# ---------- gather per-id observations ----------
obs = {}   # id -> {"ts": [(t, img)], "cls": Counter-ish, "colors": []}
for kf in tracks_raw:
    for d in kf["dets"]:
        o = obs.setdefault(d["id"], {"ts": [], "cls": {}, "colors": []})
        o["ts"].append((kf["t"], d["img"]))
        o["cls"][d["cls"]] = o["cls"].get(d["cls"], 0) + 1
        if "color" in d:
            o["colors"].append(d["color"])

def kit_of_color(bgr_mean):
    """full classifier: keepers wear green (Ederson) / orange (Lloris)."""
    b = np.uint8([[bgr_mean]])
    h, s, v = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[0, 0].astype(int)
    if h < 28 and s >= 90: return "orange"
    if 30 <= h < 78 and s >= 50: return "green"
    cyan_score = (44 <= s) * max(0, 34 - abs(int(h) - 95))
    white_score = max(0, 44 - int(s))
    return "cyan" if cyan_score > white_score else "white"

def pitch_at(o, t, tol):
    """nearest observation of this id within tol seconds, projected."""
    best = min(o["ts"], key=lambda x: abs(x[0] - t))
    if abs(best[0] - t) > tol: return None
    X, Y = to_pitch(H_at(best[0]), *best[1])
    if not (-1.5 <= X <= 106.5 and -1.5 <= Y <= 69.5): return None
    return (X, Y)

tracks = []
for tid, o in obs.items():
    o["ts"].sort()
    tot = sum(o["cls"].values())
    if o["cls"].get(3, 0) > tot / 2: continue    # referees excluded by class
    span = o["ts"][-1][0] - o["ts"][0][0]
    if span < 0.4: continue                      # blips
    kitc = kit_of_color(np.mean(o["colors"], axis=0)) if o["colors"] else None
    # keeper: enough keeper-class votes OR a keeper kit color
    keeper = o["cls"].get(1, 0) >= 0.3 * tot or kitc in ("green", "orange")
    # touchline officials project just off the pitch - drop by median position
    pts = [to_pitch(H_at(t), *im) for t, im in o["ts"][::5]]
    mx = float(np.median([p[0] for p in pts])); my = float(np.median([p[1] for p in pts]))
    if not (-0.3 <= mx <= 105.3 and -0.3 <= my <= 67.7): continue
    tracks.append({"id": tid, "o": o, "role": 1 if keeper else 2, "team": kitc,
                   "t0": o["ts"][0][0], "t1": o["ts"][-1][0], "span": span})

print(f"raw ids: {len(obs)}, kept: {len(tracks)} "
      f"(keepers {sum(1 for x in tracks if x['role']==1)})")

# ---------- stitch broken fragments (same team, small gap, plausible motion) --
def try_stitch(pool):
    pool.sort(key=lambda x: x["t0"])
    merged = True
    while merged:
        merged = False
        for a in pool:
            best = None
            for b in pool:
                if a is b: continue
                gap = b["t0"] - a["t1"]
                # small negative gap allowed: ByteTrack can briefly run the
                # old and new id in parallel around an identity break
                if not (-0.3 <= gap <= 3.0): continue
                pa = pitch_at(a["o"], a["t1"], 0.1)
                pb = pitch_at(b["o"], max(b["t0"], a["t1"]), 0.4)
                if not pa or not pb: continue
                d = np.hypot(pa[0]-pb[0], pa[1]-pb[1])
                if d <= 2.0 + 4.0 * max(gap, 0) and (best is None or d < best[1]):
                    best = (b, d)
            if best:
                b = best[0]
                a["o"]["ts"].extend(b["o"]["ts"]); a["o"]["ts"].sort()
                a["o"]["colors"].extend(b["o"]["colors"])
                a["t1"] = max(a["t1"], b["t1"])
                a["span"] = a["t1"] - a["t0"]
                pool.remove(b)
                merged = True
                break
    return pool

city  = try_stitch([x for x in tracks if x["role"] == 2 and x["team"] == "cyan"])
spurs = try_stitch([x for x in tracks if x["role"] == 2 and x["team"] == "white"])
keeps = try_stitch([x for x in tracks if x["role"] == 1])
print(f"after stitching: city {len(city)}, spurs {len(spurs)}, keepers {len(keeps)}")

# ---------- slots: longest-coverage tracks own a slot for the whole clip -----
def pick(pool, slots):
    pool = sorted(pool, key=lambda x: -x["span"])[:len(slots)]
    pool.sort(key=lambda x: x["t0"])
    return {slots[i]: tr for i, tr in enumerate(pool)}

HOME = ["H2","H3","H4","H5","H6","H7","H8","H9","H10","H11"]
AWAY = ["A2","A3","A4","A5","A6","A7","A8","A9","A10","A11"]
slot_of = {**pick(city, HOME), **pick(spurs, AWAY)}
for tr in keeps:
    pts = [to_pitch(H_at(t), *im) for t, im in tr["o"]["ts"][::5]]
    slot = "H1" if np.median([p[0] for p in pts]) < 52.5 else "A1"
    if slot not in slot_of or slot_of[slot]["span"] < tr["span"]:
        slot_of[slot] = tr
dropped = len(city) + len(spurs) - sum(1 for s in slot_of if s not in ("H1","A1"))
print(f"slots filled: {sorted(slot_of)}  (fragments left unslotted: {dropped})")

# ---------- sample every slot at the calibrated 0.5s marks ----------
frames = []
for t in OUT_T:
    positions = {}
    for slot, tr in slot_of.items():
        p = pitch_at(tr["o"], t, 0.11)          # ~3 frames at 25fps
        entry = None
        if p:
            entry = {"x": round(p[0], 2), "y": round(p[1], 2)}
        else:
            # inside a gap: interpolate between surrounding observations
            before = [x for x in tr["o"]["ts"] if x[0] < t]
            after  = [x for x in tr["o"]["ts"] if x[0] > t]
            if before and after:
                (ta, ia), (tb, ib) = before[-1], after[0]
                pa = to_pitch(H_at(ta), *ia); pb = to_pitch(H_at(tb), *ib)
                u = (t - ta) / (tb - ta)
                entry = {"x": round(pa[0] + (pb[0]-pa[0])*u, 2),
                         "y": round(pa[1] + (pb[1]-pa[1])*u, 2), "stale": True}
            elif before or after:
                tn, im = (before[-1] if before else after[0])
                q = to_pitch(H_at(tn), *im)
                entry = {"x": round(q[0], 2), "y": round(q[1], 2), "stale": True}
        if entry:
            entry["x"] = min(max(entry["x"], -1), 106)
            entry["y"] = min(max(entry["y"], -1), 68)
            positions[slot] = entry
    frames.append({"t": t, "positions": positions})

# ---------- ball: audited anchors, unchanged ----------
for f, of in zip(frames, old["frames"]):
    f["ball"] = of["ball"]

json.dump({"times": OUT_T, "frames": frames}, open("fine_positions_bt.json", "w"))

# page format
page = {"times": OUT_T, "ballSpan": [0, 22], "frames": []}
for f in frames:
    p = {s: [v["x"], v["y"]] for s, v in f["positions"].items()}
    o = [s for s, v in f["positions"].items() if not v.get("stale")]
    page["frames"].append({"p": p, "o": o, "b": [f["ball"]["x"], f["ball"]["y"]]})
json.dump(page, open("bt_page_data.json", "w"))

n_obs = sum(len(f["o"]) for f in page["frames"])
n_all = sum(len(f["p"]) for f in page["frames"])
o_obs = sum(sum(1 for v in f["positions"].values() if not v.get("stale"))
            for f in old["frames"])
print(f"v6: {n_obs}/{n_all} slot-samples directly observed")
print(f"    (old extraction had {o_obs} observed of "
      f"{sum(len(f['positions']) for f in old['frames'])})")
