#!/usr/bin/env python3
"""Turn football match footage into a tactic file for Football Tactic.html.

The pipeline samples the video every --step seconds and, for each sampled
frame:
  1. detects people (players) in the image,
  2. splits them into two teams by shirt colour,
  3. projects their feet onto real pitch coordinates (meters) using a
     homography built from a small calibration file,
  4. assigns detections to stable player slots (H1..H11 / A1..A11).

The result is a JSON file the coach imports into the board app (Import
button), reviews and edits, and then presents to the team as an animated
scene.

Usage:
  python3 match_to_tactic.py match.mp4 --calib calib.json --out tactic.json \
      --start 63 --end 75 --step 2 --name "Counter vs. high line"

Calibration file: pixel positions of at least 4 known pitch landmarks in
the (undistorted) camera image, e.g. corners of the penalty box, the
center spot, line intersections:

  {
    "points": [
      { "px": [412, 188],  "pitch": [0, 0] },
      { "px": [1710, 205], "pitch": [105, 0] },
      { "px": [1880, 940], "pitch": [105, 68] },
      { "px": [140, 910],  "pitch": [0, 68] }
    ]
  }

Pitch coordinates are meters on a 105 x 68 pitch, x left-to-right, y
top-to-bottom, matching the board app. A fixed camera needs a single
calibration; broadcast footage that cuts between cameras needs a
calibration per shot (analyse one shot at a time with --start/--end).

Detectors (--detector):
  auto   yolo if the ultralytics package is installed, otherwise hog
  yolo   YOLO person detection (pip install ultralytics) - most accurate
  hog    OpenCV HOG person detector - no extra installs, works on clear,
         reasonably zoomed footage
  color  saturated-blob detector for synthetic/test videos
"""

import argparse
import json
import sys

import cv2
import numpy as np

PITCH_W, PITCH_H = 105.0, 68.0
HOME_SLOTS = ["H1", "H2", "H3", "H4", "H5", "H6", "H8", "H10", "H7", "H9", "H11"]
AWAY_SLOTS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A11", "A9", "A10"]


def load_homography(calib_path):
    with open(calib_path) as f:
        calib = json.load(f)
    pts = calib["points"]
    if len(pts) < 4:
        sys.exit("calibration needs at least 4 point pairs")
    src = np.array([p["px"] for p in pts], dtype=np.float64)
    dst = np.array([p["pitch"] for p in pts], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    if H is None:
        sys.exit("could not compute a homography from the calibration points")
    return H


def to_pitch(H, pt):
    v = H @ np.array([pt[0], pt[1], 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


# ---------------------------------------------------------------- detectors
# Every detector returns a list of (foot_x_px, foot_y_px, mean_shirt_bgr).


def detect_hog(frame, hog):
    rects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    out = []
    for (x, y, w, h), wt in zip(rects, weights):
        if wt < 0.3:
            continue
        shirt = frame[y + h // 6 : y + h // 2, x : x + w]
        color = shirt.reshape(-1, 3).mean(axis=0) if shirt.size else np.zeros(3)
        out.append((x + w / 2.0, y + float(h), color))
    return out


def detect_yolo(frame, model, imgsz=1280, conf=0.25):
    # high imgsz matters: on tactical-camera footage players are only tens of
    # pixels tall and vanish at YOLO's default 640 inference size
    res = model.predict(frame, classes=[0], verbose=False, conf=conf, imgsz=imgsz)[0]
    out = []
    for box in res.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box[:4]
        shirt = frame[int(y1 + (y2 - y1) / 6) : int((y1 + y2) / 2), int(x1) : int(x2)]
        color = shirt.reshape(-1, 3).mean(axis=0) if shirt.size else np.zeros(3)
        out.append(((x1 + x2) / 2.0, float(y2), color))
    return out


def detect_color(frame, _=None):
    """Find saturated non-green blobs - for synthetic or highly stylised video."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(int)
    hue = hsv[:, :, 0].astype(int)
    green = (hue > 35) & (hue < 85)
    mask = ((sat > 90) & ~green).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30 or area > 8000:
            continue
        cx, cy = centroids[i]
        y2 = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
        color = frame[labels == i].mean(axis=0)
        out.append((float(cx), float(y2), color))
    return out


def detect_ball(frame):
    """Small bright white blob - used for synthetic video; real footage
    usually needs the coach to place the ball while reviewing."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 200)).astype(np.uint8) * 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if 10 <= area <= 900:
            if best is None or area < best[0]:
                best = (area, centroids[i])
    if best is None:
        return None
    (cx, cy) = best[1]
    return float(cx), float(cy)


# ------------------------------------------------------------- team split


def split_teams(detections, centers):
    """Assign each detection to the nearest of two shirt-colour centers."""
    teams = ([], [])
    for d in detections:
        dists = [np.linalg.norm(d[2] - c) for c in centers]
        teams[int(np.argmin(dists))].append(d)
    return teams


def fit_team_colors(all_colors):
    data = np.array(all_colors, dtype=np.float32)
    _, labels, centers = cv2.kmeans(
        data, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5),
        8, cv2.KMEANS_PP_CENTERS,
    )
    return centers


# ---------------------------------------------------------- slot assignment


def assign_slots(team_pts, slots, prev):
    """Map this keyframe's pitch points to stable slot ids.

    First keyframe: the point nearest its own goal line becomes the keeper,
    the rest are ordered back-to-front, top-to-bottom. Later keyframes:
    greedy nearest-neighbour match against the previous keyframe so slots
    follow the same physical player. Unseen slots keep their last position.
    """
    positions = dict(prev) if prev else {}
    pts = list(team_pts)

    if not prev:
        if not pts:
            return positions
        mean_x = np.mean([p[0] for p in pts])
        goal_x = 0.0 if mean_x < PITCH_W / 2 else PITCH_W
        pts.sort(key=lambda p: abs(p[0] - goal_x))
        keeper, rest = pts[0], pts[1:]
        rest.sort(key=lambda p: (abs(p[0] - goal_x), p[1]))
        ordered = [keeper] + rest
        for slot, p in zip(slots, ordered):
            positions[slot] = {"x": round(p[0], 2), "y": round(p[1], 2)}
        return positions

    # greedy nearest-neighbour: closest (slot, detection) pairs first
    pairs = []
    for slot in slots:
        if slot not in prev:
            continue
        sp = prev[slot]
        for i, p in enumerate(pts):
            d = ((sp["x"] - p[0]) ** 2 + (sp["y"] - p[1]) ** 2) ** 0.5
            pairs.append((d, slot, i))
    pairs.sort()
    used_slots, used_pts = set(), set()
    for d, slot, i in pairs:
        if slot in used_slots or i in used_pts or d > 20:
            continue
        used_slots.add(slot)
        used_pts.add(i)
        positions[slot] = {"x": round(pts[i][0], 2), "y": round(pts[i][1], 2)}
    # brand-new slots for leftover detections
    leftovers = [p for i, p in enumerate(pts) if i not in used_pts]
    for slot in slots:
        if slot in positions or not leftovers:
            continue
        p = leftovers.pop(0)
        positions[slot] = {"x": round(p[0], 2), "y": round(p[1], 2)}
    return positions


# ------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--calib", help="JSON file with px->pitch point pairs (fixed camera)")
    ap.add_argument("--chain", help="per-time homographies from pan_chain.py (panning camera); overrides --calib")
    ap.add_argument("--out", default="tactic.json")
    ap.add_argument("--name", default="Analysed sequence")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, default=None, help="seconds")
    ap.add_argument("--step", type=float, default=2.0, help="seconds between steps")
    ap.add_argument("--detector", choices=["auto", "yolo", "hog", "color"], default="auto")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size; raise for small/distant players")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = ap.parse_args()

    chain = None
    if args.chain:
        chain = {float(k): np.array(v) for k, v in json.load(open(args.chain)).items()}
    elif args.calib:
        H = load_homography(args.calib)
    else:
        sys.exit("pass --calib (fixed camera) or --chain (panning camera, from pan_chain.py)")

    detector = args.detector
    model = None
    if detector in ("auto", "yolo"):
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            detector = "yolo"
        except ImportError:
            if detector == "yolo":
                sys.exit("--detector yolo needs: pip install ultralytics")
            detector = "hog"
    hog = None
    if detector == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    end = args.end if args.end is not None else total / fps

    # pass 1: detect on every keyframe
    keyframes = []
    t = args.start
    while t <= end + 1e-9 and len(keyframes) < args.max_steps:
        if chain is not None:
            ct = min(chain, key=lambda u: abs(u - t))
            if abs(ct - t) > 0.5:
                t += args.step
                continue        # chain broke around this time; skip the keyframe
            H = chain[ct]
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        if detector == "yolo":
            dets = detect_yolo(frame, model, args.imgsz, args.conf)
        elif detector == "hog":
            dets = detect_hog(frame, hog)
        else:
            dets = detect_color(frame)
        # keep detections that land on (or near) the pitch
        on_pitch = []
        for (px, py, color) in dets:
            X, Y = to_pitch(H, (px, py))
            if -3 <= X <= PITCH_W + 3 and -3 <= Y <= PITCH_H + 3:
                on_pitch.append((min(max(X, -1.5), PITCH_W + 1.5), min(max(Y, -1), PITCH_H + 1), color))
        ball = detect_ball(frame)
        ball_pitch = to_pitch(H, ball) if ball else None
        keyframes.append({"t": t, "dets": on_pitch, "ball": ball_pitch})
        print(f"  t={t:6.2f}s  players detected: {len(on_pitch)}" + (f"  ball: yes" if ball_pitch else ""))
        t += args.step
    cap.release()

    if not any(k["dets"] for k in keyframes):
        sys.exit("no players detected - check the calibration, time window and detector")

    # team colours fitted over the whole sequence for stability
    all_colors = [d[2] for k in keyframes for d in k["dets"]]
    centers = fit_team_colors(all_colors)

    # decide which colour cluster is "home" = the team defending the left goal
    first = next(k for k in keyframes if k["dets"])
    team_a, team_b = split_teams(first["dets"], centers)
    mean_ax = np.mean([d[0] for d in team_a]) if team_a else PITCH_W
    mean_bx = np.mean([d[0] for d in team_b]) if team_b else 0
    home_center_idx = 0 if mean_ax <= mean_bx else 1

    # pass 2: build frames with stable slots
    frames = []
    prev_home, prev_away = None, None
    for k in keyframes:
        teams = split_teams(k["dets"], centers)
        home_dets = teams[home_center_idx]
        away_dets = teams[1 - home_center_idx]
        home = assign_slots([(d[0], d[1]) for d in home_dets], HOME_SLOTS, prev_home)
        away = assign_slots([(d[0], d[1]) for d in away_dets], AWAY_SLOTS, prev_away)
        prev_home, prev_away = home, away
        positions = {**home, **away}

        if k["ball"]:
            bx = min(max(k["ball"][0], -1.5), PITCH_W + 1.5)
            by = min(max(k["ball"][1], -1), PITCH_H + 1)
            # give possession to the nearest player when they're close enough
            nearest, nd = None, 4.0
            for slot, p in positions.items():
                d = ((p["x"] - bx) ** 2 + (p["y"] - by) ** 2) ** 0.5
                if d < nd:
                    nearest, nd = slot, d
            ball = {"holder": nearest} if nearest else {"x": round(bx, 2), "y": round(by, 2)}
        else:
            ball = {"x": PITCH_W / 2, "y": PITCH_H / 2}
        frames.append({"positions": positions, "ball": ball})

    tactic = {"name": args.name, "frames": frames}
    payload = {"app": "football-tactics", "version": 1, "current": tactic, "saved": {args.name: tactic}}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(frames)} steps to {args.out}")
    print("Import it in Football Tactic.html (Import button), review and edit each step, then Save & Present.")


if __name__ == "__main__":
    main()
