"""Automatic per-frame pitch calibration using PnLCalib (Gutierrez-Perez &
Ramon-Vigo) - no manual clicks, no reference frame, no motion chaining.

Setup (one-time):
  git clone --depth 1 https://github.com/mguti97/PnLCalib.git
  curl -L -o SV_kp    https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
  curl -L -o SV_lines https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines
  pip install torch torchvision shapely opencv-python-headless pyyaml pillow

Usage:
  python3 pnl_calibrate.py match.mp4 --pnl-dir PnLCalib --weights-kp SV_kp \
      --weights-line SV_lines --step 0.5 --out autoH.json [--crop-top 55 --crop-bottom 800]

Output: JSON of {time_s: 3x3 image->pitch homography} in this project's
pitch convention (x 0..105 left->right, y 0..68 top->bottom, meters),
directly usable as match_to_tactic.py --chain input. Optionally polish
with refine_pitch_lines.py for another ~30-50% alignment gain.
Note: PnLCalib code/weights are GPL-2.0."""
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("video")
_ap.add_argument("--pnl-dir", default="PnLCalib")
_ap.add_argument("--weights-kp", default="SV_kp")
_ap.add_argument("--weights-line", default="SV_lines")
_ap.add_argument("--step", type=float, default=0.5)
_ap.add_argument("--out", default="autoH.json")
_ap.add_argument("--crop-top", type=int, default=0, help="crop out letterboxing/chrome before inference")
_ap.add_argument("--crop-bottom", type=int, default=0, help="0 = frame height")
_args = _ap.parse_args()
import os
import sys
import json
import yaml
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as f
from PIL import Image

sys.path.insert(0, _args.pnl_dir)
from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool,
                                 get_keypoints_from_heatmap_batch_maxpool_l,
                                 complete_keypoints, coords_to_dict)

VIDEO = _args.video
CROP_Y0 = _args.crop_top
KP_TH, LINE_TH = 0.3434, 0.7867
device = "cpu"

cfg = yaml.safe_load(open(f"{_args.pnl_dir}/config/hrnetv2_w48.yaml"))
cfg_l = yaml.safe_load(open(f"{_args.pnl_dir}/config/hrnetv2_w48_l.yaml"))
model = get_cls_net(cfg)
model.load_state_dict(torch.load(_args.weights_kp, map_location=device))
model.eval()
model_l = get_cls_net_l(cfg_l)
model_l.load_state_dict(torch.load(_args.weights_line, map_location=device))
model_l.eval()
resize = T.Resize((540, 960))

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CROP_Y1 = _args.crop_bottom or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
CH = CROP_Y1 - CROP_Y0

cam = FramebyFrameCalib(iwidth=W, iheight=CH, denormalize=True)

def P_from(params):
    c = params["cam_params"]
    It = np.eye(4)[:-1]
    It[:, -1] = -np.array(c["position_meters"])
    Q = np.array([[c["x_focal_length"], 0, c["principal_point"][0]],
                  [0, c["y_focal_length"], c["principal_point"][1]],
                  [0, 0, 1]])
    return Q @ (np.array(c["rotation_matrix"]) @ It)

out = {}
t = 0.0
while t <= n / fps:
    idx = round(t * fps)
    if idx >= n: break
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok: break
    crop = frame[CROP_Y0:CROP_Y1]
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    x = f.to_tensor(img).float().unsqueeze(0)
    x = resize(x)
    with torch.no_grad():
        hm = model(x)
        hm_l = model_l(x)
    kp_c = get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1, :, :])
    ln_c = get_keypoints_from_heatmap_batch_maxpool_l(hm_l[:, :-1, :, :])
    kp_d = coords_to_dict(kp_c, threshold=KP_TH)
    ln_d = coords_to_dict(ln_c, threshold=LINE_TH)
    kp_d, ln_d = complete_keypoints(kp_d[0], ln_d[0], w=960, h=540, normalize=True)
    cam.update(kp_d, ln_d)
    params = cam.heuristic_voting(refine_lines=True)
    if params is None:
        print(f"t={t:5.2f}s  NO calibration")
        t = round(t + _args.step, 4)
        continue
    P = P_from(params)
    # ground-plane homography, world origin at pitch center -> our 0..105/0..68
    Hp2i = P[:, [0, 1, 3]]
    Tc = np.array([[1, 0, -52.5], [0, 1, -34.0], [0, 0, 1]])
    A = np.array([[1, 0, 0], [0, 1, CROP_Y0], [0, 0, 1]], dtype=float)   # un-crop
    Hp2i_full = A @ Hp2i @ Tc
    out[str(t)] = np.linalg.inv(Hp2i_full).tolist()
    print(f"t={t:5.2f}s  ok (err={params.get('final_error', -1):.3f})" if 'final_error' in params else f"t={t:5.2f}s  ok")
    t = round(t + _args.step, 4)
cap.release()
json.dump(out, open(_args.out, "w"))
print(f"wrote {_args.out} with {len(out)} frames")
