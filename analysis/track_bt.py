"""Full-rate tracking pass: football player model on every 2nd frame
(25 fps effective) through ByteTrack, so identity association happens in
image space at small inter-frame gaps instead of 0.5s jumps.
Output tracks_bt.json: [{t, dets: [{id, cls, conf, img, color}]}]"""
import json
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "clip1080.mp4"
STRIDE = 2                      # every 2nd frame of the 50fps clip
model = YOLO("rfmodels/football-player-detection.pt")

def shirt_color(frame, x1, y1, x2, y2):
    crop = frame[int(y1):int(y1 + (y2-y1)*0.55), int(x1):int(x2)]
    if crop.size == 0: return np.zeros(3)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ng = ~((hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 60))
    px = crop[ng]
    return px.mean(axis=0) if len(px) else crop.reshape(-1,3).mean(axis=0)

fps = cv2.VideoCapture(VIDEO).get(cv2.CAP_PROP_FPS)
out = []
results = model.track(source=VIDEO, stream=True, tracker="bytetrack.yaml",
                      vid_stride=STRIDE, conf=0.2, imgsz=1280,
                      classes=[1, 2, 3], verbose=False)
for i, res in enumerate(results):
    t = round(i * STRIDE / fps, 4)
    dets = []
    if res.boxes.id is not None:
        frame = res.orig_img
        for box, tid, cls, conf in zip(res.boxes.xyxy.cpu().numpy(),
                                       res.boxes.id.cpu().numpy(),
                                       res.boxes.cls.cpu().numpy(),
                                       res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box[:4]
            d = {"id": int(tid), "cls": int(cls), "conf": float(conf),
                 "img": (float((x1+x2)/2), float(y2))}
            if int(cls) in (1, 2):
                d["color"] = shirt_color(frame, x1, y1, x2, y2).tolist()
            dets.append(d)
    out.append({"t": t, "dets": dets})
    if i % 25 == 0:
        print(f"t={t:6.2f}s tracked={len(dets)}", flush=True)
json.dump(out, open("tracks_bt.json", "w"))
ids = {d["id"] for kf in out for d in kf["dets"]}
print(f"wrote tracks_bt.json: {len(out)} frames, {len(ids)} distinct track ids")
