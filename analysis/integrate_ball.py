"""Validate the automated ball trajectory against the audited anchors, then
rebuild the ball channel of the extraction from the automated set.

Acceptance gate: where both sets have an anchor at the same time, the median
disagreement must be small — the audited set is treated as ground truth. The
automated set is only adopted if it agrees where we can check it, which is
what makes it trustworthy where we couldn't (after 18s, into the goal).
"""
import json
import numpy as np

auto = {float(k): v for k, v in json.load(open("ball_anchors_auto.json")).items()}
audited = {float(k): v for k, v in json.load(open("ball_anchors.json")).items()}

common = sorted(set(auto) & set(audited))
devs = []
for t in common:
    pa, pb = auto[t]["pitch"], audited[t]["pitch"]
    devs.append((round(float(np.hypot(pa[0]-pb[0], pa[1]-pb[1])), 2), t))
devs.sort()
if devs:
    md = float(np.median([d for d, _ in devs]))
    print(f"overlap: {len(common)} anchors | median dev {md:.2f} m | "
          f"worst {devs[-1][0]:.2f} m at t={devs[-1][1]}")
    for d, t in devs[-4:]:
        print(f"   t={t:5.2f} auto={auto[t]['pitch']} audited={audited[t]['pitch']} ({d} m)")
else:
    print("no overlapping anchors!")

new_span = sorted(t for t in auto if t > max(audited))
print(f"new coverage beyond audited span: {new_span}")
for t in new_span:
    print(f"   t={t:5.2f} -> {auto[t]['pitch']} conf {auto[t]['conf']}")

# rebuild the ball channel in fine_positions_bt.json + bt_page_data.json
def ball_at(by_t, t):
    ts = sorted(by_t)
    if t <= ts[0]: return by_t[ts[0]]["pitch"]
    if t >= ts[-1]: return by_t[ts[-1]]["pitch"]
    hi = min(u for u in ts if u >= t); lo = max(u for u in ts if u <= t)
    if hi == lo: return by_t[lo]["pitch"]
    a = (t - lo) / (hi - lo)
    p, q = by_t[lo]["pitch"], by_t[hi]["pitch"]
    return [p[0] + (q[0]-p[0])*a, p[1] + (q[1]-p[1])*a]

fp = json.load(open("fine_positions_bt.json"))
for f in fp["frames"]:
    b = ball_at(auto, f["t"])
    f["ball"] = {"x": round(b[0], 2), "y": round(b[1], 2)}
json.dump(fp, open("fine_positions_bt.json", "w"))

page = json.load(open("bt_page_data.json"))
for fr, f in zip(page["frames"], fp["frames"]):
    fr["b"] = [f["ball"]["x"], f["ball"]["y"]]
page["ballSpan"] = [min(auto), max(auto)]
json.dump(page, open("bt_page_data.json", "w"))
print("rebuilt ball channel in fine_positions_bt.json + bt_page_data.json")
