# Tactics Boards

Interactive tactics whiteboards, each built as a single self-contained HTML file — no installation or build step required.

- **`Waterpolo Tactic.html`** — water polo, 7v7 pool
- **`Football Tactic.html`** — football, 11v11 full pitch (home 4-3-3 in red vs. away 4-4-2 in white, yellow goalkeepers)

Both share the same workflow: drag players and the ball into position, add steps, and play the sequence back as an animation.

## Football: from match video to an animated scene

`analysis/match_to_tactic.py` turns real match footage into a draft tactic using computer vision: it detects players in sampled frames, splits them into teams by shirt colour, projects their feet onto real pitch coordinates through a calibrated homography, and tracks them into stable player slots. The output JSON imports straight into `Football Tactic.html`, where the coach reviews and edits each step before saving and presenting it to the team.

```
pip install opencv-python-headless numpy   # optionally: ultralytics for YOLO detection
python3 analysis/match_to_tactic.py match.mp4 --calib calib.json --out tactic.json \
    --start 63 --end 75 --step 2 --name "Counter vs. high line"
```

**Panning/tactical cameras** (the camera follows play): calibrate one clear frame — the midfield view with the center circle works best — then let `analysis/pan_chain.py` track the camera motion and carry that calibration across the whole clip:

```
python3 analysis/pan_chain.py clip.mp4 --calib calib.json --ref-time 14.3 --out chainH.json
python3 analysis/refine_pitch_lines.py chainH.json clip.mp4 refinedH.json "0,2,4,6,8,10,12,14,16,18"
python3 analysis/match_to_tactic.py clip.mp4 --chain refinedH.json --out tactic.json --detector yolo --imgsz 1600
```

This works because a pilot camera rotates and zooms from a fixed point, so consecutive frames are related by a global homography. `refine_pitch_lines.py` then locks each listed frame onto the actual white lines (Levenberg-Marquardt on a distance-transform of the detected line pixels), which removes both chain drift and any imprecision in the hand calibration — without it, errors of several meters build up at the ends of the pitch. Verified on real Premier League tactical-camera footage.

The calibration file maps at least four known pitch landmarks (box corners, center spot, line intersections) from image pixels to pitch meters. Create it with **`analysis/calibrate.html`** — open it in a browser, load the match video (or a frame grab), scrub to a clear frame, click a landmark on the frame and then the same spot on the schematic pitch (clicks snap to standard landmarks), repeat four or more times, and export `calib.json`. See the script's docstring for the available detectors (`yolo`, `hog`, or `color`). A fixed tactical camera needs one calibration; broadcast footage should be analysed one camera shot at a time.

The pipeline is verified end-to-end against a synthetic match clip rendered through a known camera perspective: it recovers all 22 players and the ball with a mean position error of about 1.3 m.

# Water Polo Tactics

An interactive tactics whiteboard for water polo, built as a single self-contained HTML file — no installation or build step required.

## Features

- **Tactics board** — a water polo pool with two teams of 7 players (white and blue) plus the ball, all draggable. Designed to work well on iPad/touch devices.
- **Step-by-step animation** — build a tactic as a sequence of frames, then play it back with adjustable speed and looping. Duplicate and delete steps as you refine the play.
- **Video panel** — attach a YouTube clip (with start/end times) to a tactic so you can watch the real play next to the board.
- **Save / load** — tactics are saved in the browser, and can be exported to and imported from JSON files for sharing.
- **Presentation mode** — a clean fullscreen view for showing plays to the team.

## Getting started

The easiest way to run the app:

- **Mac** — double-click `Start Water Polo App (Mac).command`
- **Windows** — double-click `Start Water Polo App (Windows).bat`

Either launcher serves the app at [http://localhost:8787](http://localhost:8787) and opens it in your browser. Serving it over HTTP (rather than opening the file directly) is what makes the embedded YouTube video panel work. The launchers use Python's built-in web server, so Python 3 needs to be installed.

You can also open `Waterpolo Tactic.html` directly in a browser — everything works except the embedded video panel (a button to open clips on YouTube is provided as a fallback).

## Sample tactic

`Greece - Captains winner.json` contains an example tactic ("Greece — Captain's winner (vs HUN)") with a linked video clip. Load it via the **Import** button in the app.
