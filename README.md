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
