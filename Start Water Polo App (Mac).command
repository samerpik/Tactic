#!/bin/sh
# Double-click to start the Water Polo Tactics app with a working video panel.
# It serves the app at http://localhost:8787 and opens it in your browser.
cd "$(dirname "$0")"
if curl -s -o /dev/null --max-time 1 "http://localhost:8787"; then
  open "http://localhost:8787/Waterpolo%20Tactic.html"
else
  ( sleep 1; open "http://localhost:8787/Waterpolo%20Tactic.html" ) &
  echo "Water Polo Tactics is running. Keep this window open while you use the app."
  echo "Close this window (or press Ctrl+C) to stop."
  exec python3 -m http.server 8787
fi
