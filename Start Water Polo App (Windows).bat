@echo off
rem Double-click to start the Water Polo Tactics app with a working video panel.
cd /d "%~dp0"
where python >nul 2>nul
if not %errorlevel%==0 (
  echo Python was not found - opening the app directly. The video panel will be
  echo limited; install Python from python.org to enable it.
  start "" "Waterpolo Tactic.html"
  pause
  exit /b
)
start "" /min cmd /c "python -m http.server 8787"
timeout /t 1 >nul
start "" "http://localhost:8787/Waterpolo%%20Tactic.html"
echo Water Polo Tactics is running at http://localhost:8787
