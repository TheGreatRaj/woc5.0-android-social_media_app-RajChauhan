@echo off
rem Starts the app. Close this window to quit.
set "APP=%~dp0.."
if not exist "%APP%\.venv\Scripts\python.exe" (
  echo Concert Remaster is not installed yet. Run setup.bat first.
  pause
  exit /b 1
)
call "%APP%\.venv\Scripts\activate.bat"
python -m concert_remaster gui %*
if errorlevel 1 pause
