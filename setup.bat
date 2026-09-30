@echo off
rem Concert Remaster - one-time setup. Double-click this first.
title Concert Remaster setup
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0concert-remaster\windows\setup.ps1" %*
echo.
pause
