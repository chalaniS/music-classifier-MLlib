@echo off
cd /d %~dp0
start python app\server.py
timeout /t 3 /nobreak >nul
start http://localhost:5000
