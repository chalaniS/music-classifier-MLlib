@echo off
title Music Genre Classifier

echo ============================================
echo   Music Genre Classifier - Startup
echo ============================================
echo.

:: ── Change to the directory where this bat file lives ─────────────────────────
cd /d "%~dp0"

:: ── Check Python ──────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Make sure Python is in your PATH.
    pause
    exit /b 1
)

:: ── Install dependencies if needed ───────────────────────────────────────────
echo [1/3] Checking dependencies ...
pip install flask pyspark==3.4.1 --quiet

:: ── Train model if it does not exist yet ──────────────────────────────────────
if not exist "model\labels.txt" (
    echo [2/3] No trained model found. Training now ...
    python train.py
    if errorlevel 1 (
        echo [ERROR] Training failed. Check train.py and your dataset paths.
        pause
        exit /b 1
    )
) else (
    echo [2/3] Trained model found. Skipping training.
)

:: ── Start the Flask server ────────────────────────────────────────────────────
echo [3/3] Starting server on http://localhost:5000 ...
echo        Press Ctrl+C to stop.
echo.

:: Open browser after a short delay
start "" cmd /c "timeout /t 4 >nul && start http://localhost:5000"

:: Run server (blocking)
python app/server.py

pause