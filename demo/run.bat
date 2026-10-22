@echo off
setlocal
title HemoVision AI Launcher

echo ======================================================================
echo   HemoVision AI: IDA and Beta Thalassemia Trait Decision Support
echo ======================================================================
echo.

:: Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Install requirements
echo [*] Checking dependencies in requirements.txt...
pip install -r requirements.txt --quiet

:: Launch application
echo.
echo [*] Launching HemoVision AI web app...
python main.py

pause
