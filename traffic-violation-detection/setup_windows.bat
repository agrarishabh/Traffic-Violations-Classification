@echo off
REM ================================================================
REM  Traffic Violation Detection System - Windows Setup Script
REM  Run this ONCE to set up everything on your Windows machine.
REM  Double-click this file OR run it from Command Prompt.
REM ================================================================

echo.
echo  =====================================================
echo   TRAFFIC VIOLATION DETECTION SYSTEM - SETUP
echo  =====================================================
echo.

REM ---- Step 1: Check Python is installed ----
echo [STEP 1/6] Checking Python installation...
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo  ERROR: Python is not installed or not in PATH.
    echo  Please install Python 3.10 or newer from: https://www.python.org/downloads/
    echo  IMPORTANT: During install, check the box "Add Python to PATH"
    echo.
    pause
    exit /b 1
)
python --version
echo  Python found - OK
echo.

REM ---- Step 2: Check pip is available ----
echo [STEP 2/6] Checking pip (Python package installer)...
pip --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo  ERROR: pip not found. Try reinstalling Python.
    pause
    exit /b 1
)
echo  pip found - OK
echo.

REM ---- Step 3: Create virtual environment ----
echo [STEP 3/6] Creating virtual environment (venv)...
echo  This keeps project packages separate from your system Python.
IF EXIST "venv\" (
    echo  Virtual environment already exists, skipping creation.
) ELSE (
    python -m venv venv
    IF ERRORLEVEL 1 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  Virtual environment created - OK
)
echo.

REM ---- Step 4: Activate virtual environment ----
echo [STEP 4/6] Activating virtual environment...
call venv\Scripts\activate.bat
IF ERRORLEVEL 1 (
    echo  ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo  Virtual environment activated - OK
echo.

REM ---- Step 5: Upgrade pip ----
echo [STEP 5/6] Upgrading pip to latest version...
python -m pip install --upgrade pip
echo.

REM ---- Step 6: Install all dependencies ----
echo [STEP 6/6] Installing all required packages from requirements.txt...
echo  This will take 5-15 minutes depending on your internet speed.
echo  You will see a lot of text scrolling - that is normal!
echo.
pip install -r requirements.txt
IF ERRORLEVEL 1 (
    echo.
    echo  ERROR: Some packages failed to install.
    echo  Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo  =====================================================
echo   ALL PACKAGES INSTALLED SUCCESSFULLY!
echo  =====================================================
echo.
echo  Next Steps:
echo  1. Run verification:  python scripts\verify_setup.py
echo  2. Download models:   python scripts\download_models.py
echo  3. Get test images:   python scripts\create_sample_images.py
echo.
pause
