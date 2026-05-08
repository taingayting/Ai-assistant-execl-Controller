@echo off
REM ============================================================
REM  Build Sara into a standalone Windows executable
REM  Run from the project root:  build_exe.bat
REM ============================================================

cd /d "%~dp0"

REM Use the venv Python if it exists
SET PYTHON=.venv\Scripts\python.exe
IF NOT EXIST %PYTHON% SET PYTHON=python

REM Install PyInstaller if missing
%PYTHON% -m pip install pyinstaller --quiet

echo.
echo Building Sara executable...
echo.

%PYTHON% -m PyInstaller ^
    --onefile ^
    --windowed ^
    --name "Sara" ^
    --add-data "backend;backend" ^
    --add-data "desktop;desktop" ^
    --add-data "wakeword;wakeword" ^
    --add-data "backend/.env;backend" ^
    --hidden-import openwakeword ^
    --hidden-import onnxruntime ^
    --hidden-import pyaudio ^
    --hidden-import PyQt6 ^
    --hidden-import pyttsx3 ^
    --hidden-import openpyxl ^
    --hidden-import requests ^
    start_sara.pyw

echo.
IF EXIST dist\Sara.exe (
    echo SUCCESS: dist\Sara.exe created.
    echo.
    echo To add Sara to Windows Startup automatically:
    echo   1. Press Win + R
    echo   2. Type:  shell:startup
    echo   3. Copy dist\Sara.exe into the Startup folder that opens
    echo.
    echo Or run:  python install_startup.py
    echo ^(requires pywin32:  pip install pywin32^)
) ELSE (
    echo FAILED: dist\Sara.exe was not created.
    echo Check the output above for errors.
)

pause
