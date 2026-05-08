@echo off
:: Sara Desktop Launcher
:: Double-click this file to start Sara without opening VS Code.
:: It starts the backend + UI, and cleans up when you close Sara.

title Sara - Starting...

cd /d "%~dp0"

:: Use the virtual environment python
set PYTHON=.venv\Scripts\pythonw.exe
if not exist "%PYTHON%" set PYTHON=python

:: Enable wake word and TTS for real assistant experience
set SARA_WAKE_ENABLED=true
set ENABLE_TTS=true

:: Use Hey Sara model if trained, otherwise falls back to Hey Jarvis
if exist "wakeword\models\hey_sara.onnx" (
    set WAKE_MODEL_PATH=wakeword\models\hey_sara.onnx
)

:: Launch via the .pyw launcher (no console window)
start "" "%PYTHON%" start_sara.pyw

:: Close this console immediately
exit
