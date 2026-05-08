# Packaging Sara as a Desktop Application

## Prerequisites

```
pip install pyinstaller
```

Ensure **all runtime dependencies** are installed in the active environment:

```
pip install -r backend/requirements.txt
```

## Quick Build (single folder)

From the project root:

```powershell
pyinstaller --name Sara `
    --windowed `
    --icon=frontend/favicon.svg `
    --add-data "frontend;frontend" `
    --add-data "workflows;workflows" `
    --add-data "backend;backend" `
    --hidden-import=onnxruntime `
    --hidden-import=openwakeword `
    --hidden-import=pyttsx3 `
    --hidden-import=engineio.async_drivers.threading `
    desktop/sara_ui.py
```

This produces `dist/Sara/Sara.exe`.

## One-File Build

```powershell
pyinstaller --name Sara `
    --onefile `
    --windowed `
    --add-data "frontend;frontend" `
    --add-data "workflows;workflows" `
    --add-data "backend;backend" `
    --hidden-import=onnxruntime `
    --hidden-import=openwakeword `
    --hidden-import=pyttsx3 `
    desktop/sara_ui.py
```

Produces a single `dist/Sara.exe` (~150–250 MB depending on model files).

## Spec File (advanced)

For repeatable builds, generate and customize a `.spec` file:

```powershell
pyi-makespec --name Sara --windowed desktop/sara_ui.py
```

Edit `Sara.spec` to add data files, hidden imports, and exclusions.
Then build:

```powershell
pyinstaller Sara.spec
```

## What Gets Packaged

| Component | Path | Purpose |
|-----------|------|---------|
| Desktop UI | `desktop/sara_ui.py` | Main entry point |
| Backend | `backend/` | Flask API server |
| Workflows | `workflows/*.md` | Workflow definitions |
| Frontend | `frontend/` | Web UI assets |

## Runtime Architecture

The packaged `Sara.exe` launches the PyQt6 floating window.
The **backend must be started separately** (or embedded):

### Option A: Separate Backend (recommended)

1. Run the backend: `python backend/app.py`
2. Run `Sara.exe` (connects to `http://127.0.0.1:5000`)

### Option B: Embedded Backend

To start the Flask backend inside the same process, add a launcher script:

```python
# launcher.py
import threading
import sys
import os

def start_backend():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
    from app import app
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    # Start backend in background thread
    t = threading.Thread(target=start_backend, daemon=True)
    t.start()

    # Start desktop UI
    from desktop.sara_ui import main
    main()
```

Then build with:

```powershell
pyinstaller --name Sara --windowed --onefile launcher.py
```

## Environment Variables

Set these before running `Sara.exe` (or in a `.env` file):

| Variable | Default | Purpose |
|----------|---------|---------|
| `SARA_API_BASE_URL` | `http://127.0.0.1:5000` | Backend URL |
| `SARA_WAKE_ENABLED` | `true` | Enable wake-word detection |
| `GROQ_API_KEY` | — | Cloud LLM API key |
| `USE_OLLAMA` | `true` | Enable local Ollama fallback |
| `OLLAMA_MODEL` | `llama3` | Local LLM model name |
| `ENABLE_TTS` | `false` | Enable voice responses |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `onnxruntime` DLL error | Add `--hidden-import=onnxruntime` to PyInstaller command |
| PyAudio missing | Install via `pipwin install pyaudio` before building |
| Wake word models not found | Ensure openwakeword models are bundled: `--add-data ".venv/Lib/site-packages/openwakeword;openwakeword"` |
| Window doesn't appear | Check that `SARA_WAKE_ENABLED=false` or say the wake word |
| Backend connection refused | Start `python backend/app.py` first |
