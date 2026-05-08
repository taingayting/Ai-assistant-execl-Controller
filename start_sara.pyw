"""Sara Desktop Launcher — double-click to start everything.

This file uses the .pyw extension so Windows runs it without
showing a console window.  It:
  1. Starts the Flask backend (backend/app.py) in the background
  2. Waits for the backend to be ready
  3. Launches the Sara desktop UI

Close the UI (right-click tray → Quit) and the backend shuts
down automatically.
"""

import os
import sys
import subprocess
import time
import threading
import importlib

# Resolve paths relative to this script
ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(ROOT, ".venv", "Scripts", "pythonw.exe")
PYTHON = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
# Use python.exe (not pythonw.exe) for the backend since Flask needs stdio
BACKEND_PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(BACKEND_PYTHON):
    BACKEND_PYTHON = sys.executable

BACKEND_SCRIPT = os.path.join(ROOT, "backend", "app.py")
BACKEND_URL = "http://127.0.0.1:5000"


def is_backend_running() -> bool:
    """Check if the Flask backend is already responding."""
    try:
        import urllib.request
        resp = urllib.request.urlopen(BACKEND_URL + "/api/system_status", timeout=2)
        return resp.status == 200
    except Exception:
        return False


def wait_for_backend(process, timeout: int = 15) -> bool:
    """Wait up to *timeout* seconds for the backend to respond."""
    for _ in range(timeout * 4):
        if process.poll() is not None:
            # Process exited unexpectedly
            return False
        if is_backend_running():
            return True
        time.sleep(0.25)
    return False


def main() -> None:
    os.chdir(ROOT)
    print('Sara Assistant Starting...')

    # Set environment for full assistant mode
    env = os.environ.copy()
    # The desktop UI's built-in wake listener is disabled here because the
    # standalone assistant runtime owns wake-word listening.
    env["SARA_WAKE_ENABLED"] = "false"
    os.environ["SARA_WAKE_ENABLED"] = "false"
    # Enable TTS so Sara speaks responses
    env.setdefault("ENABLE_TTS", "true")
    os.environ.setdefault("ENABLE_TTS", "true")
    # Point to custom Hey Sara model if it exists
    hey_sara_model = os.path.join(ROOT, "wakeword", "models", "hey_sara.onnx")
    if os.path.exists(hey_sara_model):
        env.setdefault("WAKE_MODEL_PATH", hey_sara_model)
        os.environ.setdefault("WAKE_MODEL_PATH", hey_sara_model)

    backend_proc = None

    # Start backend only if it's not already running
    if not is_backend_running():
        # Start backend with CREATE_NO_WINDOW flag so no console flashes
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW

        backend_proc = subprocess.Popen(
            [BACKEND_PYTHON, BACKEND_SCRIPT],
            cwd=os.path.join(ROOT, "backend"),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )

        if not wait_for_backend(backend_proc):
            # Backend failed to start — show error dialog
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror(
                    "Sara Launcher",
                    "Could not start the backend server.\n\n"
                    "Make sure all dependencies are installed:\n"
                    "  pip install -r backend/requirements.txt"
                )
                root.destroy()
            except Exception:
                pass
            if backend_proc and backend_proc.poll() is None:
                backend_proc.terminate()
            sys.exit(1)

    print('Backend: running')

    # Import UI/runtime only after environment variables are finalized.
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    backend_dir = os.path.join(ROOT, 'backend')
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from PyQt6.QtWidgets import QApplication
    from desktop.sara_ui import SaraFloatingWindow
    SaraRuntime = importlib.import_module('assistant_runtime').SaraRuntime
    create_wake_detector = importlib.import_module(
        'wakeword.wake_detector_porcupine'
    ).create_wake_detector

    app = QApplication(sys.argv)
    app.setApplicationName('Sara Desktop Assistant')
    app.setQuitOnLastWindowClosed(False)

    win = SaraFloatingWindow()
    desktop = QApplication.primaryScreen().availableGeometry()
    x = desktop.x() + desktop.width() - win.width() - 30
    y = desktop.y() + 50
    win.move(x, y)
    win.hide()
    print('Desktop UI: ready')

    # Auto-select wake word engine:
    # • Porcupine ("Hey Sara" .ppn model) if PICOVOICE_ACCESS_KEY is set
    # • openwakeword (hey_jarvis demo) otherwise
    wake_detector = create_wake_detector()
    print(f'Wake word engine: {type(wake_detector).__name__}')

    runtime = SaraRuntime(ui=win, wake_detector=wake_detector)
    runtime_thread = threading.Thread(
        target=runtime.start,
        name='sara-runtime',
        daemon=True,
    )
    runtime_thread.start()

    print('Wake word detector: active')
    print('Assistant runtime: running')
    print('Sara is now listening for "Hey Sara".')

    def _cleanup() -> None:
        try:
            runtime.stop()
        except Exception:
            pass
        if backend_proc is not None and backend_proc.poll() is None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_proc.kill()

    app.aboutToQuit.connect(_cleanup)

    exit_code = app.exec()
    _cleanup()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
