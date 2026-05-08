"""Add Sara to Windows Startup so it runs automatically when the laptop boots.

Run once (as a normal user, no admin required):

    python install_startup.py          # install
    python install_startup.py --remove # uninstall

How it works:
  Creates a .lnk shortcut in the Windows Startup folder:
    %APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\Sara.lnk
  The shortcut points to pythonw.exe (from the venv) and passes start_sara.pyw
  as the argument.  pythonw.exe runs Python without a console window, so Sara
  starts silently in the background when Windows boots.
"""
from __future__ import annotations

import os
import sys
import argparse


ROOT = os.path.dirname(os.path.abspath(__file__))
STARTUP_FOLDER = os.path.join(
    os.environ.get("APPDATA", ""),
    "Microsoft", "Windows", "Start Menu", "Programs", "Startup",
)
SHORTCUT_PATH = os.path.join(STARTUP_FOLDER, "Sara.lnk")
LAUNCHER     = os.path.join(ROOT, "start_sara.pyw")
VENV_PYW     = os.path.join(ROOT, ".venv", "Scripts", "pythonw.exe")
ICON_PATH    = os.path.join(ROOT, "desktop", "icons", "sara_icon.ico")


def _find_pythonw() -> str:
    if os.path.exists(VENV_PYW):
        return VENV_PYW
    # Fall back to the Python interpreter running this script
    candidate = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if os.path.exists(candidate):
        return candidate
    return sys.executable  # last resort (will show console)


def install() -> None:
    if sys.platform != "win32":
        print("ERROR: Windows startup integration is only supported on Windows.")
        sys.exit(1)

    if not os.path.exists(LAUNCHER):
        print(f"ERROR: Launcher not found: {LAUNCHER}")
        sys.exit(1)

    try:
        import win32com.client  # type: ignore  # from pywin32
    except ImportError:
        print("ERROR: pywin32 is required.  Install it with:")
        print("  pip install pywin32")
        sys.exit(1)

    pythonw = _find_pythonw()
    print(f"Python  : {pythonw}")
    print(f"Launcher: {LAUNCHER}")
    print(f"Shortcut: {SHORTCUT_PATH}")

    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(SHORTCUT_PATH)
    shortcut.TargetPath       = pythonw
    shortcut.Arguments        = f'"{LAUNCHER}"'
    shortcut.WorkingDirectory = ROOT
    shortcut.WindowStyle      = 7   # 7 = minimized / no window
    shortcut.Description      = "Sara Voice Assistant"
    if os.path.exists(ICON_PATH):
        shortcut.IconLocation = ICON_PATH
    shortcut.save()

    print()
    print("Sara has been added to Windows Startup.")
    print("It will start automatically the next time you boot your laptop.")
    print("To test it now, run:  start_sara.pyw")


def remove() -> None:
    if os.path.exists(SHORTCUT_PATH):
        os.remove(SHORTCUT_PATH)
        print(f"Removed: {SHORTCUT_PATH}")
        print("Sara will no longer start automatically at boot.")
    else:
        print("Sara startup shortcut not found — nothing to remove.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install/remove Sara from Windows Startup")
    parser.add_argument("--remove", action="store_true", help="Remove the startup shortcut")
    args = parser.parse_args()

    if args.remove:
        remove()
    else:
        install()
