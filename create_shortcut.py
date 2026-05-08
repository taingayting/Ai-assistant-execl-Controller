"""Create a Windows Desktop shortcut for Sara.

Run once:
    python create_shortcut.py

This creates a "Sara Assistant.lnk" shortcut on your Desktop
that launches Sara with one double-click.
"""

import os
import sys


def create_shortcut() -> None:
    try:
        # pythoncom + shell32 COM for creating .lnk shortcuts
        from win32com.client import Dispatch
    except ImportError:
        print("pywin32 is required. Install it with:")
        print("  pip install pywin32")
        sys.exit(1)

    root = os.path.dirname(os.path.abspath(__file__))
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    shortcut_path = os.path.join(desktop, "Sara Assistant.lnk")

    pythonw = os.path.join(root, ".venv", "Scripts", "pythonw.exe")
    if not os.path.exists(pythonw):
        pythonw = "pythonw.exe"

    target = os.path.join(root, "start_sara.pyw")

    shell = Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.TargetPath = pythonw
    shortcut.Arguments = f'"{target}"'
    shortcut.WorkingDirectory = root
    shortcut.Description = "Sara Voice-Controlled Excel Assistant"
    # Use a generic icon; you can replace this with a custom .ico later
    shortcut.IconLocation = pythonw + ",0"
    shortcut.save()

    print(f"Shortcut created: {shortcut_path}")
    print("You can now double-click 'Sara Assistant' on your Desktop!")


if __name__ == "__main__":
    create_shortcut()
