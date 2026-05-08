"""Launch the Sara floating assistant window."""

import sys
import os

# Allow imports when run from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from floating_window import FloatingWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Sara")

    window = FloatingWindow()

    # Position bottom-right of screen
    screen = app.primaryScreen().availableGeometry()
    window.move(
        screen.width() - window.width() - 30,
        screen.height() - window.height() - 30,
    )

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
