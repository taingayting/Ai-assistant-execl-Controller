"""Sara – Voice Controlled Excel Assistant – Floating Desktop UI."""

import json
import requests
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QIcon, QColor, QTextCursor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QLineEdit, QPushButton, QApplication,
    QGraphicsDropShadowEffect,
)

BACKEND_URL = "http://localhost:5000"

# ── Stylesheet ────────────────────────────────────────────────
STYLE = """
QWidget#SaraWindow {
    background-color: #1a1b2e;
    border-radius: 18px;
}
QLabel#title {
    color: #e0e6f0;
    font-size: 22px;
    font-weight: bold;
}
QLabel#subtitle {
    color: #7b8cad;
    font-size: 11px;
}
QLabel#status {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 6px;
}
QTextEdit#transcript {
    background-color: #12132a;
    color: #c8d0e0;
    border: 1px solid #2a2d4a;
    border-radius: 10px;
    padding: 10px;
    font-size: 13px;
    selection-background-color: #3a5fe5;
}
QLineEdit#input {
    background-color: #12132a;
    color: #e0e6f0;
    border: 1px solid #2a2d4a;
    border-radius: 12px;
    padding: 8px 14px;
    font-size: 13px;
}
QLineEdit#input:focus {
    border: 1px solid #3a5fe5;
}
QPushButton#mic {
    background-color: #3a5fe5;
    border: none;
    border-radius: 20px;
    color: white;
    font-size: 18px;
    min-width: 40px;
    min-height: 40px;
    max-width: 40px;
    max-height: 40px;
}
QPushButton#mic:hover {
    background-color: #4f72f7;
}
QPushButton#mic:pressed {
    background-color: #2c4db8;
}
QPushButton#send {
    background-color: #3a5fe5;
    border: none;
    border-radius: 12px;
    color: white;
    font-size: 13px;
    font-weight: bold;
    padding: 8px 16px;
}
QPushButton#send:hover {
    background-color: #4f72f7;
}
"""


# ── Worker thread for backend calls ──────────────────────────
class BackendWorker(QThread):
    """Run parse + execute in a background thread so the UI stays responsive."""

    finished = pyqtSignal(str)   # Sara's reply text
    error = pyqtSignal(str)      # error message

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.text = text

    def run(self):
        try:
            # 1. Parse
            parse_resp = requests.post(
                f"{BACKEND_URL}/api/parse",
                json={"text": self.text},
                timeout=90,
            )
            parse_resp.raise_for_status()
            parse_data = parse_resp.json()

            action = parse_data.get("action")
            if not action:
                self.finished.emit("I couldn't understand that command.")
                return

            action_type = action.get("type", "unknown")

            # If parse returned suggestions (unknown action)
            if action_type == "unknown":
                suggestions = parse_data.get("suggestions", [])
                if suggestions:
                    msg = "I didn't understand. Try:\n• " + "\n• ".join(suggestions)
                else:
                    msg = "I didn't understand that command."
                self.finished.emit(msg)
                return

            # 2. Execute
            exec_resp = requests.post(
                f"{BACKEND_URL}/api/execute",
                json={"action": action},
                timeout=30,
            )
            exec_resp.raise_for_status()
            exec_data = exec_resp.json()

            # Handle confirmation-required responses
            if exec_data.get("requires_confirmation"):
                self.finished.emit(
                    f"⚠ {exec_data.get('message', 'Confirmation required.')}"
                )
                return

            result = exec_data.get("result", {})
            message = (
                result.get("message")
                or exec_data.get("message")
                or "Done."
            )
            self.finished.emit(f"✓ {message}")

        except requests.ConnectionError:
            self.error.emit("Backend offline — start the server on port 5000.")
        except requests.Timeout:
            self.error.emit("Request timed out. Try again.")
        except Exception as exc:
            self.error.emit(str(exc))


# ── Main window ──────────────────────────────────────────────
class FloatingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._worker = None
        self._init_window()
        self._build_ui()
        self.setStyleSheet(STYLE)

    # ── Window setup ──
    def _init_window(self):
        self.setObjectName("SaraWindow")
        self.setWindowTitle("Sara")
        self.setFixedSize(350, 450)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # For dragging
        self._drag_pos = None

    # ── Build widgets ──
    def _build_ui(self):
        # Root container with rounded background
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        container.setObjectName("SaraWindow")
        root.addWidget(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        # ── Header ──
        header = QHBoxLayout()

        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        self.title_label = QLabel("Sara")
        self.title_label.setObjectName("title")
        title_col.addWidget(self.title_label)

        self.subtitle_label = QLabel("Voice-Controlled Excel Assistant")
        self.subtitle_label.setObjectName("subtitle")
        title_col.addWidget(self.subtitle_label)

        header.addLayout(title_col)
        header.addStretch()

        # Status dot
        self.status_label = QLabel("● Online")
        self.status_label.setObjectName("status")
        self._set_status(True)
        header.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignTop)

        # Close button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet(
            "QPushButton{background:transparent;color:#7b8cad;font-size:14px;"
            "border:none;border-radius:12px;}"
            "QPushButton:hover{background:#2a2d4a;color:#e0e6f0;}"
        )
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignTop)

        layout.addLayout(header)

        # ── Transcript ──
        self.transcript = QTextEdit()
        self.transcript.setObjectName("transcript")
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("Say something or type a command…")
        layout.addWidget(self.transcript, stretch=1)

        # ── Input row ──
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self.input_box = QLineEdit()
        self.input_box.setObjectName("input")
        self.input_box.setPlaceholderText("Type a command…")
        self.input_box.returnPressed.connect(self._on_send)
        input_row.addWidget(self.input_box)

        self.mic_btn = QPushButton("🎤")
        self.mic_btn.setObjectName("mic")
        self.mic_btn.setToolTip("Voice input (coming soon)")
        input_row.addWidget(self.mic_btn)

        layout.addLayout(input_row)

        # Drop shadow on the container
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 120))
        container.setGraphicsEffect(shadow)

    # ── Status indicator ──
    def _set_status(self, online: bool):
        if online:
            self.status_label.setText("● Online")
            self.status_label.setStyleSheet(
                "color:#4ade80;background:#1a2e1a;border-radius:6px;"
                "padding:2px 8px;font-size:10px;"
            )
        else:
            self.status_label.setText("● Offline")
            self.status_label.setStyleSheet(
                "color:#f87171;background:#2e1a1a;border-radius:6px;"
                "padding:2px 8px;font-size:10px;"
            )

    # ── Send command ──
    def _on_send(self):
        text = self.input_box.text().strip()
        if not text:
            return
        self.input_box.clear()
        self._append_message("You", text, "#7b8cad")
        self._set_busy(True)

        self._worker = BackendWorker(text)
        self._worker.finished.connect(self._on_reply)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_reply(self, reply: str):
        self._append_message("Sara", reply, "#3a5fe5")
        self._set_busy(False)
        self._set_status(True)

    def _on_error(self, error: str):
        self._append_message("Sara", f"⚠ {error}", "#f87171")
        self._set_busy(False)
        if "offline" in error.lower() or "connection" in error.lower():
            self._set_status(False)

    def _set_busy(self, busy: bool):
        self.input_box.setEnabled(not busy)
        if busy:
            self.input_box.setPlaceholderText("Processing…")
        else:
            self.input_box.setPlaceholderText("Type a command…")
            self.input_box.setFocus()

    def _append_message(self, sender: str, text: str, color: str):
        self.transcript.append(
            f'<span style="color:{color};font-weight:bold;">{sender}:</span> '
            f'<span style="color:#c8d0e0;">{text}</span>'
        )
        self.transcript.moveCursor(QTextCursor.MoveOperation.End)

    # ── Runtime hook API ──
    def show_assistant(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def hide_assistant(self):
        self.hide()

    def set_listening(self):
        self.show_assistant()
        self._set_status(True)
        self.status_label.setText("● Listening")
        self.status_label.setStyleSheet(
            "color:#67e8f9;background:#122733;border-radius:6px;"
            "padding:2px 8px;font-size:10px;"
        )
        self.mic_btn.setText("🎙")
        self.input_box.setEnabled(False)
        self.input_box.setPlaceholderText("Listening…")
        self.transcript.append('<span style="color:#67e8f9;font-weight:bold;">Sara:</span> <span style="color:#c8d0e0;">Listening...</span>')
        self.transcript.moveCursor(QTextCursor.MoveOperation.End)

    def set_processing(self):
        self.show_assistant()
        self.status_label.setText("● Processing")
        self.status_label.setStyleSheet(
            "color:#fde68a;background:#2c2612;border-radius:6px;"
            "padding:2px 8px;font-size:10px;"
        )
        self.mic_btn.setText("⏳")
        self.input_box.setEnabled(False)
        self.input_box.setPlaceholderText("Processing…")

    def set_speaking(self, text: str = ""):
        """Update the status indicator while Sara is speaking via TTS."""
        self.status_label.setText("● Speaking")
        self.status_label.setStyleSheet(
            "color:#a78bfa;background:#1e1333;border-radius:6px;"
            "padding:2px 8px;font-size:10px;"
        )
        self.mic_btn.setText("🔊")
        self.input_box.setEnabled(False)
        self.input_box.setPlaceholderText("Speaking…")

    def set_response(self, text: str):
        self._append_message("Sara", text or "Done.", "#3a5fe5")
        self.mic_btn.setText("🎤")
        self.input_box.setEnabled(True)
        self.input_box.setPlaceholderText("Type a command…")
        self.input_box.setFocus()
        self._set_status(True)

    # ── Dragging ──
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
