"""PyQt6 floating desktop UI for Sara Voice-Controlled Excel Assistant.

Features:
- Small always-on-top floating window with rounded corners
- Microphone button to record audio from system mic
- Transcript display
- AI response display
- Command history
- Integration with Flask backend endpoints:
  - POST /api/upload-audio
  - POST /api/execute (manual retry of parsed action)

Run:
    py -m desktop.sara_ui

Environment variables:
    SARA_API_BASE_URL   (default: http://127.0.0.1:5000)
    SARA_RECORD_SECONDS (default: 5)
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import wave
import time
import struct
from dataclasses import dataclass, field
from datetime import datetime

# Pre-load onnxruntime before PyQt6 to avoid DLL conflict on Windows
try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass

import requests
from PyQt6.QtCore import QThread, Qt, pyqtSignal, QTimer, QRectF
from PyQt6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSystemTrayIcon,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


API_BASE_URL = os.getenv("SARA_API_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
UPLOAD_AUDIO_URL = f"{API_BASE_URL}/api/upload-audio"
EXECUTE_URL = f"{API_BASE_URL}/api/execute"
SYSTEM_STATUS_URL = f"{API_BASE_URL}/api/system_status"
WORKFLOWS_URL = f"{API_BASE_URL}/api/workflows"
RECORD_SECONDS = float(os.getenv("SARA_RECORD_SECONDS", "5"))
WAKE_ENABLED = os.getenv("SARA_WAKE_ENABLED", "true").lower() not in {"0", "false", "no"}
WAKE_MODEL_PATH = os.getenv("WAKE_MODEL_PATH")
WAKE_THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.5"))
WAKE_COOLDOWN_SECONDS = float(os.getenv("SARA_WAKE_COOLDOWN", "2.0"))
MAX_HISTORY_ITEMS = 10

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK = 1024

_lat_log = logging.getLogger('sara.latency')


def _rms(frame: bytes) -> float:
    n = len(frame) // SAMPLE_WIDTH
    if n <= 0:
        return 0.0
    vals = struct.unpack(f"<{n}h", frame[: n * SAMPLE_WIDTH])
    return (sum(v * v for v in vals) / n) ** 0.5


@dataclass
class UploadResult:
    success: bool
    transcript: str = ""
    ai_message: str = ""
    action: dict | None = None
    raw: dict | None = None
    error: str = ""
    suggestions: list = field(default_factory=list)
    workflow_id: str | None = None
    gen: int = 0  # upload generation; stale results are discarded in on_upload_completed


class RecorderThread(QThread):
    """Records microphone input into WAV bytes in a worker thread."""

    recording_finished = pyqtSignal(bytes)
    recording_failed = pyqtSignal(str)
    transcript_live = pyqtSignal(str)

    def __init__(self, seconds: float = RECORD_SECONDS):
        super().__init__()
        self.seconds = max(1.0, seconds)

    def run(self) -> None:
        _t_rec = time.perf_counter()
        try:
            import pyaudio

            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )

            frames: list[bytes] = []
            total_frames = int(SAMPLE_RATE / CHUNK * self.seconds)
            for i in range(total_frames):
                frame = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(frame)

                # Lightweight live transcript placeholder while recording.
                # Final transcript comes from backend STT after upload.
                if i % 6 == 0:
                    elapsed = (i + 1) * CHUNK / SAMPLE_RATE
                    rms = _rms(frame)
                    if rms > 500:
                        live = f"Listening... {elapsed:.1f}s (speech detected)"
                    else:
                        live = f"Listening... {elapsed:.1f}s"
                    self.transcript_live.emit(live)

            stream.stop_stream()
            stream.close()
            audio.terminate()

            wav_bytes = self._to_wav(frames)
            _lat_log.info('[LATENCY] recording=%.0fms frames=%d',
                          (time.perf_counter() - _t_rec) * 1000, len(frames))
            self.recording_finished.emit(wav_bytes)
        except Exception as exc:
            self.recording_failed.emit(f"Recording failed: {exc}")

    @staticmethod
    def _to_wav(frames: list[bytes]) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        return buffer.getvalue()


class UploadWorker(QThread):
    """Uploads WAV bytes to backend and optionally executes parsed action."""

    completed = pyqtSignal(object)

    def __init__(self, wav_bytes: bytes, gen: int = 0):
        super().__init__()
        self.wav_bytes = wav_bytes
        self._gen = gen

    def run(self) -> None:
        result = self._upload_audio(self.wav_bytes)
        result.gen = self._gen  # stamp so on_upload_completed can detect stale results
        self.completed.emit(result)

    def _upload_audio(self, wav_bytes: bytes) -> UploadResult:
        try:
            files = {"file": ("voice_command.wav", wav_bytes, "audio/wav")}
            data = {"auto_execute": "false", "language": "en-US"}
            response = requests.post(UPLOAD_AUDIO_URL, files=files, data=data, timeout=60)
            payload = response.json() if response.content else {}

            if response.status_code >= 400 or not payload.get("success", False):
                fail_suggestions = payload.get("suggestions", [])
                recovery = payload.get("recovery")
                if not fail_suggestions and isinstance(recovery, dict):
                    fail_suggestions = recovery.get("suggestions", [])
                return UploadResult(
                    success=False,
                    error=payload.get("error", f"Upload failed ({response.status_code})"),
                    raw=payload,
                    suggestions=fail_suggestions,
                )

            transcript = payload.get("transcript", "")
            action = payload.get("action")
            executed = payload.get("executed", False)
            speech_text = payload.get("speech_text")
            exec_result = payload.get("exec_result")
            exec_note = payload.get("exec_note")

            ai_message = ""
            if speech_text:
                ai_message = str(speech_text)
            elif executed and exec_result is not None:
                ai_message = self._format_exec_result(exec_result)
            elif exec_note:
                ai_message = str(exec_note)
            else:
                ai_message = "Command parsed successfully."

            suggestions = payload.get("suggestions", [])
            if not suggestions:
                recovery = payload.get("recovery")
                if isinstance(recovery, dict):
                    suggestions = recovery.get("suggestions", [])
            workflow_id = payload.get("workflow_id")

            return UploadResult(
                success=True,
                transcript=transcript,
                ai_message=ai_message,
                action=action,
                raw=payload,
                suggestions=suggestions,
                workflow_id=workflow_id,
            )
        except requests.RequestException as exc:
            return UploadResult(success=False, error=f"Network error: {exc}")
        except Exception as exc:
            return UploadResult(success=False, error=f"Unexpected error: {exc}")

    @staticmethod
    def _format_exec_result(exec_result) -> str:
        if exec_result is None:
            return "Action executed."
        if isinstance(exec_result, str):
            return exec_result
        if isinstance(exec_result, dict):
            if "message" in exec_result:
                return str(exec_result["message"])
            return json.dumps(exec_result, ensure_ascii=True)
        return str(exec_result)

    @staticmethod
    def _execute_action(action: dict) -> dict:
        try:
            payload = {"action": action, "confirmed": True}
            response = requests.post(EXECUTE_URL, json=payload, timeout=60)
            body = response.json() if response.content else {}
            if response.status_code >= 400:
                return {"success": False, "error": body.get("error", "Execution failed")}
            return body
        except requests.RequestException as exc:
            return {"success": False, "error": f"Execution request failed: {exc}"}


class TTSWorker(QThread):
    """Speak text using pyttsx3 on a background thread (fire-and-forget)."""

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def run(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.setProperty("volume", 0.9)
            engine.say(self.text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass  # TTS is a best-effort feature


class ExecuteWorker(QThread):
    """Execute a parsed action via the backend /api/execute endpoint."""

    completed = pyqtSignal(object)  # emits dict (backend JSON response)

    def __init__(self, action: dict, filename: str = ""):
        super().__init__()
        self.action = action
        self.filename = filename

    def run(self) -> None:
        try:
            payload: dict = {"action": self.action, "confirmed": True}
            if self.filename:
                payload["filename"] = self.filename
            response = requests.post(EXECUTE_URL, json=payload, timeout=60)
            body = response.json() if response.content else {}
            if response.status_code >= 400:
                body.setdefault("success", False)
                body.setdefault("error", f"Execution failed ({response.status_code})")
            self.completed.emit(body)
        except Exception as exc:
            self.completed.emit({"success": False, "error": str(exc)})


class WakeWordListenerThread(QThread):
    """Background wake-word listener that emits when wake-word is detected.

    Supports pause/resume so the mic can be released while the
    RecorderThread is capturing a command.
    """

    wake_detected = pyqtSignal(str, float)
    wake_failed = pyqtSignal(str)

    def __init__(self, model_path: str | None = WAKE_MODEL_PATH,
                 threshold: float = WAKE_THRESHOLD,
                 cooldown_seconds: float = WAKE_COOLDOWN_SECONDS):
        super().__init__()
        self.model_path = model_path
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._running = True
        self._paused = False

    def stop(self) -> None:
        self._running = False

    def pause(self) -> None:
        """Pause wake-word detection (e.g. while recording a command)."""
        self._paused = True

    def resume(self) -> None:
        """Resume wake-word detection after recording finishes."""
        self._paused = False

    @staticmethod
    def _resolve_model_path(model_path: str | None) -> str:
        """Return an absolute path to a valid .onnx wake-word model.

        Mirrors the logic in wakeword.wake_detector.WakeDetector._resolve_model_path
        so both code-paths use the same reliable resolution strategy:

        1. User-supplied WAKE_MODEL_PATH if the file exists.
        2. hey_jarvis_v0.1.onnx bundled inside the openwakeword package.
        3. Download official models then return hey_jarvis path.
        """
        import openwakeword
        from pathlib import Path as _Path

        if model_path and _Path(model_path).exists():
            return str(_Path(model_path).resolve())

        pkg_models = _Path(openwakeword.__file__).parent / "resources" / "models"
        jarvis_path = pkg_models / "hey_jarvis_v0.1.onnx"
        if jarvis_path.exists():
            return str(jarvis_path)

        openwakeword.utils.download_models()
        if jarvis_path.exists():
            return str(jarvis_path)

        raise FileNotFoundError(
            f"hey_jarvis_v0.1.onnx not found in {pkg_models}. "
            "Run: python -c \"import openwakeword; openwakeword.utils.download_models()\""
        )

    def run(self) -> None:
        try:
            import numpy as np
            import pyaudio
            import openwakeword
            from openwakeword.model import Model
            from pathlib import Path

            frame_samples = int(16000 * 80 / 1000)  # 80ms @ 16kHz

            # Resolve a full absolute .onnx path — openwakeword's Model()
            # requires a path, not a bare name string.  A bare string leaves
            # prediction_buffer={} so detection never fires.
            resolved_path = self._resolve_model_path(self.model_path)
            model = Model(
                wakeword_models=[resolved_path],
                inference_framework='onnx',
            )
            # Prime with one silent frame so prediction_buffer is populated
            # immediately (openwakeword is lazy on first predict call).
            model.predict(np.zeros(frame_samples, dtype=np.int16))

            labels = list(model.prediction_buffer.keys())
            if not labels:
                self.wake_failed.emit(
                    f"Wake-word model has no labels after loading '{resolved_path}'. "
                    "Check that openwakeword and onnxruntime are installed."
                )
                return
            target_label = labels[0]

            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=frame_samples,
            )

            _wake_log = __import__('logging').getLogger('wake_word')
            last_emit = 0.0
            while self._running:
                frame = stream.read(frame_samples, exception_on_overflow=False)
                # Skip prediction while UI is actively recording a command
                if self._paused:
                    continue
                prediction = model.predict(np.frombuffer(frame, dtype=np.int16))
                score = float(prediction.get(target_label, 0.0))
                # Log near-misses (score > 0.2) to aid threshold tuning without
                # flooding the log at every silent frame.
                if score > 0.2:
                    _accepted = score >= self.threshold
                    _wake_log.debug(
                        '[WakeWord] score=%.3f threshold=%.3f %s label=%s',
                        score, self.threshold,
                        'ACCEPTED' if _accepted else 'rejected',
                        target_label,
                    )
                if score >= self.threshold and (time.time() - last_emit) >= self.cooldown_seconds:
                    _wake_log.info('[WakeWord] triggered score=%.3f label=%s', score, target_label)
                    last_emit = time.time()
                    self.wake_detected.emit(target_label, score)
                    model.reset()

            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception as exc:
            self.wake_failed.emit(f"Wake-word listener failed: {exc}")


class StatusFetchWorker(QThread):
    """Fetches system status from backend in a worker thread."""

    completed = pyqtSignal(dict)

    def run(self) -> None:
        try:
            resp = requests.get(SYSTEM_STATUS_URL, timeout=10)
            self.completed.emit(resp.json() if resp.content else {})
        except Exception:
            self.completed.emit({})


class WorkflowFetchWorker(QThread):
    """Fetches available workflows from backend in a worker thread."""

    completed = pyqtSignal(dict)

    def run(self) -> None:
        try:
            resp = requests.get(WORKFLOWS_URL, timeout=10)
            self.completed.emit(resp.json() if resp.content else {})
        except Exception:
            self.completed.emit({})


class RoundPanel(QFrame):
    """A rounded panel that paints its own translucent background."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("roundPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor("#274257"), 1))
        painter.setBrush(QColor(17, 28, 39, 238))
        painter.drawRoundedRect(rect, 20, 20)
        super().paintEvent(event)


class StatusDot(QWidget):
    """Colored circle that indicates current assistant state."""

    COLOR_IDLE = QColor("#4CAF50")       # green
    COLOR_LISTENING = QColor("#00BCD4")  # cyan
    COLOR_PROCESSING = QColor("#FFC107") # amber
    COLOR_SPEAKING = QColor("#B39DDB")   # soft purple
    COLOR_ERROR = QColor("#F44336")      # red

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._color = self.COLOR_IDLE
        self.setFixedSize(12, 12)

    def set_state(self, state: str) -> None:
        self._color = {
            "idle": self.COLOR_IDLE,
            "done": self.COLOR_IDLE,
            "listening": self.COLOR_LISTENING,
            "processing": self.COLOR_PROCESSING,
            "speaking": self.COLOR_SPEAKING,
            "error": self.COLOR_ERROR,
        }.get(state, self.COLOR_IDLE)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._color))
        painter.drawEllipse(1, 1, 10, 10)


class SpinnerWidget(QWidget):
    """Animated ring spinner used while Sara waits on backend results."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setFixedSize(24, 24)
        self.hide()

    def start(self) -> None:
        self._angle = 0
        self.show()
        self._timer.start(45)

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._angle = (self._angle + 20) % 360
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Base ring
        base_pen = QPen(QColor("#355066"), 2.5)
        painter.setPen(base_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        rect = QRectF(2, 2, self.width() - 4, self.height() - 4)
        painter.drawEllipse(rect)

        # Active arc segment that rotates.
        active_pen = QPen(QColor("#7CC7FF"), 2.5)
        painter.setPen(active_pen)
        start = -self._angle * 16
        span = -120 * 16
        painter.drawArc(rect, start, span)


class SaraFloatingWindow(QMainWindow):
    """Main floating assistant window."""

    def __init__(self):
        super().__init__()
        self._drag_pos = None
        self._force_quit = False
        self.recorder: RecorderThread | None = None
        self.uploader: UploadWorker | None = None
        self.wake_listener: WakeWordListenerThread | None = None
        self._workflow_data: list[dict] = []

        # ── Session persistence ──────────────────────────────────────────────────
        # Mirrors run_sara.py _conversation_loop(): auto-listen after success,
        # retry on failure, sleep after 3 consecutive failures.
        self._session_active: bool = False  # True while in a voice command session
        self._session_errors: int = 0       # consecutive STT/parse failures this session
        self._upload_gen: int = 0           # incremented each recording; stale results are dropped

        # Timers drive lightweight UI animations while recording/processing.
        self._mic_anim_timer = QTimer(self)
        self._mic_anim_timer.timeout.connect(self._tick_mic_animation)
        self._mic_anim_frame = 0

        self._thinking_timer = QTimer(self)
        self._thinking_timer.timeout.connect(self._tick_thinking_animation)
        self._thinking_frame = 0

        self.setWindowTitle("Sara Assistant")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(440, 700)

        self._build_ui()
        self._apply_shadow()
        self._setup_tray()
        self._start_wake_listener_if_enabled()

    def _build_ui(self) -> None:
        root = QWidget(self)
        outer_layout = QVBoxLayout(root)
        outer_layout.setContentsMargins(12, 12, 12, 12)

        self.panel = RoundPanel()
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(16, 16, 16, 16)
        panel_layout.setSpacing(10)

        title = QLabel("Sara")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #E6F0F7;")

        subtitle = QLabel("Voice-Controlled Excel Assistant")
        subtitle.setStyleSheet("color: #A5C1D6; font-size: 11px;")

        # Minimize-to-tray button
        min_btn = QPushButton("\u2212")
        min_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        min_btn.setFixedSize(26, 26)
        min_btn.setStyleSheet(
            "QPushButton {"
            "background: #233647; color: #D7E6F2; border: 1px solid #355066;"
            "border-radius: 13px; font-weight: bold; }"
            "QPushButton:hover { background: #2d455b; }"
        )
        min_btn.clicked.connect(self._minimize_to_tray)

        close_btn = QPushButton("x")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setFixedSize(26, 26)
        close_btn.setStyleSheet(
            "QPushButton {"
            "background: #233647; color: #D7E6F2; border: 1px solid #355066;"
            "border-radius: 13px; font-weight: bold; }"
            "QPushButton:hover { background: #2d455b; }"
        )
        close_btn.clicked.connect(self._quit_app)

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header.addLayout(header_text)
        header.addStretch(1)
        header.addWidget(min_btn)
        header.addWidget(close_btn)
        panel_layout.addLayout(header)

        # Tabbed interface
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(self._tab_style())
        self.tab_widget.addTab(self._build_assistant_tab(), "\U0001F3A4 Assistant")
        self.tab_widget.addTab(self._build_status_tab(), "\U0001F4CA Status")
        self.tab_widget.addTab(self._build_workflow_tab(), "\U0001F4CB Workflows")
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        panel_layout.addWidget(self.tab_widget, 1)

        outer_layout.addWidget(self.panel)
        self.setCentralWidget(root)

    def _build_assistant_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("assistantTab")
        # Prevent the system palette Window color (gray-blue on Windows 11) from
        # bleeding through the space reserved by retainSizeWhenHidden widgets.
        tab.setStyleSheet("#assistantTab { background: #111C27; }")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        self.mic_btn = QPushButton("Start Listening")
        self.mic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mic_btn.setFixedHeight(44)
        self._apply_mic_idle_style()
        self.mic_btn.clicked.connect(self.start_recording)

        self.status_dot = StatusDot()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #9FC2DA; font-size: 11px;")

        self.thinking_label = QLabel("")
        self.thinking_label.setStyleSheet("color: #9FC2DA; font-size: 11px;")
        self.thinking_label.hide()
        self.thinking_spinner = SpinnerWidget()

        # Spinner's retained size (24 px) keeps ctrl_bar's height constant so
        # the outer layout never shifts when the thinking row appears/disappears.
        _sp = self.thinking_spinner.sizePolicy()
        _sp.setRetainSizeWhenHidden(True)
        self.thinking_spinner.setSizePolicy(_sp)

        self.success_label = QLabel("")
        self.success_label.setWordWrap(True)
        self.success_label.setStyleSheet(
            "color: #B9F6CA; font-size: 11px;"
            "background: #1B3A25; border: 1px solid #2E7D46;"
            "border-radius: 8px; padding: 6px;"
        )
        self.success_label.hide()

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(
            "color: #FFB4B4; font-size: 11px;"
            "background: #3A1D24; border: 1px solid #6A2A3A;"
            "border-radius: 8px; padding: 6px;"
        )
        self.error_label.hide()

        # Plan preview panel for multi-step plans
        self.plan_preview_label = QLabel("")
        self.plan_preview_label.setWordWrap(True)
        self.plan_preview_label.setStyleSheet(
            "color: #B3E5FC; font-size: 11px;"
            "background: #1A2A3A; border: 1px solid #2D5A7B;"
            "border-radius: 8px; padding: 6px;"
        )
        self.plan_preview_label.hide()

        self.plan_execute_btn = QPushButton("\u25B6 Execute Plan")
        self.plan_execute_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plan_execute_btn.setFixedHeight(32)
        self.plan_execute_btn.setStyleSheet(
            "QPushButton {"
            "background: #1B5E20; color: #C8E6C9; border: 1px solid #388E3C;"
            "border-radius: 8px; font-size: 12px; font-weight: 600; padding: 4px 14px; }"
            "QPushButton:hover { background: #2E7D32; }"
        )
        self.plan_execute_btn.clicked.connect(self._on_plan_execute_clicked)
        self.plan_execute_btn.hide()

        self.plan_cancel_btn = QPushButton("\u2718 Cancel")
        self.plan_cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plan_cancel_btn.setFixedHeight(32)
        self.plan_cancel_btn.setStyleSheet(
            "QPushButton {"
            "background: #4A1A1A; color: #FFCDD2; border: 1px solid #7B2D2D;"
            "border-radius: 8px; font-size: 12px; padding: 4px 14px; }"
            "QPushButton:hover { background: #5D2A2A; }"
        )
        self.plan_cancel_btn.clicked.connect(self._on_plan_cancel_clicked)
        self.plan_cancel_btn.hide()

        # Smart suggestions shown on failure or unknown commands
        self.suggestions_label = QLabel("")
        self.suggestions_label.setWordWrap(True)
        self.suggestions_label.setStyleSheet(
            "color: #FFE082; font-size: 11px;"
            "background: #2D2A1B; border: 1px solid #5D5429;"
            "border-radius: 8px; padding: 6px;"
        )
        self.suggestions_label.hide()

        transcript_lbl = QLabel("Transcript")
        transcript_lbl.setStyleSheet("color: #D5E7F5; font-weight: 600;")
        self.transcript_view = QPlainTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setFixedHeight(60)
        self.transcript_view.setStyleSheet(self._text_style())

        response_lbl = QLabel("AI Response")
        response_lbl.setStyleSheet("color: #D5E7F5; font-weight: 600;")
        self.response_view = QPlainTextEdit()
        self.response_view.setReadOnly(True)
        self.response_view.setFixedHeight(80)
        self.response_view.setStyleSheet(self._text_style())

        # Developer Mode panel (hidden by default)
        self.dev_panel = QPlainTextEdit()
        self.dev_panel.setReadOnly(True)
        self.dev_panel.setFixedHeight(90)
        self.dev_panel.setStyleSheet(
            "QPlainTextEdit {"
            "background: #0A1520; color: #7CC7FF; border: 1px solid #1A3A55;"
            "border-radius: 8px; padding: 6px; font-size: 10px;"
            "font-family: 'Consolas', 'Courier New', monospace; }"
        )
        self.dev_panel.hide()

        history_lbl = QLabel("Command History")
        history_lbl.setStyleSheet("color: #D5E7F5; font-weight: 600;")
        self.history_list = QListWidget()
        self.history_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.history_list.setStyleSheet(
            "QListWidget {"
            "background: #101d29; color: #DDEBF7; border: 1px solid #324a5f; border-radius: 10px;"
            "padding: 4px; }"
            "QListWidget::item { padding: 4px; border-bottom: 1px solid #243847; }"
            "QListWidget::item:selected { background: #28516a; }"
        )

        # Fixed-height control bar: mic button + status row + thinking row.
        # Wrapping these three items in a single QFrame gives them a shared
        # background and prevents the outer layout from shifting when the
        # thinking row shows/hides.  The spinner's retainSizeWhenHidden keeps
        # the bar's height constant so nothing below it ever moves.
        ctrl_bar = QFrame()
        ctrl_bar.setObjectName("ctrlBar")
        ctrl_bar.setStyleSheet("#ctrlBar { background: #111C27; }")
        ctrl_layout = QVBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(4)

        ctrl_layout.addWidget(self.mic_btn)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(6)
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label)
        status_row.addStretch(1)
        self.dev_toggle = QCheckBox("Dev")
        self.dev_toggle.setStyleSheet("color: #7A98AE; font-size: 10px;")
        self.dev_toggle.toggled.connect(self._toggle_dev_mode)
        status_row.addWidget(self.dev_toggle)
        ctrl_layout.addLayout(status_row)

        thinking_row = QHBoxLayout()
        thinking_row.setContentsMargins(0, 0, 0, 0)
        thinking_row.setSpacing(6)
        thinking_row.addWidget(self.thinking_spinner)
        thinking_row.addWidget(self.thinking_label)
        thinking_row.addStretch(1)
        ctrl_layout.addLayout(thinking_row)

        layout.addWidget(ctrl_bar)

        layout.addWidget(self.success_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.plan_preview_label)

        plan_btn_row = QHBoxLayout()
        plan_btn_row.setContentsMargins(0, 0, 0, 0)
        plan_btn_row.setSpacing(6)
        plan_btn_row.addWidget(self.plan_execute_btn)
        plan_btn_row.addWidget(self.plan_cancel_btn)
        plan_btn_row.addStretch(1)
        layout.addLayout(plan_btn_row)

        layout.addWidget(self.suggestions_label)
        layout.addWidget(transcript_lbl)
        layout.addWidget(self.transcript_view)
        layout.addWidget(response_lbl)
        layout.addWidget(self.response_view)
        layout.addWidget(self.dev_panel)
        layout.addWidget(history_lbl)
        layout.addWidget(self.history_list, 1)

        return tab

    def _build_status_tab(self) -> QWidget:
        """System status panel: active file, sheet, memory, recent commands."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        self.status_info = QPlainTextEdit()
        self.status_info.setReadOnly(True)
        self.status_info.setStyleSheet(self._text_style())
        self.status_info.setPlainText("Switch to this tab to load system status...")

        refresh_btn = QPushButton("\U0001F504 Refresh Status")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.setFixedHeight(34)
        refresh_btn.setStyleSheet(
            "QPushButton {"
            "background: #1A3A50; color: #B0D4ED; border: 1px solid #355066;"
            "border-radius: 8px; font-size: 12px; padding: 4px 12px; }"
            "QPushButton:hover { background: #244A62; }"
        )
        refresh_btn.clicked.connect(self._refresh_status)

        layout.addWidget(self.status_info, 1)
        layout.addWidget(refresh_btn)
        return tab

    def _build_workflow_tab(self) -> QWidget:
        """Workflow explorer: lists all registered workflows."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        self.workflow_list = QListWidget()
        self.workflow_list.setStyleSheet(
            "QListWidget {"
            "background: #101d29; color: #DDEBF7; border: 1px solid #324a5f;"
            "border-radius: 10px; padding: 4px; }"
            "QListWidget::item { padding: 8px; border-bottom: 1px solid #243847; }"
            "QListWidget::item:selected { background: #28516a; }"
        )
        self.workflow_list.itemClicked.connect(self._on_workflow_clicked)

        self.workflow_detail = QPlainTextEdit()
        self.workflow_detail.setReadOnly(True)
        self.workflow_detail.setFixedHeight(110)
        self.workflow_detail.setStyleSheet(self._text_style())
        self.workflow_detail.setPlainText("Click a workflow to see details.")

        refresh_btn = QPushButton("\U0001F504 Refresh Workflows")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.setFixedHeight(34)
        refresh_btn.setStyleSheet(
            "QPushButton {"
            "background: #1A3A50; color: #B0D4ED; border: 1px solid #355066;"
            "border-radius: 8px; font-size: 12px; padding: 4px 12px; }"
            "QPushButton:hover { background: #244A62; }"
        )
        refresh_btn.clicked.connect(self._refresh_workflows)

        layout.addWidget(self.workflow_list, 1)
        layout.addWidget(self.workflow_detail)
        layout.addWidget(refresh_btn)
        return tab

    def _apply_shadow(self) -> None:
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 170))
        self.panel.setGraphicsEffect(shadow)

    @staticmethod
    def _text_style() -> str:
        return (
            "QPlainTextEdit {"
            "background: #101d29; color: #E2EEF9; border: 1px solid #324a5f; border-radius: 10px;"
            "padding: 8px; font-size: 12px; }"
        )

    @staticmethod
    def _tab_style() -> str:
        return (
            "QTabWidget::pane { border: none; background: transparent; }"
            "QTabBar::tab {"
            "  background: #162231; color: #8AA5BA; border: 1px solid #2A4055;"
            "  border-bottom: none; border-radius: 6px 6px 0 0;"
            "  padding: 6px 10px; margin-right: 2px; font-size: 11px; }"
            "QTabBar::tab:selected {"
            "  background: #1A3045; color: #E0F0FF; border-bottom: 2px solid #15AABF; }"
            "QTabBar::tab:hover { background: #1E3650; }"
        )

    # ---- System tray ----

    def _setup_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(21, 170, 191))
        painter.drawEllipse(2, 2, 60, 60)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 30, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "S")
        painter.end()

        self.tray_icon = QSystemTrayIcon(QIcon(pixmap), self)
        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show Sara")
        show_action.triggered.connect(self._show_from_tray)
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(self._quit_app)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.setToolTip("Sara \u2013 Voice Excel Assistant")
        self.tray_icon.show()

    def _minimize_to_tray(self) -> None:
        if hasattr(self, "tray_icon") and self.tray_icon.isVisible():
            self.hide()
        else:
            self.showMinimized()

    def _show_from_tray(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _quit_app(self) -> None:
        self._force_quit = True
        if hasattr(self, "tray_icon"):
            self.tray_icon.hide()
        self.close()

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._show_from_tray()

    # ---- Tab switching ----

    def _on_tab_changed(self, index: int) -> None:
        if index == 1:
            self._refresh_status()
        elif index == 2 and self.workflow_list.count() == 0:
            self._refresh_workflows()

    # ---- Status tab helpers ----

    def _refresh_status(self) -> None:
        self.status_info.setPlainText("Loading...")
        self._status_worker = StatusFetchWorker()
        self._status_worker.completed.connect(self._on_status_fetched)
        self._status_worker.start()

    def _on_status_fetched(self, data: dict) -> None:
        if not data:
            self.status_info.setPlainText(
                "\u26A0 Could not fetch status.\nIs the backend running?"
            )
            return
        lines = [
            f"\U0001F4C1 Active File: {data.get('active_file') or '(none)'}",
            f"\U0001F4C4 Active Sheet: {data.get('active_sheet') or '(none)'}",
        ]
        recent = data.get("recent_commands", [])
        lines.append(f"\U0001F4CB Recent Commands: {len(recent)}")
        if recent:
            lines.append("")
            lines.append("Last commands:")
            for i, cmd in enumerate(recent[-5:], 1):
                if isinstance(cmd, dict):
                    detail = cmd.get("values", cmd.get("headers", ""))
                    lines.append(
                        f"  {i}. {cmd.get('type', '?')} "
                        f"{json.dumps(detail, ensure_ascii=False) if detail else ''}"
                    )
                else:
                    lines.append(f"  {i}. {cmd}")
        workflows = data.get("available_workflows", [])
        if workflows:
            lines.append("")
            lines.append(f"\U0001F527 Available Workflows: {len(workflows)}")
        self.status_info.setPlainText("\n".join(lines))

    # ---- Workflow tab helpers ----

    def _refresh_workflows(self) -> None:
        self.workflow_list.clear()
        self.workflow_detail.setPlainText("Loading workflows...")
        self._workflow_worker = WorkflowFetchWorker()
        self._workflow_worker.completed.connect(self._on_workflows_fetched)
        self._workflow_worker.start()

    def _on_workflows_fetched(self, data: dict) -> None:
        self.workflow_list.clear()
        self._workflow_data = []
        workflows = data.get("workflows", [])
        if not workflows:
            self.workflow_list.addItem(
                "\u26A0 No workflows found. Is the backend running?"
            )
            return
        for wf in workflows:
            wf_id = wf.get("id", "?")
            category = wf.get("category", "")
            prefix = "\U0001F4CC" if wf.get("destructive") else "\u25B8"
            label = wf_id.replace("_", " ").title()
            self.workflow_list.addItem(f"{prefix} {label}  [{category}]")
            self._workflow_data.append(wf)
        self.workflow_detail.setPlainText("Click a workflow to see details.")

    def _on_workflow_clicked(self, item) -> None:
        idx = self.workflow_list.row(item)
        if idx < 0 or idx >= len(self._workflow_data):
            return
        wf = self._workflow_data[idx]
        action_types = wf.get("action_types", [])
        lines = [
            f"Workflow: {wf.get('id', '?')}",
            f"Category: {wf.get('category', '?')}",
            f"Actions: {', '.join(action_types)}" if action_types else "",
            f"Destructive: {'Yes \u26A0' if wf.get('destructive') else 'No'}",
            f"Requires File: {'Yes' if wf.get('requires_file') else 'No'}",
        ]
        self.workflow_detail.setPlainText("\n".join(ln for ln in lines if ln))

    # ---- Smart suggestions ----

    def _show_suggestions(self, suggestions: list) -> None:
        if not suggestions:
            self.suggestions_label.hide()
            return
        text = "\U0001F4A1 Did you mean:\n" + "\n".join(
            f"  \u2022 {s}" for s in suggestions[:5]
        )
        self.suggestions_label.setText(text)
        self.suggestions_label.show()

    def _hide_suggestions(self) -> None:
        self.suggestions_label.hide()
        self.suggestions_label.setText("")

    # ---- Developer mode ----

    def _toggle_dev_mode(self, checked: bool) -> None:
        self.dev_panel.setVisible(checked)

    def _update_dev_panel(self, result: UploadResult) -> None:
        raw = result.raw or {}
        info = {
            "action": result.action,
            "workflow_id": result.workflow_id,
            "memory": raw.get("memory"),
            "executed": raw.get("executed"),
        }
        self.dev_panel.setPlainText(
            json.dumps(info, indent=2, ensure_ascii=False, default=str)
        )

    def start_recording(self) -> None:
        if self.recorder is not None and self.recorder.isRunning():
            return
        # Do not start a new recording while the previous upload or execution is
        # still in flight — this prevents interleaved pipelines and out-of-order
        # results from session auto-listen triggers firing too early.
        if self.uploader is not None and self.uploader.isRunning():
            return
        if getattr(self, '_execute_worker', None) is not None and self._execute_worker.isRunning():
            return
        self._upload_gen += 1   # new recording cycle — stale results from prev cycle are ignored
        self._cmd_start = time.perf_counter()
        # Activate a fresh session when none is running (manual mic click or
        # first call from on_wake_detected before _session_active was set).
        if not self._session_active:
            self._session_errors = 0
            self._session_active = True
        # Clear stale action/result from the previous command so a failed parse
        # in the new cycle can never accidentally re-execute the old action.
        self._pending_result = None
        self._pending_action = None
        # Pause wake-word listener while recording to avoid mic conflicts
        if self.wake_listener is not None:
            self.wake_listener.pause()
        self._hide_error()
        self._hide_success()
        self._stop_thinking_indicator()
        self.mic_btn.setEnabled(False)
        self.mic_btn.setText("Listening")
        self.status_dot.set_state("listening")
        self.status_label.setText("Listening...")
        self.transcript_view.setPlainText("Listening... speak now")
        self._start_mic_animation()

        self.recorder = RecorderThread(seconds=RECORD_SECONDS)
        self.recorder.recording_finished.connect(self.on_recording_finished)
        self.recorder.recording_failed.connect(self.on_recording_failed)
        self.recorder.transcript_live.connect(self.on_live_transcript)
        self.recorder.start()

    def on_live_transcript(self, text: str) -> None:
        pass  # batch STT: final transcript arrives after upload, not during recording

    def on_recording_finished(self, wav_bytes: bytes) -> None:
        self._stop_mic_animation()
        self.mic_btn.setText("Transcribing...")
        self.status_dot.set_state("processing")
        self.status_label.setText("Transcribing...")
        self.transcript_view.setPlainText("Transcribing...")
        self._start_thinking_indicator("Transcribing")
        self.uploader = UploadWorker(wav_bytes, gen=self._upload_gen)
        self.uploader.completed.connect(self.on_upload_completed)
        self.uploader.start()
        # Safety timeout: re-enable mic if upload never completes (e.g. thread crash)
        self._upload_safety_timer = QTimer(self)
        self._upload_safety_timer.setSingleShot(True)
        self._upload_safety_timer.timeout.connect(self._on_upload_timeout)
        self._upload_safety_timer.start(90_000)

    def on_recording_failed(self, error: str) -> None:
        self._stop_mic_animation()
        self._stop_thinking_indicator()
        self.status_dot.set_state("error")
        self.status_label.setText("Recording failed")
        self.mic_btn.setEnabled(True)
        self.mic_btn.setText("Start Listening")
        self.response_view.setPlainText(error)
        self._show_error(error)
        # Resume wake-word listener on recording failure
        if self.wake_listener is not None:
            self.wake_listener.resume()
        # Hardware mic error: end session — looping would be futile until mic is fixed
        self._session_active = False
        QMessageBox.warning(self, "Recording Error", error)

    def on_upload_completed(self, result: UploadResult) -> None:
        # Stale-result guard: if a newer recording cycle started while this
        # upload was in flight, discard the result silently.  Prevents
        # out-of-order actions when two pipelines overlap.
        if result.gen != self._upload_gen:
            return
        if hasattr(self, '_upload_safety_timer'):
            self._upload_safety_timer.stop()
        self._stop_thinking_indicator()
        self.mic_btn.setEnabled(True)
        self.mic_btn.setText("Start Listening")
        # Resume wake-word listener now that recording + upload is done
        if self.wake_listener is not None:
            self.wake_listener.resume()

        if not result.success:
            msg = result.error or "Unknown error"
            self.status_dot.set_state("error")
            self.status_label.setText("Command failed")
            self._hide_success()
            self._hide_plan_preview()
            self.response_view.setPlainText(msg)
            self._show_error(msg)
            if result.suggestions:
                self._show_suggestions(result.suggestions)
            else:
                self._hide_suggestions()
            self._add_history_item("(failed)", msg, success=False)
            if self.dev_toggle.isChecked():
                self._update_dev_panel(result)
            # Speak STT failure feedback once so user hears it without looking at UI.
            # Only fires for recogniser errors — not network/backend errors — to avoid
            # clashing with any spoken error text the backend already sent back.
            _stt_phrases = ("speech recognition failed", "could not transcribe",
                            "no speech detected", "whisper", "transcription failed")
            if any(p in msg.lower() for p in _stt_phrases):
                self._tts_stt_feedback = TTSWorker(
                    "Speech recognition failed. Please try again."
                )
                self._tts_stt_feedback.start()
            # Session: STT/network failure → count consecutive errors, retry or sleep
            self._session_errors += 1
            if self._session_active and self._session_errors < 3:
                QTimer.singleShot(1500, self._session_listen)
            elif self._session_errors >= 3:
                self._session_sleep()
            return

        self._hide_error()
        self.transcript_view.setPlainText(result.transcript or "")
        if self.dev_toggle.isChecked():
            self._update_dev_panel(result)

        action = result.action
        # An action may arrive as {"type": "add_row", ...} OR as {"steps": [...]}
        # (the latter when _plan_and_validate wraps a single-step in a steps list).
        # Multi-step = steps list with more than one entry (regardless of top-level type).
        # Exception: add_column is non-destructive — multiple columns execute immediately
        # without plan-review confirmation so the user gets direct feedback.
        _steps = action.get("steps", []) if isinstance(action, dict) else []
        _all_add_col = bool(_steps) and all(
            isinstance(s, dict) and s.get("type") == "add_column" for s in _steps
        )
        is_multi_step = (
            isinstance(action, dict)
            and isinstance(action.get("steps"), list)
            and len(action["steps"]) > 1
            and not _all_add_col
        )

        if is_multi_step:
            # Show plan preview and wait for user confirmation
            self.status_dot.set_state("idle")
            self.status_label.setText("Review plan before execution")
            preview_text = self._format_plan_preview(action)
            self._show_plan_preview(preview_text, action)
            self._pending_action = action
            self._pending_result = result
            if result.suggestions:
                self._show_suggestions(result.suggestions)
            else:
                self._hide_suggestions()
        elif action and isinstance(action, dict) and (action.get("type") or action.get("steps")):
            # Single-step: show brief preview then auto-execute
            # Handles both {"type": "add_row", ...} and {"steps": [single_step]}
            self._hide_plan_preview()
            self._hide_suggestions()
            preview = self._describe_single_action(action)
            self.response_view.setPlainText(f"Executing: {preview}")
            self.status_dot.set_state("processing")
            self.status_label.setText("Executing...")
            self._start_thinking_indicator("Executing")
            self._pending_result = result
            self._execute_worker = ExecuteWorker(action)
            self._execute_worker.completed.connect(self._on_execute_completed)
            self._execute_worker.start()
        else:
            # No valid action parsed
            self.status_dot.set_state("idle")
            self.status_label.setText("Could not parse command")
            self._hide_plan_preview()
            self.response_view.setPlainText(result.ai_message or "No action recognized.")
            if result.suggestions:
                self._show_suggestions(result.suggestions)
            else:
                self._hide_suggestions()
            self._add_history_item(result.transcript, "(no action)", success=False)
            # Session: heard speech but no command parsed → soft failure, retry
            self._session_errors += 1
            if self._session_active and self._session_errors < 3:
                QTimer.singleShot(1500, self._session_listen)
            elif self._session_errors >= 3:
                self._session_sleep()

    def _on_execute_completed(self, payload: dict) -> None:
        """Handle result from ExecuteWorker (thread-safe via pyqtSignal)."""
        self._stop_thinking_indicator()
        self._hide_plan_preview()
        result = getattr(self, "_pending_result", None)
        transcript = result.transcript if result else ""

        # Single-response guard: identical success payload within 2 s → skip.
        # Prevents duplicate history/TTS when the pyttsx3 engine emits a queued
        # item twice due to concurrent filler/response worker interaction.
        if payload.get("success", False):
            _now = time.monotonic()
            _key = (payload.get("speech_text", "")[:80], True)
            if (getattr(self, "_last_exec_key", None) == _key and
                    _now - getattr(self, "_last_exec_time", 0.0) < 2.0):
                return
            self._last_exec_key = _key
            self._last_exec_time = _now

        if not payload.get("success", False):
            msg = payload.get("error", "Execution failed")
            self.status_dot.set_state("error")
            self.status_label.setText("Command failed")
            self._hide_success()
            self.response_view.setPlainText(msg)
            self._show_error(msg)
            # Show recovery suggestions from execute response
            suggestions = payload.get("suggestions", [])
            recovery = payload.get("recovery")
            if not suggestions and isinstance(recovery, dict):
                suggestions = recovery.get("suggestions", [])
            if suggestions:
                self._show_suggestions(suggestions)
            else:
                self._hide_suggestions()
            self._add_history_item(transcript, msg, success=False)
            # Execution failed but user DID speak — keep session alive, retry listening
            if self._session_active:
                QTimer.singleShot(2000, self._session_listen)
            return

        ai_message = payload.get("speech_text") or ""
        if not ai_message:
            exec_result = payload.get("result")
            if isinstance(exec_result, dict) and "message" in exec_result:
                ai_message = str(exec_result["message"])
            elif isinstance(exec_result, str):
                ai_message = exec_result
            else:
                ai_message = "Action executed successfully."
        self._hide_error()
        self._hide_suggestions()
        self._session_errors = 0  # successful command resets consecutive failure count
        self._show_success("Done")           # banner: short status only
        self.response_view.setPlainText(ai_message)   # full message here
        self._add_history_item(transcript, ai_message, success=True)
        if ai_message:
            self.status_dot.set_state("speaking")
            self.status_label.setText("Speaking...")
            _total_ms = (time.perf_counter() - getattr(self, '_cmd_start', time.perf_counter())) * 1000
            _lat_log.info('[LATENCY] tts_start  total=%.0fms (wake→tts)', _total_ms)
            self._tts_worker = TTSWorker(ai_message)
            self._tts_worker.finished.connect(self._show_done_state)
            self._tts_worker.start()
        else:
            self._show_done_state()

    def _on_plan_execute_clicked(self) -> None:
        """User confirmed multi-step plan execution."""
        action = getattr(self, "_pending_action", None)
        if not action:
            return
        self._hide_plan_preview()
        self.status_dot.set_state("processing")
        self.status_label.setText("Executing plan...")
        self._start_thinking_indicator("Executing plan")
        self._execute_worker = ExecuteWorker(action)
        self._execute_worker.completed.connect(self._on_execute_completed)
        self._execute_worker.start()

    def _on_plan_cancel_clicked(self) -> None:
        """User cancelled multi-step plan."""
        self._pending_action = None
        self._hide_plan_preview()
        self.status_dot.set_state("idle")
        self.status_label.setText("Plan cancelled")
        self.response_view.setPlainText("Plan cancelled by user.")
        self._add_history_item("(cancelled plan)", "User declined execution", success=False)

    def _show_done_state(self) -> None:
        """Briefly show Done after a command completes, then transition to Ready."""
        self.status_dot.set_state("done")
        self.status_label.setText("Done")
        QTimer.singleShot(1500, self._show_ready_state)

    def _show_ready_state(self) -> None:
        """Restore idle/ready state after Done is shown.

        Within an active session, auto-schedules the next recording cycle
        after SARA_POST_TTS_DELAY so speaker echo has time to decay before
        the mic opens — mirrors the POST_TTS_DELAY gap in run_sara.py.
        """
        self.status_dot.set_state("idle")
        if self._session_active:
            self.status_label.setText("Listening for next command...")
            _delay_ms = max(0, int(float(os.getenv('SARA_POST_TTS_DELAY', '0.6')) * 1000))
            QTimer.singleShot(_delay_ms, self._session_listen)
        else:
            self.status_label.setText("Ready")

    def _session_listen(self) -> None:
        """Re-enter the recording loop within an active session (auto-listen).

        Guards against double-start: no-op if the recorder is already running.
        """
        if self._session_active and not (self.recorder and self.recorder.isRunning()):
            self.start_recording()

    def _session_sleep(self) -> None:
        """Deactivate the session after 3 consecutive failures.

        Mirrors run_sara.py: break the conversation loop after
        MAX_CONSECUTIVE_ERRORS, show a message, require wake word again.
        """
        _wake_name = "Hey Sara" if WAKE_MODEL_PATH else "Hey Jarvis"
        self._session_active = False
        self._session_errors = 0
        self.status_dot.set_state("idle")
        self.status_label.setText(f'Say "{_wake_name}" to start again.')
        self.response_view.setPlainText(
            "Session ended after repeated failures.\n"
            f'Say "{_wake_name}" or press the mic button to start a new session.'
        )

    @staticmethod
    def _format_plan_preview(action: dict) -> str:
        """Format a multi-step action into a readable plan preview."""
        steps = action.get("steps", [])
        lines = ["\U0001F4CB Multi-step plan:"]
        for i, step in enumerate(steps, 1):
            stype = step.get("type", "unknown")
            desc = stype.replace("_", " ").title()
            # Add relevant parameters for context
            details = []
            for key in ("values", "headers", "column", "range", "sheet_name", "filename"):
                val = step.get(key)
                if val is not None:
                    details.append(f"{key}={val}")
            detail_str = f" ({', '.join(details)})" if details else ""
            lines.append(f"  {i}. {desc}{detail_str}")
        return "\n".join(lines)

    @staticmethod
    def _describe_single_action(action: dict) -> str:
        """Return a short human-readable description of a single action."""
        atype = action.get("type", "unknown").replace("_", " ").title()
        details = []
        for key in ("values", "headers", "column", "range", "sheet_name", "filename"):
            val = action.get(key)
            if val is not None:
                details.append(f"{key}={val}")
        detail_str = f" ({', '.join(details)})" if details else ""
        return f"{atype}{detail_str}"

    def _show_plan_preview(self, text: str, action: dict) -> None:
        """Display plan preview panel with execute/cancel buttons."""
        self.plan_preview_label.setText(text)
        self.plan_preview_label.show()
        self.plan_execute_btn.show()
        self.plan_cancel_btn.show()

    def _hide_plan_preview(self) -> None:
        self.plan_preview_label.hide()
        self.plan_preview_label.setText("")
        self.plan_execute_btn.hide()
        self.plan_cancel_btn.hide()

    def _start_wake_listener_if_enabled(self) -> None:
        if not WAKE_ENABLED:
            self.status_dot.set_state("idle")
            self.status_label.setText("Wake-word disabled. Press Start Listening.")
            return
        self.wake_listener = WakeWordListenerThread()
        self.wake_listener.wake_detected.connect(self.on_wake_detected)
        self.wake_listener.wake_failed.connect(self.on_wake_failed)
        self.wake_listener.start()
        # Display the actual wake word being used
        if WAKE_MODEL_PATH:
            ww_name = "Hey Sara"
        else:
            ww_name = "Hey Jarvis"
        self.status_label.setText(f"\U0001F3A4 Say \"{ww_name}\" to wake Sara")

    def on_wake_detected(self, label: str, score: float) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        self._hide_error()
        self._hide_success()
        self.status_dot.set_state("listening")
        self.status_label.setText(f"Sara activated! ({label}, {score:.2f})")
        self.response_view.setPlainText("Sara is listening for your command...")
        # Fresh session on every wake word: reset the consecutive-error counter
        # so a previously-exhausted session gets a clean slate.
        self._session_errors = 0
        self.start_recording()

    def on_wake_failed(self, error: str) -> None:
        self.status_dot.set_state("error")
        self.status_label.setText("Wake-word unavailable. Use mic button.")
        self.response_view.setPlainText(error)
        self._show_error(error)

    # ---- Runtime hook API ----

    def show_assistant(self) -> None:
        """Show and focus the floating assistant window."""
        self.show()
        self.raise_()
        self.activateWindow()

    def hide_assistant(self) -> None:
        """Hide the floating assistant window when the runtime is idle."""
        self.hide()

    def set_listening(self) -> None:
        """Update the UI into the animated listening state."""
        self.show_assistant()
        self._hide_error()
        self._hide_success()
        self._hide_suggestions()
        self._stop_thinking_indicator()
        self.status_dot.set_state("listening")
        self.status_label.setText("Sara is listening...")
        self.transcript_view.setPlainText("Listening...")
        self.response_view.setPlainText("")
        self.mic_btn.setEnabled(False)
        self._start_mic_animation()

    def set_processing(self) -> None:
        """Update the UI into the animated processing state."""
        self.show_assistant()
        self._stop_mic_animation()
        self._hide_error()
        self._hide_success()
        self.status_dot.set_state("processing")
        self.status_label.setText("Processing voice command...")
        self.mic_btn.setEnabled(False)
        self.mic_btn.setText("Processing...")
        self._start_thinking_indicator("Processing")

    def set_speaking(self, text: str = "") -> None:
        """Update the UI into the TTS/speaking state.

        Called by the runtime just before starting TTS so the popup shows
        'Speaking...' while Sara is talking rather than jumping straight
        to 'Ready'.  The next set_listening() call (at the top of the next
        conversation loop iteration) restores the listening state.
        """
        self._stop_mic_animation()
        self._stop_thinking_indicator()
        self.status_dot.set_state("speaking")
        self.status_label.setText("Speaking...")
        self.mic_btn.setEnabled(False)
        self.mic_btn.setText("Speaking...")
        if text:
            self.response_view.setPlainText(text)
            self._show_success("Done")

    def set_response(self, text: str) -> None:
        """Display Sara's response text and restore the idle button state."""
        self._stop_mic_animation()
        self._stop_thinking_indicator()
        self.mic_btn.setEnabled(True)
        self.mic_btn.setText("Start Listening")
        self.status_dot.set_state("idle")
        self.status_label.setText("Ready")
        self.response_view.setPlainText(text or "")
        if text:
            self._show_success("Done")
        else:
            self._hide_success()

    def set_idle(self) -> None:
        """Stop all recording animations and restore ready state.

        Called by the runtime when it returns to idle (e.g. after [no_voice]
        or conversation end).  Guaranteed to stop _mic_anim_timer and
        _thinking_timer even if set_processing/set_speaking were never called.
        """
        self._reset_recording_ui()
        self.mic_btn.setEnabled(True)
        self.mic_btn.setText("Start Listening")
        self.status_dot.set_state("idle")
        self.status_label.setText("Ready")

    def _add_history_item(self, transcript: str, response: str, *, success: bool = True) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        icon = "\u2705" if success else "\u274C"  # green checkmark / red cross
        line = f"{icon} [{ts}] {transcript or '(no transcript)'}\n   \u2192 {response or '(no response)'}"
        item = QListWidgetItem(line)
        if not success:
            item.setForeground(QColor("#FFB4B4"))
        self.history_list.insertItem(0, item)
        # Keep history bounded to avoid unbounded UI growth
        while self.history_list.count() > MAX_HISTORY_ITEMS:
            self.history_list.takeItem(self.history_list.count() - 1)

    def _start_mic_animation(self) -> None:
        """Pulse mic visuals and show countdown to indicate active listening."""
        self._mic_anim_frame = 0
        self._mic_record_start = time.monotonic()
        self.mic_btn.setText("\U0001F534 Recording...")
        self._mic_anim_timer.start(250)

    def _stop_mic_animation(self) -> None:
        self._mic_anim_timer.stop()
        self._apply_mic_idle_style()
        # Clear any countdown text the last tick wrote ("🔴 Recording... Xs left")
        if "\U0001F534" in self.mic_btn.text():
            self.mic_btn.setText("Start Listening")

    def _tick_mic_animation(self) -> None:
        # Show elapsed/remaining time on the button text
        elapsed = time.monotonic() - self._mic_record_start
        remaining = max(0, RECORD_SECONDS - elapsed)
        self.mic_btn.setText(f"\U0001F534 Recording... {remaining:.0f}s left")

        # Alternate between two color states for a pulse effect.
        phase = self._mic_anim_frame % 4
        if phase in (0, 1):
            self.mic_btn.setStyleSheet(
                "QPushButton {"
                "background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #19B6CD, stop:1 #4B8DBB);"
                "color: white; border: 1px solid #8FDBFF; border-radius: 14px;"
                "font-size: 14px; font-weight: 700; padding: 10px 14px; }"
                "QPushButton:disabled { background: #5C7D93; color: #E6F4FF; border: 1px solid #A5DBFF; }"
            )
        else:
            self.mic_btn.setStyleSheet(
                "QPushButton {"
                "background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1198AA, stop:1 #356F98);"
                "color: white; border: 1px solid #5DA7C8; border-radius: 14px;"
                "font-size: 14px; font-weight: 700; padding: 10px 14px; }"
                "QPushButton:disabled { background: #4D677B; color: #D5E3EE; border: 1px solid #7093AA; }"
            )
        self._mic_anim_frame += 1

    def _apply_mic_idle_style(self) -> None:
        self.mic_btn.setStyleSheet(
            "QPushButton {"
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #15AABF, stop:1 #3D7EAA);"
            "color: white; border: none; border-radius: 14px; font-size: 14px; font-weight: 600;"
            "padding: 10px 14px; }"
            "QPushButton:hover { background: #2d97ae; }"
            "QPushButton:disabled { background: #556f83; color: #d3dee6; }"
        )

    def _reset_recording_ui(self) -> None:
        """Stop all recording/progress animations. Call in any terminal UI state."""
        self._stop_mic_animation()
        self._stop_thinking_indicator()

    def _start_thinking_indicator(self, phase: str = "Sara is thinking") -> None:
        """Activate the thinking indicator row (spinner + label).

        Shows the spinner and label (hidden when idle; layout space always reserved
        via retainSizeWhenHidden so no layout bounce occurs).
        """
        self._thinking_phase = phase
        self._thinking_frame = 0
        self.thinking_spinner.start()
        self.thinking_label.show()
        self._thinking_timer.start(280)
        self._tick_thinking_animation()

    def _stop_thinking_indicator(self) -> None:
        self._thinking_timer.stop()
        self.thinking_spinner.stop()
        self.thinking_label.hide()
        self.thinking_label.setText("")

    def _tick_thinking_animation(self) -> None:
        dots = "." * ((self._thinking_frame % 3) + 1)
        phase = getattr(self, '_thinking_phase', 'Sara is thinking')
        self.thinking_label.setText(f"{phase}{dots}")
        self._thinking_frame += 1

    def _show_error(self, message: str) -> None:
        self._hide_success()
        self.error_label.setText(f"\u274C Error: {message}")
        self.error_label.show()

    def _hide_error(self) -> None:
        self.error_label.hide()
        self.error_label.setText("")

    def _show_success(self, message: str) -> None:
        self._hide_error()
        self.success_label.setText(f"\u2705 {message}")
        self.success_label.show()

    def _hide_success(self) -> None:
        self.success_label.hide()
        self.success_label.setText("")

    # frameless window drag behavior
    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):  # noqa: N802
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def _on_upload_timeout(self) -> None:
        """Safety net: re-enable the UI if the upload thread never responded."""
        if self.uploader is not None and self.uploader.isRunning():
            self._reset_recording_ui()
            self.mic_btn.setEnabled(True)
            self.mic_btn.setText("Start Listening")
            self.status_dot.set_state("error")
            self.status_label.setText("Upload timed out")
            self._show_error("Upload timed out. Please try again.")
            if self.wake_listener is not None:
                self.wake_listener.resume()

    def closeEvent(self, event):  # noqa: N802
        # Minimize to tray instead of quitting (if tray is available)
        if not self._force_quit and hasattr(self, 'tray_icon') and self.tray_icon.isVisible():
            event.ignore()
            self.hide()
            return
        self._stop_mic_animation()
        self._stop_thinking_indicator()
        if hasattr(self, '_upload_safety_timer'):
            self._upload_safety_timer.stop()
        if self.recorder is not None and self.recorder.isRunning():
            self.recorder.wait(1000)
        if self.uploader is not None and self.uploader.isRunning():
            self.uploader.wait(1000)
        if self.wake_listener is not None:
            self.wake_listener.stop()
            self.wake_listener.wait(1000)
        if hasattr(self, 'tray_icon'):
            self.tray_icon.hide()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Sara Desktop Assistant")
    # Keep app running when window is hidden (tray mode)
    app.setQuitOnLastWindowClosed(False)

    win = SaraFloatingWindow()
    desktop = QApplication.primaryScreen().availableGeometry()
    x = desktop.x() + desktop.width() - win.width() - 30
    y = desktop.y() + 50
    win.move(x, y)
    if WAKE_ENABLED:
        # Siri/assistant-style behavior: popup appears after wake word.
        win.hide()
    else:
        win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
