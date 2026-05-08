"""Desktop UI tests for Sara using pytest-qt.

Tests the PyQt6 ``SaraFloatingWindow`` widget tree, signal/slot wiring,
plan-preview panel, suggestions panel, and status transitions.

Automatically skipped when PyQt6 or pytest-qt are not installed.

Run:
    pytest backend/tests/test_ui_desktop.py -v -m ui
"""
from __future__ import annotations

import importlib.util
import os
import sys
import pathlib
import json
from unittest.mock import patch, MagicMock
from dataclasses import field

import pytest

# ── path setup ──────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent
BACKEND_DIR = str(HERE.parent)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
PROJECT_ROOT = str(HERE.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HAS_PYQT6 = importlib.util.find_spec("PyQt6") is not None
HAS_PYTEST_QT = importlib.util.find_spec("pytestqt") is not None

pytestmark = [
    pytest.mark.ui,
    pytest.mark.skipif(not HAS_PYQT6, reason="PyQt6 not installed"),
    pytest.mark.skipif(not HAS_PYTEST_QT, reason="pytest-qt not installed"),
]

# Lazy import — avoid loading onnxruntime DLL at collection time which can
# cause an access-violation crash on Windows when the TTS COM thread is active.
SaraFloatingWindow = None
UploadResult = None
TTSWorker = None
ExecuteWorker = None
Qt = None  # type: ignore[assignment]
QApplication = None  # type: ignore[assignment]


def _ensure_imports():
    """Import heavy modules once, guarded by a mock for onnxruntime."""
    global SaraFloatingWindow, UploadResult, TTSWorker, ExecuteWorker, Qt, QApplication
    if SaraFloatingWindow is not None:
        return
    # Prevent onnxruntime from loading its native DLLs during testing
    if "onnxruntime" not in sys.modules:
        sys.modules["onnxruntime"] = MagicMock()
    from desktop.sara_ui import (  # noqa: E402
        SaraFloatingWindow as _SW,
        UploadResult as _UR,
        TTSWorker as _TW,
        ExecuteWorker as _EW,
    )
    from PyQt6.QtCore import Qt as _Qt  # noqa: E402
    from PyQt6.QtWidgets import QApplication as _QA  # noqa: E402
    SaraFloatingWindow = _SW
    UploadResult = _UR
    TTSWorker = _TW
    ExecuteWorker = _EW
    Qt = _Qt
    QApplication = _QA


# ── fixtures ────────────────────────────────────────────────────

@pytest.fixture
def window(qtbot):
    """Create a SaraFloatingWindow and register it with qtbot."""
    _ensure_imports()
    # Disable wake word listener to avoid mic access during tests
    with patch.dict(os.environ, {"SARA_WAKE_ENABLED": "false"}):
        # Mock tray icon and network-hitting helpers to avoid hangs
        with (
            patch.object(SaraFloatingWindow, "_setup_tray"),
            patch.object(SaraFloatingWindow, "_refresh_status"),
            patch.object(SaraFloatingWindow, "_refresh_workflows"),
        ):
            win = SaraFloatingWindow()
            qtbot.addWidget(win)
            win.show()
            yield win
            # Stop timers to prevent QThread-destroyed warnings
            win._mic_anim_timer.stop()
            win._thinking_timer.stop()
            win.close()


# ── Widget tree tests ───────────────────────────────────────────

class TestWidgetTree:
    """Verify the UI contains all expected widgets."""

    def test_mic_button_exists(self, window):
        assert window.mic_btn is not None
        assert window.mic_btn.isEnabled()
        assert "Start Listening" in window.mic_btn.text()

    def test_status_dot_exists(self, window):
        assert window.status_dot is not None

    def test_status_label_exists(self, window):
        assert window.status_label is not None

    def test_transcript_view_exists(self, window):
        assert window.transcript_view is not None
        assert window.transcript_view.isReadOnly()

    def test_response_view_exists(self, window):
        assert window.response_view is not None
        assert window.response_view.isReadOnly()

    def test_suggestions_label_exists(self, window):
        assert window.suggestions_label is not None
        assert window.suggestions_label.isHidden()

    def test_plan_preview_widgets_exist(self, window):
        assert window.plan_preview_label is not None
        assert window.plan_execute_btn is not None
        assert window.plan_cancel_btn is not None
        # All hidden initially
        assert window.plan_preview_label.isHidden()
        assert window.plan_execute_btn.isHidden()
        assert window.plan_cancel_btn.isHidden()

    def test_history_list_exists(self, window):
        assert window.history_list is not None
        assert window.history_list.count() == 0

    def test_dev_panel_hidden_by_default(self, window):
        assert window.dev_panel.isHidden()

    def test_thinking_elements_hidden(self, window):
        assert window.thinking_label.isHidden()
        assert window.thinking_spinner is not None

    def test_tabs_present(self, window):
        assert hasattr(window, "tab_widget")
        assert window.tab_widget.count() == 3


# ── Suggestions panel ──────────────────────────────────────────

class TestSuggestionsPanel:

    def test_show_suggestions(self, window):
        window._show_suggestions(["Try 'add row Tom 90'", "Try 'create excel'"])
        assert window.suggestions_label.isVisible()
        text = window.suggestions_label.text()
        assert "add row Tom 90" in text
        assert "create excel" in text

    def test_show_empty_suggestions_hides(self, window):
        window._show_suggestions(["something"])
        assert window.suggestions_label.isVisible()
        window._show_suggestions([])
        assert window.suggestions_label.isHidden()

    def test_hide_suggestions(self, window):
        window._show_suggestions(["test"])
        window._hide_suggestions()
        assert window.suggestions_label.isHidden()
        assert window.suggestions_label.text() == ""


# ── Plan preview panel ─────────────────────────────────────────

class TestPlanPreview:

    def test_show_plan_preview(self, window):
        action = {
            "type": "multi_step",
            "steps": [
                {"type": "create_excel", "headers": ["Name", "Score"]},
                {"type": "add_row", "values": ["Tom", 90]},
            ]
        }
        preview_text = window._format_plan_preview(action)
        window._show_plan_preview(preview_text, action)
        assert window.plan_preview_label.isVisible()
        assert window.plan_execute_btn.isVisible()
        assert window.plan_cancel_btn.isVisible()
        assert "create excel" in preview_text.lower()
        assert "add row" in preview_text.lower()

    def test_hide_plan_preview(self, window):
        window._show_plan_preview("test", {})
        window._hide_plan_preview()
        assert window.plan_preview_label.isHidden()
        assert window.plan_execute_btn.isHidden()
        assert window.plan_cancel_btn.isHidden()

    def test_format_plan_preview_content(self, window):
        action = {
            "steps": [
                {"type": "create_excel", "headers": ["Name", "Score"], "filename": "test.xlsx"},
                {"type": "add_row", "values": ["Alice", 95]},
                {"type": "add_row", "values": ["Bob", 88]},
            ]
        }
        text = window._format_plan_preview(action)
        assert "1." in text
        assert "2." in text
        assert "3." in text
        assert "Create Excel" in text
        assert "Add Row" in text

    def test_cancel_plan_clears_preview(self, window, qtbot):
        window._pending_action = {"steps": [{"type": "add_row", "values": ["x"]}]}
        window._show_plan_preview("test plan", window._pending_action)
        window._on_plan_cancel_clicked()
        assert window.plan_preview_label.isHidden()
        assert window._pending_action is None

    def test_describe_single_action(self, window):
        desc = window._describe_single_action({"type": "add_row", "values": ["Tom", 90]})
        assert "Add Row" in desc
        assert "values" in desc


# ── Upload completion handler ──────────────────────────────────

class TestOnUploadCompleted:

    def test_success_single_action_triggers_execute(self, window, qtbot):
        """Single-action result should trigger execution."""
        result = UploadResult(
            success=True,
            transcript="add row Tom 90",
            ai_message="Action parsed.",
            action={"type": "add_row", "values": ["Tom", 90]},
        )
        with patch.object(ExecuteWorker, "start"):
            window.on_upload_completed(result)
            assert window.status_label.text() == "Executing command..."

    def test_success_multi_step_shows_preview(self, window, qtbot):
        """Multi-step result should show plan preview, not execute immediately."""
        result = UploadResult(
            success=True,
            transcript="create and add rows",
            ai_message="Multi-step plan.",
            action={
                "type": "multi_step",
                "steps": [
                    {"type": "create_excel", "headers": ["A"]},
                    {"type": "add_row", "values": ["x"]},
                ]
            },
        )
        window.on_upload_completed(result)
        assert window.plan_preview_label.isVisible()
        assert window.plan_execute_btn.isVisible()
        assert "Review plan" in window.status_label.text()

    def test_failure_shows_error_and_suggestions(self, window, qtbot):
        """Failed upload should show error + suggestions."""
        result = UploadResult(
            success=False,
            error="No Excel file is open",
            suggestions=["Create a file first", "Use auto_correct"],
        )
        window.on_upload_completed(result)
        assert window.error_label.isVisible()
        assert "No Excel file" in window.error_label.text()
        assert window.suggestions_label.isVisible()
        assert "Create a file" in window.suggestions_label.text()

    def test_failure_no_suggestions_hides_panel(self, window, qtbot):
        result = UploadResult(success=False, error="Network error")
        window.on_upload_completed(result)
        assert window.error_label.isVisible()
        assert window.suggestions_label.isHidden()

    def test_no_action_shows_no_action_message(self, window, qtbot):
        result = UploadResult(
            success=True,
            transcript="blahblah",
            action=None,
        )
        window.on_upload_completed(result)
        assert "Could not parse" in window.status_label.text()


# ── Execute completion handler ─────────────────────────────────

class TestOnExecuteCompleted:

    def test_success_shows_message(self, window, qtbot):
        window._pending_result = UploadResult(success=True, transcript="test")
        with patch.object(TTSWorker, "start"):
            window._on_execute_completed({"success": True, "speech_text": "Done!"})
        assert window.success_label.isVisible()
        assert "Done!" in window.success_label.text()
        assert window.error_label.isHidden()

    def test_failure_shows_error_with_recovery(self, window, qtbot):
        window._pending_result = UploadResult(success=True, transcript="test")
        payload = {
            "success": False,
            "error": "File not found",
            "recovery": {"suggestions": ["Create the file first"]},
        }
        window._on_execute_completed(payload)
        assert window.error_label.isVisible()
        assert "File not found" in window.error_label.text()
        assert window.suggestions_label.isVisible()
        assert "Create the file" in window.suggestions_label.text()


# ── Status transitions ─────────────────────────────────────────

class TestStatusTransitions:

    def test_set_listening(self, window):
        _ensure_imports()
        from desktop.sara_ui import StatusDot
        window.set_listening()
        assert window.status_dot._color == StatusDot.COLOR_LISTENING
        assert "listening" in window.status_label.text().lower()

    def test_set_processing(self, window):
        _ensure_imports()
        from desktop.sara_ui import StatusDot
        window.set_processing()
        assert window.status_dot._color == StatusDot.COLOR_PROCESSING
        assert "Processing" in window.status_label.text()

    def test_set_response(self, window):
        _ensure_imports()
        from desktop.sara_ui import StatusDot
        window.set_response("Hello from Sara")
        assert window.status_dot._color == StatusDot.COLOR_IDLE
        assert window.mic_btn.isEnabled()
        assert window.response_view.toPlainText() == "Hello from Sara"

    def test_set_response_empty(self, window):
        window.set_response("")
        assert window.success_label.isHidden()


# ── History ────────────────────────────────────────────────────

class TestHistory:

    def test_add_history_item(self, window):
        window._add_history_item("hello", "world", success=True)
        assert window.history_list.count() == 1

    def test_history_bounded(self, window):
        for i in range(15):
            window._add_history_item(f"cmd {i}", f"resp {i}", success=True)
        # MAX_HISTORY_ITEMS = 10
        assert window.history_list.count() <= 10

    def test_failed_item_added(self, window):
        window._add_history_item("bad cmd", "error msg", success=False)
        assert window.history_list.count() == 1
        item_text = window.history_list.item(0).text()
        assert "\u274c" in item_text.lower() or "❌" in item_text


# ── TTSWorker / ExecuteWorker unit tests ───────────────────────

class TestWorkerClasses:

    @pytest.fixture(autouse=True)
    def _load(self):
        _ensure_imports()

    def test_tts_worker_creation(self):
        w = TTSWorker("test speech")
        assert w.text == "test speech"

    def test_execute_worker_creation(self):
        action = {"type": "add_row", "values": ["x"]}
        w = ExecuteWorker(action, filename="test.xlsx")
        assert w.action == action
        assert w.filename == "test.xlsx"

    def test_execute_worker_default_filename(self):
        w = ExecuteWorker({"type": "save"})
        assert w.filename == ""
