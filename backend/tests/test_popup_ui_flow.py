"""Safe acceptance tests for the Sara popup/UI lifecycle and wake-word flow.

Strategy
--------
* All tests run purely in-process — no real audio, no real wake-word model,
  no real backend HTTP calls.
* ``WakeWordListenerThread`` and ``RecorderThread`` are stubbed out so no
  microphone or onnxruntime is needed.
* A single QApplication is created for the whole session (Qt requires at most
  one per process).
* ``SaraFloatingWindow`` is created with SARA_WAKE_ENABLED=0 so the real
  wake-word thread never starts.
* ``FloatingWindow`` (the simpler popup) is also tested.
* Each window is created and destroyed per test class to keep state clean.

What is tested
--------------
1.  StatusDot colour states (idle / listening / processing / speaking / error)
2.  show_assistant()     → window is visible
3.  hide_assistant()     → window is hidden
4.  set_listening()      → dot cyan, label "Sara is listening...", mic disabled
5.  set_processing()     → dot amber, label "Processing voice command...", mic disabled
6.  set_speaking(text)   → label "Speaking...", mic disabled, response updated
7.  set_response(text)   → dot green, mic enabled, label "Ready", response updated
8.  on_wake_detected()   → window shows, dot listening, label updated
9.  on_wake_failed()     → dot error, error label visible
10. Multiple show_assistant() calls → one window, no duplicates
11. State cycle: listening → processing → speaking → response → idle
12. FloatingWindow counterparts of the same hooks
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Skip the entire module if PyQt6 is absent (e.g. CI without display)
# ---------------------------------------------------------------------------
HAS_PYQT6 = importlib.util.find_spec("PyQt6") is not None
pytestmark = pytest.mark.skipif(not HAS_PYQT6, reason="PyQt6 not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _process_events():
    """Pump the Qt event loop so deferred signals / timers are processed."""
    from PyQt6.QtWidgets import QApplication
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# Session-scoped QApplication (Qt allows only one per process)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Create a single QApplication for the test session."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(["pytest", "--platform=offscreen"])
    yield app
    # Do NOT call app.quit() — other tests in the same process may need it.


# ===========================================================================
# 1. StatusDot colour states
# ===========================================================================

class TestStatusDot:
    """Unit-test the StatusDot widget in isolation."""

    @pytest.fixture(autouse=True)
    def dot(self, qapp):
        from desktop.sara_ui import StatusDot
        d = StatusDot()
        yield d
        d.close()
        d.deleteLater()
        _process_events()

    def test_default_state_is_idle(self, dot):
        from PyQt6.QtGui import QColor
        from desktop.sara_ui import StatusDot
        assert dot._color == StatusDot.COLOR_IDLE

    def test_set_listening(self, dot):
        from desktop.sara_ui import StatusDot
        dot.set_state("listening")
        assert dot._color == StatusDot.COLOR_LISTENING

    def test_set_processing(self, dot):
        from desktop.sara_ui import StatusDot
        dot.set_state("processing")
        assert dot._color == StatusDot.COLOR_PROCESSING

    def test_set_speaking(self, dot):
        from desktop.sara_ui import StatusDot
        dot.set_state("speaking")
        assert dot._color == StatusDot.COLOR_SPEAKING

    def test_set_error(self, dot):
        from desktop.sara_ui import StatusDot
        dot.set_state("error")
        assert dot._color == StatusDot.COLOR_ERROR

    def test_set_idle_explicit(self, dot):
        from desktop.sara_ui import StatusDot
        dot.set_state("error")
        dot.set_state("idle")
        assert dot._color == StatusDot.COLOR_IDLE

    def test_unknown_state_falls_back_to_idle(self, dot):
        from desktop.sara_ui import StatusDot
        dot.set_state("totally_unknown_state")
        assert dot._color == StatusDot.COLOR_IDLE


# ===========================================================================
# 2–11. SaraFloatingWindow lifecycle
# ===========================================================================

@pytest.fixture(scope="class")
def sara_window(qapp):
    """Create a SaraFloatingWindow with wake-word and tray disabled.

    Two patches are required:
    * _setup_tray                    — fails on headless (no system tray)
    * _start_wake_listener_if_enabled — WAKE_ENABLED is a module-level bool
      set at import time; changing os.environ afterwards has no effect.
      Without this patch the WakeWordListenerThread starts, hits the
      onnxruntime DLL conflict, emits wake_failed, and writes the error
      message into response_view — corrupting state mid-test.
    """
    from unittest.mock import patch
    from desktop.sara_ui import SaraFloatingWindow

    with patch.object(SaraFloatingWindow, "_setup_tray"), \
         patch.object(SaraFloatingWindow, "_start_wake_listener_if_enabled"):
        win = SaraFloatingWindow()

    win.show()
    _process_events()
    yield win

    win._force_quit = True
    win.close()
    # deleteLater() removes the widget from QApplication.topLevelWidgets()
    # so class-scoped windows don't accumulate across fixture teardowns.
    win.deleteLater()
    _process_events()


class TestSaraWindowVisibility:
    """Tests for show/hide and popup singularity."""

    def test_show_assistant_makes_window_visible(self, sara_window):
        sara_window.show_assistant()
        _process_events()
        assert sara_window.isVisible()

    def test_hide_assistant_makes_window_hidden(self, sara_window):
        sara_window.show_assistant()
        _process_events()
        sara_window.hide_assistant()
        _process_events()
        assert not sara_window.isVisible()

    def test_multiple_show_calls_no_duplicates(self, sara_window):
        """Calling show_assistant() several times must not spawn extra windows."""
        from PyQt6.QtWidgets import QApplication
        sara_window.show_assistant()
        sara_window.show_assistant()
        sara_window.show_assistant()
        _process_events()
        # Count top-level windows of SaraFloatingWindow type
        from desktop.sara_ui import SaraFloatingWindow
        count = sum(
            1 for w in QApplication.topLevelWidgets()
            if isinstance(w, SaraFloatingWindow)
        )
        assert count == 1, f"Expected 1 SaraFloatingWindow, found {count}"
        assert sara_window.isVisible()

    def test_show_after_hide_restores_visibility(self, sara_window):
        sara_window.hide_assistant()
        _process_events()
        assert not sara_window.isVisible()
        sara_window.show_assistant()
        _process_events()
        assert sara_window.isVisible()


class TestSaraListeningState:
    """Tests for set_listening() hook."""

    @pytest.fixture(autouse=True)
    def reset_after(self, sara_window):
        yield
        # Always reset to idle so tests don't bleed into each other
        sara_window.set_response("")
        _process_events()

    def test_set_listening_shows_window(self, sara_window):
        sara_window.set_listening()
        _process_events()
        assert sara_window.isVisible()

    def test_set_listening_dot_state(self, sara_window):
        from desktop.sara_ui import StatusDot
        sara_window.set_listening()
        _process_events()
        assert sara_window.status_dot._color == StatusDot.COLOR_LISTENING

    def test_set_listening_label_text(self, sara_window):
        sara_window.set_listening()
        _process_events()
        assert "listening" in sara_window.status_label.text().lower()

    def test_set_listening_mic_disabled(self, sara_window):
        sara_window.set_listening()
        _process_events()
        assert not sara_window.mic_btn.isEnabled()

    def test_set_listening_transcript_updated(self, sara_window):
        sara_window.set_listening()
        _process_events()
        assert "listening" in sara_window.transcript_view.toPlainText().lower()


class TestSaraProcessingState:
    """Tests for set_processing() hook."""

    @pytest.fixture(autouse=True)
    def reset_after(self, sara_window):
        yield
        sara_window.set_response("")
        _process_events()

    def test_set_processing_shows_window(self, sara_window):
        sara_window.set_processing()
        _process_events()
        assert sara_window.isVisible()

    def test_set_processing_dot_state(self, sara_window):
        from desktop.sara_ui import StatusDot
        sara_window.set_processing()
        _process_events()
        assert sara_window.status_dot._color == StatusDot.COLOR_PROCESSING

    def test_set_processing_label_text(self, sara_window):
        sara_window.set_processing()
        _process_events()
        assert "processing" in sara_window.status_label.text().lower()

    def test_set_processing_mic_disabled(self, sara_window):
        sara_window.set_processing()
        _process_events()
        assert not sara_window.mic_btn.isEnabled()

    def test_set_processing_mic_button_text(self, sara_window):
        sara_window.set_processing()
        _process_events()
        assert "processing" in sara_window.mic_btn.text().lower()


class TestSaraSpeakingState:
    """Tests for set_speaking() hook."""

    @pytest.fixture(autouse=True)
    def reset_after(self, sara_window):
        yield
        sara_window.set_response("")
        _process_events()

    def test_set_speaking_label_text(self, sara_window):
        sara_window.set_speaking("Hello there")
        _process_events()
        assert "speaking" in sara_window.status_label.text().lower()

    def test_set_speaking_mic_disabled(self, sara_window):
        sara_window.set_speaking("Hello there")
        _process_events()
        assert not sara_window.mic_btn.isEnabled()

    def test_set_speaking_mic_button_text(self, sara_window):
        sara_window.set_speaking("Hello there")
        _process_events()
        assert "speaking" in sara_window.mic_btn.text().lower()

    def test_set_speaking_response_text_shown(self, sara_window):
        sara_window.set_speaking("Test response")
        _process_events()
        assert "Test response" in sara_window.response_view.toPlainText()

    def test_set_speaking_no_text_does_not_crash(self, sara_window):
        sara_window.set_speaking()
        _process_events()
        # No assertion — just must not raise


class TestSaraResponseState:
    """Tests for set_response() hook (idle restore)."""

    @pytest.fixture(autouse=True)
    def enter_processing(self, sara_window):
        sara_window.set_processing()
        _process_events()
        yield
        sara_window.set_response("")
        _process_events()

    def test_set_response_mic_re_enabled(self, sara_window):
        sara_window.set_response("Done.")
        _process_events()
        assert sara_window.mic_btn.isEnabled()

    def test_set_response_mic_button_text_reset(self, sara_window):
        sara_window.set_response("Done.")
        _process_events()
        assert "start listening" in sara_window.mic_btn.text().lower()

    def test_set_response_dot_idle(self, sara_window):
        from desktop.sara_ui import StatusDot
        sara_window.set_response("Done.")
        _process_events()
        assert sara_window.status_dot._color == StatusDot.COLOR_IDLE

    def test_set_response_status_label_ready(self, sara_window):
        sara_window.set_response("Done.")
        _process_events()
        assert "ready" in sara_window.status_label.text().lower()

    def test_set_response_text_appears(self, sara_window):
        sara_window.set_response("File created successfully.")
        _process_events()
        assert "File created successfully." in sara_window.response_view.toPlainText()

    def test_set_response_empty_does_not_crash(self, sara_window):
        sara_window.set_response("")
        _process_events()
        assert sara_window.mic_btn.isEnabled()


class TestSaraWakeCallbacks:
    """Tests for on_wake_detected() and on_wake_failed() signals."""

    @pytest.fixture(autouse=True)
    def reset_after(self, sara_window):
        yield
        sara_window.set_response("")
        _process_events()

    def test_on_wake_detected_shows_window(self, sara_window):
        sara_window.hide()
        _process_events()
        # Simulate wake callback without starting recording (patch start_recording)
        from unittest.mock import patch
        with patch.object(sara_window, "start_recording"):
            sara_window.on_wake_detected("hey_jarvis", 0.87)
        _process_events()
        assert sara_window.isVisible()

    def test_on_wake_detected_dot_listening(self, sara_window):
        from unittest.mock import patch
        from desktop.sara_ui import StatusDot
        with patch.object(sara_window, "start_recording"):
            sara_window.on_wake_detected("hey_jarvis", 0.87)
        _process_events()
        assert sara_window.status_dot._color == StatusDot.COLOR_LISTENING

    def test_on_wake_detected_label_updated(self, sara_window):
        from unittest.mock import patch
        with patch.object(sara_window, "start_recording"):
            sara_window.on_wake_detected("hey_jarvis", 0.95)
        _process_events()
        label = sara_window.status_label.text().lower()
        assert "sara" in label or "activated" in label or "hey_jarvis" in label

    def test_on_wake_detected_second_call_no_duplicate(self, sara_window):
        """Two rapid wake events must not open a second window."""
        from unittest.mock import patch
        from PyQt6.QtWidgets import QApplication
        from desktop.sara_ui import SaraFloatingWindow
        with patch.object(sara_window, "start_recording"):
            sara_window.on_wake_detected("hey_jarvis", 0.8)
            sara_window.on_wake_detected("hey_jarvis", 0.9)
        _process_events()
        count = sum(
            1 for w in QApplication.topLevelWidgets()
            if isinstance(w, SaraFloatingWindow)
        )
        assert count == 1

    def test_on_wake_failed_dot_error(self, sara_window):
        sara_window.on_wake_failed("Model not found")
        _process_events()
        from desktop.sara_ui import StatusDot
        assert sara_window.status_dot._color == StatusDot.COLOR_ERROR

    def test_on_wake_failed_error_label_visible(self, sara_window):
        sara_window.on_wake_failed("Model not found")
        _process_events()
        assert sara_window.error_label.isVisible()

    def test_on_wake_failed_label_text(self, sara_window):
        sara_window.on_wake_failed("Model not found")
        _process_events()
        label = sara_window.status_label.text().lower()
        assert "wake" in label or "unavailable" in label or "button" in label


class TestSaraFullStateCycle:
    """Tests for the complete listening → processing → speaking → idle cycle."""

    def test_full_cycle_ends_idle(self, sara_window):
        from desktop.sara_ui import StatusDot
        sara_window.set_listening()
        _process_events()
        sara_window.set_processing()
        _process_events()
        sara_window.set_speaking("Here is your answer.")
        _process_events()
        sara_window.set_response("Here is your answer.")
        _process_events()

        assert sara_window.mic_btn.isEnabled()
        assert sara_window.status_dot._color == StatusDot.COLOR_IDLE
        assert "ready" in sara_window.status_label.text().lower()
        assert "start listening" in sara_window.mic_btn.text().lower()

    def test_second_cycle_works_after_first(self, sara_window):
        """Verify the window is not stuck after one complete cycle."""
        from desktop.sara_ui import StatusDot

        for _ in range(2):
            sara_window.set_listening()
            _process_events()
            sara_window.set_processing()
            _process_events()
            sara_window.set_response("OK done.")
            _process_events()

        assert sara_window.mic_btn.isEnabled()
        assert sara_window.status_dot._color == StatusDot.COLOR_IDLE

    def test_mic_never_stuck_disabled_after_response(self, sara_window):
        sara_window.set_listening()
        _process_events()
        assert not sara_window.mic_btn.isEnabled(), "Should be disabled while listening"
        sara_window.set_response("done")
        _process_events()
        assert sara_window.mic_btn.isEnabled(), "Must be re-enabled after response"


# ===========================================================================
# 12. FloatingWindow (simpler popup in ui/floating_window.py)
# ===========================================================================

@pytest.fixture(scope="class")
def floating_window(qapp):
    """Create the simple FloatingWindow."""
    from ui.floating_window import FloatingWindow
    win = FloatingWindow()
    win.show()
    _process_events()
    yield win
    win.close()
    _process_events()


class TestFloatingWindowLifecycle:
    """Tests for the simpler FloatingWindow hook API."""

    def test_show_assistant_visible(self, floating_window):
        floating_window.show_assistant()
        _process_events()
        assert floating_window.isVisible()

    def test_hide_assistant_hidden(self, floating_window):
        floating_window.show_assistant()
        _process_events()
        floating_window.hide_assistant()
        _process_events()
        assert not floating_window.isVisible()

    def test_set_listening_disables_input(self, floating_window):
        floating_window.set_listening()
        _process_events()
        assert not floating_window.input_box.isEnabled()

    def test_set_listening_status_text(self, floating_window):
        floating_window.set_listening()
        _process_events()
        assert "listening" in floating_window.status_label.text().lower()

    def test_set_processing_disables_input(self, floating_window):
        floating_window.set_processing()
        _process_events()
        assert not floating_window.input_box.isEnabled()

    def test_set_processing_status_text(self, floating_window):
        floating_window.set_processing()
        _process_events()
        assert "processing" in floating_window.status_label.text().lower()

    def test_set_speaking_disables_input(self, floating_window):
        floating_window.set_speaking("Hello")
        _process_events()
        assert not floating_window.input_box.isEnabled()

    def test_set_speaking_status_text(self, floating_window):
        floating_window.set_speaking("Hello")
        _process_events()
        assert "speaking" in floating_window.status_label.text().lower()

    def test_set_response_re_enables_input(self, floating_window):
        floating_window.set_processing()
        _process_events()
        floating_window.set_response("Done.")
        _process_events()
        assert floating_window.input_box.isEnabled()

    def test_set_response_appends_transcript(self, floating_window):
        floating_window.set_processing()
        _process_events()
        floating_window.set_response("File created.")
        _process_events()
        html = floating_window.transcript.toHtml()
        assert "File created." in html or "File created." in floating_window.transcript.toPlainText()

    def test_full_cycle_restores_input(self, floating_window):
        floating_window.set_listening()
        _process_events()
        assert not floating_window.input_box.isEnabled()
        floating_window.set_processing()
        _process_events()
        assert not floating_window.input_box.isEnabled()
        floating_window.set_speaking("Speaking now")
        _process_events()
        assert not floating_window.input_box.isEnabled()
        floating_window.set_response("All done.")
        _process_events()
        assert floating_window.input_box.isEnabled()
