"""Thin proxy to the existing WakeDetector implementation.

The full openwakeword loop already lives in ``wakeword/wake_detector.py``.
This module exposes a cleaner, self-contained API for use from run_sara.py
without duplicating any detection logic.

Usage::

    from voice.wakeword_listener import WakeWordListener

    def on_wake():
        print("Wake word detected — listening for command...")

    listener = WakeWordListener(on_wake_callback=on_wake)
    listener.start()   # background thread — non-blocking
    ...
    listener.stop()

Wake word:
    The model used is determined by the WAKE_MODEL_PATH environment variable
    (a .onnx file).  When the variable is not set, the listener falls back to
    the ``hey_jarvis_v0.1.onnx`` model bundled with the openwakeword package.
    Point WAKE_MODEL_PATH at a custom "hey_sara.onnx" to use "Hey Sara".

Environment variables (all optional):
    WAKE_MODEL_PATH  — path to a custom .onnx wake-word model
    WAKE_THRESHOLD   — detection confidence threshold (default 0.5)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Callable

log = logging.getLogger('voice.wakeword_listener')

# Resolve paths so this module can be imported from backend/voice/
_VOICE_DIR   = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_VOICE_DIR)
_ROOT_DIR    = os.path.dirname(_BACKEND_DIR)

for _p in (_BACKEND_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class WakeWordListener:
    """Wrapper around :class:`wakeword.wake_detector.WakeDetector`.

    Provides a minimal start/stop interface that run_sara.py can use without
    importing the detector directly.  All openwakeword model loading, audio
    streaming, and threshold logic remain inside WakeDetector.

    Parameters
    ----------
    on_wake_callback:
        Callable with no arguments called each time the wake word is detected.
        Runs on the wake-detector thread, so keep it short (e.g. set a flag or
        put an item on a queue) and do the heavy work elsewhere.
    model_path:
        Path to a .onnx wake-word model.  Defaults to the WAKE_MODEL_PATH
        environment variable, then falls back to the bundled hey_jarvis model.
    threshold:
        Detection confidence threshold (0.0–1.0).  Defaults to WAKE_THRESHOLD
        env var or 0.5.
    """

    def __init__(
        self,
        on_wake_callback: Callable[[], None],
        model_path: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self._callback = on_wake_callback

        # Import here so the module can be imported even when openwakeword
        # is not installed (the error only surfaces when start() is called).
        from wakeword.wake_detector import WakeDetector, WAKE_MODEL_PATH, WAKE_THRESHOLD

        _model   = model_path   or WAKE_MODEL_PATH
        _thresh  = threshold    if threshold is not None else WAKE_THRESHOLD

        self._detector = WakeDetector(model_path=_model, threshold=_thresh)
        log.info('[WakeWordListener] Initialised (threshold=%.2f)', _thresh)

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start wake-word detection on a background daemon thread."""
        log.info('[WakeWordListener] Starting background listener')
        self._detector.start(self._callback)

    def stop(self) -> None:
        """Stop detection and release the microphone."""
        log.info('[WakeWordListener] Stopping')
        self._detector.stop()

    @property
    def is_alive(self) -> bool:
        """True while the detector thread is running."""
        t = self._detector._thread
        return t is not None and t.is_alive()
