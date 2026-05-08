"""Porcupine-based wake word detector for Sara.

Uses the Picovoice Porcupine engine to listen for a custom "Hey Sara"
keyword.  Porcupine runs entirely on-device (no cloud), uses < 5 % CPU
on a modern laptop, and has best-in-class false-positive rejection.

Setup (one-time)
----------------
1. pip install pvporcupine pvrecorder
2. Get a free access key at https://picovoice.ai  (free for personal/academic use)
3. Set PICOVOICE_ACCESS_KEY in backend/.env
4. Either:
   a) Use the built-in "Hey Sara"-equivalent model provided by Picovoice.
      Go to https://picovoice.ai/console → Porcupine → Wake Word → "Hey Sara"
      Download the .ppn file for your platform (Windows).
      Set WAKE_MODEL_PATH=path/to/hey_sara_windows.ppn  in .env
   b) Use a built-in keyword (e.g. "jarvis") as a demo:
      Set WAKE_BUILTIN_KEYWORD=jarvis  in .env  (no .ppn file needed)

Environment variables
---------------------
PICOVOICE_ACCESS_KEY  : (required) Picovoice account access key
WAKE_MODEL_PATH       : (optional) path to .ppn keyword file for "Hey Sara"
WAKE_BUILTIN_KEYWORD  : (optional) built-in keyword to use when no .ppn file
                        is provided.  Choices: alexa, hey google, hey siri,
                        jarvis, ok google, picovoice, porcupine, terminator,
                        bumblebee, computer, grapefruit, grasshopper, blueberry,
                        americano, hey barista, hey mycroft.
                        Default: "jarvis"  (sounds similar to "Hey Sara" for demo)
WAKE_THRESHOLD        : detection sensitivity 0.0-1.0  (default: 0.5)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger('wake_detector_porcupine')

PICOVOICE_ACCESS_KEY: str = os.getenv('PICOVOICE_ACCESS_KEY', '')
WAKE_MODEL_PATH: str | None = os.getenv('WAKE_MODEL_PATH') or None
WAKE_BUILTIN_KEYWORD: str = os.getenv('WAKE_BUILTIN_KEYWORD', 'jarvis')
WAKE_THRESHOLD: float = float(os.getenv('WAKE_THRESHOLD', '0.5'))


class WakeDetectorPorcupine:
    """Wake word detector backed by Picovoice Porcupine.

    Identical public interface to WakeDetector (openwakeword-based) so that
    either class can be dropped into SaraRuntime without code changes.

    Usage::

        detector = WakeDetectorPorcupine()
        detector.start(callback=runtime.handle_wake_word)
        # ... later ...
        detector.stop()
    """

    def __init__(
        self,
        access_key: str = PICOVOICE_ACCESS_KEY,
        model_path: str | None = WAKE_MODEL_PATH,
        builtin_keyword: str = WAKE_BUILTIN_KEYWORD,
        threshold: float = WAKE_THRESHOLD,
    ):
        if not access_key:
            raise ValueError(
                'PICOVOICE_ACCESS_KEY is not set. '
                'Get a free key at https://picovoice.ai and add it to backend/.env'
            )

        self._access_key = access_key
        self._model_path = model_path
        self._builtin_keyword = builtin_keyword
        self._threshold = threshold
        self._running = False
        self._thread: threading.Thread | None = None
        self._callback = None

        # Validate pvporcupine is installed
        try:
            import pvporcupine  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'pvporcupine is not installed. Run: pip install pvporcupine pvrecorder'
            ) from exc

        # Validate pvrecorder is installed
        try:
            import pvrecorder  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'pvrecorder is not installed. Run: pip install pvrecorder'
            ) from exc

        log.info('[PorcupineWake] Initialised — keyword=%s  threshold=%.2f',
                 model_path or builtin_keyword, threshold)
        print(f'[PorcupineWake] Keyword: {model_path or builtin_keyword!r}  '
              f'threshold={threshold}')

    # ------------------------------------------------------------------ #
    # Public API (matches WakeDetector interface)
    # ------------------------------------------------------------------ #

    def start(self, callback=None) -> None:
        """Start detection on a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(
            target=self.run,
            name='wake-detector-porcupine',
            daemon=True,
        )
        self._thread.start()

    def run(self) -> None:
        """Blocking loop: listen → detect → call callback → repeat."""
        import pvporcupine
        from pvrecorder import PvRecorder

        # Build Porcupine instance
        if self._model_path and Path(self._model_path).exists():
            porcupine = pvporcupine.create(
                access_key=self._access_key,
                keyword_paths=[self._model_path],
                sensitivities=[self._threshold],
            )
            keyword_label = Path(self._model_path).stem
        else:
            if self._model_path:
                log.warning('[PorcupineWake] Model path %r not found — using built-in %r',
                            self._model_path, self._builtin_keyword)
                print(f'[PorcupineWake] Model not found, using built-in "{self._builtin_keyword}"')
            porcupine = pvporcupine.create(
                access_key=self._access_key,
                keywords=[self._builtin_keyword],
                sensitivities=[self._threshold],
            )
            keyword_label = self._builtin_keyword

        recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)

        log.info('[PorcupineWake] Listening for %r', keyword_label)
        print(f'[PorcupineWake] Listening for "{keyword_label}" ...')

        recorder.start()
        self._running = True

        try:
            while self._running:
                pcm = recorder.read()
                result = porcupine.process(pcm)
                if result >= 0:
                    log.info('[PorcupineWake] Wake word detected!')
                    print('[PorcupineWake] Wake word detected!')
                    if self._callback is not None:
                        recorder.stop()
                        try:
                            self._callback()
                        except Exception as exc:
                            log.exception('[PorcupineWake] Callback failed: %s', exc)
                        if not self._running:
                            break
                        recorder.start()
                        log.info('[PorcupineWake] Listening for %r', keyword_label)
        except KeyboardInterrupt:
            log.info('[PorcupineWake] Stopped by user')
        finally:
            self._running = False
            try:
                recorder.delete()
            except Exception:
                pass
            try:
                porcupine.delete()
            except Exception:
                pass

    def stop(self) -> None:
        """Signal the run loop to exit and join the background thread."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None


def create_wake_detector(prefer_porcupine: bool = True):
    """Factory: return a WakeDetectorPorcupine if Porcupine is available and
    configured, otherwise fall back to the openwakeword WakeDetector.

    This is the recommended way to construct the wake detector — it
    automatically selects the best available engine::

        detector = create_wake_detector()
        runtime = SaraRuntime(wake_detector=detector)

    Porcupine is preferred when:
    • ``pvporcupine`` and ``pvrecorder`` are installed
    • ``PICOVOICE_ACCESS_KEY`` is set in the environment / .env

    Otherwise openwakeword (hey_jarvis demo model) is used.
    """
    if prefer_porcupine and PICOVOICE_ACCESS_KEY:
        try:
            import pvporcupine  # noqa: F401
            import pvrecorder   # noqa: F401
            detector = WakeDetectorPorcupine()
            log.info('[WakeFactory] Using Porcupine wake word engine')
            return detector
        except (ImportError, Exception) as exc:
            log.warning('[WakeFactory] Porcupine unavailable (%s) — falling back to openwakeword', exc)

    from wakeword.wake_detector import WakeDetector
    log.info('[WakeFactory] Using openwakeword engine (hey_jarvis demo)')
    return WakeDetector()
