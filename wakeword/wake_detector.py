"""
Wake-word detector for Sara assistant.

Listens to the microphone for a configurable wake word using openwakeword,
records the subsequent voice command, and POSTs the audio to the Flask API.

Model loading strategy (in priority order)
-------------------------------------------
1. WAKE_MODEL_PATH env var  — full path to any .onnx file you supply
                              (use this once you train a "Hey Sara" model)
2. hey_jarvis_v0.1.onnx     — the pre-trained model bundled with openwakeword,
                              resolved via the package's resources/models dir
                              so it works without any extra download step
3. Full download fallback   — if neither of the above exists, download all
                              official models and use hey_jarvis

The IMPORTANT difference vs. the previous version: openwakeword's Model()
requires a full file path, not a bare model name string, when using the onnx
inference framework.  Passing a name-only string leaves prediction_buffer
empty, meaning the detector never fires.  This version always resolves to an
absolute .onnx path before constructing the Model.

Environment variables
---------------------
WAKE_MODEL_PATH  : full path to a custom .onnx wake-word model
                   (e.g. /path/to/hey_sara.onnx)
WAKE_THRESHOLD   : detection threshold 0.0-1.0  (default: 0.5)
API_URL          : upload-audio endpoint         (default: http://localhost:5000/api/upload-audio)
RECORD_SECONDS   : max seconds to record after wake word (default: 5)
SILENCE_TIMEOUT  : seconds of silence before stopping early (default: 1.5)
SILENCE_ENERGY   : RMS energy threshold for silence (default: 300)
"""

from __future__ import annotations

import io
import logging
import os
import struct
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import requests

log = logging.getLogger('wake_detector')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WAKE_MODEL_PATH: str | None = os.getenv("WAKE_MODEL_PATH")
# Default threshold lowered from 0.5 → 0.42.
# Realtek laptop mics produce lower RMS than studio/external mics, so the
# hey_jarvis model routinely scores 0.35–0.48 for clear utterances on laptop
# hardware.  0.42 is still high enough to avoid random false triggers while
# firing reliably on genuine "hey jarvis" speech.  Override via WAKE_THRESHOLD.
WAKE_THRESHOLD: float = float(os.getenv("WAKE_THRESHOLD", "0.42"))
API_URL: str = os.getenv("API_URL", "http://localhost:5000/api/upload-audio")
RECORD_SECONDS: float = float(os.getenv("RECORD_SECONDS", "5"))
SILENCE_TIMEOUT: float = float(os.getenv("SILENCE_TIMEOUT", "1.5"))
SILENCE_ENERGY: float = float(os.getenv("SILENCE_ENERGY", "300"))

# Audio parameters expected by openwakeword
SAMPLE_RATE = 16_000       # 16 kHz
SAMPLE_WIDTH = 2           # 16-bit (2 bytes)
CHANNELS = 1               # mono
FRAME_MS = 80              # openwakeword expects 80 ms frames
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)   # 1280 samples
CHUNK_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH             # 2560 bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms(frame_bytes: bytes) -> float:
    """Return root-mean-square energy of a 16-bit PCM frame."""
    n = len(frame_bytes) // SAMPLE_WIDTH
    if n == 0:
        return 0.0
    shorts = struct.unpack(f"<{n}h", frame_bytes[: n * SAMPLE_WIDTH])
    return (sum(s * s for s in shorts) / n) ** 0.5


def _frames_to_wav(frames: list[bytes]) -> bytes:
    """Combine raw PCM frames into a WAV byte string."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def _post_audio(wav_bytes: bytes) -> dict | None:
    """POST a WAV buffer to the Sara API and return the JSON response."""
    try:
        resp = requests.post(
            API_URL,
            files={"file": ("command.wav", wav_bytes, "audio/wav")},
            data={"auto_execute": "true", "language": "en-US"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"[WakeDetector] API request failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# WakeDetector
# ---------------------------------------------------------------------------

class WakeDetector:
    """Continuously listens for a wake word and can activate the runtime.

    The detector can run either in the foreground via ``run()`` or on a
    background thread via ``start()``. When a callback is provided, wake-word
    detection triggers that callback instead of posting recorded audio to the
    Flask API directly.
    """

    def __init__(
        self,
        model_path: str | None = WAKE_MODEL_PATH,
        threshold: float = WAKE_THRESHOLD,
        api_url: str = API_URL,
        record_seconds: float = RECORD_SECONDS,
        silence_timeout: float = SILENCE_TIMEOUT,
        silence_energy: float = SILENCE_ENERGY,
    ):
        self.threshold = threshold
        self.api_url = api_url
        self.record_seconds = record_seconds
        self.silence_timeout = silence_timeout
        self.silence_energy = silence_energy
        self._running = False
        self._thread: threading.Thread | None = None
        self._callback = None

        # ---- Load openwakeword model ----
        import openwakeword  # noqa: E402
        from openwakeword.model import Model  # noqa: E402

        resolved_path = self._resolve_model_path(model_path)
        log.info('[WakeWord] Using model: %s', resolved_path)
        print(f"[WakeDetector] Loading model: {resolved_path}")

        # Always pass a full file path and request the onnx framework.
        # Passing a bare name string leaves prediction_buffer empty on
        # Windows/onnxruntime setups — the model never fires.
        self.model = Model(
            wakeword_models=[resolved_path],
            inference_framework='onnx',
        )

        # Prime the model with one silent frame so prediction_buffer
        # is populated immediately (openwakeword is lazy on first call).
        import numpy as np  # noqa: E402
        self.model.predict(np.zeros(FRAME_SAMPLES, dtype=np.int16))

        # Identify which model/label to watch
        self._target_label: str | None = None
        for label in self.model.prediction_buffer.keys():
            self._target_label = label
            break

        if self._target_label is None:
            raise RuntimeError(
                f"No wake-word labels found after loading '{resolved_path}'. "
                "Ensure openwakeword and onnxruntime are correctly installed."
            )
        print(f"[WakeDetector] Listening for '{self._target_label}' "
              f"(threshold={self.threshold})")

        log.info('[WakeWord] Listening for wake word')

    # ------------------------------------------------------------------ #
    # Model path resolution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_model_path(model_path: str | None) -> str:
        """Return an absolute path to a valid .onnx wake-word model file.

        Priority:
          1. ``model_path`` argument (from WAKE_MODEL_PATH env var) if it
             points to an existing file.
          2. The ``hey_jarvis_v0.1.onnx`` file already bundled inside the
             openwakeword package (no download needed if package is installed).
          3. Download all official models first, then return hey_jarvis path.

        Returns the resolved absolute path string.
        """
        import openwakeword  # noqa: E402

        # 1. User-supplied custom model
        if model_path and Path(model_path).exists():
            return str(Path(model_path).resolve())

        if model_path:
            print(
                f"[WakeDetector] WAKE_MODEL_PATH '{model_path}' not found — "
                "falling back to built-in hey_jarvis.",
                file=sys.stderr,
            )

        # Locate the openwakeword resources/models directory
        pkg_models_dir = Path(openwakeword.__file__).parent / "resources" / "models"

        # 2. Built-in hey_jarvis already present on disk
        jarvis_path = pkg_models_dir / "hey_jarvis_v0.1.onnx"
        if jarvis_path.exists():
            return str(jarvis_path)

        # 3. Download official models (idempotent — skips files that exist)
        print("[WakeDetector] Downloading openwakeword models (first run)…")
        openwakeword.utils.download_models()
        if jarvis_path.exists():
            return str(jarvis_path)

        raise FileNotFoundError(
            f"Could not locate hey_jarvis_v0.1.onnx in {pkg_models_dir}. "
            "Try: python -c \"import openwakeword; openwakeword.utils.download_models()\""
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self, callback=None) -> None:
        """Start wake-word detection on a background thread.

        When *callback* is provided, it is called after a wake word is
        detected. This is used by the assistant runtime to activate the full
        voice pipeline.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(
            target=self.run,
            name='wake-detector',
            daemon=True,
        )
        self._thread.start()

    def run(self) -> None:
        """Blocking loop: listen → detect → activate callback/API → repeat.

        The outer while loop auto-restarts the audio stream if the microphone
        device crashes (e.g. USB disconnect, driver hiccup).  The assistant
        never silently dies — it logs the error, waits 2 s, and tries again.
        """
        self._running = True
        while self._running:
            try:
                self._run_audio_loop()
            except KeyboardInterrupt:
                log.info('[WakeWord] Stopped by user')
                break
            except Exception as exc:
                if not self._running:
                    break
                log.exception('[WakeWord] Audio error — restarting in 2 s: %s', exc)
                time.sleep(2.0)

        self._running = False
        log.info('[WakeWord] Run loop exited')

    def _run_audio_loop(self) -> None:
        """Inner audio loop — opens mic, detects wake word, repeats."""
        import pyaudio  # noqa: E402

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SAMPLES,
        )
        log.info('[WakeWord] Microphone stream opened — listening for wake word')

        # Score logging: log periodically at DEBUG level so developers can
        # watch the detector without flooding INFO logs.
        _SCORE_LOG_INTERVAL = 50   # frames (~4 s at 80 ms/frame)
        _score_log_counter = 0
        _DEBUG_SCORE_THRESHOLD = 0.15  # log scores above this at debug level
        # Near-miss tracking: log at INFO once when score crosses 0.30 so the
        # user can see in logs that the wake word is almost reaching the threshold.
        # Resets to False when score drops back below 0.15.
        _NEAR_MISS_THRESHOLD = 0.30
        _near_miss_logged = False

        try:
            while self._running:
                frame = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                prediction = self.model.predict(
                    np.frombuffer(frame, dtype=np.int16)
                )

                score = prediction.get(self._target_label, 0.0)

                # Periodic debug logging — visible with --verbose / DEBUG level
                _score_log_counter += 1
                if score >= _DEBUG_SCORE_THRESHOLD:
                    log.debug('[WakeWord] score=%.3f (threshold=%.2f) label=%s',
                              score, self.threshold, self._target_label)
                elif _score_log_counter >= _SCORE_LOG_INTERVAL:
                    log.debug('[WakeWord] listening... score=%.3f', score)
                    _score_log_counter = 0

                # Near-miss: score meaningful but below threshold — log once
                # per peak so the user can diagnose wake-word reliability issues.
                if score >= _NEAR_MISS_THRESHOLD and not _near_miss_logged:
                    log.info('[WakeWord] near-miss score=%.3f threshold=%.2f — '
                             'say wake word louder/clearer if Sara does not respond',
                             score, self.threshold)
                    _near_miss_logged = True
                elif score < _DEBUG_SCORE_THRESHOLD:
                    _near_miss_logged = False  # reset when score drops to baseline

                if score >= self.threshold:
                    log.info('[WakeWord] Wake word detected (score=%.3f threshold=%.2f)',
                             score, self.threshold)
                    self.model.reset()
                    if self._callback is not None:
                        stream.stop_stream()
                        stream.close()
                        pa.terminate()
                        time.sleep(0.35)  # let Windows release the audio device fully
                        try:
                            self._callback()
                        except Exception as exc:
                            log.exception('[WakeWord] Runtime callback failed: %s', exc)
                        if not self._running:
                            return
                        # Re-open mic for next detection cycle
                        pa = pyaudio.PyAudio()
                        stream = pa.open(
                            format=pyaudio.paInt16,
                            channels=CHANNELS,
                            rate=SAMPLE_RATE,
                            input=True,
                            frames_per_buffer=FRAME_SAMPLES,
                        )
                        log.info('[WakeWord] Listening for wake word')
                        continue

                    recorded = self._record_command(stream)
                    if recorded:
                        wav = _frames_to_wav(recorded)
                        result = self._send_audio(wav)
                        if result:
                            log.info('[WakeWord] API response received')
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            try:
                pa.terminate()
            except Exception:
                pass

    def stop(self) -> None:
        """Signal the run loop to exit and join the background thread."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _record_command(self, stream) -> list[bytes]:
        """Record audio until silence or timeout, return list of raw frames."""
        frames: list[bytes] = []
        max_frames = int(self.record_seconds * SAMPLE_RATE / FRAME_SAMPLES)
        silent_frames = 0
        silent_limit = int(self.silence_timeout * SAMPLE_RATE / FRAME_SAMPLES)

        print("[WakeDetector] Recording command …")
        for _ in range(max_frames):
            data = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
            frames.append(data)

            if _rms(data) < self.silence_energy:
                silent_frames += 1
            else:
                silent_frames = 0

            if silent_frames >= silent_limit:
                print("[WakeDetector] Silence detected, stopping recording.")
                break

        return frames

    def _send_audio(self, wav_bytes: bytes) -> dict | None:
        """Send recorded WAV to the API endpoint."""
        try:
            resp = requests.post(
                self.api_url,
                files={"file": ("command.wav", wav_bytes, "audio/wav")},
                data={"auto_execute": "true", "language": "en-US"},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            print(f"[WakeDetector] API request failed: {exc}", file=sys.stderr)
            return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the wake-word detector from the command line."""
    detector = WakeDetector()
    detector.run()


if __name__ == "__main__":
    main()
