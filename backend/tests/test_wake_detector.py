"""Tests for wakeword.wake_detector (all hardware/network mocked)."""

import struct
import types
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import requests

# ---------------------------------------------------------------------------
# We need to mock heavy imports (openwakeword, pyaudio, numpy) so the tests
# can run in CI without those packages installed.
# ---------------------------------------------------------------------------

# Build a minimal numpy stub that supports frombuffer and zeros
_np_stub = MagicMock()
_np_stub.int16 = "int16"
_np_stub.bool_ = bool  # pytest.approx needs a real type for isinstance()
_np_stub.frombuffer = MagicMock(side_effect=lambda buf, dtype: buf)
_np_stub.zeros = MagicMock(return_value=b"\x00" * 2560)

# Build an openwakeword stub — __file__ must point somewhere real so that
# _resolve_model_path can find the resources/models directory.
import pathlib, tempfile as _tempfile

_oww_model_cls = MagicMock()
_oww_utils = MagicMock()
_oww = types.ModuleType("openwakeword")
_oww.utils = _oww_utils
# Give the stub module a __file__ inside a tmp dir so path logic works
_oww_tmp_pkg = pathlib.Path(_tempfile.gettempdir()) / "_oww_stub"
(_oww_tmp_pkg / "resources" / "models").mkdir(parents=True, exist_ok=True)
# Create a fake hey_jarvis onnx file so _resolve_model_path finds it
(_oww_tmp_pkg / "resources" / "models" / "hey_jarvis_v0.1.onnx").write_bytes(b"fake")
_oww.__file__ = str(_oww_tmp_pkg / "__init__.py")

_oww_model_mod = types.ModuleType("openwakeword.model")
_oww_model_mod.Model = _oww_model_cls

# Build a pyaudio stub
_pyaudio_stub = types.ModuleType("pyaudio")
_pyaudio_stub.PyAudio = MagicMock
_pyaudio_stub.paInt16 = 8  # same value as real constant

import sys

sys.modules.setdefault("numpy", _np_stub)
sys.modules.setdefault("openwakeword", _oww)
sys.modules.setdefault("openwakeword.model", _oww_model_mod)
sys.modules.setdefault("openwakeword.utils", _oww_utils)
sys.modules.setdefault("pyaudio", _pyaudio_stub)

# Now we can import the module under test
from wakeword.wake_detector import (
    _rms,
    _frames_to_wav,
    WakeDetector,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    CHANNELS,
)


# ---------------------------------------------------------------------------
# Helper: build raw PCM frames with known energy
# ---------------------------------------------------------------------------

def _silent_frame() -> bytes:
    """Return a single 80 ms frame of silence (zeros)."""
    return b"\x00" * (FRAME_SAMPLES * SAMPLE_WIDTH)


def _loud_frame(amplitude: int = 5000) -> bytes:
    """Return a single 80 ms frame of constant-amplitude PCM."""
    return struct.pack(f"<{FRAME_SAMPLES}h", *([amplitude] * FRAME_SAMPLES))


# ========================== Unit tests ====================================


class TestRms:
    def test_silent(self):
        assert _rms(_silent_frame()) == 0.0

    def test_loud(self):
        rms = _rms(_loud_frame(5000))
        assert rms == pytest.approx(5000.0, abs=1)

    def test_empty(self):
        assert _rms(b"") == 0.0


class TestFramesToWav:
    def test_produces_valid_wav(self):
        import wave, io

        frames = [_silent_frame(), _loud_frame()]
        wav = _frames_to_wav(frames)
        with wave.open(io.BytesIO(wav), "rb") as wf:
            assert wf.getnchannels() == CHANNELS
            assert wf.getsampwidth() == SAMPLE_WIDTH
            assert wf.getframerate() == SAMPLE_RATE
            total_samples = wf.getnframes()
            assert total_samples == FRAME_SAMPLES * 2


class TestWakeDetectorInit:
    """Test WakeDetector.__init__ with mocked openwakeword."""

    def _make_model_mock(self):
        """Return a mock Model whose prediction_buffer is populated after
        the first predict() call (mimicking the prime-frame behaviour)."""
        m = MagicMock()
        # Start empty; predict() populates it (prime call in __init__)
        m.prediction_buffer = {}

        def _predict(frame):
            m.prediction_buffer["hey_jarvis_v0.1"] = []
            return {"hey_jarvis_v0.1": 0.0}

        m.predict.side_effect = _predict
        return m

    def test_default_init_uses_bundled_model(self):
        """Default init resolves to hey_jarvis_v0.1.onnx without downloading."""
        _oww_utils.download_models.reset_mock()
        _oww_model_cls.reset_mock()
        _oww_model_cls.return_value = self._make_model_mock()

        det = WakeDetector(model_path=None)

        # download_models should NOT be called because the .onnx file already
        # exists in the stub package (created in module setup above).
        _oww_utils.download_models.assert_not_called()
        # Model must be called with a path string and onnx framework
        call_kwargs = _oww_model_cls.call_args
        assert call_kwargs.kwargs.get('inference_framework') == 'onnx'
        ww = call_kwargs.kwargs.get('wakeword_models') or call_kwargs.args[0]
        assert len(ww) == 1
        assert ww[0].endswith('hey_jarvis_v0.1.onnx')
        assert det._target_label == "hey_jarvis_v0.1"
        assert det.threshold == 0.5

    def test_custom_model_path(self, tmp_path):
        fake_model = tmp_path / "hey_sara.onnx"
        fake_model.write_bytes(b"fake")

        _oww_model_cls.reset_mock()
        _oww_model_cls.return_value = self._make_model_mock()

        det = WakeDetector(model_path=str(fake_model))
        call_kwargs = _oww_model_cls.call_args
        ww = call_kwargs.kwargs.get('wakeword_models') or call_kwargs.args[0]
        assert ww[0] == str(fake_model.resolve())
        assert call_kwargs.kwargs.get('inference_framework') == 'onnx'

    def test_custom_model_missing_falls_back_to_bundled(self):
        """Non-existent WAKE_MODEL_PATH falls back to bundled hey_jarvis."""
        _oww_utils.download_models.reset_mock()
        _oww_model_cls.reset_mock()
        _oww_model_cls.return_value = self._make_model_mock()

        det = WakeDetector(model_path="/nonexistent/model.onnx")

        # download_models must NOT be called — bundled file exists in stub
        _oww_utils.download_models.assert_not_called()
        call_kwargs = _oww_model_cls.call_args
        ww = call_kwargs.kwargs.get('wakeword_models') or call_kwargs.args[0]
        assert ww[0].endswith('hey_jarvis_v0.1.onnx')

    def test_no_labels_raises(self):
        m = MagicMock()
        m.prediction_buffer = {}
        m.predict.return_value = {}  # prime call yields nothing
        _oww_model_cls.return_value = m

        with pytest.raises(RuntimeError, match="No wake-word labels found"):
            WakeDetector(model_path=None)

    def test_custom_threshold(self):
        _oww_model_cls.return_value = self._make_model_mock()
        det = WakeDetector(model_path=None, threshold=0.8)
        assert det.threshold == 0.8


def _make_primed_model_mock(label="hey_jarvis_v0.1"):
    """Return a model mock that populates prediction_buffer on first predict()."""
    m = MagicMock()
    m.prediction_buffer = {}

    def _predict(frame):
        m.prediction_buffer[label] = []
        return {label: 0.0}

    m.predict.side_effect = _predict
    return m


class TestRecordCommand:
    """Test the _record_command helper (stream is mocked)."""

    def _make_detector(self):
        _oww_model_cls.return_value = _make_primed_model_mock()
        return WakeDetector(
            model_path=None,
            record_seconds=0.5,
            silence_timeout=0.2,
            silence_energy=100,
        )

    def test_records_until_timeout(self):
        det = self._make_detector()
        stream = MagicMock()
        stream.read.return_value = _loud_frame(5000)

        frames = det._record_command(stream)
        max_frames = int(det.record_seconds * SAMPLE_RATE / FRAME_SAMPLES)
        assert len(frames) == max_frames

    def test_stops_on_silence(self):
        det = self._make_detector()
        stream = MagicMock()
        # Return one loud frame then silence
        stream.read.side_effect = [_loud_frame(5000)] + [_silent_frame()] * 100

        frames = det._record_command(stream)
        # Should stop early – fewer frames than max
        max_frames = int(det.record_seconds * SAMPLE_RATE / FRAME_SAMPLES)
        assert len(frames) < max_frames


class TestSendAudio:
    """Test _send_audio with mocked requests."""

    def _make_detector(self):
        _oww_model_cls.return_value = _make_primed_model_mock()
        return WakeDetector(model_path=None)

    @patch("wakeword.wake_detector.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "ok"}),
        )
        mock_post.return_value.raise_for_status = MagicMock()

        det = self._make_detector()
        result = det._send_audio(b"RIFF fake wav")

        assert result == {"status": "ok"}
        mock_post.assert_called_once()

    @patch("wakeword.wake_detector.requests.post", side_effect=requests.RequestException("timeout"))
    def test_failure_returns_none(self, mock_post):
        det = self._make_detector()
        result = det._send_audio(b"RIFF fake wav")
        assert result is None


class TestRunLoop:
    """Test the main run() loop with mocked PyAudio + model."""

    def _make_detector(self):
        model = _make_primed_model_mock()
        model.reset = MagicMock()
        _oww_model_cls.return_value = model
        det = WakeDetector(model_path=None, record_seconds=0.2, silence_timeout=0.1)
        # After init, reset predict so the run loop can use its own side_effect
        det.model.predict.reset_mock()
        det.model.predict.return_value = {"hey_jarvis_v0.1": 0.0}
        det.model.predict.side_effect = None
        return det

    @patch("wakeword.wake_detector.requests.post")
    def test_detection_triggers_record_and_send(self, mock_post):
        det = self._make_detector()

        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "ok"}),
        )
        mock_post.return_value.raise_for_status = MagicMock()

        # First call: high score (triggers recording).
        # All subsequent: low score, then raise KeyboardInterrupt to exit.
        call_count = 0
        def predict_side_effect(frame):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"hey_jarvis_v0.1": 0.9}
            if call_count > 20:
                raise KeyboardInterrupt
            return {"hey_jarvis_v0.1": 0.0}

        det.model.predict.side_effect = predict_side_effect

        stream_mock = MagicMock()
        stream_mock.read.return_value = _silent_frame()

        pa_instance = MagicMock()
        pa_instance.open.return_value = stream_mock

        with patch("pyaudio.PyAudio", return_value=pa_instance):
            det.run()

        # Should have called reset after detection
        det.model.reset.assert_called()
        # Should have attempted to POST
        mock_post.assert_called()

    def test_stop_flag_exits_loop(self):
        det = self._make_detector()
        det.model.predict.return_value = {"hey_jarvis": 0.0}

        stream_mock = MagicMock()
        stream_mock.read.return_value = _silent_frame()

        pa_instance = MagicMock()
        pa_instance.open.return_value = stream_mock

        # Stop after a few iterations
        call_count = 0
        original_predict = det.model.predict.return_value

        def predict_then_stop(frame):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                det.stop()
            return {"hey_jarvis_v0.1": 0.0}

        det.model.predict.side_effect = predict_then_stop

        with patch("pyaudio.PyAudio", return_value=pa_instance):
            det.run()

        assert det._running is False
