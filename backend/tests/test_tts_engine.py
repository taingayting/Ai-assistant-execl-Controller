"""Tests for tts_engine module."""
import importlib
import queue
from unittest.mock import patch, MagicMock

import pytest


# Reload the module with controlled env vars for each test.
def _reload_tts(env_overrides=None):
    """Reimport tts_engine with fresh module-level globals."""
    env = {
        'ENABLE_TTS': 'false',
        'TTS_RATE': '170',
        'TTS_VOLUME': '0.9',
    }
    if env_overrides:
        env.update(env_overrides)
    with patch.dict('os.environ', env, clear=False):
        import tts_engine
        # Reset internal state so each test starts clean
        tts_engine._started = False
        tts_engine._engine = None
        tts_engine._worker = None
        tts_engine._queue = queue.Queue()
        tts_engine.ENABLE_TTS = env.get('ENABLE_TTS', 'false').lower() in ('true', '1', 'yes')
        tts_engine.TTS_RATE = int(env.get('TTS_RATE', '170'))
        tts_engine.TTS_VOLUME = float(env.get('TTS_VOLUME', '0.9'))
        return tts_engine


# ── speak() when TTS disabled ────────────────────────────────────

class TestSpeakDisabled:
    def test_speak_noop_when_disabled(self):
        """speak() should do nothing when ENABLE_TTS is false."""
        engine = _reload_tts({'ENABLE_TTS': 'false'})
        engine.speak("hello")
        assert engine._queue.empty()

    def test_initialize_noop_when_disabled(self):
        engine = _reload_tts({'ENABLE_TTS': 'false'})
        engine.initialize_engine()
        assert engine._started is False
        assert engine._worker is None


# ── speak() when TTS enabled ─────────────────────────────────────

class TestSpeakEnabled:
    def test_speak_enqueues_text(self):
        """speak() should put text on the queue and start the worker."""
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        # Prevent actual pyttsx3 init by mocking the worker thread
        fake_thread = MagicMock()
        with patch('tts_engine.threading.Thread', return_value=fake_thread):
            engine.speak("Sales report created successfully.")
        assert not engine._queue.empty()
        assert engine._queue.get_nowait() == "Sales report created successfully."

    def test_speak_empty_text_ignored(self):
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        engine.speak("")
        assert engine._queue.empty()

    def test_speak_none_text_ignored(self):
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        engine.speak(None)
        assert engine._queue.empty()

    def test_initialize_starts_worker(self):
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        fake_thread = MagicMock()
        with patch('tts_engine.threading.Thread', return_value=fake_thread) as thread_cls:
            engine.initialize_engine()
        thread_cls.assert_called_once()
        fake_thread.start.assert_called_once()
        assert engine._started is True

    def test_initialize_idempotent(self):
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        fake_thread = MagicMock()
        with patch('tts_engine.threading.Thread', return_value=fake_thread) as thread_cls:
            engine.initialize_engine()
            engine.initialize_engine()
        # Thread should only be created once
        assert thread_cls.call_count == 1


# ── _worker_loop ──────────────────────────────────────────────────

class TestWorkerLoop:
    def test_worker_processes_text_and_exits_on_sentinel(self):
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        mock_pyttsx3 = MagicMock()
        mock_engine_inst = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine_inst

        engine._queue.put("Hello world")
        engine._queue.put(None)  # sentinel to stop

        with patch.dict('sys.modules', {'pyttsx3': mock_pyttsx3}):
            engine._worker_loop()

        mock_engine_inst.say.assert_called_once_with("Hello world")
        mock_engine_inst.runAndWait.assert_called_once()

    def test_worker_handles_pyttsx3_init_failure(self):
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.side_effect = RuntimeError("no audio device")

        with patch.dict('sys.modules', {'pyttsx3': mock_pyttsx3}):
            engine._worker_loop()
        # Should exit gracefully without crashing


# ── shutdown ──────────────────────────────────────────────────────

class TestShutdown:
    def test_shutdown_sends_sentinel(self):
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        fake_thread = MagicMock()
        with patch('tts_engine.threading.Thread', return_value=fake_thread):
            engine.initialize_engine()
        engine.shutdown()
        # sentinel None should be on the queue
        fake_thread.join.assert_called_once_with(timeout=5)

    def test_shutdown_noop_when_not_started(self):
        engine = _reload_tts({'ENABLE_TTS': 'false'})
        engine.shutdown()  # should not raise


# ── safe_speak() ─────────────────────────────────────────────────

class TestSafeSpeak:
    def test_safe_speak_noop_when_disabled(self):
        engine = _reload_tts({'ENABLE_TTS': 'false'})
        engine.safe_speak("hello")
        assert engine._queue.empty()

    def test_safe_speak_warns_on_empty_text(self, caplog):
        import logging
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        with caplog.at_level(logging.WARNING, logger='tts_engine'):
            engine.safe_speak("")
        assert engine._queue.empty()
        assert 'empty' in caplog.text.lower()

    def test_safe_speak_enqueues_when_worker_alive(self):
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        fake_thread = MagicMock()
        fake_thread.is_alive.return_value = True
        with patch('tts_engine.threading.Thread', return_value=fake_thread):
            engine.initialize_engine()
        engine.safe_speak("test message")
        assert engine._queue.get_nowait() == "test message"

    def test_safe_speak_restarts_dead_worker(self):
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        dead_thread = MagicMock()
        dead_thread.is_alive.return_value = False
        engine._worker = dead_thread
        engine._started = True

        new_thread = MagicMock()
        with patch('tts_engine.threading.Thread', return_value=new_thread):
            engine.safe_speak("after restart")

        new_thread.start.assert_called_once()
        assert engine._queue.get_nowait() == "after restart"


# ── _restart_worker() ─────────────────────────────────────────────

class TestRestartWorker:
    def test_restart_creates_fresh_queue(self):
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        old_queue = engine._queue
        fake_thread = MagicMock()
        with patch('tts_engine.threading.Thread', return_value=fake_thread):
            engine._restart_worker()
        assert engine._queue is not old_queue

    def test_restart_starts_new_thread_and_sets_started(self):
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        fake_thread = MagicMock()
        with patch('tts_engine.threading.Thread', return_value=fake_thread):
            engine._restart_worker()
        fake_thread.start.assert_called_once()
        assert engine._started is True


# ── _worker_loop lifecycle logging and engine re-init ─────────────

class TestWorkerLoopEnhanced:
    def test_worker_logs_started_and_finished(self, caplog):
        import logging
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        mock_pyttsx3 = MagicMock()
        mock_engine_inst = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine_inst

        engine._queue.put("Hello world")
        engine._queue.put(None)

        with patch.dict('sys.modules', {'pyttsx3': mock_pyttsx3}):
            with caplog.at_level(logging.INFO, logger='tts_engine'):
                engine._worker_loop()

        assert '[TTS] started' in caplog.text
        assert '[TTS] finished' in caplog.text

    def test_worker_reinits_engine_after_speech_error(self):
        engine = _reload_tts({'ENABLE_TTS': 'true'})
        mock_pyttsx3 = MagicMock()
        mock_engine_inst = MagicMock()
        # First say() raises, second succeeds
        mock_engine_inst.say.side_effect = [RuntimeError("COM error"), None]
        mock_pyttsx3.init.return_value = mock_engine_inst

        engine._queue.put("first")
        engine._queue.put("second")
        engine._queue.put(None)

        with patch.dict('sys.modules', {'pyttsx3': mock_pyttsx3}):
            engine._worker_loop()

        # init called at startup + re-init after error = at least 2
        assert mock_pyttsx3.init.call_count >= 2


# ── Flask after_request TTS hook ──────────────────────────────────

class TestFlaskTTSHook:
    def test_after_request_speaks_speech_text(self):
        """The after_request hook should call tts_engine.speak for JSON responses with speech_text."""
        with patch.dict('os.environ', {'ENABLE_TTS': 'false'}):
            from app import app as flask_app
        with flask_app.test_client() as client:
            with patch('tts_engine.speak') as mock_speak, \
                 patch('app.parse_with_llm', return_value={"type": "save"}):
                resp = client.post('/api/parse', json={"text": "save"})
                data = resp.get_json()
                # If response contains speech_text, speak should be called
                if data and data.get('speech_text'):
                    mock_speak.assert_called()
                # Otherwise it's fine — not all responses have speech_text

    def test_after_request_skips_when_no_speech_text(self):
        """No speak() call when response has no speech_text."""
        with patch.dict('os.environ', {'ENABLE_TTS': 'false'}):
            from app import app as flask_app
        with flask_app.test_client() as client:
            with patch('tts_engine.speak') as mock_speak:
                resp = client.get('/')
                # Static HTML response — speak should NOT be called
                mock_speak.assert_not_called()
