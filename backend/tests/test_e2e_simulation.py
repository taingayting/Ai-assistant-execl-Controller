"""End-to-end simulation of the Sara assistant pipeline.

Mocks hardware (microphone, speakers, wake detector) and feeds a scripted
sequence of voice commands through the full runtime pipeline to verify all
subsystems are wired correctly.

Run:
    python tests/test_e2e_simulation.py
"""
import sys
import os
import logging
import threading
import time

# Ensure backend/ is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force heuristic fallback so test doesn't wait for LLM APIs
os.environ['GROQ_API_KEY'] = ''
os.environ['USE_OLLAMA'] = 'false'

# When running under pytest, llm_router is already imported by conftest → app.
# The module-level USE_OLLAMA constant is already set to True from that import,
# so patching the env var alone is not enough.  Force the cached constants and
# spoof offline mode so Groq is skipped entirely — no real HTTP calls in the test.
try:
    import llm_router as _llm_router
    _llm_router.USE_OLLAMA = False
    _llm_router.GROQ_MAX_RETRIES = 0
    # Set the connectivity cache to "offline" so route_llm() skips Groq.
    _llm_router._conn_cache['result'] = False
    _llm_router._conn_cache['ts'] = float('inf')
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-5s %(name)s %(message)s',
)

# ---- Mock hardware dependencies ----

COMMANDS = [
    'create sales sheet',
    'add row John 100',
    'add row Mary 200',
    'thank you',
]
_cmd_idx = 0

import speech_recognition as sr  # noqa: E402

def mock_listen_once(language='en', seconds=5.0):
    global _cmd_idx
    if _cmd_idx >= len(COMMANDS):
        return 'goodbye'
    cmd = COMMANDS[_cmd_idx]
    _cmd_idx += 1
    print(f'  [MOCK-MIC] User said: "{cmd}"')
    return cmd

sr.listen_once = mock_listen_once

import tts_engine  # noqa: E402

tts_engine.ENABLE_TTS = False
_spoken: list[str] = []

def mock_speak(text):
    _spoken.append(text)
    print(f'  [MOCK-TTS] Sara says: "{text}"')

tts_engine.speak = mock_speak
tts_engine.wait_until_done = lambda: None
tts_engine.initialize_engine = lambda: None


class MockWakeDetector:
    """Fire callback once per command (single-turn loop: one wake → one command)."""

    def __init__(self):
        self._callback = None
        self._running = False

    def start(self, callback=None):
        self._callback = callback
        self._running = True
        t = threading.Thread(target=self._fire_loop, daemon=True)
        t.start()

    def _fire_loop(self):
        """Fire the callback once for each command, with a small gap between."""
        for _ in range(len(COMMANDS)):
            if not self._running:
                break
            time.sleep(0.3)
            if self._callback and self._running:
                self._callback()
                # Wait a moment for the command cycle to complete before
                # firing the next wake event.
                time.sleep(0.5)

    def stop(self):
        self._running = False


# ---- Run ----

from assistant_runtime import SaraRuntime  # noqa: E402

print()
print('=' * 60)
print('  SARA END-TO-END SIMULATION TEST')
print('=' * 60)
print()

runtime = SaraRuntime(
    ui=None,
    wake_detector=MockWakeDetector(),
)


def _run():
    runtime.start()


t = threading.Thread(target=_run, daemon=True)
t.start()
t.join(timeout=30)
runtime.stop()

print()
print('=' * 60)
print('  RESULTS')
print('=' * 60)
print(f'  Commands consumed: {_cmd_idx}')
print(f'  TTS responses:     {len(_spoken)}')
for i, s in enumerate(_spoken):
    print(f'    {i + 1}. {s[:120]}')
print()

assert _cmd_idx == len(COMMANDS), (
    f'Expected all {len(COMMANDS)} commands consumed, got {_cmd_idx}'
)
# greeting + one response per command + farewell
assert len(_spoken) >= len(COMMANDS), (
    f'Expected at least {len(COMMANDS)} TTS responses, got {len(_spoken)}'
)
# None of the TTS responses should be the raw "steps_executed" sentinel
for resp in _spoken:
    assert resp != 'steps_executed', (
        f'Raw "steps_executed" should not be spoken — got "{resp}"'
    )
# The non-greeting, non-farewell responses should not be "I could not understand"
non_meta = [r for r in _spoken if r not in ("I'm listening.", "Goodbye! Let me know when you need me again.")]
for resp in non_meta:
    assert resp != 'I could not understand that command.', (
        f'Heuristic fallback failed — got "I could not understand"'
    )
print('  ALL ASSERTIONS PASSED')
print('=' * 60)
