"""Text-to-Speech engine for Sara.

Uses **pyttsx3** for offline TTS.  All speech runs on a dedicated daemon
thread so that API responses are never blocked.

Per-utterance engine lifecycle
-------------------------------
Each queued text item gets its own ``pyttsx3.init()`` / ``runAndWait()`` /
``stop()`` cycle.  This prevents the Windows SAPI5 driver from entering a
silently-inaudible state after the first ``runAndWait()`` event loop
completes — a known issue where subsequent calls log [TTS] finished but
produce no audio output.

Configuration (environment / .env)::

    ENABLE_TTS=true   – master switch (default: false)
    TTS_RATE=170      – words-per-minute (default: 170)
    TTS_VOLUME=0.9    – 0.0 … 1.0 (default: 0.9)

Usage::

    from tts_engine import safe_speak
    safe_speak("Sales report created successfully.")   # non-blocking, self-healing
"""
from __future__ import annotations

import logging
import os
import queue
import threading

log = logging.getLogger('tts_engine')

# --------------- Configuration ---------------

ENABLE_TTS: bool = os.environ.get('ENABLE_TTS', 'false').lower() in ('true', '1', 'yes')
TTS_RATE: int = int(os.environ.get('TTS_RATE', '170'))
TTS_VOLUME: float = float(os.environ.get('TTS_VOLUME', '0.9'))

# --------------- Internal state ---------------

_engine = None          # current pyttsx3 engine (None between utterances)
_queue: queue.Queue[str | None] = queue.Queue()
_worker: threading.Thread | None = None
_started = False


def initialize_engine() -> None:
    """Start the background TTS worker thread (idempotent).

    Called once at app startup.  If ``ENABLE_TTS`` is False the call is a
    no-op so the rest of the application is unaffected.
    """
    global _started, _worker

    if not ENABLE_TTS:
        log.info('initialize_engine: TTS disabled (ENABLE_TTS=%s)', os.environ.get('ENABLE_TTS'))
        return

    if _started:
        return

    _worker = threading.Thread(target=_worker_loop, name='tts-worker', daemon=True)
    _worker.start()
    _started = True
    log.info('initialize_engine: TTS worker started (rate=%d, volume=%.1f)', TTS_RATE, TTS_VOLUME)


def _restart_worker() -> None:
    """Replace the worker thread and its queue with fresh instances.

    Called by ``safe_speak()`` when the existing worker is found dead (e.g.
    after a pyttsx3 init or COM failure).  A new ``queue.Queue`` is created
    so that ``unfinished_tasks`` is reset to zero — the old queue may have a
    non-zero counter from items fetched by the dead worker that never reached
    ``task_done()``, which would cause ``_queue.join()`` to block forever.

    This function must only be called when TTS is enabled.
    """
    global _worker, _started, _queue
    log.warning('[TTS] worker not alive, restarting')
    _queue = queue.Queue()
    _worker = threading.Thread(target=_worker_loop, name='tts-worker', daemon=True)
    _worker.start()
    _started = True
    log.info('[TTS] worker restarted')


def speak(text: str) -> None:
    """Enqueue *text* for speech.  Returns immediately (non-blocking).

    If TTS is disabled or *text* is empty the call is silently ignored.
    Prefer ``safe_speak()`` for assistant responses — it adds worker-death
    recovery and full lifecycle logging.
    """
    if not ENABLE_TTS or not text:
        return
    # Lazily start the worker if caller forgot initialize_engine()
    if not _started:
        initialize_engine()
    _queue.put(text)
    log.debug('speak: enqueued %r', text[:80])


def safe_speak(text: str) -> None:
    """Speak *text* with worker-death recovery and full lifecycle logging.

    Unlike ``speak()``, this function:
    - Logs the request at INFO level so it is always visible in production logs
    - Warns explicitly when *text* is empty (not silent)
    - Detects a dead worker thread and restarts it once before enqueueing
    - Never raises an exception — TTS failures must not crash the assistant

    The companion ``_worker_loop`` logs ``[TTS] started`` / ``[TTS] finished``
    around each ``pyttsx3.runAndWait()`` call so the full lifecycle is visible.
    """
    if not ENABLE_TTS:
        return
    if not text:
        log.warning('[TTS] speak requested but text is empty — skipping')
        return

    log.info('[TTS] speak requested: %r', text[:80])

    # Lazily start the worker if caller forgot initialize_engine()
    if not _started:
        initialize_engine()

    # Detect dead worker and restart once before enqueueing.
    # A fresh queue is created so unfinished_tasks resets to zero.
    if _worker is not None and not _worker.is_alive():
        _restart_worker()

    try:
        _queue.put(text)
        log.info('[TTS] queued: %r', text[:60])
    except Exception as exc:
        log.error('[TTS] failed to enqueue: %s', exc)


def wait_until_done() -> None:
    """Block until all queued speech has been spoken.

    Returns immediately when TTS is disabled or the worker is not running.
    """
    if not ENABLE_TTS or not _started:
        return
    # Guard: if the worker thread died (e.g. pyttsx3 init failed), joining the
    # queue would block forever because task_done() is never called for items
    # that were enqueued after the worker exited.  Check liveness first.
    if _worker is not None and not _worker.is_alive():
        log.warning('wait_until_done: TTS worker is not alive — skipping queue join')
        return
    _queue.join()


def is_busy() -> bool:
    """Return True if TTS is enabled, the worker is alive, and the queue
    still has unprocessed items.

    Use this to avoid stacking a "thinking" hint on top of an already-queued
    response — callers can skip the hint when TTS is already active.
    """
    if not ENABLE_TTS or not _started:
        return False
    if _worker is not None and not _worker.is_alive():
        return False
    return _queue.unfinished_tasks > 0


def shutdown() -> None:
    """Signal the worker to exit and wait for it to drain."""
    if not _started or _worker is None:
        return
    _queue.put(None)          # sentinel
    _worker.join(timeout=5)
    log.info('shutdown: TTS worker stopped')


# --------------- Worker thread ---------------

def _worker_loop() -> None:
    """Run on the daemon thread.  Processes the speech queue until a ``None``
    sentinel is received.

    Each utterance gets its own pyttsx3 engine instance — created fresh,
    used, stopped, and released.  This prevents the Windows SAPI5 driver from
    entering a silently-inaudible state that can occur when a long-lived
    ISpVoice COM object is reused across multiple ``runAndWait()`` calls.

    Lifecycle logs (INFO level so they appear in production):
    - ``[TTS] engine created``  — fresh pyttsx3 instance ready
    - ``[TTS] started``         — just before pyttsx3 speaks
    - ``[TTS] finished``        — immediately after pyttsx3 returns
    - ``[TTS] engine released`` — COM object freed
    - ``[TTS] failed``          — exception during speech
    - ``[TTS] restarting engine`` — recovering from per-utterance error
    """
    global _engine
    import pyttsx3

    # Probe once at worker startup to fail-fast when pyttsx3 is completely
    # unavailable (e.g. no audio driver, missing SAPI5).  This keeps the
    # existing behaviour for test_worker_handles_pyttsx3_init_failure.
    try:
        _probe = pyttsx3.init()
        _probe.stop()
        del _probe
        log.info('_worker_loop: pyttsx3 available — per-utterance engine lifecycle active')
    except Exception as exc:
        log.error('_worker_loop: pyttsx3 unavailable: %s', exc)
        # Drain queue so wait_until_done() doesn't hang on _queue.join()
        while True:
            try:
                _queue.get_nowait()
                _queue.task_done()
            except Exception:
                break
        return

    while True:
        try:
            text = _queue.get()
            if text is None:
                _queue.task_done()
                break                       # shutdown sentinel

            # Fresh engine per utterance — avoids stale SAPI5 COM state
            log.info('[TTS] engine created')
            _engine = pyttsx3.init()
            _engine.setProperty('rate', TTS_RATE)
            _engine.setProperty('volume', max(0.0, min(1.0, TTS_VOLUME)))

            log.info('[TTS] started: %r', text[:60])
            _engine.say(text)
            _engine.runAndWait()
            log.info('[TTS] finished: %r', text[:60])

            try:
                _engine.stop()
            except Exception:
                pass
            _engine = None
            log.info('[TTS] engine released')

            _queue.task_done()

        except Exception as exc:
            log.error('[TTS] failed: %s', exc)
            try:
                _queue.task_done()
            except Exception:
                pass
            # Release any partially-initialised engine
            try:
                if _engine is not None:
                    _engine.stop()
            except Exception:
                pass
            _engine = None
            log.info('[TTS] restarting engine')
