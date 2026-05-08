"""Real microphone input with local transcription.

Wraps the existing ``speech_recognition`` module's ``listen_once()`` and
``transcribe_audio_file()`` functions so callers get a simple two-function API:

    from voice.mic_listener import record_audio, transcribe_audio

    wav_path = record_audio()           # blocks until silence or timeout
    text     = transcribe_audio(wav_path)  # returns transcript string

Transcription backends (tried in order, same as speech_recognition.py):

1. **Google Cloud STT REST** — fast, accurate, requires GOOGLE_API_KEY.
2. **Google Cloud STT SDK**  — requires GOOGLE_APPLICATION_CREDENTIALS.
3. **Groq Whisper**          — cloud fallback, requires GROQ_API_KEY.
4. **faster-whisper (local)**— fully offline, no API key needed.
           Install: pip install faster-whisper
           Model:   set WHISPER_MODEL env var (default: "base.en")

If all cloud backends fail and faster-whisper is installed, transcription
works with zero internet access.

Environment variables (all optional):
    SARA_LANGUAGE        — BCP-47 language tag (default: en-US)
    SARA_RECORD_SECONDS  — max recording duration in seconds (default: 5)
    WHISPER_MODEL        — faster-whisper model size (default: base.en)
                           options: tiny, tiny.en, base, base.en,
                                    small, small.en, medium, large-v3
    WHISPER_DEVICE       — "cpu" or "cuda" (default: cpu)
    WHISPER_COMPUTE_TYPE — "int8", "float16", etc. (default: int8)
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import wave

log = logging.getLogger('voice.mic_listener')

# Resolve paths so this module can be imported from backend/voice/
_VOICE_DIR   = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_VOICE_DIR)
_ROOT_DIR    = os.path.dirname(_BACKEND_DIR)

for _p in (_BACKEND_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-use constants already defined in speech_recognition
from speech_recognition import (
    listen_once as _listen_once,
    transcribe_audio_file as _transcribe_cloud,
    LISTEN_SAMPLE_RATE,
    LISTEN_CHANNELS,
    LISTEN_SAMPLE_WIDTH,
    LISTEN_SECONDS,
)

LANGUAGE     = os.environ.get('SARA_LANGUAGE', 'en-US')
WHISPER_MODEL        = os.environ.get('WHISPER_MODEL', 'base.en')
WHISPER_DEVICE       = os.environ.get('WHISPER_DEVICE', 'cpu')
WHISPER_COMPUTE_TYPE = os.environ.get('WHISPER_COMPUTE_TYPE', 'int8')

# Lazy-loaded to avoid import error when faster-whisper is not installed
_faster_whisper_model = None


def _load_faster_whisper():
    """Load (and cache) the faster-whisper model on first use."""
    global _faster_whisper_model
    if _faster_whisper_model is not None:
        return _faster_whisper_model
    try:
        from faster_whisper import WhisperModel  # type: ignore
        log.info('[mic_listener] Loading faster-whisper model=%s device=%s compute=%s',
                 WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
        _faster_whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        log.info('[mic_listener] faster-whisper model loaded')
        return _faster_whisper_model
    except ImportError:
        log.debug('[mic_listener] faster-whisper not installed — skipping local STT')
        return None
    except Exception as exc:
        log.warning('[mic_listener] faster-whisper failed to load: %s', exc)
        return None


def _transcribe_faster_whisper(wav_path: str, language: str = 'en') -> str | None:
    """Transcribe *wav_path* locally using faster-whisper.

    Returns the transcript string, or None if faster-whisper is unavailable.
    """
    model = _load_faster_whisper()
    if model is None:
        return None
    try:
        # faster-whisper accepts short ISO lang codes ('en', not 'en-US')
        iso_lang = language.split('-')[0] if '-' in language else language
        segments, info = model.transcribe(
            wav_path,
            language=iso_lang,
            beam_size=5,
            vad_filter=True,           # skip silent segments
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text = ' '.join(seg.text.strip() for seg in segments).strip()
        log.info('[mic_listener] faster-whisper: lang=%s result=%r', iso_lang, text[:200])
        return text if text else ''
    except Exception as exc:
        log.warning('[mic_listener] faster-whisper transcription failed: %s', exc)
        return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def record_audio(
    language: str = LANGUAGE,
    seconds: float = LISTEN_SECONDS,
) -> str:
    """Record one utterance from the microphone and return the transcript.

    This is the primary entry point for run_sara.py.  It delegates to the
    already-working ``listen_once()`` in speech_recognition.py which handles:
      - pyaudio stream management
      - silence detection (stops early after ~1.2 s of quiet)
      - WAV encoding
      - cloud STT (Google → Groq fallback)

    Returns the transcript string, or a ``[transcription_error]`` / ``[no_voice]``
    sentinel on failure.

    Parameters
    ----------
    language:
        BCP-47 tag (e.g. "en-US") passed to the STT backend.
    seconds:
        Maximum recording duration before silence detection kicks in.
    """
    log.info('[mic_listener] record_audio: language=%s max=%.1fs', language, seconds)
    result = _listen_once(language=language, seconds=seconds)
    log.info('[mic_listener] record_audio: transcript=%r', result[:120] if result else '')
    return result


def transcribe_audio(
    wav_path: str,
    language: str = LANGUAGE,
) -> str:
    """Transcribe an existing WAV file and return the text.

    Tries the same cloud backends as ``speech_recognition.transcribe_audio_file``
    and then falls back to local faster-whisper if all cloud calls fail.

    Parameters
    ----------
    wav_path:
        Absolute path to a .wav file.
    language:
        Language code (ISO or BCP-47).

    Returns
    -------
    str
        The recognised text, or a ``[transcription_error]`` string on total failure.
    """
    log.info('[mic_listener] transcribe_audio: path=%s language=%s', wav_path, language)

    # 1–3. Cloud backends (Google REST → Google SDK → Groq)
    text = _transcribe_cloud(wav_path, language=language)
    if text is not None and not text.startswith('[transcription_error]'):
        return text

    # 4. Local fallback: faster-whisper (fully offline)
    local = _transcribe_faster_whisper(wav_path, language=language)
    if local is not None:
        return local

    # All backends failed
    log.error('[mic_listener] All transcription backends failed for %s', wav_path)
    return '[transcription_error] All transcription backends failed'


def record_and_transcribe(
    language: str = LANGUAGE,
    seconds: float = LISTEN_SECONDS,
) -> str:
    """Convenience: record one utterance and return the transcript.

    Equivalent to calling ``record_audio()`` — included for callers that
    prefer a single-function interface.
    """
    return record_audio(language=language, seconds=seconds)
