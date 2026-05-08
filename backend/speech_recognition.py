"""Utilities to transcribe audio files.

Active STT pipeline (transcribe_audio_file)::

    WAV input → _normalize_wav (mono 16 kHz 16-bit PCM)
              → faster-whisper  (primary, offline, CPU)
              → Google Cloud STT SDK  (fallback, requires GOOGLE_APPLICATION_CREDENTIALS)

Disabled / not in active path:
  - Google Cloud STT REST API (_transcribe_google_rest) — preserved for reference
  - Groq Whisper API          (_transcribe_groq)        — preserved for reference

Language codes:
  Pass either a 2-letter ISO-639-1 code ('en', 'zh', 'th') or a full BCP-47 tag
  ('en-US', 'cmn-Hans-CN', 'th-TH'). Both forms are accepted; the module maps
  them to the BCP-47 format required by Google Speech-to-Text automatically.
"""
import base64
import concurrent.futures
import logging
import os
import struct
import tempfile
import threading
import time
import wave

# Load .env so GOOGLE_API_KEY / GROQ_API_KEY are available whether this module
# is imported by the Flask app (which already called load_dotenv) or directly
# by the assistant runtime in start_sara.pyw (which does not).
try:
    from dotenv import load_dotenv as _load_dotenv
    _env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(_env_file):
        _load_dotenv(_env_file, override=False)   # don't override values already set by app.py
except ImportError:
    pass

log = logging.getLogger('speech')

LISTEN_SAMPLE_RATE = 16_000
LISTEN_CHANNELS = 1
LISTEN_SAMPLE_WIDTH = 2
LISTEN_CHUNK = 1024
LISTEN_SECONDS = float(os.environ.get('SARA_RECORD_SECONDS', os.environ.get('RECORD_SECONDS', '5')))

# ---------------------------------------------------------------------------
# Speech context phrases for Google STT biasing.
# Listing Sara's command vocabulary boosts recognition probability for these
# words and dramatically reduces errors like "a role" → "add row".
# $OOV_CLASS_DIGIT_SEQUENCE is a special Google token that improves number
# accuracy (e.g. prevents "ninety" from being transcribed as "nineteen").
# ---------------------------------------------------------------------------
_SARA_SPEECH_CONTEXT = {
    'phrases': [
        # ── Core commands ──────────────────────────────────────────────────
        'open excel',
        'close excel',
        'create excel',
        'create sheet',
        'switch sheet',
        'add row',
        'add column',
        'write cell',
        'delete row',
        'delete column',
        'delete sheet',
        'freeze header',
        'freeze panes',
        'unfreeze panes',
        'write formula',
        'calculate sum',
        'calculate average',
        'find and replace',
        'save',
        'undo',
        # ── Slot values used in demos ─────────────────────────────────────
        'score', 'report', 'sales',
        'John', 'Lisa', 'Alice', 'Bob',
        # ── Cell references used in demos ─────────────────────────────────
        # Biasing these nudges Google to output "A1" not "a one" / "eight one"
        'A1', 'B2', 'C2', 'C12',
        # ── Digit sequences — keeps numbers as digits, not words ──────────
        '$OOV_CLASS_DIGIT_SEQUENCE',
    ],
    # boost=20 is the Google STT maximum.
    'boost': 20,
}


# ---------- Language code mapping ----------

# Maps 2-letter ISO-639-1 → BCP-47 language tag accepted by Google STT.
# Extend this dict to support more languages.
_ISO_TO_BCP47 = {
    'en': 'en-US',
    'zh': 'cmn-Hans-CN',   # Mandarin simplified
    'th': 'th-TH',
    'ja': 'ja-JP',
    'ko': 'ko-KR',
    'fr': 'fr-FR',
    'de': 'de-DE',
    'es': 'es-ES',
    'it': 'it-IT',
    'pt': 'pt-BR',
    'ru': 'ru-RU',
    'ar': 'ar-SA',
    'hi': 'hi-IN',
    'vi': 'vi-VN',
    'ms': 'ms-MY',
    'id': 'id-ID',
}


def _to_bcp47(lang: str) -> str:
    """Normalise a language code to a BCP-47 tag for Google Speech-to-Text.

    Accepts both short forms ('en', 'zh') and full BCP-47 tags ('en-US').
    Returns 'en-US' as the default when the input is unrecognised.
    """
    if not lang:
        return 'en-US'
    lang = lang.strip()
    # Already a BCP-47 tag (contains a hyphen) — return as-is.
    if '-' in lang:
        return lang
    # Map 2-letter ISO code using the table above.
    return _ISO_TO_BCP47.get(lang.lower(), 'en-US')


# ---------- Audio format → Google STT encoding ----------

# Maps lowercase file extension → (AudioEncoding name, sample_rate_hertz).
# sample_rate_hertz of 0 tells the Google library to infer from the stream
# header (supported for FLAC and OGG_OPUS; set explicitly for others).
_EXT_TO_ENCODING = {
    '.webm': ('WEBM_OPUS', 48000),   # Chrome/Firefox MediaRecorder default
    '.wav':  ('LINEAR16', 16000),
    '.flac': ('FLAC', 0),            # FLAC headers carry sample rate
    '.ogg':  ('OGG_OPUS', 0),        # Ogg headers carry sample rate
    '.mp3':  ('MP3', 16000),
    '.m4a':  ('MP3', 16000),         # AAC in M4A — use MP3 as closest
}


# ---------- Google Cloud Speech-to-Text REST backend (API key) ----------

_GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
_GOOGLE_STT_REST_URL = 'https://speech.googleapis.com/v1/speech:recognize'


def _transcribe_google_rest(path: str, language: str = 'en') -> str | None:
    """Transcribe using the Google Cloud Speech-to-Text v1 REST API with an API key.

    This requires a Google Cloud API key (AIzaSy...) with the Speech-to-Text
    API enabled — no service account JSON file needed.

    Returns:
        The transcript string on success, or None if the key is missing or
        the request fails.
    """
    if not _GOOGLE_API_KEY:
        log.warning(
            'transcribe(google-rest): GOOGLE_API_KEY is not set. '
            'Trying SDK backend.'
        )
        return None
    try:
        import requests
        lang_code = _to_bcp47(language)
        ext = os.path.splitext(path)[1].lower()
        enc_name, sample_rate = _EXT_TO_ENCODING.get(ext, ('WEBM_OPUS', 48000))

        with open(path, 'rb') as fh:
            audio_b64 = base64.b64encode(fh.read()).decode('utf-8')

        config: dict = {
            'encoding': enc_name,
            'languageCode': lang_code,
            # command_and_search is optimised for short voice commands —
            # significantly more accurate than latest_long for 1–5 word phrases.
            'model': 'command_and_search',
            # Punctuation adds commas/periods that confuse the command parser.
            'enableAutomaticPunctuation': False,
            # Boost recognition of Sara's command vocabulary and digit sequences.
            'speechContexts': [_SARA_SPEECH_CONTEXT],
            # Use the enhanced telephony/command model where available.
            'useEnhanced': True,
        }
        if sample_rate:
            config['sampleRateHertz'] = sample_rate

        payload = {
            'config': config,
            'audio': {'content': audio_b64},
        }

        log.info('transcribe(google-rest): POST %s lang=%s enc=%s',
                 _GOOGLE_STT_REST_URL, lang_code, enc_name)
        _t = time.perf_counter()
        resp = requests.post(
            _GOOGLE_STT_REST_URL,
            params={'key': _GOOGLE_API_KEY},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        _elapsed_ms = (time.perf_counter() - _t) * 1000
        results = data.get('results', [])
        if not results:
            log.warning('transcribe(google-rest): empty results for %s (%.0f ms)', path, _elapsed_ms)
            return ''
        text = ' '.join(
            r['alternatives'][0]['transcript']
            for r in results
            if r.get('alternatives')
        )
        log.info('transcribe(google-rest): lang=%s %.0f ms result=%r', lang_code, _elapsed_ms, text[:200])
        return text
    except Exception as exc:
        log.warning('transcribe(google-rest): failed: %s', exc)
        return None


# ---------- Google Cloud Speech-to-Text SDK backend (service account) ----------

def _transcribe_google_cloud(path: str, language: str = 'en') -> str | None:
    """Transcribe using the Google Cloud Speech-to-Text v1 API.

    Returns:
        The transcript string on success, or None if the backend is
        unavailable (library not installed / credentials missing) or fails.
    """
    try:
        from google.cloud import speech  # type: ignore
    except ImportError:
        log.warning(
            'transcribe(google-cloud): google-cloud-speech is not installed. '
            'Run: pip install google-cloud-speech'
        )
        return None

    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
    if not creds_path:
        log.warning(
            'transcribe(google-cloud): GOOGLE_APPLICATION_CREDENTIALS is not set — '
            'set it to a service-account JSON path to enable this fallback.'
        )
        return None

    try:
        lang_code = _to_bcp47(language)
        ext = os.path.splitext(path)[1].lower()
        enc_name, sample_rate = _EXT_TO_ENCODING.get(ext, ('WEBM_OPUS', 48000))
        encoding = getattr(
            speech.RecognitionConfig.AudioEncoding,
            enc_name,
            speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        )

        client = speech.SpeechClient()

        with open(path, 'rb') as fh:
            content = fh.read()

        audio = speech.RecognitionAudio(content=content)
        config_kwargs = dict(
            encoding=encoding,
            language_code=lang_code,
            # command_and_search gives better accuracy on short voice commands.
            model='command_and_search',
            enable_automatic_punctuation=False,
            use_enhanced=True,
            speech_contexts=[
                speech.SpeechContext(
                    phrases=_SARA_SPEECH_CONTEXT['phrases'],
                    boost=float(_SARA_SPEECH_CONTEXT['boost']),
                )
            ],
        )
        # Only set sample_rate_hertz when we have an explicit value;
        # leaving it unset lets the API infer it from stream headers.
        if sample_rate:
            config_kwargs['sample_rate_hertz'] = sample_rate

        config = speech.RecognitionConfig(**config_kwargs)
        _t = time.perf_counter()
        response = client.recognize(config=config, audio=audio, timeout=4.0)
        _elapsed_ms = (time.perf_counter() - _t) * 1000
        log.info('transcribe(google-cloud): lang=%s %.0f ms raw_response', lang_code, _elapsed_ms)


        if not response.results:
            log.warning('transcribe(google-cloud): empty results for %s', path)
            return ''

        text = ' '.join(
            result.alternatives[0].transcript
            for result in response.results
            if result.alternatives
        )
        log.info('transcribe(google-cloud): lang=%s result=%r', lang_code, text[:200])
        return text

    except Exception as exc:
        log.warning('transcribe(google-cloud): failed: %s', exc)
        return None


# ---------- Groq Whisper fallback ----------

_GROQ_KEY = os.environ.get('GROQ_API_KEY', '')
_GROQ_WHISPER_URL = 'https://api.groq.com/openai/v1/audio/transcriptions'


def _transcribe_groq(path: str, language: str = 'en') -> str | None:
    """Fallback: Groq Whisper large-v3.

    Groq Whisper accepts 2-letter ISO-639-1 codes (not BCP-47 tags).
    Returns text on success, None on failure or missing credentials.
    """
    # Groq STT returns 403 (endpoint restricted). Skip immediately so the
    # fallback chain does not pay a 30 s timeout on every failed utterance.
    log.debug('transcribe(groq): disabled — returning None')
    return None
    if not _GROQ_KEY:
        log.warning('transcribe(groq): GROQ_API_KEY is not set')
        return None
    try:
        import requests
        # Groq Whisper expects short ISO codes ('en', not 'en-US')
        iso_lang = language.split('-')[0] if '-' in language else language
        fname = os.path.basename(path)
        log.info('transcribe(groq): POST %s lang=%s', _GROQ_WHISPER_URL, iso_lang)
        _t = time.perf_counter()
        with open(path, 'rb') as fh:
            resp = requests.post(
                _GROQ_WHISPER_URL,
                headers={'Authorization': f'Bearer {_GROQ_KEY}'},
                files={'file': (fname, fh)},
                data={'model': 'whisper-large-v3', 'language': iso_lang},
                timeout=30,
            )
        resp.raise_for_status()
        _elapsed_ms = (time.perf_counter() - _t) * 1000
        text = resp.json().get('text', '')
        log.info('transcribe(groq): %.0f ms result=%r', _elapsed_ms, text[:200] if text else '')
        return text
    except Exception as exc:
        # Surface 401 explicitly — the most common failure is an expired key.
        if hasattr(exc, 'response') and getattr(exc.response, 'status_code', None) == 401:
            log.warning(
                'transcribe(groq): 401 Unauthorized — GROQ_API_KEY is invalid or expired. '
                'Get a new key at console.groq.com and update your .env file.'
            )
        else:
            log.warning('transcribe(groq): failed: %s', exc)
        return None


# ---------- Local Whisper STT (faster-whisper, fully offline) ----------

_WHISPER_MODEL_SIZE = os.environ.get('WHISPER_MODEL', 'base.en')
_WHISPER_DEVICE = os.environ.get('WHISPER_DEVICE', 'cpu')
_WHISPER_COMPUTE_TYPE = os.environ.get('WHISPER_COMPUTE_TYPE', 'int8')

_whisper_model = None   # lazy-loaded on first use
_whisper_model_lock = threading.Lock()  # prevents concurrent loads


def _preload_whisper_model() -> None:
    """Load the Whisper model at startup so the first voice command is not slow.

    Called once in a background daemon thread immediately after module import.
    Sets the module-level ``_whisper_model`` cache so subsequent calls to
    ``_load_whisper_model()`` return instantly.
    """
    model = _load_whisper_model()
    if model is not None:
        log.info('[Startup] faster-whisper model warm and ready — primary STT active')
    else:
        log.warning(
            '[Startup] faster-whisper model is NOT available.  '
            'Every voice command will depend on cloud STT.  '
            'Check the log above for the reason (ImportError or load failure).'
        )


def _load_whisper_model():
    """Load (and cache) the faster-whisper model. Returns None if unavailable."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_model_lock:
        if _whisper_model is not None:  # re-check after acquiring lock
            return _whisper_model
        try:
            from faster_whisper import WhisperModel  # type: ignore
            log.info(
                'transcribe(whisper): loading model=%s device=%s compute=%s',
                _WHISPER_MODEL_SIZE, _WHISPER_DEVICE, _WHISPER_COMPUTE_TYPE,
            )
            _t = time.perf_counter()
            _whisper_model = WhisperModel(
                _WHISPER_MODEL_SIZE,
                device=_WHISPER_DEVICE,
                compute_type=_WHISPER_COMPUTE_TYPE,
            )
            log.info('transcribe(whisper): model loaded in %.1f s', time.perf_counter() - _t)
            return _whisper_model
        except ImportError:
            log.warning(
                'transcribe(whisper): faster-whisper is not installed — '
                'STT will depend entirely on cloud backends.  '
                'Fix: pip install faster-whisper'
            )
            return None
        except Exception as exc:
            log.warning('transcribe(whisper): failed to load model: %s', exc)
            return None


# Kick off model loading in a background thread immediately at import time.
# Placed after _load_whisper_model is defined so the thread can call it safely.
# Daemon=True: this thread will not block Python from exiting.
threading.Thread(
    target=_preload_whisper_model,
    daemon=True,
    name='whisper-preload',
).start()


def transcribe_local_whisper(path: str, language: str = 'en') -> str | None:
    """Transcribe *path* using faster-whisper (fully offline, CPU-friendly).

    Returns the transcript string on success, or None if faster-whisper is
    not installed or the transcription fails.

    Target latency: < 3 seconds for short voice commands on CPU.

    Args:
        path:     Path to a WAV file (16 kHz mono recommended).
        language: 2-letter ISO code or BCP-47 tag; only the ISO prefix is
                  passed to faster-whisper (e.g. 'en-US' → 'en').
    """
    model = _load_whisper_model()
    if model is None:
        return None
    try:
        iso_lang = language.split('-')[0] if '-' in language else language
        log.info('transcribe(whisper): transcribing %s lang=%s', path, iso_lang)
        _t = time.perf_counter()
        segments, info = model.transcribe(
            path,
            language=iso_lang,
            beam_size=5,
            vad_filter=False,
        )
        text = ' '.join(seg.text.strip() for seg in segments).strip()
        _elapsed_ms = (time.perf_counter() - _t) * 1000
        log.info(
            'transcribe(whisper): %.0f ms  detected_lang=%s  result=%r',
            _elapsed_ms, info.language, text[:200],
        )
        return text if text else ''
    except Exception as exc:
        log.warning('transcribe(whisper): transcription failed: %s', exc)
        return None


# ---------- Whisper usability constants ----------

_WHISPER_TIMEOUT: float = float(os.environ.get('WHISPER_TIMEOUT', '8'))

# Words that by themselves indicate STT captured noise rather than a real command.
_STT_NOISE_WORDS: frozenset[str] = frozenset({
    'uh', 'um', 'hmm', 'hm', 'mm', 'ah', 'oh', 'eh',
    'okay', 'ok', 'yeah', 'yep', 'nope', 'bye',
    'thank', 'thanks', 'please',
})

# Single-token Whisper results that are definitively junk (filler, social words,
# random mis-fires).  When Whisper returns exactly one of these words we skip
# the Google Cloud fallback entirely — there is nothing useful to recover, and
# the network round-trip only makes the user feel stuck.
# Contrast with _SUSPICIOUS_FRAGMENTS (corrupted command words like "adro",
# "dalit") where Google fallback genuinely helps.
_GOOGLE_SKIP_SINGLE_WORDS: frozenset[str] = frozenset({
    'yes', 'yeah', 'yep', 'nope', 'no',
    'ok', 'okay',
    'oh', 'ah', 'uh', 'um', 'hmm', 'hm', 'mm',
    'hi', 'hey', 'bye',
    'i', 'a', 'the', 'it', 'is', 'in', 'and', 'or',
    'what', 'huh', 'sorry',
    'thanks', 'thank', 'please',
    'makeup', 'music', 'noise',
})


def _is_whisper_usable(text: str) -> bool:
    """Return True if *text* looks like a real command worth acting on.

    Returns False (trigger Google fallback) when:
      - text is empty or whitespace-only
      - fewer than 2 tokens survive after removing noise words
      - every token is a noise/filler word
    """
    if not text or not text.strip():
        return False
    tokens = [t for t in text.strip().lower().split() if t.isalpha() or t.isdigit()]
    meaningful = [t for t in tokens if t not in _STT_NOISE_WORDS]
    if len(meaningful) < 2 and not (len(meaningful) == 1 and len(tokens) == 1):
        return False
    return True


# ---------- WAV normaliser ----------

def _normalize_wav(src_path: str) -> tuple[str, bool]:
    """Convert a WAV file to mono 16 kHz 16-bit PCM if needed.

    Returns (path, is_temp):
        is_temp=False → src_path is already correct; nothing was written.
        is_temp=True  → a new temp file was written; caller must os.remove it.

    Non-WAV files (webm, flac …) are returned unchanged so faster-whisper
    can handle them natively.  If audioop is unavailable (Python ≥ 3.13) the
    original path is returned with a warning.
    """
    if not src_path.lower().endswith('.wav'):
        return src_path, False

    TARGET_RATE, TARGET_WIDTH, TARGET_CHANNELS = 16_000, 2, 1
    try:
        import audioop  # stdlib, deprecated in 3.13 but still importable
        with wave.open(src_path, 'rb') as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if nch == TARGET_CHANNELS and sw == TARGET_WIDTH and rate == TARGET_RATE:
            return src_path, False  # already correct — nothing to do

        log.info(
            'transcribe: normalizing WAV ch=%d→%d sw=%d→%d rate=%d→%d',
            nch, TARGET_CHANNELS, sw, TARGET_WIDTH, rate, TARGET_RATE,
        )

        # 1. Convert bit depth to 16-bit PCM
        if sw != TARGET_WIDTH:
            frames = audioop.lin2lin(frames, sw, TARGET_WIDTH)
            sw = TARGET_WIDTH

        # 2. Stereo (or multi-channel) → mono by averaging channels
        if nch != TARGET_CHANNELS:
            frames = audioop.tomono(frames, sw, 0.5, 0.5)
            nch = TARGET_CHANNELS

        # 3. Resample to 16 kHz
        if rate != TARGET_RATE:
            frames, _ = audioop.ratecv(frames, sw, nch, rate, TARGET_RATE, None)

        tmp = tempfile.NamedTemporaryFile(suffix='_norm.wav', delete=False)
        tmp.close()
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(TARGET_CHANNELS)
            wf.setsampwidth(TARGET_WIDTH)
            wf.setframerate(TARGET_RATE)
            wf.writeframes(frames)
        return tmp.name, True

    except Exception as exc:
        log.warning('transcribe: WAV normalization failed — using original: %s', exc)
        return src_path, False


# ---------- Public API ----------

def transcribe_audio_file(path: str, language: str = 'en') -> str:
    """Transcribe an audio file and return the recognised text.

    Two-stage pipeline::

        1. faster-whisper    — local, CPU, ~1-3 s, fully offline (primary)
        2. Google Cloud STT  — SDK fallback, requires GOOGLE_APPLICATION_CREDENTIALS

    Every WAV input is normalised to mono 16 kHz 16-bit PCM before dispatch so
    Google Cloud STT receives the exact format it expects.  Non-WAV files are
    passed unchanged to faster-whisper which handles them natively.

    Google REST STT and Groq audio STT are not used in this path.

    Args:
        path:     Absolute or relative path to the audio file.
                  Supported formats: .wav (normalised), .webm, .flac, .ogg, .mp3
        language: Language code — 2-letter ISO ('en', 'zh') or BCP-47 ('en-US').

    Returns:
        The transcript as a plain string.
        On total failure returns a string prefixed with '[transcription_error]'.
    """
    log.info('transcribe: path=%s language=%s', path, language)
    norm_path, is_temp = _normalize_wav(path)
    try:
        # Stage 1: local faster-whisper (offline, no API cost, ~1-3 s on CPU)
        # Hard timeout prevents rare 20-60 s CPU stalls.
        # Decision tree after Whisper finishes:
        #   usable text  → return immediately (Whisper is primary)
        #   empty        → fall to Google (nothing to work with)
        #   non-empty but unusable (noise/filler) → fall to Google if credentials
        #                  are configured so it can attempt recovery; if credentials
        #                  are absent Google will return None immediately anyway, and
        #                  the conversation loop's describe_low_confidence() can filter
        #                  the noise result better than a sentinel.
        text = None
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _exe:
                _fut = _exe.submit(transcribe_local_whisper, norm_path, language)
                text = _fut.result(timeout=_WHISPER_TIMEOUT)
        except concurrent.futures.TimeoutError:
            log.warning('transcribe: faster-whisper exceeded %.1fs timeout — '
                        'falling through to Google Cloud STT', _WHISPER_TIMEOUT)
            text = None
        except Exception as _exc:
            log.warning('transcribe: faster-whisper error: %s', _exc)
            text = None

        usable = _is_whisper_usable(text) if text else False
        if usable:
            log.info('transcribe(whisper): usable result — returning immediately')
            return text

        if text:
            # Whisper returned something but it is unusable (noise/filler word).
            # Skip Google fallback when the result is a single definitively-junk
            # token (yes, yeah, oh, makeup …).  These are never recoverable by a
            # second STT engine and the network round-trip only makes the user feel
            # stuck.  Genuine STT corruptions of command words (adro, dalit …) are
            # NOT in _GOOGLE_SKIP_SINGLE_WORDS and still reach Google as before.
            _norm_token = text.strip().lower().rstrip('.,!?')
            if _norm_token in _GOOGLE_SKIP_SINGLE_WORDS:
                log.info(
                    'transcribe(whisper): junk single-word %r — skipping Google fallback',
                    text[:80]
                )
                return text

            # Allow Google to attempt recovery only when credentials are configured;
            # _transcribe_google_cloud() performs the same check internally, so this
            # avoids a pointless call and guarantees we return the Whisper result
            # (for the conversation loop to filter) rather than the all-failed sentinel.
            _google_configured = bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''))
            if not _google_configured:
                log.info(
                    'transcribe(whisper): unusable result %r, Google not configured — '
                    'returning for conversation loop filtering', text[:80]
                )
                return text
            log.info(
                'transcribe(whisper): unusable result %r — Google credentials found, '
                'attempting recovery', text[:80]
            )
        elif text:
            log.info('transcribe: faster-whisper returned empty — trying Google Cloud STT')
        

        # Stage 2: Google Cloud STT SDK (GOOGLE_APPLICATION_CREDENTIALS)
        text = _transcribe_google_cloud(norm_path, language=language)
        if text:
            return text
        if text == '':
            log.info('transcribe: google-cloud returned empty — no speech detected')

        log.error('transcribe: all backends failed for %s', path)
        return '[transcription_error] No speech detected'

    except Exception as exc:
        log.error('transcribe: unhandled error: %s', exc)
        return f'[transcription_error] {exc}'
    finally:
        if is_temp:
            try:
                os.remove(norm_path)
            except OSError:
                pass


def _rms_energy(frame_bytes: bytes) -> float:
    """Return RMS energy of a 16-bit PCM frame (used for silence detection)."""
    n = len(frame_bytes) // 2
    if n == 0:
        return 0.0
    shorts = struct.unpack(f'<{n}h', frame_bytes[:n * 2])
    return (sum(s * s for s in shorts) / n) ** 0.5


def listen_once(language: str = 'en', seconds: float = LISTEN_SECONDS) -> str:
    """Record one microphone utterance and pass it through the transcription pipeline.

    Records up to *seconds* of audio, stopping early after 1.2 s of silence
    (RMS energy below 200) so that the STT request is sent as soon as the user
    stops talking rather than waiting out the full duration.

    Returns the transcript string, or a ``[transcription_error]`` message when
    microphone capture or transcription fails.
    """
    log.info('[Speech] listen_once called — language=%s max_seconds=%.1f', language, seconds)
    try:
        import pyaudio
    except ImportError:
        log.error('[Speech] PyAudio is not installed')
        return '[transcription_error] PyAudio is not installed'

    pa = None
    stream = None
    tmp_path = None
    try:
        pa = pyaudio.PyAudio()

        # Log the default input device so the user can verify the right mic is used.
        try:
            dev_info = pa.get_default_input_device_info()
            log.info('[Speech] Microphone: index=%d name=%r channels=%d rate=%d',
                     dev_info['index'], dev_info['name'],
                     dev_info['maxInputChannels'], int(dev_info['defaultSampleRate']))
        except Exception as dev_exc:
            log.warning('[Speech] Could not query default input device: %s', dev_exc)

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=LISTEN_CHANNELS,
            rate=LISTEN_SAMPLE_RATE,
            input=True,
            frames_per_buffer=LISTEN_CHUNK,
        )
        log.info('[Speech] Microphone opened — recording up to %.1f s at %d Hz',
                 seconds, LISTEN_SAMPLE_RATE)

        # Warmup drain: discard the first ~65 ms of audio after the stream opens.
        # On Windows/Realtek the driver needs a brief settling period after open():
        # the very first frames can contain initialization noise or lingering TTS
        # echo from the previous output.  1 chunk (64 ms) is enough to clear the
        # driver buffer; 2 chunks (128 ms) was clipping short commands like "save"
        # and "undo" that are spoken immediately after wake-word detection.
        _WARMUP_CHUNKS = 1  # 1 × (LISTEN_CHUNK / LISTEN_SAMPLE_RATE) ≈ 64 ms
        for _ in range(_WARMUP_CHUNKS):
            stream.read(LISTEN_CHUNK, exception_on_overflow=False)
        log.debug('[Speech] Mic warmup done — discarded %d startup chunks', _WARMUP_CHUNKS)

        frames: list[bytes] = []
        total_frames = max(1, int(LISTEN_SAMPLE_RATE / LISTEN_CHUNK * max(1.0, seconds)))
        # Silence detection: stop early after SILENCE_FRAMES consecutive quiet frames.
        # Default lowered from 300 → 200 → 150: Realtek laptop mics produce 150–280 RMS
        # for normal conversational speech; 200 still caused [no_voice] (logs showed
        # peak_rms=193 < threshold=200).  Raise SARA_SILENCE_THRESHOLD if background
        # noise is high enough to be misdetected as voice.
        SILENCE_ENERGY_THRESHOLD = int(os.environ.get('SARA_SILENCE_THRESHOLD', '150'))
        # Minimum number of above-threshold frames required before the recording
        # is treated as real speech (had_voice=True).  Each frame is ~64 ms.
        # Default 3 → catches "save" / "undo" (monosyllables) while rejecting
        # single-frame noise spikes from keyboard clicks or fan bursts.
        _VOICE_FRAMES_REQUIRED = int(os.environ.get('SARA_VOICE_FRAMES_MIN', '3'))
        # Seconds of post-voice silence that triggers early recording stop.
        # 0.8 s gives a comfortable window for natural mid-command pauses
        # (e.g. "save... and close") without waiting out the full max duration.
        # Tune via SARA_SILENCE_AFTER_SPEECH_SECONDS in .env.
        _SILENCE_AFTER_SPEECH_S = float(os.environ.get('SARA_SILENCE_AFTER_SPEECH_SECONDS', '0.8'))
        SHORT_SILENCE_LIMIT = int(_SILENCE_AFTER_SPEECH_S * LISTEN_SAMPLE_RATE / LISTEN_CHUNK)
        silent_streak = 0
        had_voice = False
        _voice_frame_count = 0

        _rec_start = time.perf_counter()
        for _ in range(total_frames):
            chunk = stream.read(LISTEN_CHUNK, exception_on_overflow=False)
            frames.append(chunk)
            energy = _rms_energy(chunk)
            if energy >= SILENCE_ENERGY_THRESHOLD:
                _voice_frame_count += 1
                if _voice_frame_count >= _VOICE_FRAMES_REQUIRED:
                    had_voice = True
                silent_streak = 0
            else:
                silent_streak += 1

            # Stop early only after we have heard at least some voice activity
            # followed by sustained silence, so we don't cut off before the user speaks.
            if had_voice and silent_streak >= SHORT_SILENCE_LIMIT:
                log.debug('[Speech] Silence detected after %.2f s — stopping early',
                          time.perf_counter() - _rec_start)
                break

        _rec_elapsed = time.perf_counter() - _rec_start
        _recorded_seconds = len(frames) * LISTEN_CHUNK / LISTEN_SAMPLE_RATE

        # Diagnostic: compute peak RMS so the user can tune SARA_SILENCE_THRESHOLD.
        # If peak_rms is well above 0 but below the threshold you see here, the
        # threshold is too high for your microphone — lower SARA_SILENCE_THRESHOLD.
        _peak_rms = max((_rms_energy(f) for f in frames), default=0.0)
        log.info(
            '[Speech] Recording complete — %.2f s captured (wall %.2f s) '
            'peak_rms=%.0f threshold=%d had_voice=%s',
            _recorded_seconds, _rec_elapsed,
            _peak_rms, SILENCE_ENERGY_THRESHOLD, had_voice,
        )

        # Fast-exit: no voice energy at all → skip STT and return sentinel.
        # The conversation loop treats this as "user went quiet" and exits
        # cleanly instead of waiting through MAX_CONSECUTIVE_ERRORS retries.
        if not had_voice:
            log.info(
                '[Speech] No voice detected (peak_rms=%.0f < threshold=%d) — '
                'returning [no_voice].  If you spoke, lower SARA_SILENCE_THRESHOLD '
                '(currently %d) in your .env file.',
                _peak_rms, SILENCE_ENERGY_THRESHOLD, SILENCE_ENERGY_THRESHOLD,
            )
            return '[no_voice]'

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(LISTEN_CHANNELS)
            wf.setsampwidth(LISTEN_SAMPLE_WIDTH)
            wf.setframerate(LISTEN_SAMPLE_RATE)
            wf.writeframes(b''.join(frames))

        wav_bytes = os.path.getsize(tmp_path)
        log.info('[Speech] WAV written to %s (%d bytes) — sending to STT', tmp_path, wav_bytes)

        _stt_start = time.perf_counter()
        transcript = transcribe_audio_file(tmp_path, language=language)
        _stt_elapsed = (time.perf_counter() - _stt_start) * 1000

        if not transcript or transcript.startswith('[transcription_error]'):
            log.warning('[Speech] STT failed after %.0f ms: %s', _stt_elapsed,
                        transcript or '(empty)')
            return transcript or '[transcription_error] Empty response from STT'

        log.info('[Speech] Transcript received in %.0f ms (%d chars): %r',
                 _stt_elapsed, len(transcript), transcript[:200])
        return transcript

    except Exception as exc:
        log.warning('[Speech] Microphone error: %s', exc)
        return f'[transcription_error] Microphone error: {exc}'
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
                log.debug('[Speech] Microphone stream closed')
            except Exception:
                pass
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
