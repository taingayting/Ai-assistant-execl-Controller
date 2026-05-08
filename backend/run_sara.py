"""Sara headless runtime — command-line entry point.

Runs Sara fully locally without the Flask server or the PyQt desktop UI.
All systems (wake word, speech recognition, LLM routing, agent, TTS) are
wired together through the existing SaraRuntime class in assistant_runtime.py.

Usage::

    python backend/run_sara.py

    # With options
    python backend/run_sara.py --language en-US --threshold 0.5 --no-tts

Environment variables (loaded from backend/.env automatically):

    GOOGLE_API_KEY        — Google Cloud STT API key (primary STT backend)
    GROQ_API_KEY          — Groq API key (STT + LLM fallback)
    ENABLE_TTS            — "true" / "false" (default: true in this script)
    WAKE_MODEL_PATH       — path to custom .onnx wake-word model
    WAKE_THRESHOLD        — detection threshold 0.0–1.0 (default: 0.5)
    SARA_LANGUAGE         — BCP-47 language tag (default: en-US)
    WHISPER_MODEL         — faster-whisper model for offline STT (default: base.en)
    USE_OLLAMA            — "true" to enable Ollama LLM backend

Interaction flow (one iteration)::

    [idle]  ─── wake word detected ──▶  [show "Listening..."]
         ▲                                       │
         │   stop phrase / error                 ▼
         │                            record + transcribe speech
         │                                       │
         └──────────── respond ◀── agent executes Excel command
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — must happen before any project imports
# ---------------------------------------------------------------------------

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.dirname(_BACKEND_DIR)

for _p in (_BACKEND_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Load .env before importing modules that read env vars at module level
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(_BACKEND_DIR, '.env')
    if os.path.exists(_env_path):
        load_dotenv(_env_path, override=True)
        print(f'[Sara] Loaded environment from {_env_path}')
except ImportError:
    pass   # python-dotenv not installed — rely on shell environment


# ---------------------------------------------------------------------------
# Logging setup (before any project imports so all loggers are configured)
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s',
        datefmt='%H:%M:%S',
    )
    # Suppress noisy third-party loggers
    for _noisy in ('urllib3', 'requests', 'httpx', 'httpcore',
                   'onnxruntime', 'openwakeword'):
        logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Sara voice assistant — headless CLI runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--language', '-l',
        default=os.environ.get('SARA_LANGUAGE', 'en-US'),
        metavar='LANG',
        help='BCP-47 language tag for speech recognition (default: en-US)',
    )
    p.add_argument(
        '--threshold', '-t',
        type=float,
        default=float(os.environ.get('WAKE_THRESHOLD', '0.5')),
        metavar='FLOAT',
        help='Wake-word detection threshold 0.0–1.0 (default: 0.5)',
    )
    p.add_argument(
        '--model', '-m',
        default=os.environ.get('WAKE_MODEL_PATH'),
        metavar='PATH',
        help='Path to a custom .onnx wake-word model (default: bundled hey_jarvis)',
    )
    p.add_argument(
        '--no-tts',
        action='store_true',
        help='Disable text-to-speech output (print responses to terminal only)',
    )
    p.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable DEBUG-level logging',
    )
    p.add_argument(
        '--files-dir',
        default=None,
        metavar='DIR',
        help='Directory for Excel files (default: backend/files/)',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _setup_logging(verbose=args.verbose)

    log = logging.getLogger('run_sara')

    # Apply CLI overrides to the environment so downstream modules pick them up
    os.environ['SARA_LANGUAGE'] = args.language
    os.environ['WAKE_THRESHOLD'] = str(args.threshold)
    if args.model:
        os.environ['WAKE_MODEL_PATH'] = args.model
    if args.no_tts:
        os.environ['ENABLE_TTS'] = 'false'
    else:
        # Default to TTS enabled when running headlessly — unlike the test suite
        os.environ.setdefault('ENABLE_TTS', 'true')

    # ------------------------------------------------------------------ #
    # Print banner
    # ------------------------------------------------------------------ #

    print()
    print('┌─────────────────────────────────────────────────────┐')
    print('│              S A R A  Voice Assistant               │')
    print('│         Say "Hey Jarvis" (or your wake word)        │')
    print('│         to activate · Ctrl+C to quit                │')
    print('└─────────────────────────────────────────────────────┘')
    print()
    print(f'  Language  : {args.language}')
    print(f'  TTS       : {"disabled (--no-tts)" if args.no_tts else "enabled (pyttsx3)"}')
    print(f'  Threshold : {args.threshold}')
    print()

    # ------------------------------------------------------------------ #
    # Import project modules (after env vars are set)
    # ------------------------------------------------------------------ #

    log.info('Importing project modules...')

    try:
        from wakeword.wake_detector_porcupine import create_wake_detector
        log.info('wake detector factory: wakeword.wake_detector_porcupine')
    except (ImportError, ModuleNotFoundError):
        # Fallback: use openwakeword WakeDetector directly
        from wakeword.wake_detector import WakeDetector
        def create_wake_detector():
            return WakeDetector(
                model_path=os.environ.get('WAKE_MODEL_PATH'),
                threshold=float(os.environ.get('WAKE_THRESHOLD', '0.5')),
            )
        log.info('wake detector factory: wakeword.wake_detector (openwakeword)')

    from assistant_runtime import SaraRuntime
    import tts_engine

    # ------------------------------------------------------------------ #
    # Build runtime (no UI — headless mode)
    # ------------------------------------------------------------------ #

    log.info('Initialising TTS engine...')
    tts_engine.initialize_engine()

    log.info('Loading wake-word model...')
    try:
        wake_detector = create_wake_detector()
    except FileNotFoundError as exc:
        print()
        print(f'ERROR: Could not load wake-word model:\n  {exc}')
        print()
        print('Fix options:')
        print('  1. Install openwakeword models:')
        print('       python -c "import openwakeword; openwakeword.utils.download_models()"')
        print('  2. Set WAKE_MODEL_PATH to a custom .onnx file')
        sys.exit(1)

    log.info('Wake-word engine: %s', type(wake_detector).__name__)

    # headless=True → no PyQt popup; the runtime still speaks via pyttsx3
    runtime = SaraRuntime(
        ui=None,                       # no desktop popup — headless CLI mode
        wake_detector=wake_detector,
        language=args.language,
        hide_ui_when_idle=False,       # no UI to hide
        files_dir=args.files_dir,
    )

    print('  Wake word detector : active')
    print('  Assistant runtime  : starting')
    print()
    print('  Say the wake word to begin...')
    print()

    # ------------------------------------------------------------------ #
    # Run (blocks until Ctrl+C)
    # ------------------------------------------------------------------ #

    try:
        runtime.start()        # blocking — contains the watchdog + wake-word loop
    except KeyboardInterrupt:
        print()
        print('Stopping Sara...')
    finally:
        runtime.stop()
        tts_engine.shutdown()
        log.info('Sara stopped cleanly.')
        print('Goodbye.')


if __name__ == '__main__':
    main()
