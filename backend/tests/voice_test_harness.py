#!/usr/bin/env python
"""
voice_test_harness.py — Real end-to-end voice pipeline validation for Sara.

Tests the ACTUAL microphone path.  No mocks, no fakes:

    microphone capture             (speech_recognition.listen_once)
        ↓  raw STT transcript
    transcript correction          (pipeline_sanitizer.correct_transcript)
        ↓  corrected transcript
    intent detection               (pipeline_sanitizer.detect_command_intent)
        ↓  intent name
    slot completeness check        (pipeline_sanitizer.check_command_slots)
        ↓  clarification or continue
    parse / route                  (llm_router.route_llm — heuristic first)
        ↓  action dict
    execution response             (agent.process_command — only with --execute)

Run from the project root
─────────────────────────
    python backend/tests/voice_test_harness.py            # parse only (safe)
    python backend/tests/voice_test_harness.py --execute  # real Excel actions
    python backend/tests/voice_test_harness.py --timeout 10
    python backend/tests/voice_test_harness.py --tests 1 3 5
    python backend/tests/voice_test_harness.py --language en-US
    python backend/tests/voice_test_harness.py --skip-ready

What to say
───────────
Test 1 → open excel
Test 2 → add row John 90
Test 3 → write cell C2 12
Test 4 → go to cell B3
Test 5 → switch sheet Sheet2
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# ── Path bootstrap ──────────────────────────────────────────────────────────
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND   = os.path.dirname(_TESTS_DIR)
_PROJECT   = os.path.dirname(_BACKEND)
for _p in (_BACKEND, _PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Load .env before any backend module imports that read env vars at module level ──
try:
    from dotenv import load_dotenv as _load_dotenv
    import pathlib as _pathlib
    _env_path = _pathlib.Path(_BACKEND) / '.env'
    if _env_path.exists():
        _load_dotenv(_env_path, override=True)
except ImportError:
    pass

# Suppress chatty module logs during the interactive test run.
# Only WARNING and above is shown; set SARA_TEST_VERBOSE=1 to see DEBUG.
import logging
if os.environ.get('SARA_TEST_VERBOSE'):
    logging.basicConfig(level=logging.DEBUG, format='%(name)s %(levelname)s: %(message)s')
else:
    logging.basicConfig(level=logging.WARNING, format='%(name)s: %(message)s')

# ── ANSI colour helpers (TTY-aware) ─────────────────────────────────────────
_TTY = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
def _c(code: str, t: str) -> str: return f'\033[{code}m{t}\033[0m' if _TTY else t
def grn(t: str) -> str: return _c('32',  t)
def red(t: str) -> str: return _c('31',  t)
def ylw(t: str) -> str: return _c('33',  t)
def bld(t: str) -> str: return _c('1',   t)
def cyn(t: str) -> str: return _c('36',  t)
def dim(t: str) -> str: return _c('2',   t)
def mgn(t: str) -> str: return _c('35',  t)

SEP  = '─' * 64
SEP2 = '═' * 64


# ── Test-case definitions ───────────────────────────────────────────────────

@dataclass
class TestCase:
    num: int
    say: str             # exactly what to say aloud
    expected_intent: str
    hint: str = ''       # shown as a reminder before recording


DEFAULT_TESTS: list[TestCase] = [
    # ── file commands ────────────────────────────────────────────────────────
    TestCase(1, 'open excel',          'open_excel',
             hint='triggers the open-Excel action'),
    TestCase(2, 'close excel',         'close_excel',
             hint='file session teardown — must not route to open_excel'),
    # ── save / undo ──────────────────────────────────────────────────────────
    TestCase(3, 'save',                'save',
             hint='single-word command — Whisper sometimes returns "safe" or "sake"'),
    TestCase(4, 'undo',                'undo',
             hint='single-word command — must not route to add_row or write_cell'),
    # ── row commands ─────────────────────────────────────────────────────────
    TestCase(5, 'add row John 90',     'add_row',
             hint='name=John, score=90 — tests phonetic correction path'),
    TestCase(6, 'delete row 2',        'delete_row',
             hint='index=2 — tests numeric slot capture'),
    # ── cell commands ────────────────────────────────────────────────────────
    TestCase(7, 'write cell C2 12',    'write_cell',
             hint='cell=C2, value=12 — tests cell-ref normalisation'),
    TestCase(8, 'go to cell B3',       'read_cell',
             hint='cell navigation — must NOT route to switch_sheet'),
    # ── sheet commands ───────────────────────────────────────────────────────
    TestCase(9, 'switch sheet Sheet2', 'switch_sheet',
             hint='sheet name = Sheet2 — tests sheet-slot path'),
    # ── filter command ───────────────────────────────────────────────────────
    TestCase(10, 'filter by column B value 90', 'filter_table',
             hint='multi-word — tests full capture at SARA_SILENCE_AFTER_SPEECH_SECONDS=1.5'),
]


# ── Per-test result ─────────────────────────────────────────────────────────

@dataclass
class TestResult:
    case: TestCase
    # Stage outputs
    raw_transcript: str      = ''
    corrected:      str      = ''
    intent:         str|None = None
    slot_msg:       str|None = None
    action:         dict|None = None
    route_source:   str      = ''   # heuristic / ollama / groq / unknown
    response:       str      = ''   # only set when --execute
    # Stage pass/fail
    voice_ok:      bool = False   # Stage 1: real audio captured
    stt_ok:        bool = False   # Stage 1: STT returned non-empty string
    stt_empty:     bool = False   # STT returned '' (VAD filtered)
    intent_match:  bool = False   # Stage 3: intent == expected
    slot_complete: bool = False   # Stage 4: no clarification needed
    action_ok:     bool = False   # Stage 5: action.type matches expected
    execute_ok:    bool = False   # Stage 6: execution returned a response
    # Error info
    error: str = ''

    @property
    def overall_pass(self) -> bool:
        return (self.voice_ok and self.stt_ok and
                self.intent_match and self.slot_complete and self.action_ok)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _mic_info() -> str:
    """Return a one-line string describing the default input device, or '?'."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        info = pa.get_default_input_device_info()
        pa.terminate()
        return f"[{info['index']}] {info['name']}  ({int(info['defaultSampleRate'])} Hz)"
    except Exception as exc:
        return f'(could not query mic: {exc})'


def _wait_for_enter(skip: bool) -> None:
    """Either wait for Enter or skip the prompt when --skip-ready is set."""
    if skip:
        time.sleep(0.3)   # small gap so the Enter keypress from the last test settles
        return
    try:
        input(f'  {dim("Press ENTER, then speak immediately...")}  ')
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    # Brief pause so the Enter-key click is past the warmup drain window.
    time.sleep(0.25)


# ── Core: run one test through the full pipeline ────────────────────────────

def run_one_test(case: TestCase, args: argparse.Namespace) -> TestResult:
    result = TestResult(case=case)

    # ── Imports (lazy so startup is fast) ─────────────────────────────────
    from speech_recognition import listen_once
    from pipeline_sanitizer import (
        correct_transcript,
        detect_command_intent,
        check_command_slots,
    )
    from llm_router import route_llm

    # ── Stage 1: microphone capture + STT ─────────────────────────────────
    print(f'  {cyn("Stage 1")}  Listening (up to {args.timeout}s)…', flush=True)
    t0 = time.perf_counter()
    transcript = listen_once(language=args.language, seconds=float(args.timeout))
    elapsed = time.perf_counter() - t0
    result.raw_transcript = transcript

    if transcript == '[no_voice]':
        print(f'  {cyn("Stage 1")}  {red("✗ No voice detected")}  '
              f'({elapsed:.1f}s)  — check mic / threshold')
        result.error = 'no_voice'
        return result

    if transcript.startswith('[transcription_error]'):
        print(f'  {cyn("Stage 1")}  {red("✗ STT error")}:  {transcript}')
        result.error = 'stt_error'
        return result

    result.voice_ok = True

    if not transcript.strip():
        result.stt_empty = True
        print(f'  {cyn("Stage 1")}  {red("✗ STT returned empty string")}  '
              f'({elapsed:.1f}s)  — faster-whisper VAD filtered or cloud returned nothing')
        result.error = 'stt_empty'
        return result

    result.stt_ok = True
    print(f'  {cyn("Stage 1")}  {grn("✓ Voice + STT OK")}  '
          f'({elapsed:.1f}s)  raw: {bld(repr(transcript[:72]))}')

    # ── Stage 2: transcript correction ────────────────────────────────────
    corrected = correct_transcript(transcript)
    result.corrected = corrected
    if corrected != transcript:
        print(f'  {cyn("Stage 2")}  {grn("Corrected")}:  '
              f'{dim(repr(transcript[:50]))} → {bld(repr(corrected[:50]))}')
    else:
        print(f'  {cyn("Stage 2")}  No correction:  {bld(repr(corrected[:60]))}')

    # ── Stage 3: intent detection ──────────────────────────────────────────
    intent = detect_command_intent(corrected)
    result.intent = intent
    if intent is None:
        print(f'  {cyn("Stage 3")}  {red("✗ No intent detected")}  '
              f'(expected: {case.expected_intent})')
    elif intent == case.expected_intent:
        result.intent_match = True
        print(f'  {cyn("Stage 3")}  {grn("✓ Intent match")}:  {bld(intent)}')
    else:
        print(f'  {cyn("Stage 3")}  {ylw("≠ Intent mismatch")}:  '
              f'got {bld(intent)}  expected {case.expected_intent}')

    # ── Stage 4: slot completeness ─────────────────────────────────────────
    slot_msg = check_command_slots(intent, corrected)
    result.slot_msg = slot_msg
    if slot_msg:
        print(f'  {cyn("Stage 4")}  {ylw("Slot gap")}:  {slot_msg}')
    else:
        result.slot_complete = True
        print(f'  {cyn("Stage 4")}  {grn("✓ Slots complete")}')

    # ── Stage 5: route / parse ─────────────────────────────────────────────
    try:
        t_r = time.perf_counter()
        action = route_llm(corrected)
        route_elapsed = (time.perf_counter() - t_r) * 1000
    except Exception as exc:
        print(f'  {cyn("Stage 5")}  {red("✗ route_llm error")}:  {exc}')
        result.error = f'route_llm: {exc}'
        return result

    result.action       = action
    result.route_source = action.get('_router_source', '?')

    # route_llm returns EITHER a single-action dict  {'type': 'add_row', ...}
    # OR a steps-based plan dict                     {'steps': [...], ...}
    # Extract the primary action type from whichever shape was returned so
    # that the pass/fail check and display are both correct.
    if action.get('steps'):
        _steps = action['steps']
        action_type = _steps[0].get('type', 'unknown') if _steps else 'unknown'
        _action_display = {'steps': [
            {k: v for k, v in s.items() if not k.startswith('_')}
            for s in _steps[:2]          # show at most 2 steps to keep output readable
        ]}
    else:
        action_type    = action.get('type', 'unknown')
        _action_display = {k: v for k, v in action.items()
                           if not k.startswith('_') and k != 'confidence'}

    # The heuristic parser uses more specific action names than the intent labels.
    # This table maps each expected_intent to all action types that count as a pass.
    # Add entries here when the router legitimately returns an alias.
    _INTENT_ACTION_ALIASES: dict[str, set[str]] = {
        'open_excel':        {'open_excel',   'open_excel_app', 'open_file'},
        'close_excel':       {'close_excel',  'close_excel_app'},
        # Heuristic returns 'undo'; LLM prompt names it 'undo_last' — accept both.
        'undo':              {'undo', 'undo_last'},
        'calculate_sum':     {'calculate_sum',     'write_sum_formula',     'calculate_row_sum'},
        'calculate_average': {'calculate_average', 'write_average_formula'},
    }
    _accepted = _INTENT_ACTION_ALIASES.get(case.expected_intent,
                                           {case.expected_intent})
    result.action_ok = (action_type in _accepted)
    _source_tag = dim(f'[{result.route_source}  {route_elapsed:.0f}ms]')

    if result.action_ok:
        print(f'  {cyn("Stage 5")}  {grn("✓ Action")}:  '
              f'{bld(str(_action_display)[:80])}  {_source_tag}')
    elif action_type == 'unknown':
        print(f'  {cyn("Stage 5")}  {red("✗ Action unknown")}  '
              f'{_source_tag}  — heuristic miss + LLM offline/exhausted')
    else:
        print(f'  {cyn("Stage 5")}  {ylw("≠ Action type")}:  '
              f'got {bld(action_type)}  expected {case.expected_intent}  {_source_tag}')

    # ── Stage 6: execute (--execute only) ──────────────────────────────────
    if args.execute:
        try:
            from agent import process_command
            t_x = time.perf_counter()
            response = process_command(corrected)
            exec_elapsed = (time.perf_counter() - t_x) * 1000
            result.response   = str(response) if response else ''
            result.execute_ok = bool(response)
            symbol = grn('✓') if result.execute_ok else ylw('?')
            print(f'  {cyn("Stage 6")}  {symbol} Executed ({exec_elapsed:.0f}ms):  '
                  f'{bld(str(response)[:80])}')
        except Exception as exc:
            result.error = f'execute: {exc}'
            print(f'  {cyn("Stage 6")}  {red("✗ Execution error")}:  {str(exc)[:80]}')
    else:
        print(f'  {cyn("Stage 6")}  {dim("Skipped  (add --execute to run real Excel actions)")}')

    return result


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary(results: list[TestResult], args: argparse.Namespace,
                  log_path: str) -> None:
    print()
    print(SEP2)
    print(bld('  SUMMARY'))
    print(SEP2)

    # Column headers
    cols = ['#', 'Say', 'Voice', 'STT', 'Intent', 'Slots', 'Action', 'Result']
    widths = [3, 28, 6, 4, 14, 6, 8, 7]
    header = '  ' + '  '.join(c.ljust(w) for c, w in zip(cols, widths))
    print(dim(header))
    print(dim('  ' + '  '.join('-' * w for w in widths)))

    total = stt_empty = no_voice = stt_error = 0
    correct_intent = slot_ok_n = action_ok_n = passed = 0

    for r in results:
        total += 1
        no_voice  += int(not r.voice_ok and r.error == 'no_voice')
        stt_empty += int(r.stt_empty)
        stt_error += int(r.error == 'stt_error')

        v = grn('OK') if r.voice_ok else red('MISS')
        s = grn('OK') if r.stt_ok   else (ylw('EMPTY') if r.stt_empty else red('ERR'))

        if r.intent_match:
            correct_intent += 1
            i = grn(str(r.intent or '')[:14])
        elif r.intent:
            i = ylw(str(r.intent)[:14])
        else:
            i = red('none')

        if r.slot_complete:
            slot_ok_n += 1
            sl = grn('OK')
        elif r.slot_msg:
            sl = ylw('GAP')
        else:
            sl = dim('—')

        if r.action_ok:
            action_ok_n += 1
            ac = grn(str(r.action.get('type', '?'))[:8])
        elif r.action:
            ac = ylw(str(r.action.get('type', '?'))[:8])
        else:
            ac = red('none')

        if r.overall_pass:
            passed += 1
            res = grn('PASS ✓')
        else:
            res = red('FAIL ✗')

        row = [
            str(r.case.num),
            f'"{r.case.say}"'[:28],
            v, s, i, sl, ac, res,
        ]
        print('  ' + '  '.join(str(c).ljust(w) for c, w in zip(row, widths)))

    print()
    print(f'  Total tests    : {bld(str(total))}')
    print(f'  No voice       : {(red if no_voice  else grn)(str(no_voice))}')
    print(f'  STT empty      : {(red if stt_empty else grn)(str(stt_empty))}')
    print(f'  STT error      : {(red if stt_error else grn)(str(stt_error))}')
    print(f'  Intent correct : {bld(str(correct_intent))}/{total}')
    print(f'  Slots complete : {bld(str(slot_ok_n))}/{total}')
    print(f'  Action correct : {bld(str(action_ok_n))}/{total}')
    print(f'  {bld("Overall PASS")}   : {bld(str(passed))}/{total}  '
          + (grn('ALL PASS ✓') if passed == total else ylw(f'{total-passed} failing')))
    print()
    if log_path:
        print(f'  Full log: {dim(log_path)}')
    print(SEP2)


# ── Pre-flight mic check ─────────────────────────────────────────────────────

def preflight_check() -> bool:
    """Return True if pyaudio is available and a mic device can be found."""
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        print(red('  ERROR: pyaudio is not installed.'))
        print('  Run:  pip install pyaudio')
        return False
    info = _mic_info()
    print(f'  Microphone : {bld(info)}')
    return True


# ── Log file setup ───────────────────────────────────────────────────────────

class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, path: str):
        self._orig  = sys.stdout
        self._fh    = open(path, 'w', encoding='utf-8', errors='replace')
        sys.stdout  = self

    def write(self, s: str) -> int:
        # Strip ANSI codes for the file copy
        import re
        clean = re.sub(r'\033\[\d+m', '', s)
        self._fh.write(clean)
        return self._orig.write(s)

    def flush(self) -> None:
        self._fh.flush()
        self._orig.flush()

    def isatty(self) -> bool:
        return self._orig.isatty()

    def restore(self) -> str:
        sys.stdout = self._orig
        self._fh.close()
        return self._fh.name


# ── Main ─────────────────────────────────────────────────────────────────────

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Sara real-microphone end-to-end voice pipeline test harness.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--execute', action='store_true',
                   help='Also run Excel actions for real (default: parse only).')
    p.add_argument('--timeout', type=float, default=7.0, metavar='SEC',
                   help='Max seconds to record each utterance (default: 7).')
    p.add_argument('--language', default='en', metavar='LANG',
                   help='Language code passed to STT backends (default: en).')
    p.add_argument('--tests', nargs='+', type=int, metavar='N',
                   help='Run only these test numbers, e.g. --tests 1 3 5.')
    p.add_argument('--skip-ready', action='store_true',
                   help='Skip the "Press ENTER to record" prompt (auto-start).')
    p.add_argument('--no-log', action='store_true',
                   help='Do not write a log file.')
    return p.parse_args()


def main() -> None:
    args = build_args()

    # ── Log file ──────────────────────────────────────────────────────────
    log_path = ''
    tee = None
    if not args.no_log:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(_TESTS_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'voice_test_{ts}.log')
        tee = _Tee(log_path)

    try:
        _run(args, log_path)
    finally:
        if tee:
            tee.restore()


def _run(args: argparse.Namespace, log_path: str) -> None:
    # ── Banner ─────────────────────────────────────────────────────────────
    print()
    print(SEP2)
    print(bld('  Sara Voice Pipeline — End-to-End Real Microphone Test'))
    print(SEP2)
    print(f'  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'  Mode    : {"EXECUTE (real Excel actions)" if args.execute else "PARSE ONLY (safe)"}')
    print(f'  Timeout : {args.timeout}s per utterance')
    print(f'  Language: {args.language}')
    print()

    # ── Pre-flight ─────────────────────────────────────────────────────────
    if not preflight_check():
        sys.exit(1)
    print()

    if args.execute:
        print(ylw('  ⚠  --execute is ON — spoken commands will run against Excel.'))
        print(ylw('     Make sure a test Excel file is ready or accept default files.'))
        print()

    # ── Select tests ───────────────────────────────────────────────────────
    selected: list[TestCase]
    if args.tests:
        num_set = set(args.tests)
        selected = [t for t in DEFAULT_TESTS if t.num in num_set]
        if not selected:
            print(red(f'  No tests match --tests {args.tests}'))
            sys.exit(1)
    else:
        selected = DEFAULT_TESTS

    # Show test plan
    print(f'  Running {bld(str(len(selected)))} test(s):')
    for t in selected:
        print(f'    {t.num}.  Say: {bld(repr(t.say))}  {dim("→")}  {dim(t.hint)}')
    print()

    # ── Run each test ──────────────────────────────────────────────────────
    results: list[TestResult] = []
    for idx, case in enumerate(selected, 1):
        print(SEP)
        print(bld(f'  Test {idx}/{len(selected)}  ▸  Say: "{case.say}"'))
        if case.hint:
            print(dim(f'  ({case.hint})'))
        print()

        _wait_for_enter(args.skip_ready)
        print(f'  {mgn("● RECORDING")} — speak now', flush=True)
        print()

        r = run_one_test(case, args)
        results.append(r)

        # Per-test verdict
        print()
        if r.overall_pass:
            print(f'  {grn("PASS ✓")}')
        else:
            fails = []
            if not r.voice_ok:      fails.append('no voice')
            if not r.stt_ok:        fails.append('STT empty/error')
            if not r.intent_match:  fails.append(f'intent={r.intent or "none"}')
            if not r.slot_complete: fails.append('slot gap')
            if not r.action_ok:     fails.append(f'action={r.action.get("type","?") if r.action else "none"}')
            print(f'  {red("FAIL ✗")}  →  {", ".join(fails)}')

        # Pause between tests (unless last)
        if idx < len(selected):
            print()
            time.sleep(0.5)

    # ── Summary ────────────────────────────────────────────────────────────
    print_summary(results, args, log_path)


if __name__ == '__main__':
    main()
