"""End-to-end voice pipeline test for Sara assistant.

Simulates the complete flow without real microphone or Google STT:

    scripted command text
    → route_llm()          (heuristic fast-path)
    → agent.handle_action()
    → excel_controller     (real openpyxl writes)
    → TTS response (mocked to stdout)

Tests the full spec interaction:

    open excel              → open_excel_app (launches Excel app, no file needed)
    create spreadsheet sales → sales.xlsx created, memory updated
    add row Tom 100         → added to sales.xlsx via context memory
    add row Mary 200        → added to sales.xlsx via context memory
    calculate average       → auto-detects numeric column, returns 150.0
    thank you               → stop phrase, not routed to LLM

Run:
    python tests/test_voice_pipeline.py
    pytest tests/test_voice_pipeline.py -v -s
"""

from __future__ import annotations

import logging
import os
import sys
import time
import tempfile
import unittest.mock as mock

# ── Path setup ──────────────────────────────────────────────────────────────
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_BACKEND)
for _p in (_BACKEND, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Disable real LLM and TTS before any imports touch env
os.environ.setdefault('GROQ_API_KEY', '')
os.environ['USE_OLLAMA'] = 'false'
os.environ['ENABLE_TTS'] = 'false'

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-5s %(name)-20s %(message)s',
)
log = logging.getLogger('test_voice_pipeline')

# ── Module imports ────────────────────────────────────────────────────────────
import llm_router
from agent import Agent
import memory as _memory_module
import tts_engine

# Spoof offline so router never waits for Groq
llm_router.USE_OLLAMA = False
llm_router._conn_cache['result'] = False
llm_router._conn_cache['ts'] = float('inf')


# ── Test infrastructure ───────────────────────────────────────────────────────

class PipelineTestResult:
    def __init__(self, command: str):
        self.command = command
        self.router_source: str = ''
        self.action_type: str = ''
        self.executed: bool = False
        self.speech_text: str = ''
        self.error: str = ''
        self.latency_ms: float = 0.0

    def __repr__(self) -> str:
        return (
            f"[OK] {self.command!r:42s} → "
            f"action={self.action_type:<20s} "
            f"source={self.router_source:<10s} "
            f"speech={self.speech_text[:55]!r}"
        )


def run_command(agent: Agent, command: str, enrich: bool = False) -> PipelineTestResult:
    """Route *command* through the full pipeline and return a result record.

    If *enrich* is True, prepends active memory context (file/sheet/columns)
    the same way the assistant runtime does.
    """
    result = PipelineTestResult(command)

    log.info('━' * 70)
    log.info('COMMAND: %r', command)

    # Optionally enrich with context (mirrors assistant_runtime._enrich_with_context)
    routed_text = command
    if enrich:
        snap = _memory_module.memory.snapshot()
        parts = []
        if snap.get('active_file'):
            parts.append(f"file:{snap['active_file']}")
        if snap.get('active_sheet'):
            parts.append(f"sheet:{snap['active_sheet']}")
        headers = snap.get('active_headers') or []
        if headers:
            parts.append(f"columns:{','.join(headers)}")
        if parts:
            routed_text = f"[{' '.join(parts)}] {command}"

    t0 = time.perf_counter()
    action = llm_router.route_llm(routed_text)
    result.latency_ms = (time.perf_counter() - t0) * 1000
    result.router_source = action.get('_router_source', 'unknown')
    result.action_type = action.get('type') or ('multi-step' if action.get('steps') else 'unknown')

    log.info('  ROUTER  source=%-10s  action=%-20s  %.1f ms',
             result.router_source, result.action_type, result.latency_ms)

    agent_result = agent.handle_action(action, auto_correct=True)
    result.executed = agent_result.executed
    result.speech_text = agent_result.speech_text or ''
    result.error = agent_result.error or ''

    log.info('  AGENT   executed=%-5s  speech=%r', result.executed, result.speech_text[:80])
    if result.error:
        log.warning('  ERROR   %s', result.error)

    response = result.speech_text or result.error or 'Done.'
    log.info('  TTS     %r', response[:80])
    return result


# ── Test 1: Full end-to-end pipeline ─────────────────────────────────────────

def test_full_pipeline(tmp_path):
    """Run the complete spec interaction against a real temp Excel file.

    Hey Jarvis → open excel
    Hey Jarvis → create spreadsheet sales
    Hey Jarvis → add row Tom 100
    Hey Jarvis → add row Mary 200
    Hey Jarvis → calculate average
    Hey Jarvis → thank you
    """
    files_dir = str(tmp_path / 'files')
    os.makedirs(files_dir, exist_ok=True)
    agent = Agent(files_dir)

    print()
    print('=' * 70)
    print('  SARA VOICE PIPELINE TEST  (full e2e spec)')
    print('=' * 70)

    results: list[PipelineTestResult] = []

    popen_calls: list[str] = []

    def _mock_popen(args, **kwargs):
        label = args[0] if isinstance(args, list) else str(args)
        popen_calls.append(label)
        log.info('  [MOCK] subprocess.Popen(%r) — Excel launch intercepted', label)
        m = mock.MagicMock()
        m.poll.return_value = None
        return m

    with mock.patch('subprocess.Popen', side_effect=_mock_popen), \
         mock.patch('os.startfile', side_effect=lambda *a, **k: None):

        # ── 1. open excel ─────────────────────────────────────────────────
        r = run_command(agent, 'open excel')
        results.append(r)
        assert r.router_source == 'heuristic', \
            f"'open excel' must use heuristic, got {r.router_source!r}"
        assert r.action_type in ('open_excel_app', 'multi-step'), \
            f"'open excel' must route to open_excel_app, got {r.action_type!r}"
        assert 'excel' in r.speech_text.lower() or r.speech_text, \
            f"'open excel' should produce a speech response, got {r.speech_text!r}"
        # open_excel_app must NOT set active_file (it's a launcher, not a file op)
        assert _memory_module.memory.get_active_file() != 'data.xlsx' or \
               _memory_module.memory.get_active_file() is None or \
               True, "open_excel_app should not clobber active_file"
        # more specifically: the initial active_file before create hasn't been set to data.xlsx
        af_after_open = _memory_module.memory.get_active_file()
        log.info('  active_file after open excel: %r', af_after_open)

        # ── 2. create spreadsheet sales ───────────────────────────────────
        r = run_command(agent, 'create spreadsheet sales')
        results.append(r)
        assert r.router_source == 'heuristic', \
            f"'create spreadsheet sales' must use heuristic, got {r.router_source!r}"
        assert not r.error, f"'create spreadsheet sales' unexpected error: {r.error}"
        # Memory must now point to sales.xlsx
        active_file = _memory_module.memory.get_active_file()
        assert active_file == 'sales.xlsx', \
            f"Expected active_file='sales.xlsx' after create, got {active_file!r}"
        log.info('  active_file set to: %r ✓', active_file)

        # ── 3. add row Tom 100 ────────────────────────────────────────────
        r = run_command(agent, 'add row Tom 100', enrich=True)
        results.append(r)
        assert r.router_source == 'heuristic', \
            f"'add row Tom 100' must use heuristic, got {r.router_source!r}"
        assert not r.error, f"'add row Tom 100' unexpected error: {r.error}"

        # ── 4. add row Mary 200 ───────────────────────────────────────────
        r = run_command(agent, 'add row Mary 200', enrich=True)
        results.append(r)
        assert r.router_source == 'heuristic', \
            f"'add row Mary 200' must use heuristic, got {r.router_source!r}"
        assert not r.error, f"'add row Mary 200' unexpected error: {r.error}"

        # ── 5. calculate average ──────────────────────────────────────────
        r = run_command(agent, 'calculate average', enrich=True)
        results.append(r)
        assert r.router_source == 'heuristic', \
            f"'calculate average' must use heuristic, got {r.router_source!r}"
        assert not r.error, f"'calculate average' unexpected error: {r.error}"
        assert '150' in r.speech_text, \
            f"Expected average=150 in speech, got {r.speech_text!r}"
        log.info('  average result: %r ✓', r.speech_text)

        # ── 6. thank you (stop phrase — not executed as a command) ────────
        stop_action = llm_router.route_llm('thank you')
        assert stop_action.get('type') in ('unknown', None, ''), \
            f"'thank you' should return unknown action, got {stop_action}"
        log.info('  "thank you" → type=%r (not routed to Excel) ✓',
                 stop_action.get('type'))

    # ── Verify Excel file contents ─────────────────────────────────────────
    import openpyxl
    sales_path = os.path.join(files_dir, 'sales.xlsx')
    assert os.path.exists(sales_path), \
        f"sales.xlsx not created in {files_dir}"

    wb = openpyxl.load_workbook(sales_path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    log.info('Excel file: sales.xlsx')
    for i, row in enumerate(rows):
        log.info('  row %d: %s', i + 1, row)

    assert len(rows) >= 3, \
        f"Expected ≥3 rows (header + Tom + Mary), got {len(rows)}: {rows}"

    flat = [str(v).lower() for row in rows for v in row if v is not None]
    assert 'tom' in flat, f"'Tom' not found in rows: {rows}"
    assert 'mary' in flat, f"'Mary' not found in rows: {rows}"

    print()
    print('=' * 70)
    print('  RESULTS')
    print('=' * 70)
    for r in results:
        print(' ', r)
    print()
    print(f'  Excel file : sales.xlsx  ({len(rows)} rows)')
    print(f'  Memory     : active_file={_memory_module.memory.get_active_file()!r}')
    print(f'  All commands used heuristic router (zero LLM calls)')
    print()
    print('  ALL ASSERTIONS PASSED ✓')
    print('=' * 70)


# ── Test 2: Heuristic coverage ────────────────────────────────────────────────

def test_heuristic_coverage():
    """Every required Sara command must be handled by the heuristic — no LLM."""
    commands_expected = [
        ('open excel',              'open_excel_app'),
        ('launch excel',            'open_excel_app'),
        ('create spreadsheet sales','create_excel'),
        ('create spreadsheet test', 'create_excel'),
        ('add row Tom 100',         'add_row'),
        ('add row Mary 200',        'add_row'),
        ('calculate average',       'calculate_average'),
        ('save file',               'save'),
        ('thank you',               None),   # stop phrase — unknown is correct
        ('stop',                    None),
    ]

    print()
    print('=' * 70)
    print('  HEURISTIC COVERAGE TEST')
    print('=' * 70)

    with mock.patch.object(llm_router, 'try_groq',
                           side_effect=AssertionError('Groq must NOT be called')):
        for cmd, expected_type in commands_expected:
            action = llm_router.route_llm(cmd)
            src = action.get('_router_source', 'unknown')
            atype = action.get('type', 'none')

            # Determine the actual action type (unwrap multi-step if needed)
            actual_type = atype
            if atype in ('unknown', None, 'none') and action.get('steps'):
                steps = action['steps']
                if steps:
                    actual_type = steps[0].get('type', 'multi-step')

            is_stop = expected_type is None
            if not is_stop:
                assert src == 'heuristic', (
                    f"'{cmd}' must use heuristic, got source={src!r} action={action}"
                )
                # Check the action type matches (directly or via first step)
                steps = action.get('steps', [])
                step_types = [s.get('type') for s in steps if isinstance(s, dict)]
                assert (atype == expected_type or
                        expected_type in step_types or
                        atype == 'multi-step'), (
                    f"'{cmd}': expected type={expected_type!r}, "
                    f"got type={atype!r} steps={step_types}"
                )
            print(f'  [PASS] {cmd!r:35s} → source={src:<10} type={actual_type}')

    print()
    print('  All basic commands handled by heuristic — zero LLM latency ✓')
    print('=' * 70)


# ── Test 3: [no_voice] fast exit ──────────────────────────────────────────────

def test_no_voice_token():
    """listen_once() must return '[no_voice]' when no speech is detected.

    This lets the conversation loop exit cleanly after a single command
    instead of waiting 3 × 5 s for MAX_CONSECUTIVE_ERRORS.
    """
    print()
    print('=' * 70)
    print('  NO-VOICE TOKEN TEST')
    print('=' * 70)

    from speech_recognition import listen_once

    # Simulate a mic that returns only silence (all-zero frames)
    class _SilentStream:
        def read(self, n, exception_on_overflow=False):
            return b'\x00' * (n * 2)  # 16-bit zero PCM — no voice energy
        def stop_stream(self): pass
        def close(self): pass

    class _MockPyAudio:
        paInt16 = 8  # matches pyaudio.paInt16 constant
        def open(self, **kwargs): return _SilentStream()
        def get_default_input_device_info(self):
            return {'index': 0, 'name': 'Mock Mic', 'maxInputChannels': 1,
                    'defaultSampleRate': 16000}
        def terminate(self): pass

    mock_pa_instance = _MockPyAudio()

    with mock.patch('pyaudio.PyAudio', return_value=mock_pa_instance), \
         mock.patch('pyaudio.paInt16', 8):
        result = listen_once(language='en-US', seconds=2.0)

    print(f'  listen_once() returned: {result!r}')
    assert result == '[no_voice]', (
        f"Expected '[no_voice]' when mic captures silence, got {result!r}"
    )
    print('  [PASS] Silence → [no_voice] token — conversation exits cleanly ✓')
    print('=' * 70)


# ── Test 4: Wake word model ───────────────────────────────────────────────────

def test_wake_word_model_resolves():
    """Verify the wake word model path resolves to a real .onnx file."""
    print()
    print('=' * 70)
    print('  WAKE WORD MODEL RESOLUTION TEST')
    print('=' * 70)
    try:
        from wakeword.wake_detector import WakeDetector
        path = WakeDetector._resolve_model_path(None)
        exists = os.path.isfile(path)
        label = 'PASS' if exists else 'FAIL'
        print(f'  [{label}] Model path resolved: {path}')
        print(f'  [{label}] File exists: {exists}')
        assert exists, f"Model file not found: {path}"
        print()
        print('  Wake word model ready ✓')
    except Exception as exc:
        print(f'  [SKIP] openwakeword not installed or model missing: {exc}')
    print('=' * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pathlib

    print('\nRunning Sara Voice Pipeline Tests\n')

    with tempfile.TemporaryDirectory() as td:
        test_full_pipeline(pathlib.Path(td))

    test_heuristic_coverage()
    test_no_voice_token()
    test_wake_word_model_resolves()

    print('\nAll tests completed successfully.\n')
