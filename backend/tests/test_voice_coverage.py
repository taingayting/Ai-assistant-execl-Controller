"""test_voice_coverage.py — Dataset-driven voice pipeline coverage tests.

Loads every case from voice_test_dataset.json and runs it through the
pipeline correction → intent → low-confidence → slot layers without a
microphone.  Adding a new test case requires only editing the JSON file.

Run from the repo root:
    python -m pytest backend/tests/test_voice_coverage.py -v
    python -m pytest backend/tests/test_voice_coverage.py -v -k FA       # file/app only
    python -m pytest backend/tests/test_voice_coverage.py -v -k RC001    # single case
    python -m pytest backend/tests/test_voice_coverage.py --co -q        # list tests
    python -m pytest backend/tests/test_voice_coverage.py -v --tb=short  # short tracebacks
"""
from __future__ import annotations

import json
import os
import sys
import unittest

# ── Path bootstrap ──────────────────────────────────────────────────────────
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND   = os.path.dirname(_TESTS_DIR)
for _p in (_BACKEND, os.path.dirname(_BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Dataset loading ─────────────────────────────────────────────────────────
_DATASET_PATH = os.path.join(_TESTS_DIR, 'voice_test_dataset.json')

def _load_cases() -> list[dict]:
    with open(_DATASET_PATH, encoding='utf-8') as fh:
        data = json.load(fh)
    return data['cases']

_ALL_CASES: list[dict] = _load_cases()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_pipeline(raw: str) -> dict:
    """Run *raw* through correction → intent → low-conf → slot layers."""
    from pipeline_sanitizer import (
        correct_transcript,
        detect_command_intent,
        is_low_confidence_transcript,
        check_command_slots,
    )
    corrected = correct_transcript(raw)
    intent    = detect_command_intent(corrected)
    low_conf  = is_low_confidence_transcript(corrected)
    slot_msg  = check_command_slots(intent, corrected) if intent else None
    return {
        'corrected': corrected,
        'intent':    intent,
        'low_conf':  low_conf,
        'slot_msg':  slot_msg,
    }


# ── Parametrized test class ──────────────────────────────────────────────────

def _make_test(case: dict):
    """Return a bound test method for *case*."""
    cid  = case['id']
    raw  = case['raw_stt']
    note = case.get('note', '')

    def test_fn(self: unittest.TestCase) -> None:  # type: ignore[type-arg]
        result = _run_pipeline(raw)
        corrected = result['corrected']
        intent    = result['intent']
        low_conf  = result['low_conf']
        slot_msg  = result['slot_msg']

        ctx = (
            f"\n  id={cid!r}  note={note!r}"
            f"\n  raw_stt    = {raw!r}"
            f"\n  corrected  = {corrected!r}"
            f"\n  intent     = {intent!r}"
            f"\n  low_conf   = {low_conf}"
            f"\n  slot_msg   = {slot_msg!r}"
        )

        # ── Check 1: transcript correction substring ─────────────────────
        substr = case.get('expected_corrected_contains')
        if substr is not None:
            self.assertIn(
                substr.lower(), corrected.lower(),
                f"[{cid}] corrected transcript missing {substr!r}{ctx}",
            )

        # ── Check 2: low-confidence filter ───────────────────────────────
        lc_expected = case.get('low_conf_expected')
        if lc_expected is not None:
            if lc_expected:
                self.assertTrue(
                    low_conf,
                    f"[{cid}] expected low_conf=True but filter passed{ctx}",
                )
            else:
                self.assertFalse(
                    low_conf,
                    f"[{cid}] expected low_conf=False but transcript was blocked{ctx}",
                )

        # ── Check 3: intent ──────────────────────────────────────────────
        expected_intent = case.get('expected_intent')
        if expected_intent is not None:
            self.assertEqual(
                intent, expected_intent,
                f"[{cid}] intent mismatch: got {intent!r}, expected {expected_intent!r}{ctx}",
            )
        # When expected_intent is null the intent check is skipped (LLM handles it)

        # ── Check 4: slot completeness ───────────────────────────────────
        slot_complete = case.get('slot_complete')
        if slot_complete is not None:
            if slot_complete:
                self.assertIsNone(
                    slot_msg,
                    f"[{cid}] expected no slot gap but got: {slot_msg!r}{ctx}",
                )
            else:
                self.assertIsNotNone(
                    slot_msg,
                    f"[{cid}] expected a slot clarification but got None{ctx}",
                )

    test_fn.__name__ = f'test_{cid}'
    test_fn.__doc__  = f'{cid}: {raw!r}  →  {case.get("expected_intent","?")!r}  ({note})'
    return test_fn


class TestVoiceCoverage(unittest.TestCase):
    """Dataset-driven coverage tests for the full Sara command set.

    Each method corresponds to one case in voice_test_dataset.json.
    To add a new case, append an entry to the JSON file.
    """


# Dynamically attach one test method per dataset case.
for _case in _ALL_CASES:
    _method = _make_test(_case)
    setattr(TestVoiceCoverage, _method.__name__, _method)


# ── Standalone summary runner ────────────────────────────────────────────────

def _run_summary() -> None:
    """Print a per-family coverage summary when run as a script."""
    import re

    # Colour helpers (TTY-aware)
    _TTY = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    def _c(code: str, t: str) -> str: return f'\033[{code}m{t}\033[0m' if _TTY else t
    grn = lambda t: _c('32', t)
    red = lambda t: _c('31', t)
    ylw = lambda t: _c('33', t)
    bld = lambda t: _c('1', t)
    dim = lambda t: _c('2', t)

    print()
    print('═' * 70)
    print(bld('  Sara Voice Coverage — Quick Pipeline Trace'))
    print('═' * 70)
    print(f'  Dataset: {_DATASET_PATH}')
    print(f'  Cases  : {len(_ALL_CASES)}')
    print()

    family_pass: dict[str, list] = {}
    family_fail: dict[str, list] = {}

    for case in _ALL_CASES:
        cid    = case['id']
        family = re.match(r'[A-Z]+', cid).group(0) if re.match(r'[A-Z]+', cid) else 'XX'
        result = _run_pipeline(case['raw_stt'])

        errors = []
        substr = case.get('expected_corrected_contains')
        if substr and substr.lower() not in result['corrected'].lower():
            errors.append(f"corrected missing {substr!r}")

        lc_exp = case.get('low_conf_expected')
        if lc_exp is not None:
            if lc_exp and not result['low_conf']:
                errors.append('expected low_conf=True')
            elif not lc_exp and result['low_conf']:
                errors.append('blocked unexpectedly')

        exp_intent = case.get('expected_intent')
        if exp_intent is not None and result['intent'] != exp_intent:
            errors.append(f"intent={result['intent']!r} want={exp_intent!r}")

        slot_complete = case.get('slot_complete')
        if slot_complete is True and result['slot_msg'] is not None:
            errors.append(f"slot gap: {result['slot_msg'][:50]!r}")
        elif slot_complete is False and result['slot_msg'] is None:
            errors.append('expected slot gap but got None')

        bucket = family_fail if errors else family_pass
        bucket.setdefault(family, []).append((cid, errors, result))

    all_families = sorted(set(list(family_pass) + list(family_fail)))
    total_pass = total_fail = 0

    for fam in all_families:
        p = len(family_pass.get(fam, []))
        f = len(family_fail.get(fam, []))
        total_pass += p
        total_fail += f
        sym = grn('PASS') if f == 0 else red('FAIL')
        print(f'  {fam:5s}  {sym}  {grn(str(p))} pass  {(red(str(f)) if f else dim("0")) + " fail":>10s}')
        for cid, errs, res in family_fail.get(fam, []):
            print(f'         {red("✗")} {cid}: {", ".join(errs)}')
            print(f'           raw={case["raw_stt"]!r:40s}  corrected={res["corrected"]!r}')

    total = total_pass + total_fail
    print()
    print('─' * 70)
    pct = int(100 * total_pass / total) if total else 0
    bar = grn('█' * (pct // 5)) + dim('░' * (20 - pct // 5))
    result_line = grn(f'{total_pass}/{total} PASS ({pct}%)') if total_fail == 0 else \
                  ylw(f'{total_pass}/{total} ({pct}%) — {total_fail} failing')
    print(f'  {bar}  {result_line}')
    print('═' * 70)
    print()


if __name__ == '__main__':
    _run_summary()
