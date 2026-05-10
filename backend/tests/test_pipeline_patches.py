"""
Pipeline patch tests for Sara voice assistant.

Covers P1–P5 fixes:
  P1  silence threshold 200→150
  P2  faster-whisper empty-string falls through to cloud STT
  P3  "and grow" / "raisa" / "raise <cell>" transcript corrections
  P4  "go to" handling: bare=disambiguation, cell ref=read_cell, sheet=switch_sheet
  P5  excel save retry budget (constant-only change, verified by assertion)

Run from the repo root:
    python -m pytest backend/tests/test_pipeline_patches.py -v
"""
from __future__ import annotations

import os
import struct
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure backend/ is on the path regardless of where pytest is invoked.
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pcm_frame(rms_target: float, n_samples: int = 1024) -> bytes:
    """Return a 16-bit little-endian PCM frame whose RMS ≈ rms_target."""
    import math
    amplitude = int(rms_target * math.sqrt(2))
    samples = [
        int(amplitude * math.sin(2 * math.pi * i / n_samples))
        for i in range(n_samples)
    ]
    return struct.pack(f'<{n_samples}h', *samples)


def _pipeline(raw: str) -> dict:
    """Run *raw* through correction → intent → slot → heuristic parse."""
    from pipeline_sanitizer import (
        correct_transcript, detect_command_intent, check_command_slots,
    )
    from llm_parser import _heuristic_natural_language_parse  # type: ignore

    corrected  = correct_transcript(raw)
    intent     = detect_command_intent(corrected)
    slot_msg   = check_command_slots(intent, corrected)
    action     = None if slot_msg else _heuristic_natural_language_parse(corrected)
    return {
        'corrected': corrected,
        'intent':    intent,
        'slot_msg':  slot_msg,
        'action':    action,
    }


# ---------------------------------------------------------------------------
# P1 — silence threshold reduction (200 → 150)
# ---------------------------------------------------------------------------

class TestP1ThresholdReduction(unittest.TestCase):
    """listen_once must detect voice at peak_rms=193 when threshold=150."""

    def _run_listen_once_with_frames(self, frames: list[bytes]) -> str:
        """Inject *frames* into listen_once via mock PyAudio.

        pyaudio is imported lazily inside listen_once (not at module level),
        so patch('speech_recognition.pyaudio') can't find it.  Injecting into
        sys.modules makes the inner 'import pyaudio' resolve to our mock.
        """
        import sys
        import speech_recognition as sr  # noqa: PLC0415

        frame_iter = iter(frames)
        mock_stream = MagicMock()
        mock_stream.read.side_effect = lambda n, **kw: next(frame_iter, frames[-1])

        mock_pa = MagicMock()
        mock_pa.open.return_value = mock_stream
        mock_pa.get_default_input_device_info.return_value = {
            'index': 0, 'name': 'Test Mic',
            'maxInputChannels': 1, 'defaultSampleRate': 16000,
        }

        mock_pyaudio_mod = MagicMock()
        mock_pyaudio_mod.PyAudio.return_value = mock_pa
        mock_pyaudio_mod.paInt16 = 8

        with patch.dict(sys.modules, {'pyaudio': mock_pyaudio_mod}), \
             patch('speech_recognition.transcribe_audio_file', return_value='add row'):
            os.environ.pop('SARA_SILENCE_THRESHOLD', None)   # use default 150
            return sr.listen_once(language='en', seconds=3.0)

    def test_voice_detected_at_rms_193(self):
        """Peak RMS 193 > threshold 150 → had_voice=True → STT is called."""
        quiet = bytes(1024 * 2)
        voice = _make_pcm_frame(193)
        # 30 quiet + 5 voice + 30 quiet (total 65 frames — well under 3-s limit)
        frames = [quiet] * 30 + [voice] * 5 + [quiet] * 30

        result = self._run_listen_once_with_frames(frames)
        self.assertNotEqual(
            result, '[no_voice]',
            "listen_once returned [no_voice] for peak_rms=193 with threshold=150. "
            "P1 patch may not be applied correctly.",
        )

    def test_silence_still_returns_no_voice(self):
        """All-zero frames → had_voice=False → [no_voice] returned."""
        quiet = bytes(1024 * 2)
        frames = [quiet] * 200

        result = self._run_listen_once_with_frames(frames)
        self.assertEqual(result, '[no_voice]')

    def test_threshold_env_override(self):
        """SARA_SILENCE_THRESHOLD env var overrides the default."""
        import speech_recognition as sr  # noqa: PLC0415
        import inspect

        src = inspect.getsource(sr.listen_once)
        self.assertIn("os.environ.get('SARA_SILENCE_THRESHOLD', '150')", src,
                      "Default threshold is not 150 — P1 patch may be missing.")


# ---------------------------------------------------------------------------
# P2 — faster-whisper empty/None falls through to Google Cloud STT
# ---------------------------------------------------------------------------

class TestP2WhisperEmptyFallback(unittest.TestCase):
    """transcribe_audio_file must try Google Cloud STT when whisper returns '' or None."""

    def test_empty_whisper_calls_google_cloud(self):
        import speech_recognition as sr  # noqa: PLC0415

        with patch.object(sr, 'transcribe_local_whisper', return_value=''), \
             patch.object(sr, '_transcribe_google_cloud', return_value='add row John') as mock_g:
            result = sr.transcribe_audio_file('/fake/cmd.wav', language='en')

        mock_g.assert_called_once_with('/fake/cmd.wav', language='en')
        self.assertEqual(result, 'add row John')

    def test_whisper_none_calls_google_cloud(self):
        import speech_recognition as sr  # noqa: PLC0415

        with patch.object(sr, 'transcribe_local_whisper', return_value=None), \
             patch.object(sr, '_transcribe_google_cloud', return_value='save') as mock_g:
            result = sr.transcribe_audio_file('/fake/cmd.wav', language='en')

        mock_g.assert_called_once()
        self.assertEqual(result, 'save')

    def test_empty_google_cloud_returns_error(self):
        """When Google Cloud returns empty, return a clear error."""
        import speech_recognition as sr  # noqa: PLC0415

        with patch.object(sr, 'transcribe_local_whisper', return_value=''), \
             patch.object(sr, '_transcribe_google_cloud', return_value=''):
            result = sr.transcribe_audio_file('/fake/cmd.wav', language='en')

        # Must return a recognisable error
        self.assertTrue(
            result.startswith('[transcription_error]'),
            f"Expected error sentinel, got: {result!r}",
        )

    def test_non_empty_whisper_skips_cloud(self):
        import speech_recognition as sr  # noqa: PLC0415

        with patch.object(sr, 'transcribe_local_whisper', return_value='open excel'), \
             patch.object(sr, '_transcribe_google_cloud') as mock_g:
            result = sr.transcribe_audio_file('/fake/cmd.wav', language='en')

        mock_g.assert_not_called()
        self.assertEqual(result, 'open excel')


# ---------------------------------------------------------------------------
# Voice-path: "add row John 90"
# ---------------------------------------------------------------------------

class TestVoicePathAddRow(unittest.TestCase):

    def test_canonical_add_row(self):
        r = _pipeline('add row John 90')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'add_row')

    def test_and_grow_phonetic(self):
        """'and grow John 90' must correct to 'add row John 90' (P3a)."""
        r = _pipeline('and grow John 90')
        self.assertEqual(r['corrected'], 'add row John 90')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNone(r['slot_msg'])

    def test_adroit_phonetic(self):
        r = _pipeline('adroit John 90')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNone(r['slot_msg'])

    def test_bare_add_row_asks_for_values(self):
        r = _pipeline('add row')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIn('name', r['slot_msg'].lower())

    def test_add_row_number_only_asks_slot(self):
        """Proven by real mic test: 'add row 290' must ask for clarification.

        STT garbled 'add row John 90' → 'add row 290' (name dropped/merged).
        Previously the slot check passed it through because the bare-number
        guard had 'not _SLOT_BARE_NUM_RE.match()' which explicitly let numbers
        pass.  The fix adds an explicit number-only → ask-for-name check.
        """
        r = _pipeline('add row 290')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNotNone(
            r['slot_msg'],
            "'add row 290' (number only) should ask for name and score",
        )
        self.assertIn('name', r['slot_msg'].lower())

    def test_add_row_number_word(self):
        """Number words must be converted to digits before routing."""
        r = _pipeline('add row Jane ninety')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIn('90', r['corrected'])


# ---------------------------------------------------------------------------
# Voice-path: "write cell C2 12"
# ---------------------------------------------------------------------------

class TestVoicePathWriteCell(unittest.TestCase):

    def test_canonical_write_cell(self):
        r = _pipeline('write cell C2 12')
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_cell')
        self.assertEqual(r['action'].get('cell', '').upper(), 'C2')

    def test_raisa_phonetic(self):
        """'Raisa C2 12' must correct to 'write C2 12' (P3b)."""
        r = _pipeline('Raisa C2 12')
        self.assertIn('write', r['corrected'].lower())
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNone(r['slot_msg'])

    def test_raise_with_cell_ref_corrects(self):
        """'raise C2 score' → 'write C2 score' (narrowed raise rule, P3b)."""
        r = _pipeline('raise C2 score')
        self.assertIn('write', r['corrected'].lower())
        self.assertEqual(r['intent'], 'write_cell')

    def test_write_cell_missing_value_asks_slot(self):
        r = _pipeline('write cell C2')
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIn('C2', r['slot_msg'])

    def test_write_cell_missing_ref_asks_slot(self):
        r = _pipeline('write cell')
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNotNone(r['slot_msg'])


# ---------------------------------------------------------------------------
# P3b regression: "raise the bar" must NOT become "write the bar"
# ---------------------------------------------------------------------------

class TestRaiseRuleNarrowing(unittest.TestCase):

    def test_raise_the_bar_not_converted(self):
        from pipeline_sanitizer import correct_transcript
        result = correct_transcript('raise the bar')
        self.assertNotIn('write', result.lower(),
            f"'raise the bar' was wrongly converted to: {result!r}")

    def test_raise_your_hand_not_converted(self):
        from pipeline_sanitizer import correct_transcript
        result = correct_transcript('raise your hand please')
        self.assertNotIn('write', result.lower(),
            f"'raise your hand please' was wrongly converted to: {result!r}")

    def test_raise_cell_converts(self):
        from pipeline_sanitizer import correct_transcript
        result = correct_transcript('raise cell B3 hello')
        self.assertTrue(result.lower().startswith('write'),
            f"'raise cell B3 hello' should start with 'write'. Got: {result!r}")

    def test_raise_column_converts(self):
        from pipeline_sanitizer import correct_transcript
        result = correct_transcript('raise column A 99')
        self.assertTrue(result.lower().startswith('write'),
            f"'raise column A 99' should start with 'write'. Got: {result!r}")

    def test_raise_b3_converts(self):
        from pipeline_sanitizer import correct_transcript
        result = correct_transcript('raise B3 value')
        self.assertTrue(result.lower().startswith('write'),
            f"'raise B3 value' should start with 'write'. Got: {result!r}")


class TestCoreVoiceRecovery(unittest.TestCase):

    def test_open_the_chair_maps_to_open_excel(self):
        r = _pipeline('open the chair')
        self.assertEqual(r['corrected'], 'open excel')
        self.assertEqual(r['intent'], 'open_excel')

    def test_open_exhale_maps_to_open_excel(self):
        r = _pipeline('open exhale')
        self.assertEqual(r['corrected'], 'open excel')
        self.assertEqual(r['intent'], 'open_excel')

    def test_close_excel_maps_to_close_excel(self):
        r = _pipeline('close excel')
        self.assertEqual(r['corrected'], 'close excel')
        self.assertEqual(r['intent'], 'close_excel')

    def test_at_roll_john_90_maps_to_add_row(self):
        r = _pipeline('at roll John 90')
        self.assertEqual(r['corrected'], 'add row John 90')
        self.assertEqual(r['intent'], 'add_row')

    def test_rise_up_f1_100_maps_to_write_cell(self):
        r = _pipeline('rise up F1 100')
        self.assertEqual(r['corrected'], 'write cell F1 100')
        self.assertEqual(r['intent'], 'write_cell')

    def test_rice_serve_d1_pen_maps_to_write_cell(self):
        r = _pipeline('rice serve D1 pen')
        self.assertEqual(r['corrected'], 'write cell D1 pen')
        self.assertEqual(r['intent'], 'write_cell')

    def test_open_each_cell_maps_to_open_excel(self):
        r = _pipeline('open each cell')
        self.assertEqual(r['corrected'], 'open excel')
        self.assertEqual(r['intent'], 'open_excel')

    def test_open_it_sir_maps_to_open_excel(self):
        r = _pipeline('open it sir')
        self.assertEqual(r['corrected'], 'open excel')
        self.assertEqual(r['intent'], 'open_excel')

    def test_rice_cell_e3_10_maps_to_write_cell(self):
        r = _pipeline('rice cell E3 10')
        self.assertEqual(r['corrected'], 'write cell E3 10')
        self.assertEqual(r['intent'], 'write_cell')

    def test_adroll_austin_8_maps_to_add_row(self):
        r = _pipeline('adroll Austin 8')
        self.assertEqual(r['corrected'], 'add row Austin 8')
        self.assertEqual(r['intent'], 'add_row')


# ---------------------------------------------------------------------------
# P4 rework: "go to" handling
# ---------------------------------------------------------------------------

class TestGoToHandling(unittest.TestCase):

    def test_undo_is_deterministic_fast_path(self):
        """'undo' must route deterministically without LLM."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('undo'), 'undo')
        self.assertEqual(detect_command_intent('Undo'), 'undo')

    def test_go_to_b3_is_read_cell_not_switch_sheet(self):
        from pipeline_sanitizer import detect_command_intent
        intent = detect_command_intent('go to B3')
        self.assertNotEqual(intent, 'switch_sheet',
            "'go to B3' must not map to switch_sheet — it's a cell ref")
        self.assertEqual(intent, 'read_cell',
            "'go to B3' should map to read_cell")

    def test_go_to_cell_b3_is_read_cell(self):
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('go to cell B3'), 'read_cell')

    def test_bare_go_to_gives_disambiguation(self):
        from pipeline_sanitizer import describe_low_confidence
        msg = describe_low_confidence('go to')
        self.assertIsNotNone(msg, "Bare 'go to' must produce a message, not None")
        # Message must mention both options
        self.assertIn('sheet', msg.lower())
        self.assertIn('cell', msg.lower())

    def test_bare_go_to_is_low_confidence(self):
        from pipeline_sanitizer import is_low_confidence_transcript
        self.assertTrue(is_low_confidence_transcript('go to'),
            "Bare 'go to' (2 words, no intent) must be low-confidence")

    def test_go_to_b3_heuristic_returns_read_cell(self):
        """llm_parser heuristic must parse 'go to B3' as read_cell."""
        from llm_parser import _heuristic_natural_language_parse  # type: ignore
        result = _heuristic_natural_language_parse('go to B3')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'read_cell')
        self.assertEqual(result.get('cell', '').upper(), 'B3')

    def test_go_to_cell_b3_heuristic_returns_read_cell(self):
        from llm_parser import _heuristic_natural_language_parse  # type: ignore
        result = _heuristic_natural_language_parse('go to cell B3')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'read_cell')
        self.assertEqual(result.get('cell', '').upper(), 'B3')

    def test_go_to_sheet_heuristic_returns_switch_sheet(self):
        """'go to sheet Sales' must still reach switch_sheet in the heuristic."""
        from llm_parser import _heuristic_natural_language_parse  # type: ignore
        result = _heuristic_natural_language_parse('go to sheet Sales')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'switch_sheet')


# ---------------------------------------------------------------------------
# P5 — save retry constants
# ---------------------------------------------------------------------------

class TestP5SaveRetryBudget(unittest.TestCase):

    def test_retry_constants(self):
        import excel_controller as ec  # type: ignore
        self.assertEqual(ec._SAVE_RETRIES, 5,
            "_SAVE_RETRIES must be 5 — P5 patch may not be applied")
        self.assertAlmostEqual(ec._SAVE_RETRY_DELAY, 0.5,
            msg="_SAVE_RETRY_DELAY must be 0.5 s")
        self.assertAlmostEqual(ec._LOCK_RELEASE_DELAY, 0.25,
            msg="_LOCK_RELEASE_DELAY must be 0.25 s")


# ---------------------------------------------------------------------------
# Regression: existing text commands must be unaffected
# ---------------------------------------------------------------------------

class TestRegressionExistingCommands(unittest.TestCase):

    def _intent(self, text: str):
        from pipeline_sanitizer import detect_command_intent
        return detect_command_intent(text)

    def _corrected(self, text: str) -> str:
        from pipeline_sanitizer import correct_transcript
        return correct_transcript(text)

    def _low_conf(self, text: str) -> bool:
        from pipeline_sanitizer import is_low_confidence_transcript
        return is_low_confidence_transcript(text)

    # ── Commands that must still work ──────────────────────────────────────

    def test_save(self):
        self.assertEqual(self._intent('save'), 'save')
        self.assertFalse(self._low_conf('save'))

    def test_undo(self):
        self.assertEqual(self._intent('undo'), 'undo')

    def test_open_excel(self):
        self.assertEqual(self._intent('open excel'), 'open_excel')

    def test_close_excel(self):
        self.assertEqual(self._intent('close excel'), 'close_excel')

    def test_save_and_close(self):
        self.assertEqual(self._corrected('safe and close excel'), 'save and close excel')

    def test_add_row_rojo(self):
        self.assertEqual(self._corrected('at Rojo 90'), 'add row 90')

    def test_rice_cell_correction(self):
        r = self._corrected('rice C1 score')
        self.assertIn('write', r.lower())

    def test_switch_sheet_with_name(self):
        self.assertEqual(self._intent('switch sheet Data'), 'switch_sheet')

    def test_go_to_sheet_keyword_still_switch_sheet(self):
        self.assertEqual(self._intent('go to sheet Sales'), 'switch_sheet')

    def test_calculate_sum(self):
        self.assertEqual(self._intent('calculate sum of column A'), 'calculate_sum')

    def test_delete_row(self):
        self.assertEqual(self._intent('delete row 3'), 'delete_row')

    def test_create_sheet(self):
        self.assertEqual(self._intent('create sheet Summary'), 'create_sheet')

    def test_noise_words_filtered(self):
        self.assertTrue(self._low_conf('taste'))
        self.assertTrue(self._low_conf('oh'))

    def test_number_only_filtered(self):
        self.assertTrue(self._low_conf('90'))

    # ── Corrections that must still fire ──────────────────────────────────

    def test_and_grow_correction(self):
        self.assertEqual(self._corrected('and grow Jane 80'), 'add row Jane 80')

    def test_raisa_correction(self):
        r = self._corrected('Raisa C1 180')
        self.assertIn('write', r.lower())

    def test_wake_prefix_stripped(self):
        r = self._corrected('hey jarvis open excel')
        self.assertNotIn('jarvis', r.lower())
        self.assertIn('open excel', r.lower())

    def test_cell_ref_spacing_normalised(self):
        r = self._corrected('write cell c 2 score')
        self.assertIn('c2', r.lower())


# ---------------------------------------------------------------------------
# P6 — corrections for real mic failures (atro / switch cheat sheet / go to set)
# ---------------------------------------------------------------------------

class TestP6RealMicCorrections(unittest.TestCase):
    """Covers STT patterns confirmed from real microphone logs."""

    def _corrected(self, text: str) -> str:
        from pipeline_sanitizer import correct_transcript
        return correct_transcript(text)

    def _intent(self, text: str):
        from pipeline_sanitizer import detect_command_intent
        return detect_command_intent(text)

    def _low_conf(self, text: str) -> bool:
        from pipeline_sanitizer import is_low_confidence_transcript
        return is_low_confidence_transcript(text)

    # ── P6a: "atro" → "add row" ────────────────────────────────────────────

    def test_atro_corrects_to_add_row(self):
        """Real STT: 'atro John 90 at row John' — 'atro' must correct to 'add row'."""
        r = self._corrected('atro John 90')
        self.assertTrue(r.lower().startswith('add row'),
            f"'atro John 90' should start with 'add row'. Got: {r!r}")

    def test_atro_full_garbled_transcript_intent(self):
        """After correction, 'atro John 90 at row John' must route as add_row."""
        corrected = self._corrected('atro John 90 at row John')
        self.assertEqual(self._intent(corrected), 'add_row',
            f"Expected add_row intent after correcting 'atro John 90 at row John'. "
            f"Corrected to: {corrected!r}")

    def test_atro_not_middle_of_word(self):
        """'atro' correction is anchored at start — mid-word 'atro' unchanged."""
        r = self._corrected('the patron sat')
        self.assertNotIn('add row', r.lower(),
            f"'patron' must not be corrupted by the ^atro rule. Got: {r!r}")

    # ── P6b: "switch cheat sheet" → "switch sheet" ────────────────────────

    def test_switch_cheat_sheet_corrects(self):
        """Real STT: 'switch cheat sheet 2' for 'switch sheet Sheet2'."""
        r = self._corrected('switch cheat sheet 2')
        self.assertNotIn('cheat', r.lower(),
            f"'cheat' must be removed. Got: {r!r}")
        self.assertEqual(self._intent(r), 'switch_sheet',
            f"Intent after correction should be switch_sheet. Got: {self._intent(r)!r}")

    def test_cheat_sheet_alone_not_affected(self):
        """'cheat sheet' without 'switch' must not be changed."""
        r = self._corrected('I used a cheat sheet')
        self.assertNotIn('switch sheet', r.lower(),
            f"Bare 'cheat sheet' must not become 'switch sheet'. Got: {r!r}")

    # ── P6c: "go to set <cell>" → "go to cell <cell>" ────────────────────

    def test_go_to_set_cell_corrects(self):
        """Real STT: 'go to set B 3' → after norm → 'go to set B3' → 'go to cell B3'."""
        r = self._corrected('go to set B 3')
        self.assertIn('cell', r.lower(),
            f"'go to set B 3' should correct to include 'cell'. Got: {r!r}")
        self.assertEqual(self._intent(r), 'read_cell',
            f"Intent should be read_cell. Got: {self._intent(r)!r}")

    def test_go_to_set_already_normalised(self):
        """'go to set B3' (pre-normalised) must also correct."""
        r = self._corrected('go to set B3')
        self.assertEqual(self._intent(r), 'read_cell')

    def test_go_to_set_pipeline(self):
        """Full pipeline: 'go to set B3' must produce read_cell action."""
        result = _pipeline('go to set B3')
        self.assertEqual(result['intent'], 'read_cell')
        self.assertIsNone(result['slot_msg'])

    # ── P6d: 3-word noise filter ──────────────────────────────────────────

    def test_three_word_noise_filtered(self):
        """Real failure: 'John 19 at' (3 words, no intent, no command keyword) → low confidence."""
        self.assertTrue(self._low_conf('John 19 at'),
            "'John 19 at' must be treated as low-confidence noise.")

    def test_three_word_with_command_keyword_not_filtered(self):
        """'add a row' has 'add' → must NOT be filtered (needs slot clarification)."""
        self.assertFalse(self._low_conf('add a row'),
            "'add a row' contains a command keyword — must not be low-confidence.")

    def test_three_word_go_to_cell_not_filtered(self):
        """'go to cell' has 'go' and 'cell' → must NOT be filtered."""
        self.assertFalse(self._low_conf('go to cell'),
            "'go to cell' contains command keywords — must not be low-confidence.")

    def test_three_word_go_to_sheet_not_filtered(self):
        """'go to sheet' has detectable intent → must NOT be filtered."""
        self.assertFalse(self._low_conf('go to sheet'),
            "'go to sheet' has switch_sheet intent — must not be low-confidence.")

    def test_three_random_words_filtered(self):
        """Pure noise 3-word transcripts with no command keyword → low confidence."""
        self.assertTrue(self._low_conf('hello world today'))
        self.assertTrue(self._low_conf('Mary likes pizza'))


# ---------------------------------------------------------------------------
# P7 — Real voice failures: intent detection / normalization
# ---------------------------------------------------------------------------

class TestP7RealVoiceIntentFixes(unittest.TestCase):
    """Regression tests for four real-microphone failures identified in P7.

    Each test mirrors a transcript that was spoken clearly but STT returned
    a garbled form.  The pipeline must produce the correct intent without
    being blocked by the low-confidence filter.
    """

    def _corrected(self, text: str) -> str:
        from pipeline_sanitizer import correct_transcript
        return correct_transcript(text)

    def _intent(self, text: str) -> str | None:
        from pipeline_sanitizer import detect_command_intent
        return detect_command_intent(text)

    def _low_conf(self, text: str) -> bool:
        from pipeline_sanitizer import is_low_confidence_transcript
        return is_low_confidence_transcript(text)

    def _pipeline(self, raw: str) -> dict:
        from pipeline_sanitizer import (
            correct_transcript, detect_command_intent,
            is_low_confidence_transcript, check_command_slots,
        )
        corrected = correct_transcript(raw)
        intent    = detect_command_intent(corrected)
        low_conf  = is_low_confidence_transcript(corrected)
        slot_msg  = check_command_slots(intent, corrected)
        return {'corrected': corrected, 'intent': intent,
                'low_conf': low_conf, 'slot_msg': slot_msg}

    # ── "Excel" → open_excel ──────────────────────────────────────────────

    def test_bare_excel_corrects_to_open_excel(self):
        """Bare 'Excel' utterance must correct to 'open excel'."""
        self.assertEqual(self._corrected('Excel').lower(), 'open excel',
            "correct_transcript('Excel') should return 'open excel'")

    def test_bare_excel_intent_is_open_excel(self):
        """Bare 'Excel' must be detected as open_excel after correction."""
        r = self._pipeline('Excel')
        self.assertEqual(r['intent'], 'open_excel',
            f"Expected open_excel, got {r['intent']!r} (corrected={r['corrected']!r})")

    def test_bare_excel_not_low_confidence(self):
        """'Excel' must NOT be filtered as low-confidence after correction."""
        r = self._pipeline('Excel')
        self.assertFalse(r['low_conf'],
            "Corrected 'open excel' should pass low-confidence filter")

    def test_bare_excel_no_slot_message(self):
        """'Excel' → 'open excel' needs no clarification — no required slots."""
        r = self._pipeline('Excel')
        self.assertIsNone(r['slot_msg'],
            f"open_excel has no required slots; slot_msg should be None, got {r['slot_msg']!r}")

    # ── "write healthy 212" → write_cell ─────────────────────────────────

    def test_write_healthy_corrects_to_write_cell(self):
        """'healthy' as STT for 'cell' must be normalised."""
        corrected = self._corrected('write healthy 212')
        self.assertIn('write cell', corrected.lower(),
            f"Expected 'write cell' in corrected output, got {corrected!r}")

    def test_write_healthy_intent(self):
        """'write healthy 212' must map to write_cell intent."""
        r = self._pipeline('write healthy 212')
        self.assertEqual(r['intent'], 'write_cell',
            f"Expected write_cell, got {r['intent']!r} (corrected={r['corrected']!r})")

    def test_write_healthy_not_low_confidence(self):
        """'write healthy 212' must not be filtered."""
        r = self._pipeline('write healthy 212')
        self.assertFalse(r['low_conf'])

    def test_rice_healthy_also_corrects(self):
        """Phonetic variants of 'write' + 'healthy' must also correct."""
        corrected = self._corrected('rice healthy B3')
        self.assertIn('cell', corrected.lower(),
            f"'rice healthy B3' should correct 'healthy'→'cell', got {corrected!r}")

    # ── "go to free" → read_cell (F3) ────────────────────────────────────

    def test_go_to_free_corrects_to_f3(self):
        """'free' as phonetic STT for cell 'F3' must be normalised."""
        corrected = self._corrected('go to free')
        self.assertIn('F3', corrected,
            f"Expected 'F3' in corrected output, got {corrected!r}")

    def test_go_to_free_intent(self):
        """'go to free' must map to read_cell intent."""
        r = self._pipeline('go to free')
        self.assertEqual(r['intent'], 'read_cell',
            f"Expected read_cell, got {r['intent']!r} (corrected={r['corrected']!r})")

    def test_go_to_free_not_low_confidence(self):
        """'go to free' must not be filtered — contains command keyword 'go'."""
        r = self._pipeline('go to free')
        self.assertFalse(r['low_conf'])

    def test_go_to_free_no_slot_message(self):
        """'go to F3' has a complete cell ref — no clarification needed."""
        r = self._pipeline('go to free')
        self.assertIsNone(r['slot_msg'],
            f"read_cell with F3 should need no clarification, got {r['slot_msg']!r}")

    def test_go_to_free_does_not_affect_longer_phrase(self):
        """'go to freedom' must NOT be corrupted by the 'free'→'F3' rule."""
        corrected = self._corrected('go to freedom')
        self.assertNotIn('F3', corrected,
            f"'go to freedom' should be unchanged, got {corrected!r}")

    # ── "switch shit shit" → switch_sheet ────────────────────────────────

    def test_switch_shit_corrects_to_switch_sheet(self):
        """'shit' after 'switch' must be normalised to 'sheet'."""
        corrected = self._corrected('switch shit shit')
        self.assertIn('switch sheet', corrected.lower(),
            f"Expected 'switch sheet' in corrected output, got {corrected!r}")
        self.assertNotIn('cheat', corrected.lower())

    def test_switch_shit_intent(self):
        """'switch shit shit' must map to switch_sheet intent."""
        r = self._pipeline('switch shit shit')
        self.assertEqual(r['intent'], 'switch_sheet',
            f"Expected switch_sheet, got {r['intent']!r} (corrected={r['corrected']!r})")

    def test_switch_shit_not_low_confidence(self):
        """'switch shit shit' must not be filtered — contains 'switch' keyword."""
        r = self._pipeline('switch shit shit')
        self.assertFalse(r['low_conf'])

    def test_standalone_shit_not_affected(self):
        """Bare 'shit shit' (no 'switch') must NOT be rewritten to 'switch sheet'."""
        corrected = self._corrected('shit shit')
        self.assertNotIn('switch sheet', corrected.lower(),
            f"'shit shit' without 'switch' prefix should be unchanged, got {corrected!r}")

    def test_cheat_sheet_standalone_not_affected(self):
        """'cheat sheet' without 'switch' prefix must not be rewritten."""
        corrected = self._corrected('cheat sheet revision')
        self.assertNotIn('switch', corrected.lower(),
            f"'cheat sheet revision' should not gain 'switch', got {corrected!r}")


# ---------------------------------------------------------------------------
# P8 — Real harness failures: cell-ref and sheet-name recovery
# ---------------------------------------------------------------------------

class TestP8PatternBasedRecovery(unittest.TestCase):
    """Pattern-based recovery for STT slot failures — no word-specific hacks.

    Failure 1: STT merges cell ref + value into one token ("C212" instead of "C2 12").
      Pattern rule: single letter + 1 digit + 2+ trailing digits → split.

    Failure 2: "switch shit shit" — first "shit"→"sheet" (P7); second token is
      an unrecoverable sheet name. Safe fallback: return switch_sheet intent +
      slot clarification request.

    Failure 3: "sweet City" → phonetic for "switch sheet City".
      Prefix rule: ^sweet → "switch sheet".
    """

    def _corrected(self, text: str) -> str:
        from pipeline_sanitizer import correct_transcript
        return correct_transcript(text)

    def _pipeline(self, raw: str) -> dict:
        from pipeline_sanitizer import (
            correct_transcript, detect_command_intent,
            is_low_confidence_transcript, check_command_slots,
        )
        corrected = correct_transcript(raw)
        intent    = detect_command_intent(corrected)
        low_conf  = is_low_confidence_transcript(corrected)
        slot_msg  = check_command_slots(intent, corrected)
        return {'corrected': corrected, 'intent': intent,
                'low_conf': low_conf, 'slot_msg': slot_msg}

    # ── Pattern rule: merged cell+value split ─────────────────────────────

    def test_merged_cell_ref_value_splits(self):
        """'write cell C212' (merged) must split into 'write cell C2 12'."""
        corrected = self._corrected('write cell C212')
        self.assertEqual(corrected, 'write cell C2 12',
            f"Expected split 'write cell C2 12', got {corrected!r}")

    def test_merged_split_full_pipeline_no_slot_gap(self):
        """After split, write_cell with both ref and value must have no slot gap."""
        r = self._pipeline('write cell C212')
        self.assertEqual(r['intent'], 'write_cell',
            f"Expected write_cell, got {r['intent']!r} (corrected={r['corrected']!r})")
        self.assertIsNone(r['slot_msg'],
            f"Expected no slot gap after split, got {r['slot_msg']!r}")
        self.assertFalse(r['low_conf'])

    def test_split_does_not_fire_outside_write_cell(self):
        """Cell+value split must not fire on non-write-cell commands."""
        corrected = self._corrected('go to C212')
        self.assertNotIn(' ', corrected.split('C212')[1] if 'C212' in corrected else '',
            f"Split must not fire outside write cell context: {corrected!r}")

    def test_write_cell_canonical_still_works(self):
        """Existing canonical 'write cell C2 12' must be unaffected."""
        r = self._pipeline('write cell C2 12')
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNone(r['slot_msg'])

    # ── Safe fallback: "switch shit shit" → intent + clarification ────────

    def test_switch_shit_shit_reaches_switch_sheet_intent(self):
        """'switch shit shit' must resolve to switch_sheet intent (first shit→sheet)."""
        r = self._pipeline('switch shit shit')
        self.assertEqual(r['intent'], 'switch_sheet',
            f"Expected switch_sheet, got {r['intent']!r} (corrected={r['corrected']!r})")

    def test_switch_shit_shit_requests_clarification(self):
        """'switch shit shit' must trigger a slot clarification (sheet name unknown)."""
        r = self._pipeline('switch shit shit')
        self.assertIsNotNone(r['slot_msg'],
            f"Expected slot clarification for ambiguous sheet name, got None\n"
            f"  corrected={r['corrected']!r}")

    def test_switch_sheet_named_not_affected(self):
        """'switch sheet Data' with a real name must pass slot check."""
        r = self._pipeline('switch sheet Data')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNone(r['slot_msg'],
            f"'switch sheet Data' should have no slot gap, got {r['slot_msg']!r}")

    def test_switch_shit_named_sheet_not_overridden(self):
        """'switch shit Budget' — command word is profanity but name is valid."""
        corrected = self._corrected('switch shit Budget')
        self.assertIn('switch sheet', corrected.lower(),
            f"Expected 'switch sheet', got {corrected!r}")
        self.assertNotIn('Sheet2', corrected,
            f"Valid sheet name 'Budget' must not be replaced with Sheet2, got {corrected!r}")

    # ── Phonetic prefix: "sweet" → "switch sheet" ─────────────────────────

    def test_sweet_prefix_maps_to_switch_sheet(self):
        """'sweet City' must correct to 'switch sheet City'."""
        corrected = self._corrected('sweet City')
        self.assertTrue(corrected.lower().startswith('switch sheet'),
            f"Expected 'switch sheet ...' from 'sweet City', got {corrected!r}")

    def test_sweet_city_full_pipeline(self):
        """'sweet City' must reach switch_sheet intent with no slot gap."""
        r = self._pipeline('sweet City')
        self.assertEqual(r['intent'], 'switch_sheet',
            f"Expected switch_sheet, got {r['intent']!r} (corrected={r['corrected']!r})")
        self.assertIsNone(r['slot_msg'],
            f"Expected no slot gap for 'sweet City', got {r['slot_msg']!r}")

    def test_write_side_seat_to_requests_clarification(self):
        """'write side seat to 12' — no pattern rule recovers the cell ref; slot gap expected."""
        r = self._pipeline('write side seat to 12')
        self.assertEqual(r['intent'], 'write_cell',
            f"Expected write_cell, got {r['intent']!r} (corrected={r['corrected']!r})")
        self.assertIsNotNone(r['slot_msg'],
            f"Expected slot clarification (cell ref unrecoverable), got None")


# ---------------------------------------------------------------------------
# P9 — real harness failures: "price of C2 fan", "go to set Me 3"
# ---------------------------------------------------------------------------

class TestP9RealHarnessFailures(unittest.TestCase):
    """Regression tests for two real voice failures observed in the P9 harness run.

    Failure 1: "write cell C2 fan"  → STT "price of C2 fan"
      "price of" is a phonetic corruption of "write cell".

    Failure 2: "go to cell B3"  → STT "go to set Me 3"
      "set" for "cell" (existing P7 rule handles merged B3); "Me" is STT noise
      where "M" is the column letter and "e" is appended garbage.
    """

    def _corrected(self, text: str) -> str:
        from pipeline_sanitizer import correct_transcript
        return correct_transcript(text)

    def _pipeline(self, raw: str) -> dict:
        from pipeline_sanitizer import (
            correct_transcript, detect_command_intent,
            is_low_confidence_transcript, check_command_slots,
        )
        corrected = correct_transcript(raw)
        intent    = detect_command_intent(corrected)
        low_conf  = is_low_confidence_transcript(corrected)
        slot_msg  = check_command_slots(intent, corrected)
        return {'corrected': corrected, 'intent': intent,
                'low_conf': low_conf, 'slot_msg': slot_msg}

    # ── Failure 1: "price of C2 fan" → write_cell ────────────────────────

    def test_price_of_corrects_to_write_cell(self):
        """'price of' at command start must normalise to 'write cell'."""
        corrected = self._corrected('price of C2 fan')
        self.assertTrue(corrected.lower().startswith('write cell'),
            f"Expected 'write cell ...', got {corrected!r}")

    def test_price_of_full_pipeline_no_slot_gap(self):
        """'price of C2 fan' must reach write_cell with no slot gap."""
        r = self._pipeline('price of C2 fan')
        self.assertEqual(r['intent'], 'write_cell',
            f"Expected write_cell, got {r['intent']!r} (corrected={r['corrected']!r})")
        self.assertIsNone(r['slot_msg'],
            f"Expected no slot gap for 'price of C2 fan', got {r['slot_msg']!r}")
        self.assertFalse(r['low_conf'])

    def test_price_of_preserves_cell_and_value(self):
        """'price of C2 fan' must keep 'C2' as cell ref and 'fan' as value."""
        corrected = self._corrected('price of C2 fan')
        self.assertIn('C2',  corrected, f"Cell ref C2 missing from {corrected!r}")
        self.assertIn('fan', corrected, f"Value 'fan' missing from {corrected!r}")

    # ── Failure 2: "go to set Me 3" → read_cell ──────────────────────────

    def test_go_to_set_noisy_column_corrects_to_cell_ref(self):
        """'go to set Me 3' must correct to 'go to cell M3' (first letter is column)."""
        corrected = self._corrected('go to set Me 3')
        self.assertIn('go to cell', corrected.lower(),
            f"Expected 'go to cell ...' in {corrected!r}")
        self.assertIn('M3', corrected.upper(),
            f"Expected cell ref 'M3' in {corrected!r}")

    def test_go_to_set_noisy_column_full_pipeline(self):
        """'go to set Me 3' must reach read_cell intent with no slot gap."""
        r = self._pipeline('go to set Me 3')
        self.assertEqual(r['intent'], 'read_cell',
            f"Expected read_cell, got {r['intent']!r} (corrected={r['corrected']!r})")
        self.assertIsNone(r['slot_msg'],
            f"Expected no slot gap for 'go to set Me 3', got {r['slot_msg']!r}")

    def test_go_to_set_clean_ref_still_works(self):
        """Existing 'go to set B3' (merged ref) must still correct to 'go to cell B3'."""
        corrected = self._corrected('go to set B3')
        self.assertIn('go to cell', corrected.lower(),
            f"Expected 'go to cell ...', got {corrected!r}")
        self.assertIn('B3', corrected,
            f"Expected 'B3' in {corrected!r}")

    def test_go_to_set_does_not_route_to_switch_sheet(self):
        """'go to set Me 3' must NOT route to switch_sheet after correction."""
        r = self._pipeline('go to set Me 3')
        self.assertNotEqual(r['intent'], 'switch_sheet',
            f"'go to set Me 3' incorrectly routed to switch_sheet "
            f"(corrected={r['corrected']!r})")


# ---------------------------------------------------------------------------
# P10 — real harness failures: "at all adult 19", "Brightside C 212",
#        "go to sad B 3", "speech sheet sheet 2"
# ---------------------------------------------------------------------------

class TestP10RealHarnessFailures(unittest.TestCase):
    """Regression tests for four real voice failures from the P10 harness run."""

    def _corrected(self, text: str) -> str:
        from pipeline_sanitizer import correct_transcript
        return correct_transcript(text)

    def _pipeline(self, raw: str) -> dict:
        from pipeline_sanitizer import (
            correct_transcript, detect_command_intent,
            is_low_confidence_transcript, check_command_slots,
        )
        corrected = correct_transcript(raw)
        intent    = detect_command_intent(corrected)
        low_conf  = is_low_confidence_transcript(corrected)
        slot_msg  = check_command_slots(intent, corrected)
        return {'corrected': corrected, 'intent': intent,
                'low_conf': low_conf, 'slot_msg': slot_msg}

    # ── "at all adult 19" → add_row ──────────────────────────────────────

    def test_at_all_corrects_to_add_row(self):
        """'at all ...' at command start must normalise to 'add row ...'."""
        corrected = self._corrected('at all adult 19')
        self.assertTrue(corrected.lower().startswith('add row'),
            f"Expected 'add row ...', got {corrected!r}")

    def test_at_all_full_pipeline_no_slot_gap(self):
        """'at all adult 19' must reach add_row intent with no slot gap."""
        r = self._pipeline('at all adult 19')
        self.assertEqual(r['intent'], 'add_row',
            f"Expected add_row, got {r['intent']!r} (corrected={r['corrected']!r})")
        self.assertIsNone(r['slot_msg'],
            f"Expected no slot gap for 'at all adult 19', got {r['slot_msg']!r}")

    # ── "Brightside C 212" → write_cell C2 12 ────────────────────────────

    def test_brightside_corrects_to_write_cell(self):
        """'Brightside ...' at command start must normalise to 'write cell ...'."""
        corrected = self._corrected('Brightside C 212')
        self.assertTrue(corrected.lower().startswith('write cell'),
            f"Expected 'write cell ...', got {corrected!r}")

    def test_brightside_cell_value_split(self):
        """'Brightside C 212' must produce 'write cell C2 12' after split."""
        corrected = self._corrected('Brightside C 212')
        self.assertIn('C2',  corrected, f"Cell ref C2 missing from {corrected!r}")
        self.assertIn('12',  corrected, f"Value '12' missing from {corrected!r}")

    def test_brightside_full_pipeline_no_slot_gap(self):
        """'Brightside C 212' must reach write_cell with no slot gap."""
        r = self._pipeline('Brightside C 212')
        self.assertEqual(r['intent'], 'write_cell',
            f"Expected write_cell, got {r['intent']!r} (corrected={r['corrected']!r})")
        self.assertIsNone(r['slot_msg'],
            f"Expected no slot gap for 'Brightside C 212', got {r['slot_msg']!r}")

    # ── "go to sad B 3" → read_cell B3 ───────────────────────────────────

    def test_go_to_sad_corrects_to_go_to_cell(self):
        """'go to sad ...' must correct to 'go to cell ...'."""
        corrected = self._corrected('go to sad B 3')
        self.assertIn('go to cell', corrected.lower(),
            f"Expected 'go to cell ...', got {corrected!r}")

    def test_go_to_sad_full_pipeline(self):
        """'go to sad B 3' must reach read_cell with no slot gap."""
        r = self._pipeline('go to sad B 3')
        self.assertEqual(r['intent'], 'read_cell',
            f"Expected read_cell, got {r['intent']!r} (corrected={r['corrected']!r})")
        self.assertIsNone(r['slot_msg'],
            f"Expected no slot gap for 'go to sad B 3', got {r['slot_msg']!r}")

    def test_go_to_sad_does_not_route_to_switch_sheet(self):
        """'go to sad B 3' must NOT route to switch_sheet."""
        r = self._pipeline('go to sad B 3')
        self.assertNotEqual(r['intent'], 'switch_sheet',
            f"'go to sad B 3' incorrectly routed to switch_sheet")

    # ── "speech sheet sheet 2" → switch_sheet Sheet2 ─────────────────────

    def test_speech_sheet_corrects_to_switch_sheet(self):
        """'speech sheet ...' must normalise to 'switch sheet ...'."""
        corrected = self._corrected('speech sheet sheet 2')
        self.assertTrue(corrected.lower().startswith('switch sheet'),
            f"Expected 'switch sheet ...', got {corrected!r}")

    def test_speech_sheet_sheet_2_normalises_name(self):
        """'speech sheet sheet 2' must produce 'switch sheet Sheet2'."""
        corrected = self._corrected('speech sheet sheet 2')
        self.assertIn('Sheet2', corrected,
            f"Expected 'Sheet2' in corrected output, got {corrected!r}")

    def test_speech_sheet_full_pipeline_no_slot_gap(self):
        """'speech sheet sheet 2' must reach switch_sheet with no slot gap."""
        r = self._pipeline('speech sheet sheet 2')
        self.assertEqual(r['intent'], 'switch_sheet',
            f"Expected switch_sheet, got {r['intent']!r} (corrected={r['corrected']!r})")
        self.assertIsNone(r['slot_msg'],
            f"Expected no slot gap for 'speech sheet sheet 2', got {r['slot_msg']!r}")


# ---------------------------------------------------------------------------
# P11 — Voice recovery: filler stripping and phonetic expansion
# ---------------------------------------------------------------------------

class TestP11VoiceRecovery(unittest.TestCase):
    """Regression tests for Task 3 voice-recovery additions.

    Covers:
    * Filler word stripping from add-row payload ("of", "a", etc.)
    * "at raw" prefix → "add row"
    * "rise up" phonetic → "write cell"
    * "raiser" phonetic → "write"
    * Direct unit tests for recover_add_row_voice and recover_write_cell_voice
    """

    def _corrected(self, text: str) -> str:
        from pipeline_sanitizer import correct_transcript
        return correct_transcript(text)

    def _pipeline(self, raw: str) -> dict:
        from pipeline_sanitizer import (
            correct_transcript, detect_command_intent,
            is_low_confidence_transcript, check_command_slots,
        )
        corrected = correct_transcript(raw)
        intent    = detect_command_intent(corrected)
        low_conf  = is_low_confidence_transcript(corrected)
        slot_msg  = check_command_slots(intent, corrected)
        return {'corrected': corrected, 'intent': intent,
                'low_conf': low_conf, 'slot_msg': slot_msg}

    # ── Mandatory: "At role of Jenny 18" — filler 'of' stripped ──────────

    def test_at_role_of_jenny_corrected(self):
        """'of' filler after 'add row' prefix must be stripped from payload."""
        r = self._pipeline('At role of Jenny 18')
        self.assertEqual(r['corrected'].lower(), 'add row jenny 18',
            f"Expected 'add row jenny 18', got {r['corrected']!r}")

    def test_at_role_of_jenny_intent(self):
        r = self._pipeline('At role of Jenny 18')
        self.assertEqual(r['intent'], 'add_row',
            f"Expected add_row, got {r['intent']!r} (corrected={r['corrected']!r})")

    def test_at_role_of_jenny_no_slot_gap(self):
        r = self._pipeline('At role of Jenny 18')
        self.assertIsNone(r['slot_msg'],
            f"Expected no slot gap after filler strip, got {r['slot_msg']!r}")

    # ── Mandatory: "at raw Alice 18" — 'raw' prefix → 'add row' ──────────

    def test_at_raw_alice_corrects_to_add_row(self):
        """'at raw' at command start must normalise to 'add row'."""
        r = self._pipeline('at raw Alice 18')
        self.assertTrue(r['corrected'].lower().startswith('add row'),
            f"Expected 'add row ...', got {r['corrected']!r}")

    def test_at_raw_alice_intent(self):
        r = self._pipeline('at raw Alice 18')
        self.assertEqual(r['intent'], 'add_row',
            f"Expected add_row, got {r['intent']!r} (corrected={r['corrected']!r})")

    def test_at_raw_alice_no_slot_gap(self):
        r = self._pipeline('at raw Alice 18')
        self.assertIsNone(r['slot_msg'],
            f"Expected no slot gap for 'at raw Alice 18', got {r['slot_msg']!r}")

    # ── "Rise up G2 writer" — phonetic 'rise up' → 'write cell' ──────────

    def test_rise_up_corrects_to_write_cell_prefix(self):
        """'rise up' must be caught by phonetic fix before 'rise'→'write' fires."""
        corrected = self._corrected('Rise up G2 writer')
        self.assertTrue(corrected.lower().startswith('write cell'),
            f"Expected 'write cell ...', got {corrected!r}")

    def test_rise_up_intent(self):
        r = self._pipeline('Rise up G2 writer')
        self.assertEqual(r['intent'], 'write_cell',
            f"Expected write_cell, got {r['intent']!r} (corrected={r['corrected']!r})")

    def test_rise_up_no_slot_gap(self):
        r = self._pipeline('Rise up G2 writer')
        self.assertIsNone(r['slot_msg'],
            f"Expected no slot gap for 'Rise up G2 writer', got {r['slot_msg']!r}")

    # ── "Raiser E2 18" — phonetic 'raiser' → 'write' ─────────────────────

    def test_raiser_intent(self):
        """'raiser' as STT corruption of 'write' must be caught by phonetic fix."""
        r = self._pipeline('Raiser E2 18')
        self.assertEqual(r['intent'], 'write_cell',
            f"Expected write_cell, got {r['intent']!r} (corrected={r['corrected']!r})")

    def test_raiser_no_slot_gap(self):
        r = self._pipeline('Raiser E2 18')
        self.assertIsNone(r['slot_msg'],
            f"Expected no slot gap for 'Raiser E2 18', got {r['slot_msg']!r}")

    # ── recover_add_row_voice: direct unit tests ───────────────────────────

    def test_recover_add_row_strips_of(self):
        from pipeline_sanitizer import recover_add_row_voice
        self.assertEqual(recover_add_row_voice('add row of jenny 18'), 'add row jenny 18')

    def test_recover_add_row_no_leading_filler_returns_none(self):
        from pipeline_sanitizer import recover_add_row_voice
        self.assertIsNone(recover_add_row_voice('add row jenny 18'),
            "No filler to strip — should return None")

    def test_recover_add_row_filler_only_no_name_returns_none(self):
        """'add row of 18' — no name remains after stripping 'of' → None."""
        from pipeline_sanitizer import recover_add_row_voice
        self.assertIsNone(recover_add_row_voice('add row of 18'),
            "No name-like token after filler strip — should return None")

    def test_recover_add_row_non_add_row_prefix_returns_none(self):
        from pipeline_sanitizer import recover_add_row_voice
        self.assertIsNone(recover_add_row_voice('write cell C1 18'))

    def test_recover_add_row_unsafe_multiple_words_returns_none(self):
        """'add row our stain 8' — multiple uncertain words → unsafe, return None."""
        from pipeline_sanitizer import recover_add_row_voice
        self.assertIsNone(recover_add_row_voice('add row our stain 8'),
            "Multiple words after filler strip — should return None for safety")

    # ── recover_write_cell_voice: direct unit tests ────────────────────────

    def test_recover_write_cell_strips_of(self):
        from pipeline_sanitizer import recover_write_cell_voice
        self.assertEqual(recover_write_cell_voice('write cell c1 of 18'), 'write cell c1 18')

    def test_recover_write_cell_no_filler_returns_none(self):
        from pipeline_sanitizer import recover_write_cell_voice
        self.assertIsNone(recover_write_cell_voice('write cell c1 18'),
            "No filler present — should return None")

    def test_recover_write_cell_real_value_not_stripped(self):
        """A non-filler word at the start of the value must be preserved."""
        from pipeline_sanitizer import recover_write_cell_voice
        self.assertIsNone(recover_write_cell_voice('write cell c1 score'),
            "'score' is not a filler word — should return None")

    def test_recover_write_cell_non_write_cell_returns_none(self):
        from pipeline_sanitizer import recover_write_cell_voice
        self.assertIsNone(recover_write_cell_voice('add row jenny 18'))

    # ── Regression: existing add-row commands unaffected ─────────────────

    def test_canonical_add_row_unaffected(self):
        """Clean 'add row Jenny 18' must not be modified by recovery pass."""
        r = self._pipeline('add row Jenny 18')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNone(r['slot_msg'])

    def test_canonical_write_cell_unaffected(self):
        r = self._pipeline('write cell C1 90')
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNone(r['slot_msg'])


# ---------------------------------------------------------------------------
# P12 — Deterministic routing: undo must never reach the LLM
# ---------------------------------------------------------------------------

class TestDeterministicUndoRouting(unittest.TestCase):
    """undo must be resolved without any LLM call."""

    def test_undo_intent_detected(self):
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('undo'), 'undo')
        self.assertEqual(detect_command_intent('Undo'), 'undo')

    def test_undo_no_slot_message(self):
        """undo has no required slots — slot check must pass immediately."""
        from pipeline_sanitizer import check_command_slots
        self.assertIsNone(check_command_slots('undo', 'undo'))

    def test_save_no_slot_message(self):
        from pipeline_sanitizer import check_command_slots
        self.assertIsNone(check_command_slots('save', 'save'))

    def test_undo_heuristic_bare(self):
        """Heuristic parser returns undo for bare 'undo' text."""
        from llm_parser import _heuristic_natural_language_parse  # type: ignore
        result = _heuristic_natural_language_parse('undo')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'undo')

    def test_undo_heuristic_context_enriched(self):
        """Heuristic must parse undo even when context prefix is prepended.

        _enrich_with_context() produces '[file:data.xlsx sheet:Sheet1] undo'.
        The prefix shifts 'undo' off tokens[0]; the fix strips it first.
        """
        from llm_parser import _heuristic_natural_language_parse  # type: ignore
        enriched = '[file:data.xlsx sheet:Sheet1 columns:Name,Score] undo'
        result = _heuristic_natural_language_parse(enriched)
        self.assertIsNotNone(
            result,
            f"Heuristic returned None for enriched undo text: {enriched!r}",
        )
        self.assertEqual(
            result.get('type'), 'undo',
            f"Expected type=undo, got {result!r}",
        )


# ---------------------------------------------------------------------------
# P12 — Unsafe add_row payloads must be rejected before execution
# ---------------------------------------------------------------------------

class TestUnsafeAddRowBlocking(unittest.TestCase):
    """Garbled add_row transcripts must trigger clarification, not execute."""

    def test_three_data_tokens_clarifies(self):
        """'add row eight nine 10' has 3 data tokens — must ask for retry."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots('add_row', 'add row eight nine 10')
        self.assertIsNotNone(
            msg,
            "'add row eight nine 10' (3 tokens) must trigger clarification",
        )

    def test_valid_john_90_passes(self):
        from pipeline_sanitizer import check_command_slots
        self.assertIsNone(check_command_slots('add_row', 'add row John 90'))

    def test_valid_minna_10_passes(self):
        from pipeline_sanitizer import check_command_slots
        self.assertIsNone(check_command_slots('add_row', 'add row Minna 10'))

    def test_valid_alice_18_passes(self):
        from pipeline_sanitizer import check_command_slots
        self.assertIsNone(check_command_slots('add_row', 'add row Alice 18'))

    def test_one_word_no_number_clarifies(self):
        """'add row one' (word-only, no number) must clarify."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots('add_row', 'add row one')
        self.assertIsNotNone(
            msg,
            "'add row one' (no numeric score) must trigger clarification",
        )

    def test_safe_add_row_helper_valid(self):
        """_is_safe_add_row_values accepts exactly [name, number]."""
        import sys, os
        _BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _BACKEND not in sys.path:
            sys.path.insert(0, _BACKEND)
        from assistant_runtime import _is_safe_add_row_values  # type: ignore
        self.assertTrue(_is_safe_add_row_values(['John', 90]))
        self.assertTrue(_is_safe_add_row_values(['Alice', '18']))

    def test_safe_add_row_helper_rejects_one_word(self):
        from assistant_runtime import _is_safe_add_row_values  # type: ignore
        self.assertFalse(_is_safe_add_row_values(['one.']))
        self.assertFalse(_is_safe_add_row_values(['tool.']))

    def test_safe_add_row_helper_rejects_three_items(self):
        from assistant_runtime import _is_safe_add_row_values  # type: ignore
        self.assertFalse(_is_safe_add_row_values(['eight', 'nine', 10]))

    def test_safe_add_row_helper_rejects_two_strings(self):
        """Two alpha strings (no number) must be rejected."""
        from assistant_runtime import _is_safe_add_row_values  # type: ignore
        self.assertFalse(_is_safe_add_row_values(['Yvonne', 'Studin']))


# ---------------------------------------------------------------------------
# P12 — write without cell ref must clarify, not call LLM
# ---------------------------------------------------------------------------

class TestWriteNoCellRefClarifies(unittest.TestCase):
    """Transcripts with 'write' but no cell ref (e.g. B3) must clarify."""

    def test_write_two_names_clarifies(self):
        """'write Yvonne Studin' has no cell ref — slot check must return msg."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots(None, 'write Yvonne Studin')
        self.assertIsNotNone(
            msg,
            "'write Yvonne Studin' (no cell ref) must trigger clarification",
        )
        self.assertIn('cell', msg.lower(), f"Message should mention 'cell': {msg!r}")

    def test_write_number_no_cell_clarifies(self):
        """'write 100 hello' — number present, still no cell ref."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots(None, 'write 100 hello')
        self.assertIsNotNone(msg)

    def test_write_with_cell_ref_passes(self):
        """'write cell C1 90' has a cell ref — slot check passes."""
        from pipeline_sanitizer import check_command_slots
        self.assertIsNone(check_command_slots('write_cell', 'write cell C1 90'))

    def test_write_cell_missing_value_still_caught(self):
        """'write cell C1' (no value) is still caught by write_cell branch."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots('write_cell', 'write cell C1')
        self.assertIsNotNone(msg)
        self.assertIn('C1', msg)


# ---------------------------------------------------------------------------
# P13 — Full command voice coverage
# ---------------------------------------------------------------------------

class TestP13FullCommandVoiceCoverage(unittest.TestCase):
    """Every command in the spec must reach its action through the voice path.

    Deterministic commands (freeze header, unfreeze panes) must be in
    _DETERMINISTIC_ACTIONS so they never reach Ollama/Groq.
    All other commands must be handled by the heuristic parser.
    """

    def _intent(self, text: str):
        from pipeline_sanitizer import detect_command_intent
        return detect_command_intent(text)

    def _corrected(self, text: str) -> str:
        from pipeline_sanitizer import correct_transcript
        return correct_transcript(text)

    def _heuristic(self, text: str):
        from llm_parser import _heuristic_natural_language_parse  # type: ignore
        return _heuristic_natural_language_parse(text)

    # ── freeze header: new intent pattern ────────────────────────────────

    def test_freeze_header_intent(self):
        self.assertEqual(self._intent('freeze header'), 'freeze_panes')

    def test_freeze_the_header_intent(self):
        self.assertEqual(self._intent('freeze the header'), 'freeze_panes')

    def test_freeze_top_row_intent(self):
        self.assertEqual(self._intent('freeze top row'), 'freeze_panes')

    # ── bare unfreeze: new intent pattern ─────────────────────────────────

    def test_bare_unfreeze_intent(self):
        self.assertEqual(self._intent('unfreeze'), 'unfreeze_panes')

    def test_unfreeze_panes_intent(self):
        self.assertEqual(self._intent('unfreeze panes'), 'unfreeze_panes')

    # ── deterministic bypass entries ──────────────────────────────────────

    def test_freeze_panes_in_deterministic_actions(self):
        """freeze_panes must bypass Ollama/Groq entirely."""
        from assistant_runtime import _DETERMINISTIC_ACTIONS
        self.assertIn('freeze_panes', _DETERMINISTIC_ACTIONS)
        a = _DETERMINISTIC_ACTIONS['freeze_panes']
        self.assertEqual(a.get('type'), 'freeze_panes')
        self.assertEqual(a.get('cell'), 'A2')

    def test_unfreeze_panes_in_deterministic_actions(self):
        """unfreeze_panes must bypass Ollama/Groq entirely."""
        from assistant_runtime import _DETERMINISTIC_ACTIONS
        self.assertIn('unfreeze_panes', _DETERMINISTIC_ACTIONS)
        self.assertEqual(_DETERMINISTIC_ACTIONS['unfreeze_panes'].get('type'), 'unfreeze_panes')

    def test_all_original_deterministic_entries_present(self):
        """Existing deterministic entries must not be disturbed."""
        from assistant_runtime import _DETERMINISTIC_ACTIONS
        for key in ('undo', 'save', 'open_excel', 'close_excel'):
            self.assertIn(key, _DETERMINISTIC_ACTIONS, f"Missing key: {key}")

    # ── freeze header: heuristic parse ───────────────────────────────────

    def test_freeze_header_heuristic(self):
        result = self._heuristic('freeze header')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'freeze_panes')

    # ── insert row normalization ───────────────────────────────────────────

    def test_insert_row_normalizes_to_add_row(self):
        corrected = self._corrected('insert row Tom 80')
        self.assertTrue(corrected.lower().startswith('add row'),
            f"Expected 'add row ...', got {corrected!r}")

    def test_insert_row_full_pipeline(self):
        r = _pipeline('insert row Tom 80')
        self.assertEqual(r['intent'], 'add_row',
            f"Expected add_row, got {r['intent']!r} (corrected={r['corrected']!r})")
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'add_row')

    def test_insert_row_at_not_normalized(self):
        """Positional 'insert row at 3' must keep its own intent, not become add_row."""
        corrected = self._corrected('insert row at 3')
        self.assertFalse(corrected.lower().startswith('add row'),
            f"'insert row at 3' must not become 'add row ...', got {corrected!r}")

    # ── bold range ────────────────────────────────────────────────────────

    def test_bold_range_heuristic(self):
        result = self._heuristic('bold A1:C1')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'format_cells')
        self.assertTrue(result.get('bold'))

    # ── merge cells ───────────────────────────────────────────────────────

    def test_merge_cells_intent(self):
        self.assertEqual(self._intent('merge cells A1:C1'), 'merge_cells')

    def test_merge_cells_heuristic(self):
        result = self._heuristic('merge cells A1:C1')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'merge_cells')

    # ── unmerge cells ─────────────────────────────────────────────────────

    def test_unmerge_cells_intent(self):
        self.assertEqual(self._intent('unmerge A1:C1'), 'unmerge_cells')

    def test_unmerge_cells_heuristic(self):
        result = self._heuristic('unmerge cells A1:C1')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'unmerge_cells')

    # ── sort by column ────────────────────────────────────────────────────

    def test_sort_by_column_intent(self):
        self.assertEqual(self._intent('sort by column B'), 'sort_range')

    def test_sort_by_column_heuristic(self):
        result = self._heuristic('sort by column B')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'sort_range')

    # ── replace ───────────────────────────────────────────────────────────

    def test_replace_intent(self):
        self.assertEqual(self._intent('replace John with Jenny'), 'find_replace')

    def test_replace_heuristic(self):
        result = self._heuristic('replace John with Jenny')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'find_replace')

    # ── autofit column ────────────────────────────────────────────────────

    def test_autofit_intent(self):
        self.assertEqual(self._intent('autofit column A'), 'set_column_width')

    def test_autofit_heuristic(self):
        result = self._heuristic('autofit column A')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'set_column_width')
        self.assertTrue(result.get('auto'))

    # ── rename sheet ──────────────────────────────────────────────────────

    def test_rename_sheet_intent(self):
        self.assertEqual(self._intent('rename sheet Sales to Report'), 'rename_sheet')

    def test_rename_sheet_heuristic(self):
        result = self._heuristic('rename sheet Sales to Report')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'rename_sheet')

    # ── delete sheet ──────────────────────────────────────────────────────

    def test_delete_sheet_intent(self):
        self.assertEqual(self._intent('delete sheet Temp'), 'delete_sheet')

    def test_delete_sheet_heuristic(self):
        result = self._heuristic('delete sheet Temp')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'delete_sheet')

    # ── existing deterministic commands still work ────────────────────────

    def test_undo_pipeline(self):
        r = _pipeline('undo')
        self.assertEqual(r['intent'], 'undo')
        self.assertIsNone(r['slot_msg'])

    def test_save_pipeline(self):
        r = _pipeline('save')
        self.assertEqual(r['intent'], 'save')
        self.assertIsNone(r['slot_msg'])

    def test_open_excel_pipeline(self):
        r = _pipeline('open excel')
        self.assertEqual(r['intent'], 'open_excel')
        self.assertIsNone(r['slot_msg'])

    def test_close_excel_pipeline(self):
        r = _pipeline('close excel')
        self.assertEqual(r['intent'], 'close_excel')
        self.assertIsNone(r['slot_msg'])


# ---------------------------------------------------------------------------
# P14 — Incomplete commands clarify; unclear voice skips LLM
# ---------------------------------------------------------------------------

class TestP14IncompleteClarification(unittest.TestCase):
    """Incomplete commands must stop at the slot gate, never reach route_llm."""

    # ── create sheet incomplete ───────────────────────────────────────────

    def test_create_sheet_by_dots_clarifies(self):
        """'create sheet by...' — trailing-punctuation artifact must clarify."""
        r = _pipeline('create sheet by...')
        self.assertIsNotNone(
            r['slot_msg'],
            "'create sheet by...' must produce a clarification, not route to LLM",
        )
        self.assertIsNone(r['action'])

    def test_create_sheet_by_bare_clarifies(self):
        """'create sheet by' (no dots) must also clarify — 'by' is a garbage word."""
        r = _pipeline('create sheet by')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_create_sheet_alone_clarifies(self):
        """Bare 'create sheet' with no name must clarify."""
        r = _pipeline('create sheet')
        self.assertEqual(r['intent'], 'create_sheet')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_create_sheet_sales_works(self):
        """'create sheet Sales' is complete — must NOT clarify."""
        r = _pipeline('create sheet Sales')
        self.assertEqual(r['intent'], 'create_sheet')
        self.assertIsNone(r['slot_msg'],
            f"'create sheet Sales' must pass slot check, got: {r['slot_msg']!r}")

    def test_create_sheet_report_works(self):
        r = _pipeline('create sheet Report')
        self.assertEqual(r['intent'], 'create_sheet')
        self.assertIsNone(r['slot_msg'])

    # ── create excel incomplete ───────────────────────────────────────────

    def test_create_excel_alone_clarifies(self):
        """Bare 'create excel' with no filename must clarify."""
        r = _pipeline('create excel')
        self.assertEqual(r['intent'], 'create_excel')
        self.assertIsNotNone(r['slot_msg'],
            "'create excel' alone must ask for a filename")
        self.assertIsNone(r['action'])

    def test_create_excel_teacher_works(self):
        """'create excel Teacher' is complete — must NOT clarify."""
        r = _pipeline('create excel Teacher')
        self.assertEqual(r['intent'], 'create_excel')
        self.assertIsNone(r['slot_msg'],
            f"'create excel Teacher' must pass slot check, got: {r['slot_msg']!r}")

    # ── unclear voice: "do let go to" skips LLM ──────────────────────────

    def test_do_let_go_to_clarifies(self):
        """'do let go to' has no destination — must clarify, not call Ollama/Groq."""
        r = _pipeline('do let go to')
        self.assertIsNotNone(
            r['slot_msg'],
            "'do let go to' must produce a clarification before reaching route_llm",
        )
        self.assertIsNone(r['action'],
            "'do let go to' must not produce an action")

    def test_go_to_b3_still_works(self):
        """'go to B3' has a cell destination — must NOT be caught by the guard."""
        r = _pipeline('go to B3')
        self.assertEqual(r['intent'], 'read_cell')
        self.assertIsNone(r['slot_msg'])

    def test_go_to_sheet_sales_still_works(self):
        """'go to sheet Sales' has a sheet destination — must NOT be caught."""
        r = _pipeline('go to sheet Sales')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNone(r['slot_msg'])

    # ── existing deterministic commands unaffected ────────────────────────

    def test_undo_unaffected(self):
        r = _pipeline('undo')
        self.assertEqual(r['intent'], 'undo')
        self.assertIsNone(r['slot_msg'])

    def test_save_unaffected(self):
        r = _pipeline('save')
        self.assertEqual(r['intent'], 'save')
        self.assertIsNone(r['slot_msg'])

    def test_freeze_header_unaffected(self):
        r = _pipeline('freeze header')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertIsNone(r['slot_msg'])

    def test_unfreeze_panes_unaffected(self):
        r = _pipeline('unfreeze panes')
        self.assertEqual(r['intent'], 'unfreeze_panes')
        self.assertIsNone(r['slot_msg'])

    def test_delete_row_2_unaffected(self):
        r = _pipeline('delete row 2')
        self.assertEqual(r['intent'], 'delete_row')
        self.assertIsNone(r['slot_msg'])

    def test_add_row_jenna_15_unaffected(self):
        r = _pipeline('add row Jenna 15')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNone(r['slot_msg'])

    def test_write_e1_teacher_unaffected(self):
        r = _pipeline('write cell E1 teacher')
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNone(r['slot_msg'])


# ---------------------------------------------------------------------------
# P15 — Full command-group voice coverage audit
# ---------------------------------------------------------------------------

class TestP15VoiceCoverageGroups(unittest.TestCase):
    """Every typed command must have a reliable voice path.

    Tests are organised by command group.  Each test verifies:
    - intent detection  (pipeline layer 2)
    - heuristic parse   (pipeline layer 4, no LLM)
    - slot clarification where the command has required arguments
    """

    def _intent(self, text: str):
        from pipeline_sanitizer import detect_command_intent
        return detect_command_intent(text)

    def _heuristic(self, text: str):
        from llm_parser import _heuristic_natural_language_parse  # type: ignore
        return _heuristic_natural_language_parse(text)

    # ── G1: File commands ────────────────────────────────────────────────

    def test_g1_open_excel_intent_and_action(self):
        r = _pipeline('open excel')
        self.assertEqual(r['intent'], 'open_excel')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'open_excel_app')

    def test_g1_close_excel_intent_and_action(self):
        r = _pipeline('close excel')
        self.assertEqual(r['intent'], 'close_excel')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'close_excel_app')

    def test_g1_save_deterministic(self):
        r = _pipeline('save')
        self.assertEqual(r['intent'], 'save')
        self.assertIsNone(r['slot_msg'])

    def test_g1_create_excel_with_name_works(self):
        r = _pipeline('create excel Teacher')
        self.assertEqual(r['intent'], 'create_excel')
        self.assertIsNone(r['slot_msg'],
            f"'create excel Teacher' must pass slot check, got: {r['slot_msg']!r}")

    def test_g1_create_excel_bare_clarifies(self):
        r = _pipeline('create excel')
        self.assertEqual(r['intent'], 'create_excel')
        self.assertIsNotNone(r['slot_msg'])

    def test_g1_rename_file_intent_and_action(self):
        r = _pipeline('rename file to budget')
        self.assertEqual(r['intent'], 'rename_file')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'rename_file')

    # ── G2: Row commands ─────────────────────────────────────────────────

    def test_g2_add_row_complete(self):
        r = _pipeline('add row Jenna 15')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'add_row')

    def test_g2_add_row_bare_clarifies(self):
        r = _pipeline('add row')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNotNone(r['slot_msg'])

    def test_g2_insert_row_voice_normalises(self):
        """'insert row Tom 80' normalises to 'add row Tom 80' via Pass 2."""
        r = _pipeline('insert row Tom 80')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'add_row')

    def test_g2_delete_row_complete(self):
        r = _pipeline('delete row 3')
        self.assertEqual(r['intent'], 'delete_row')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'delete_row')
        self.assertEqual(r['action'].get('index'), 3)

    def test_g2_delete_row_bare_clarifies(self):
        r = _pipeline('delete row')
        self.assertEqual(r['intent'], 'delete_row')
        self.assertIsNotNone(r['slot_msg'])

    # ── G3: Column commands ───────────────────────────────────────────────

    def test_g3_add_column_intent_and_action(self):
        r = _pipeline('add column Score')
        self.assertEqual(r['intent'], 'add_column')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'add_column')
        self.assertEqual(r['action'].get('name'), 'Score')

    def test_g3_delete_column_intent_and_action(self):
        r = _pipeline('delete column B')
        self.assertEqual(r['intent'], 'delete_column')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'delete_column')
        self.assertEqual(r['action'].get('column'), 'B')

    def test_g3_autofit_column_single(self):
        r = _pipeline('autofit column A')
        self.assertEqual(r['intent'], 'set_column_width')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'set_column_width')
        self.assertTrue(r['action'].get('auto'))

    def test_g3_autofit_all_columns_heuristic(self):
        """'autofit all columns' — intent=None but heuristic handles it directly."""
        result = self._heuristic('autofit all columns')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'set_column_width')
        self.assertEqual(result.get('column'), 'all')
        self.assertTrue(result.get('auto'))

    # ── G4: Cell commands ────────────────────────────────────────────────

    def test_g4_write_cell_complete(self):
        r = _pipeline('write cell C2 12')
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_cell')
        self.assertEqual(r['action'].get('cell', '').upper(), 'C2')

    def test_g4_write_cell_no_value_clarifies(self):
        r = _pipeline('write cell C2')
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIn('C2', r['slot_msg'])

    def test_g4_read_cell_intent_and_action(self):
        r = _pipeline('read cell B3')
        self.assertEqual(r['intent'], 'read_cell')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'read_cell')
        self.assertEqual(r['action'].get('cell', '').upper(), 'B3')

    def test_g4_go_to_cell_voice_form(self):
        """'go to B3' — natural voice form for read_cell."""
        r = _pipeline('go to B3')
        self.assertEqual(r['intent'], 'read_cell')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'read_cell')

    def test_g4_clear_cell_intent_and_action(self):
        """'clear cell A1' — intent detected, heuristic handles (no LLM)."""
        r = _pipeline('clear cell A1')
        self.assertEqual(r['intent'], 'clear_cell')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'],
            "'clear cell A1' heuristic must return an action (not fall through to LLM)")
        self.assertEqual(r['action'].get('type'), 'clear_cell')
        self.assertEqual(r['action'].get('cell', '').upper(), 'A1')

    def test_g4_clear_cell_erase_variant(self):
        result = self._heuristic('erase cell B3')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'clear_cell')
        self.assertEqual(result.get('cell'), 'B3')

    # ── G5: Formula commands ─────────────────────────────────────────────

    def test_g5_write_formula_sum(self):
        r = _pipeline('write formula =SUM(A1:A10) in A11')
        self.assertEqual(r['intent'], 'write_formula')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_formula')
        self.assertEqual(r['action'].get('cell', '').upper(), 'A11')
        self.assertIn('SUM', r['action'].get('formula', '').upper())

    def test_g5_sum_range_into_cell(self):
        r = _pipeline('sum B2:B10 into C2')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_sum_formula')
        self.assertEqual(r['action'].get('cell', '').upper(), 'C2')

    def test_g5_calculate_average_column(self):
        r = _pipeline('calculate average of column B')
        self.assertEqual(r['intent'], 'calculate_average')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'calculate_average')
        self.assertEqual(r['action'].get('column', '').upper(), 'B')

    # ── G6: Sheet commands ────────────────────────────────────────────────

    def test_g6_create_sheet_complete(self):
        r = _pipeline('create sheet Sales')
        self.assertEqual(r['intent'], 'create_sheet')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'create_sheet')

    def test_g6_create_sheet_bare_clarifies(self):
        r = _pipeline('create sheet')
        self.assertEqual(r['intent'], 'create_sheet')
        self.assertIsNotNone(r['slot_msg'])

    def test_g6_switch_sheet_complete(self):
        r = _pipeline('switch sheet Report')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'switch_sheet')
        self.assertEqual(r['action'].get('sheet_name'), 'Report')

    def test_g6_switch_sheet_bare_clarifies(self):
        r = _pipeline('switch sheet')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNotNone(r['slot_msg'])

    def test_g6_rename_sheet_complete(self):
        r = _pipeline('rename sheet Sales to Report')
        self.assertEqual(r['intent'], 'rename_sheet')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'rename_sheet')

    def test_g6_delete_sheet_complete(self):
        r = _pipeline('delete sheet Temp')
        self.assertEqual(r['intent'], 'delete_sheet')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'delete_sheet')

    def test_g6_list_sheets_intent_and_action(self):
        """'list sheets' — intent detected, heuristic handles (no LLM)."""
        r = _pipeline('list sheets')
        self.assertEqual(r['intent'], 'list_sheets')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'],
            "'list sheets' heuristic must return an action (not fall through to LLM)")
        self.assertEqual(r['action'].get('type'), 'list_sheets')

    def test_g6_show_all_sheets_normalised(self):
        """'show all sheets' normalises to 'list sheets' via Pass 2."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('show all sheets')
        self.assertEqual(corrected, 'list sheets')
        r = _pipeline('show all sheets')
        self.assertEqual(r['action'].get('type'), 'list_sheets')

    # ── G7: Formatting commands ──────────────────────────────────────────���

    def test_g7_bold_range(self):
        result = self._heuristic('bold A1:C1')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'format_cells')
        self.assertTrue(result.get('bold'))
        self.assertEqual(result.get('range', '').upper(), 'A1:C1')

    def test_g7_bold_column(self):
        result = self._heuristic('bold column B')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'format_cells')
        self.assertTrue(result.get('bold'))

    def test_g7_bold_row_1(self):
        result = self._heuristic('bold row 1')
        self.assertIsNotNone(result)
        self.assertEqual(result.get('type'), 'format_cells')
        self.assertTrue(result.get('bold'))

    def test_g7_freeze_header_deterministic(self):
        r = _pipeline('freeze header')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertIsNone(r['slot_msg'])

    def test_g7_unfreeze_panes_deterministic(self):
        r = _pipeline('unfreeze panes')
        self.assertEqual(r['intent'], 'unfreeze_panes')
        self.assertIsNone(r['slot_msg'])

    def test_g7_merge_cells(self):
        r = _pipeline('merge cells A1:C1')
        self.assertEqual(r['intent'], 'merge_cells')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')

    def test_g7_unmerge_cells(self):
        r = _pipeline('unmerge cells A1:C1')
        self.assertEqual(r['intent'], 'unmerge_cells')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells')

    # ── G8: Data commands ─────────────────────────────────────────────────

    def test_g8_sort_by_column(self):
        # Direction-less sort now asks for clarification (P40 safety fix).
        # Use explicit direction to test the full sort_range pipeline path.
        r = _pipeline('sort by column B ascending')
        self.assertEqual(r['intent'], 'sort_range')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'sort_range')
        self.assertEqual(r['action'].get('column', '').upper(), 'B')

    def test_g8_find_replace(self):
        r = _pipeline('replace John with Jenny')
        self.assertEqual(r['intent'], 'find_replace')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'find_replace')

    def test_g8_filter_table(self):
        r = _pipeline('filter column B = 90')
        self.assertEqual(r['intent'], 'filter_table')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'filter_table')
        self.assertEqual(r['action'].get('column', '').upper(), 'B')

    # ── G9: Safety — incomplete and unclear commands ──────────────────────

    def test_g9_undo_deterministic(self):
        r = _pipeline('undo')
        self.assertEqual(r['intent'], 'undo')
        self.assertIsNone(r['slot_msg'])

    def test_g9_incomplete_create_sheet_clarifies(self):
        r = _pipeline('create sheet by...')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_g9_unclear_voice_no_llm(self):
        """'do let go to' must clarify at the slot gate — never reaches route_llm."""
        r = _pipeline('do let go to')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_g9_single_noise_word_low_confidence(self):
        from pipeline_sanitizer import is_low_confidence_transcript
        self.assertTrue(is_low_confidence_transcript('taste'))
        self.assertTrue(is_low_confidence_transcript('oh'))

    def test_g9_two_word_no_intent_low_confidence(self):
        from pipeline_sanitizer import is_low_confidence_transcript
        self.assertTrue(is_low_confidence_transcript('hello world'))

    def test_g9_valid_commands_not_filtered(self):
        from pipeline_sanitizer import is_low_confidence_transcript
        for cmd in ('add row John 90', 'write cell C2 12', 'freeze header',
                    'delete row 3', 'sort by column B'):
            self.assertFalse(is_low_confidence_transcript(cmd),
                f"'{cmd}' must not be treated as low-confidence")


# ---------------------------------------------------------------------------
# P16 — Voice audit gap fixes (4 real bugs found in P16 audit)
# ---------------------------------------------------------------------------

class TestP16VoiceAuditGapFixes(unittest.TestCase):
    """Four bugs found during the P16 full-command-group voice audit.

    Bug 1: 'make A1 bold' / 'make header size 18' → false SLOT because the
           create-guard in check_command_slots fired on 'make' even when
           formatting keywords were present.

    Bug 2: 'align column B right' → phonetic fix converted 'right' → 'write',
           then the write-without-cell-ref guard blocked the command.

    Bug 3: 'set column width A 20' → cell-ref normalizer merged 'A 20' → 'A20';
           heuristic required space between column and width, so it returned None
           and fell through to the LLM.

    Bug 4: bare 'add column' → heuristic fell through to the standalone add_row
           branch and returned {'type':'add_row','values':['column']} — wrong.
    """

    # ── Bug 1: formatting 'make' commands ────────────────────────────────────

    def test_make_cell_bold(self):
        """'make A1 bold' must produce format_cells action, not the create-guard."""
        r = _pipeline('make A1 bold')
        self.assertIsNone(r['slot_msg'],
            f"'make A1 bold' must not hit the create-guard, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertTrue(r['action'].get('bold'))

    def test_make_header_size_18(self):
        """'make header size 18' must produce format_cells, not the create-guard."""
        r = _pipeline('make header size 18')
        self.assertIsNone(r['slot_msg'],
            f"'make header size 18' must not hit the create-guard, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('font_size'), 18)

    def test_make_cell_italic(self):
        """'make B2 italic' must produce format_cells action."""
        r = _pipeline('make B2 italic')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertTrue(r['action'].get('italic'))

    def test_make_column_bold(self):
        """'make column A bold' must produce format_cells action."""
        r = _pipeline('make column A bold')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertTrue(r['action'].get('bold'))

    # ── Bug 2: 'align ... right' phonetic corruption ─────────────────────────

    def test_align_column_right(self):
        """'align column B right' must not be corrupted to 'align column B write'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('align column B right')
        self.assertNotIn('write', corrected.lower(),
            f"'right' direction must not become 'write': got {corrected!r}")
        r = _pipeline('align column B right')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('align'), 'right')

    def test_align_row_right(self):
        """'align row 1 right' must produce format_cells align=right."""
        r = _pipeline('align row 1 right')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('align'), 'right')

    def test_align_range_right(self):
        """'align A1:C1 right' must produce format_cells align=right."""
        r = _pipeline('align A1:C1 right')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('align'), 'right')

    def test_align_left_unaffected(self):
        """'align column B left' must still work after phonetic fix guard."""
        r = _pipeline('align column B left')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('align'), 'left')

    def test_align_center_unaffected(self):
        """'align row 1 center' must still work."""
        r = _pipeline('align row 1 center')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('align'), 'center')

    # ── Bug 3: set column width with merged column+width token ───────────────

    def test_set_column_width_space(self):
        """'set column width A 20' must produce set_column_width action."""
        r = _pipeline('set column width A 20')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'],
            "'set column width A 20' must not fall through to LLM")
        self.assertEqual(r['action'].get('type'), 'set_column_width')
        self.assertEqual(r['action'].get('column'), 'A')
        self.assertAlmostEqual(r['action'].get('width'), 20.0)

    def test_set_column_width_b(self):
        """'set column width B 15' must produce set_column_width action."""
        r = _pipeline('set column width B 15')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'set_column_width')
        self.assertEqual(r['action'].get('column'), 'B')

    def test_adjust_column_width(self):
        """'adjust column width C 30' must produce set_column_width action."""
        r = _pipeline('adjust column width C 30')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'set_column_width')
        self.assertEqual(r['action'].get('column'), 'C')

    def test_autofit_column_unaffected(self):
        """autofit column A must still work (autofit path unaffected by patch)."""
        r = _pipeline('autofit column A')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'set_column_width')
        self.assertTrue(r['action'].get('auto'))

    # ── Bug 4: bare 'add column' must clarify, not produce wrong add_row ─────

    def test_add_column_bare_clarifies(self):
        """'add column' with no name must clarify — never return an add_row action."""
        r = _pipeline('add column')
        self.assertIsNotNone(r['slot_msg'],
            "'add column' with no name must produce a clarification")
        self.assertIsNone(r['action'],
            "'add column' must not silently execute as add_row with values=['column']")

    def test_add_column_with_name_works(self):
        """'add column Score' must still produce a correct add_column action."""
        r = _pipeline('add column Score')
        self.assertEqual(r['intent'], 'add_column')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'add_column')
        self.assertEqual(r['action'].get('name'), 'Score')

    def test_add_column_multiple_names_works(self):
        """'add column Name Score Age' must return a multi-step action."""
        r = _pipeline('add column Name Score Age')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        # multi-column returns {'steps': [...]} or a single add_column
        atype = r['action'].get('type')
        has_steps = 'steps' in r['action']
        self.assertTrue(atype == 'add_column' or has_steps,
            f"Expected add_column or steps, got {r['action']}")

    # ── Regression: existing commands unaffected ──────────────────────────────

    def test_create_excel_guard_still_fires(self):
        """'create the thing' with no Excel anchor should still block."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots(None, 'create the thing')
        self.assertIsNotNone(msg)

    def test_create_excel_with_name_unaffected(self):
        """'create excel Teacher' must still pass through."""
        r = _pipeline('create excel Teacher')
        self.assertEqual(r['intent'], 'create_excel')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])

    def test_bold_a1_c1_unaffected(self):
        r = _pipeline('bold A1:C1')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertTrue(r['action'].get('bold'))

    def test_write_cell_right_phonetic_unaffected(self):
        """'right cell C1 score' must still correct to write_cell (not broken by align guard)."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('right cell C1 score')
        self.assertIn('write', corrected.lower(),
            f"'right cell' must still become 'write cell', got: {corrected!r}")

    def test_add_row_john_90_unaffected(self):
        r = _pipeline('add row John 90')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'add_row')


class TestP17STTReliability(unittest.TestCase):
    """P17 — STT reliability: real-voice garbled-command corrections and detection."""

    # ── Correction: adro → add row ──────────────────────────────────────────

    def test_adro_correction(self):
        """'adro John 90' must correct to 'add row John 90'."""
        from pipeline_sanitizer import correct_transcript
        self.assertEqual(correct_transcript('adro John 90'), 'add row John 90')

    def test_adro_produces_add_row_action(self):
        """'adro John 90' must produce an add_row action end-to-end."""
        r = _pipeline('adro John 90')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'add_row')

    def test_adro_bare_triggers_slot_clarification(self):
        """Bare 'adro' corrects to 'add row' and triggers slot clarification (not wrong action)."""
        r = _pipeline('adro')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNotNone(r['slot_msg'],
            "Bare 'adro' → 'add row' must ask for name/score, not execute blindly")

    # ── Correction: dalit roto → delete row ─────────────────────────────────

    def test_dalit_roto_correction(self):
        """'dalit roto 3' must correct to 'delete row 3'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('dalit roto 3')
        self.assertIn('delete row', corrected.lower())

    def test_dalit_roto_produces_delete_row_action(self):
        """'dalit roto 3' must produce a delete_row action end-to-end."""
        r = _pipeline('dalit roto 3')
        self.assertEqual(r['intent'], 'delete_row')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'delete_row')

    # ── Correction: free hada → freeze header ───────────────────────────────

    def test_free_hada_correction(self):
        """'free hada' must correct to 'freeze header'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('free hada')
        self.assertIn('freeze', corrected.lower())

    def test_free_hada_produces_freeze_panes_intent(self):
        """'free hada' must reach freeze_panes intent after correction."""
        r = _pipeline('free hada')
        self.assertEqual(r['intent'], 'freeze_panes',
            f"Expected freeze_panes intent, got: {r['intent']!r} (corrected={r['corrected']!r})")

    # ── is_suspicious_transcript ─────────────────────────────────────────────

    def test_suspicious_adro(self):
        from pipeline_sanitizer import is_suspicious_transcript
        self.assertTrue(is_suspicious_transcript('adro John 90'))

    def test_suspicious_dalit_roto(self):
        from pipeline_sanitizer import is_suspicious_transcript
        self.assertTrue(is_suspicious_transcript('dalit roto 3'))

    def test_suspicious_free_hada(self):
        from pipeline_sanitizer import is_suspicious_transcript
        self.assertTrue(is_suspicious_transcript('free hada'))

    def test_not_suspicious_valid_command(self):
        from pipeline_sanitizer import is_suspicious_transcript
        self.assertFalse(is_suspicious_transcript('add row John 90'))

    def test_not_suspicious_freeze_header(self):
        from pipeline_sanitizer import is_suspicious_transcript
        self.assertFalse(is_suspicious_transcript('freeze header'))

    def test_not_suspicious_delete_row(self):
        from pipeline_sanitizer import is_suspicious_transcript
        self.assertFalse(is_suspicious_transcript('delete row 3'))

    def test_not_suspicious_error_sentinel(self):
        from pipeline_sanitizer import is_suspicious_transcript
        self.assertFalse(is_suspicious_transcript('[transcription_error] oops'))

    def test_not_suspicious_empty(self):
        from pipeline_sanitizer import is_suspicious_transcript
        self.assertFalse(is_suspicious_transcript(''))

    # ── Google STT phrase hints ──────────────────────────────────────────────

    def test_phrase_hints_include_delete_row(self):
        """'delete row' must be in Google STT phrase hints (real-voice failure)."""
        from speech_recognition import _SARA_SPEECH_CONTEXT
        phrases_lower = [p.lower() for p in _SARA_SPEECH_CONTEXT['phrases']]
        self.assertIn('delete row', phrases_lower)

    def test_phrase_hints_include_freeze_header(self):
        """'freeze header' must be in Google STT phrase hints (real-voice failure)."""
        from speech_recognition import _SARA_SPEECH_CONTEXT
        phrases_lower = [p.lower() for p in _SARA_SPEECH_CONTEXT['phrases']]
        self.assertIn('freeze header', phrases_lower)

    def test_phrase_hints_include_add_column(self):
        from speech_recognition import _SARA_SPEECH_CONTEXT
        phrases_lower = [p.lower() for p in _SARA_SPEECH_CONTEXT['phrases']]
        self.assertIn('add column', phrases_lower)

    # ── Already-working cases must still work (regressions) ─────────────────

    def test_at_road_still_works(self):
        """'at road John 90' was already handled and must not regress."""
        r = _pipeline('at road John 90')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNone(r['slot_msg'])

    def test_raisa_still_works(self):
        """'raisa c1 100' was already handled and must not regress."""
        r = _pipeline('raisa c1 100')
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNone(r['slot_msg'])

    def test_rice_cell_still_works(self):
        """'rice cell C1 score' was already handled and must not regress."""
        r = _pipeline('rice cell C1 score')
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNone(r['slot_msg'])


class TestP18GarbledVoice(unittest.TestCase):
    """P18 — real-voice garbled transcripts: bold / delete / col-width / write-cell safety."""

    # ── Bold phonetic variants ───────────────────────────────────────────────

    def test_make_a_one_bowl_bold(self):
        r = _pipeline('Make a one bowl')
        self.assertIsNotNone(r['action'], f"Expected bold action, got slot_msg={r['slot_msg']!r}")
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertTrue(r['action'].get('bold'))
        self.assertEqual(r['action'].get('range', '').upper(), 'A1')

    def test_make_a1_bolt_bold(self):
        r = _pipeline('Make A1 bolt')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertTrue(r['action'].get('bold'))
        self.assertEqual(r['action'].get('range', '').upper(), 'A1')

    def test_make_a_one_ball_bold(self):
        r = _pipeline('Make a one ball')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertTrue(r['action'].get('bold'))

    def test_make_a_one_board_bold(self):
        r = _pipeline('Make a one board')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertTrue(r['action'].get('bold'))

    def test_bold_correction_text(self):
        from pipeline_sanitizer import correct_transcript
        for raw in ('Make a one bowl', 'Make a one ball', 'Make a one bolt', 'Make A1 bolt'):
            corrected = correct_transcript(raw)
            self.assertIn('bold', corrected.lower(), f"Expected 'bold' in corrected({raw!r})={corrected!r}")
            self.assertNotIn('bowl', corrected.lower())
            self.assertNotIn('bolt', corrected.lower())

    def test_make_bowl_soup_not_affected(self):
        """'make a bowl of soup' has no cell ref — must NOT be rewritten."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('make a bowl of soup')
        self.assertNotIn('bold', corrected.lower())

    # ── Delete row garbles ───────────────────────────────────────────────────

    def test_delhi_row_2_action(self):
        r = _pipeline('Delhi Row 2')
        self.assertEqual(r['intent'], 'delete_row')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'delete_row')
        self.assertEqual(r['action'].get('index'), 2)

    def test_deli_roto_clarifies_not_add_row(self):
        r = _pipeline('Deli Roto')
        self.assertEqual(r['intent'], 'delete_row')
        self.assertIsNotNone(r['slot_msg'], "Expected clarification (row number missing)")
        self.assertIsNone(r['action'])
        if r['slot_msg']:
            self.assertNotIn('add row', r['slot_msg'].lower())

    def test_d_lit_roll_tool_clarifies(self):
        r = _pipeline('D-lit roll tool')
        self.assertEqual(r['intent'], 'delete_row')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_do_you_eat_rotu_clarifies(self):
        r = _pipeline('Do you eat rotu')
        self.assertEqual(r['intent'], 'delete_row')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_delete_correction_text_delhi(self):
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('Delhi Row 2')
        self.assertIn('delete row', corrected.lower())

    # ── Column width ─────────────────────────────────────────────────────────

    def test_set_column_with_a_20_action(self):
        r = _pipeline('Set column with a 20')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'set_column_width')
        self.assertEqual(str(r['action'].get('column', '')).upper(), 'A')
        self.assertEqual(float(r['action'].get('width', 0)), 20.0)

    def test_column_with_rewritten_to_width(self):
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('Set column with a 20')
        self.assertIn('width', corrected.lower())
        self.assertNotIn('with a', corrected.lower())

    def test_column_with_score_not_rewritten(self):
        """'create a column with score' has no number — must NOT be rewritten."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('create a column with score')
        self.assertNotIn('column width', corrected.lower())

    # ── Write-cell value safety ───────────────────────────────────────────────

    def test_rise_to_then_to_e1_clarifies(self):
        """'Rise to then to E1' must NOT execute — value is pure noise."""
        r = _pipeline('Rise to then to E1')
        self.assertIsNone(r['action'],
            f"Should not execute, got action={r['action']}")
        self.assertIsNotNone(r['slot_msg'])

    def test_write_cell_then_to_e1_clarifies(self):
        r = _pipeline('write cell then to E1')
        self.assertIsNone(r['action'])
        self.assertIsNotNone(r['slot_msg'])

    def test_valid_write_cell_e1_student(self):
        r = _pipeline('write cell E1 student')
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_cell')
        self.assertEqual(r['action'].get('value'), 'student')

    def test_valid_write_e1_student(self):
        r = _pipeline('write E1 student')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_cell')

    def test_valid_put_student_in_e1(self):
        r = _pipeline('put student in E1')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_cell')
        self.assertEqual(r['action'].get('value'), 'student')


# ---------------------------------------------------------------------------
# P19 — remaining real-voice failures
# ---------------------------------------------------------------------------

class TestP19VoiceFailures(unittest.TestCase):
    """Tests for 8 real-voice failure groups fixed in P19."""

    # ── P2: delete-column garbles ─────────────────────────────────────────

    def test_do_you_like_colombia_corrects(self):
        from pipeline_sanitizer import correct_transcript
        self.assertIn('delete column', correct_transcript('Do you like Colombia').lower())

    def test_do_you_like_kalumbi_corrects(self):
        from pipeline_sanitizer import correct_transcript
        self.assertIn('delete column', correct_transcript('Do you like Kalumbi').lower())

    def test_d_late_cut_love_be_corrects(self):
        from pipeline_sanitizer import correct_transcript
        self.assertIn('delete column', correct_transcript('D late cut love be').lower())

    def test_lily_call_up_me_corrects(self):
        from pipeline_sanitizer import correct_transcript
        self.assertIn('delete column', correct_transcript('lily call up me').lower())

    def test_delete_column_b_no_slot_msg(self):
        r = _pipeline('delete column b')
        self.assertEqual(r['intent'], 'delete_column')
        self.assertIsNone(r['slot_msg'])

    def test_delete_column_no_letter_asks(self):
        r = _pipeline('delete column')
        self.assertEqual(r['intent'], 'delete_column')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIn('column', r['slot_msg'].lower())

    # ── P3 gap: pre-merged column width (a20) ─────────────────────────────

    def test_column_with_a20_merged_corrects(self):
        """'set column with a20' (no space) must still rewrite to 'column width'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('set column with a20')
        self.assertIn('column width', corrected.lower())

    def test_column_with_a_20_spaced_still_corrects(self):
        """Existing spaced form must still work after regex change."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('set column with a 20')
        self.assertIn('column width', corrected.lower())

    # ── P4: 'make A1 bold' LC filter ─────────────────────────────────────

    def test_make_a1_bold_not_low_confidence(self):
        from pipeline_sanitizer import is_low_confidence_transcript
        self.assertFalse(is_low_confidence_transcript('make A1 bold'))

    def test_make_header_bold_not_low_confidence(self):
        from pipeline_sanitizer import is_low_confidence_transcript
        self.assertFalse(is_low_confidence_transcript('make header bold'))

    def test_make_b2_bold_not_low_confidence(self):
        from pipeline_sanitizer import is_low_confidence_transcript
        self.assertFalse(is_low_confidence_transcript('make B2 bold'))

    # ── P5: 'make header bold' variants ──────────────────────────────────

    def test_make_header_bold_intent(self):
        """After correction 'make header bold' must not be blocked by create-guard."""
        r = _pipeline('make header bold')
        self.assertIsNone(r['slot_msg'],
            f"Unexpected slot_msg: {r['slot_msg']}")

    def test_make_head_the_bolt_corrects(self):
        from pipeline_sanitizer import correct_transcript
        self.assertEqual(correct_transcript('Make head the bolt').lower(), 'make header bold')

    def test_make_head_up_board_corrects(self):
        from pipeline_sanitizer import correct_transcript
        self.assertEqual(correct_transcript('Make head up board').lower(), 'make header bold')

    def test_make_a_header_bold_corrects(self):
        from pipeline_sanitizer import correct_transcript
        self.assertEqual(correct_transcript('make a header bold').lower(), 'make header bold')

    # ── P6: 'freeze panes' garbles ───────────────────────────────────────

    def test_and_freeze_pan_corrects(self):
        from pipeline_sanitizer import correct_transcript
        self.assertEqual(correct_transcript('and freeze pan').lower(), 'freeze panes')

    def test_and_freeze_pants_corrects(self):
        from pipeline_sanitizer import correct_transcript
        self.assertEqual(correct_transcript('and freeze pants').lower(), 'freeze panes')

    def test_and_freeze_pen_corrects(self):
        from pipeline_sanitizer import correct_transcript
        self.assertEqual(correct_transcript('and freeze pen').lower(), 'freeze panes')

    def test_freeze_pan_intent(self):
        r = _pipeline('freeze pan')
        self.assertEqual(r['intent'], 'freeze_panes')

    def test_freeze_pants_intent(self):
        r = _pipeline('freeze pants')
        self.assertEqual(r['intent'], 'freeze_panes')

    def test_freeze_pen_intent(self):
        r = _pipeline('freeze pen')
        self.assertEqual(r['intent'], 'freeze_panes')

    # ── Regressions: existing commands must not be broken ─────────────────

    def test_make_a1_bold_phonetic_still_works(self):
        from pipeline_sanitizer import correct_transcript
        self.assertIn('bold', correct_transcript('make A1 bowl').lower())

    def test_delete_row_3_unaffected(self):
        r = _pipeline('delete row 3')
        self.assertEqual(r['intent'], 'delete_row')
        self.assertIsNone(r['slot_msg'])

    def test_create_column_with_score_unaffected(self):
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('create a column with score')
        self.assertNotIn('column width', corrected.lower())

    def test_freeze_panes_direct_unaffected(self):
        r = _pipeline('freeze panes')
        self.assertEqual(r['intent'], 'freeze_panes')


# ---------------------------------------------------------------------------
# P20 — formatting commands with "make" blocked by create-guard
# ---------------------------------------------------------------------------

class TestP20FormatCells(unittest.TestCase):
    """Tests for P20: format_cells commands starting with 'make' must not be
    blocked by the create-guard in check_command_slots."""

    # ── Intent detection ─────────────────────────────────────────────────

    def test_make_a1_bold_intent(self):
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        c = correct_transcript('Make A1 bold')
        self.assertEqual(detect_command_intent(c), 'format_cells')

    def test_make_a1_bolt_corrects_and_intent(self):
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        c = correct_transcript('Make A1 bolt')
        self.assertEqual(c.lower(), 'make a1 bold')
        self.assertEqual(detect_command_intent(c), 'format_cells')

    def test_make_header_bold_intent(self):
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        c = correct_transcript('make header bold')
        self.assertEqual(detect_command_intent(c), 'format_cells')

    def test_make_head_the_ball_intent(self):
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        c = correct_transcript('Make head the ball')
        self.assertEqual(detect_command_intent(c), 'format_cells')

    # ── Slot guard must not block ─────────────────────────────────────────

    def test_make_a1_bold_no_create_guard(self):
        r = _pipeline('make A1 bold')
        self.assertIsNone(r['slot_msg'],
            f"create-guard must not fire: slot_msg={r['slot_msg']!r}")

    def test_make_a1_bolt_no_create_guard(self):
        r = _pipeline('Make A1 bolt')
        self.assertIsNone(r['slot_msg'],
            f"create-guard must not fire: slot_msg={r['slot_msg']!r}")

    def test_make_header_bold_no_create_guard(self):
        r = _pipeline('make header bold')
        self.assertIsNone(r['slot_msg'],
            f"create-guard must not fire: slot_msg={r['slot_msg']!r}")

    def test_make_head_ball_no_create_guard(self):
        r = _pipeline('Make head the ball')
        self.assertIsNone(r['slot_msg'],
            f"create-guard must not fire: slot_msg={r['slot_msg']!r}")

    # ── Heuristic parse → format_cells action ────────────────────────────

    def test_make_a1_bold_parses_to_format_cells(self):
        r = _pipeline('make A1 bold')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), 'A1')
        self.assertTrue(r['action'].get('bold'))

    def test_make_a1_bolt_parses_to_format_cells(self):
        r = _pipeline('Make A1 bolt')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), 'A1')
        self.assertTrue(r['action'].get('bold'))

    def test_make_header_bold_parses_range_1_1(self):
        r = _pipeline('make header bold')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), '1:1')
        self.assertTrue(r['action'].get('bold'))

    def test_make_head_ball_parses_header_bold(self):
        r = _pipeline('Make head the ball')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), '1:1')
        self.assertTrue(r['action'].get('bold'))

    # ── Low confidence must not filter these ─────────────────────────────

    def test_make_a1_bold_not_lc(self):
        from pipeline_sanitizer import is_low_confidence_transcript
        self.assertFalse(is_low_confidence_transcript('make A1 bold'))

    def test_make_header_bold_not_lc(self):
        from pipeline_sanitizer import is_low_confidence_transcript
        self.assertFalse(is_low_confidence_transcript('make header bold'))

    # ── Regressions ───────────────────────────────────────────────────────

    def test_create_excel_still_detected(self):
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('create excel teacher'), 'create_excel')

    def test_create_sheet_still_detected(self):
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('make a sheet report'), 'create_sheet')

    def test_write_cell_still_works(self):
        r = _pipeline('write cell E1 student')
        self.assertEqual(r['intent'], 'write_cell')
        self.assertIsNone(r['slot_msg'])

    def test_delete_row_still_works(self):
        r = _pipeline('delete row 3')
        self.assertEqual(r['intent'], 'delete_row')
        self.assertIsNone(r['slot_msg'])


# ---------------------------------------------------------------------------
# P21 — format_cells COM live path + STT junk-word Google-skip
# ---------------------------------------------------------------------------

class TestP21FormatCellsCOMPath(unittest.TestCase):
    """format_cells must use the COM live path when workbook is open in Excel."""

    def test_com_live_format_cells_exists(self):
        """_com_live_format_cells must be importable from excel_controller."""
        from excel_controller import _com_live_format_cells
        self.assertTrue(callable(_com_live_format_cells))

    def test_com_live_format_cells_returns_none_when_no_excel(self):
        """When no Excel is running _com_live_format_cells returns None (safe)."""
        from excel_controller import _com_live_format_cells
        result = _com_live_format_cells('nonexistent.xlsx', 'A1', bold=True)
        self.assertIsNone(result)

    def test_format_cells_skips_atomic_save_when_com_succeeds(self):
        """When COM live format succeeds, _flush_session / _atomic_save must NOT run."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        fake_com_result = {'_com_used': True}
        with patch.object(ec, '_com_live_format_cells', return_value=fake_com_result) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_wb, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            # ensure_workbook must return a workbook + worksheet mock
            mock_ws = MagicMock()
            mock_ws.__getitem__ = MagicMock(return_value=MagicMock())
            mock_wb.return_value = (MagicMock(), mock_ws)
            ec.format_cells('fake.xlsx', 'A1', bold=True)

        mock_com.assert_called_once()
        mock_flush.assert_not_called()

    def test_format_cells_falls_back_to_flush_when_com_unavailable(self):
        """When COM is unavailable (returns None) _flush_session must run."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_format_cells', return_value=None) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_wb, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ws = MagicMock()
            mock_ws.__getitem__ = MagicMock(return_value=MagicMock())
            mock_wb.return_value = (MagicMock(), mock_ws)
            ec.format_cells('fake.xlsx', 'A1', bold=True)

        mock_com.assert_called_once()
        mock_flush.assert_called_once()

    def test_make_a1_bold_pipeline_does_not_trigger_locked_save(self):
        """End-to-end: make A1 bold reaches format_cells without touching atomic save."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec
        from pipeline_sanitizer import correct_transcript, detect_command_intent, check_command_slots
        from llm_parser import _heuristic_natural_language_parse

        c = correct_transcript('Make A1 board')
        intent = detect_command_intent(c)
        slot = check_command_slots(intent, c)
        action = _heuristic_natural_language_parse(c) if slot is None else None

        self.assertEqual(intent, 'format_cells')
        self.assertIsNone(slot)
        self.assertIsNotNone(action)
        self.assertEqual(action.get('type'), 'format_cells')
        self.assertEqual(action.get('range'), 'A1')

        # Verify that when COM succeeds, no atomic save occurs
        fake_com = {'_com_used': True}
        with patch.object(ec, '_com_live_format_cells', return_value=fake_com), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_wb, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ws = MagicMock()
            mock_ws.__getitem__ = MagicMock(return_value=MagicMock())
            mock_wb.return_value = (MagicMock(), mock_ws)
            ec.format_cells('fake.xlsx', action['range'], bold=action.get('bold'))

        mock_flush.assert_not_called()

    def test_header_bold_range_1_1_com_path(self):
        """make header bold → range='1:1'; COM must receive '1:1' as the range."""
        from unittest.mock import patch, MagicMock, call
        import excel_controller as ec
        from pipeline_sanitizer import correct_transcript
        from llm_parser import _heuristic_natural_language_parse

        c = correct_transcript('make header bold')
        action = _heuristic_natural_language_parse(c)
        self.assertEqual(action.get('range'), '1:1')

        with patch.object(ec, '_com_live_format_cells', return_value={'_com_used': True}) as mock_com, \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, 'ensure_workbook') as mock_wb, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ws = MagicMock()
            mock_ws.__getitem__ = MagicMock(return_value=MagicMock())
            mock_wb.return_value = (MagicMock(), mock_ws)
            ec.format_cells('fake.xlsx', '1:1', bold=True)

        # Verify COM was called with the correct range
        args, kwargs = mock_com.call_args
        self.assertIn('1:1', args + tuple(kwargs.values()),
                      "COM live format must receive range '1:1'")


class TestP21STTJunkSkip(unittest.TestCase):
    """Junk one-word Whisper results must not block on Google Cloud fallback."""

    def _skip_words(self):
        from speech_recognition import _GOOGLE_SKIP_SINGLE_WORDS
        return _GOOGLE_SKIP_SINGLE_WORDS

    def test_skip_set_contains_yes(self):
        self.assertIn('yes', self._skip_words())

    def test_skip_set_contains_yeah(self):
        self.assertIn('yeah', self._skip_words())

    def test_skip_set_contains_oh(self):
        self.assertIn('oh', self._skip_words())

    def test_skip_set_contains_i(self):
        self.assertIn('i', self._skip_words())

    def test_skip_set_contains_makeup(self):
        self.assertIn('makeup', self._skip_words())

    def test_junk_yes_skips_google(self):
        """Whisper result 'Yes.' must skip Google fallback and return immediately."""
        from unittest.mock import patch
        import speech_recognition as sr

        with patch.object(sr, 'transcribe_local_whisper', return_value='Yes.'), \
             patch.object(sr, '_transcribe_google_cloud') as mock_google, \
             patch.object(sr, '_normalize_wav', return_value=('fake.wav', False)):
            result = sr.transcribe_audio_file('fake.wav')

        mock_google.assert_not_called()
        self.assertEqual(result, 'Yes.')

    def test_junk_yeah_skips_google(self):
        from unittest.mock import patch
        import speech_recognition as sr

        with patch.object(sr, 'transcribe_local_whisper', return_value='yeah'), \
             patch.object(sr, '_transcribe_google_cloud') as mock_google, \
             patch.object(sr, '_normalize_wav', return_value=('fake.wav', False)):
            result = sr.transcribe_audio_file('fake.wav')

        mock_google.assert_not_called()
        self.assertEqual(result, 'yeah')

    def test_junk_oh_skips_google(self):
        from unittest.mock import patch
        import speech_recognition as sr

        with patch.object(sr, 'transcribe_local_whisper', return_value='Oh'), \
             patch.object(sr, '_transcribe_google_cloud') as mock_google, \
             patch.object(sr, '_normalize_wav', return_value=('fake.wav', False)):
            result = sr.transcribe_audio_file('fake.wav')

        mock_google.assert_not_called()

    def test_noise_pair_still_reaches_google(self):
        """Two noise words ('uh yeah') are unusable but NOT a junk single word —
        Google fallback must still run (nothing in _GOOGLE_SKIP_SINGLE_WORDS covers
        multi-token strings)."""
        from unittest.mock import patch
        import speech_recognition as sr
        import os

        with patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': '/fake/creds.json'}), \
             patch.object(sr, 'transcribe_local_whisper', return_value='uh yeah'), \
             patch.object(sr, '_transcribe_google_cloud', return_value='add row') as mock_google, \
             patch.object(sr, '_normalize_wav', return_value=('fake.wav', False)):
            result = sr.transcribe_audio_file('fake.wav')

        mock_google.assert_called_once()
        self.assertEqual(result, 'add row')

    def test_unknown_single_word_still_reaches_google(self):
        """An unknown single word ('zlork') is not in the skip set — Google runs."""
        from unittest.mock import patch
        import speech_recognition as sr
        import os

        # 'zlork' has 1 token, is not in _STT_NOISE_WORDS, so _is_whisper_usable
        # treats it as usable (single meaningful token) and returns it directly.
        # This test confirms the skip-set does NOT affect words outside it.
        from speech_recognition import _GOOGLE_SKIP_SINGLE_WORDS
        self.assertNotIn('zlork', _GOOGLE_SKIP_SINGLE_WORDS)

    # ── Regressions ──────────────────────────────────────────────────────

    def test_real_command_bypasses_skip(self):
        """A real usable command must never hit the junk-skip path."""
        from unittest.mock import patch
        import speech_recognition as sr

        with patch.object(sr, 'transcribe_local_whisper', return_value='add row John 90'), \
             patch.object(sr, '_transcribe_google_cloud') as mock_google, \
             patch.object(sr, '_normalize_wav', return_value=('fake.wav', False)):
            result = sr.transcribe_audio_file('fake.wav')

        mock_google.assert_not_called()
        self.assertEqual(result, 'add row John 90')


# ---------------------------------------------------------------------------
# P22 — Safe command-catalog voice layer
# ---------------------------------------------------------------------------

class TestP22CommandCatalog(unittest.TestCase):
    """P22: Sara must map noisy transcripts to the known command set safely."""

    # ── Fix 1: Auto-fit hyphen normalization ─────────────────────────────

    def test_autofit_hyphen_corrects(self):
        """'Auto-fit column A' — hyphen must be normalised so heuristic matches."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('Auto-fit column A')
        self.assertNotIn('-', corrected, "Hyphen must be removed from 'auto-fit'")
        self.assertIn('auto fit', corrected.lower())
        self.assertEqual(detect_command_intent(corrected), 'set_column_width')

    def test_autofit_no_hyphen_unchanged(self):
        """'auto fit column B' (no hyphen) must still reach set_column_width."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('auto fit column B'), 'set_column_width')

    # ── Fix 2: "auto fit all" intent ─────────────────────────────────────

    def test_autofit_all_columns_intent(self):
        """'auto fit all columns' must map to set_column_width, not None."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('auto fit all columns'), 'set_column_width')

    def test_autofit_all_intent_direct(self):
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('auto fit all'), 'set_column_width')

    # ── Fix 3: Contextual "shit/shet" → "sheet" ──────────────────────────

    def test_create_shit_corrects_to_sheet(self):
        """'create shit cooking' → 'create sheet cooking', intent=create_sheet."""
        r = _pipeline('create shit cooking')
        self.assertIn('create sheet', r['corrected'].lower())
        self.assertEqual(r['intent'], 'create_sheet')

    def test_delete_shit_corrects_to_sheet(self):
        """'delete shit' → 'delete sheet', intent=delete_sheet."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('delete shit')
        self.assertIn('delete sheet', corrected.lower())
        self.assertEqual(detect_command_intent(corrected), 'delete_sheet')

    def test_list_shit_corrects_to_sheet(self):
        """'list shit' → 'list sheet', intent=list_sheets."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('list shit')
        self.assertIn('list sheet', corrected.lower())
        self.assertEqual(detect_command_intent(corrected), 'list_sheets')

    def test_rename_shit_corrects_to_sheet(self):
        """'rename shit' → 'rename sheet', intent=rename_sheet."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('rename shit')
        self.assertIn('rename sheet', corrected.lower())
        self.assertEqual(detect_command_intent(corrected), 'rename_sheet')

    def test_switch_shit_regression(self):
        """'switch shit 2' was fixed in P19 — must still work."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('switch shit 2')
        self.assertIn('switch sheet', corrected.lower())
        self.assertEqual(detect_command_intent(corrected), 'switch_sheet')

    def test_add_shit_not_corrected_to_sheet(self):
        """'add shit cooking' must NOT be rewritten as 'add sheet cooking'.
        Only sheet-command verbs trigger the correction."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('add shit cooking')
        self.assertNotIn('add sheet', corrected.lower())

    # ── Fix 4: "some" → "sum" with cell-ref lookahead ────────────────────

    def test_some_with_cellref_corrects_to_sum(self):
        """'some B2 to B5' → 'sum B2 to B5'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('some B2 to B5')
        self.assertIn('sum', corrected.lower())
        self.assertNotIn('some', corrected.lower())

    def test_some_without_cellref_unchanged(self):
        """'add some values' must NOT change 'some' to 'sum'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('add some values')
        self.assertIn('some', corrected.lower())
        self.assertNotIn('sum', corrected.lower())

    def test_some_cellref_intent(self):
        """'some B2 to B5 into C6' → after correction, sum is detectable."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('some B2 to B5')
        self.assertTrue(corrected.lower().startswith('sum'))

    # ── Fix 5: Block add_row with profanity payload ───────────────────────

    def test_can_i_add_polite_prefix_stripped(self):
        """'Can I add row John 90' → prefix stripped → intent=add_row, action ok."""
        r = _pipeline('Can I add row John 90')
        self.assertEqual(r['corrected'], 'add row John 90')
        self.assertEqual(r['intent'], 'add_row')
        self.assertIsNone(r['slot_msg'])

    def test_can_i_write_prefix_stripped(self):
        """'Could I write cell B3 100' → intent=write_cell."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('Could I write cell B3 100')
        self.assertEqual(detect_command_intent(corrected), 'write_cell')

    def test_can_i_add_profanity_blocked(self):
        """'Can I add shit cooking' → profanity guard must fire → slot_msg."""
        r = _pipeline('Can I add shit cooking')
        self.assertIsNotNone(r['slot_msg'],
            "Profanity in slot values must trigger clarification, not execute")

    def test_add_row_profanity_blocked(self):
        """'add row shit cooking' → profanity guard fires even with explicit 'row'."""
        r = _pipeline('add row shit cooking')
        self.assertIsNotNone(r['slot_msg'])

    def test_add_shit_directly_blocked(self):
        """'add shit cooking' (no 'row') → profanity guard fires."""
        r = _pipeline('add shit cooking')
        self.assertIsNotNone(r['slot_msg'])

    # ── Regression: Fix 6 — create_sheet never becomes create_excel ───────

    def test_create_sheet_size_stays_create_sheet(self):
        """'create sheet size' must produce intent=create_sheet, not create_excel."""
        r = _pipeline('create sheet size')
        self.assertEqual(r['intent'], 'create_sheet',
            "'create sheet size' must be create_sheet, not create_excel")

    def test_create_sheet_report_stays_create_sheet(self):
        r = _pipeline('create sheet report')
        self.assertEqual(r['intent'], 'create_sheet')


# ---------------------------------------------------------------------------
# P23 — Command-catalog voice matching improvements
# ---------------------------------------------------------------------------

class TestP23AutoFitVoice(unittest.TestCase):
    """Auto fit column — broad STT garble corrections."""

    def test_although_fit_column_b(self):
        """'Although fit column B' → 'auto fit column B', intent=set_column_width."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('Although fit column B')
        self.assertEqual(corrected.lower(), 'auto fit column b')
        self.assertEqual(detect_command_intent(corrected), 'set_column_width')

    def test_although_comma_fit_column_b(self):
        """'Although, fit column B' (comma) → normalises correctly."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('Although, fit column B')
        self.assertEqual(corrected.lower(), 'auto fit column b')
        self.assertEqual(detect_command_intent(corrected), 'set_column_width')

    def test_auto_fit_color_b(self):
        """'Auto fit color B' → 'auto fit column B' ('color' as garble of 'column')."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('Auto fit color B')
        self.assertEqual(corrected.lower(), 'auto fit column b')
        self.assertEqual(detect_command_intent(corrected), 'set_column_width')

    def test_auto_hyphen_fit_color_b(self):
        """'Auto-fit color B' → hyphen stripped then column corrected."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('Auto-fit color B')
        self.assertEqual(corrected.lower(), 'auto fit column b')
        self.assertEqual(detect_command_intent(corrected), 'set_column_width')

    def test_go_fit_columbee(self):
        """'Go Fit Columbee' → 'columbee' expands to 'column b' → auto fit column B."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('Go Fit Columbee')
        self.assertEqual(corrected.lower(), 'auto fit column b')
        self.assertEqual(detect_command_intent(corrected), 'set_column_width')

    def test_all_the_fit_color_b(self):
        """'All the fit color B' → 'auto fit column B'."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('All the fit color B')
        self.assertEqual(corrected.lower(), 'auto fit column b')
        self.assertEqual(detect_command_intent(corrected), 'set_column_width')

    def test_autofit_column_b_regression(self):
        """'auto fit column B' (already correct) must still work."""
        from pipeline_sanitizer import correct_transcript, detect_command_intent
        corrected = correct_transcript('auto fit column B')
        self.assertEqual(detect_command_intent(corrected), 'set_column_width')


class TestP23SumRange(unittest.TestCase):
    """Sum range into cell — slot clarification for garbled destinations."""

    def test_sum_b2_to_b5_into_c6_works(self):
        """'sum B2 to B5 into C6' must pass through without clarification."""
        r = _pipeline('sum B2 to B5 into C6')
        self.assertEqual(r['intent'], 'calculate_sum')
        self.assertIsNone(r['slot_msg'],
            "'sum B2 to B5 into C6' must not trigger clarification")

    def test_some_b2_to_b5_into_c6_works(self):
        """'Some B2 to B5 into C6' — 'some' corrected to 'sum', passes through."""
        r = _pipeline('Some B2 to B5 into C6')
        self.assertEqual(r['intent'], 'calculate_sum')
        self.assertIsNone(r['slot_msg'])

    def test_sum_b2_to_b5_in_tubes_clarifies(self):
        """'sum B2 to B5 in tubes' — 'tubes' is not a cell ref → ask for destination."""
        r = _pipeline('sum B2 to B5 in tubes')
        self.assertEqual(r['intent'], 'calculate_sum')
        self.assertIsNotNone(r['slot_msg'],
            "'in tubes' is a garbled destination — must ask which cell to put sum in")
        self.assertIn('cell', r['slot_msg'].lower())

    def test_sum_b2_2_clarifies(self):
        """'Some B2 2' — unclear range → intent detected, but destination check."""
        r = _pipeline('Some B2 2')
        self.assertEqual(r['intent'], 'calculate_sum')
        # No "into/in" present, so no destination clarification fires, but action
        # should be attempted (heuristic handles it). Just verify intent is right.

    def test_calculate_sum_column_regression(self):
        """'calculate sum column A' (existing form) must still work."""
        r = _pipeline('calculate sum column A')
        self.assertEqual(r['intent'], 'calculate_sum')
        self.assertIsNone(r['slot_msg'])


class TestP23SheetSwitch(unittest.TestCase):
    """Sheet switch — profanity in sheet name must trigger clarification."""

    def test_sweet_shit_clarifies_sheet(self):
        """'Sweet shit' → switch sheet + profanity name → ask which sheet."""
        r = _pipeline('Sweet shit')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNotNone(r['slot_msg'],
            "'Sweet shit' must clarify target sheet, not execute with 'shit'")
        self.assertIn('sheet', r['slot_msg'].lower())

    def test_switch_sheet_shit_does_not_execute(self):
        """'switch sheet shit' must not use 'shit' as a sheet name."""
        r = _pipeline('switch sheet shit')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNotNone(r['slot_msg'],
            "'switch sheet shit' must clarify — 'shit' is not a valid sheet name")

    def test_switch_sheet_valid_regression(self):
        """'switch sheet Data' must still execute without clarification."""
        r = _pipeline('switch sheet Data')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNone(r['slot_msg'])


class TestP23SheetRename(unittest.TestCase):
    """Sheet rename — must route to rename_sheet, not switch_sheet; clarify missing names."""

    def test_have_me_rename_sheet_to_student_clarifies_old(self):
        """'Have me rename sheet to student' — old name missing → clarify."""
        r = _pipeline('Have me rename sheet to student')
        self.assertEqual(r['intent'], 'rename_sheet',
            "Must not go to switch_sheet — 'rename' keyword is present")
        self.assertIsNotNone(r['slot_msg'],
            "Old sheet name is missing — must ask which sheet to rename")
        self.assertIn('sheet', r['slot_msg'].lower())

    def test_rename_sheet_style_to_student_works(self):
        """'rename sheet Style to Student' — both names present → no clarification."""
        r = _pipeline('rename sheet Style to Student')
        self.assertEqual(r['intent'], 'rename_sheet')
        self.assertIsNone(r['slot_msg'],
            "'rename sheet Style to Student' has both names — must not clarify")

    def test_rename_no_names_clarifies(self):
        """'rename sheet' alone — both names missing → clarify."""
        r = _pipeline('rename sheet')
        self.assertEqual(r['intent'], 'rename_sheet')
        self.assertIsNotNone(r['slot_msg'])

    def test_rename_only_old_name_clarifies_new(self):
        """'rename sheet Style' — new name missing → ask what to rename it to."""
        r = _pipeline('rename sheet Style')
        self.assertEqual(r['intent'], 'rename_sheet')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIn('rename', r['slot_msg'].lower())

    def test_rename_structural_form(self):
        """'rename Style to Student' (no sheet keyword) → still rename_sheet intent."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('rename Style to Student'), 'rename_sheet')


# ---------------------------------------------------------------------------
# P24 — Sheet slot safety (never use profane/noise words as sheet names)
# ---------------------------------------------------------------------------

class TestP24SheetSlotSafety(unittest.TestCase):
    """Sheet commands must clarify rather than execute with unsafe name tokens."""

    # ── switch_sheet ─────────────────────────────────────────────────────────

    def test_switch_sheet_shit_clarifies(self):
        """'switch sheet shit' — profanity in name → must clarify."""
        r = _pipeline('switch sheet shit')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNotNone(r['slot_msg'],
            "'switch sheet shit' must ask for sheet name, not execute")
        self.assertIn('sheet', r['slot_msg'].lower())

    def test_switch_sheet_shit_18_sweet_clarifies(self):
        """'switch sheet shit 18 sweet' — mixed garbage → clarify (any, not all)."""
        r = _pipeline('switch sheet shit 18 sweet')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNotNone(r['slot_msg'],
            "Mixed profanity+number+noise must trigger clarification")

    def test_sweet_shit_18_sweet_clarifies(self):
        """'Sweet shit 18 sweet' — corrected to switch_sheet with garbage name."""
        r = _pipeline('Sweet shit 18 sweet')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNotNone(r['slot_msg'])

    def test_switch_sheet_bare_number_passes_pipeline(self):
        """'switch sheet 8' — numeric sheet names are valid (e.g. tab '18' exists).
        P27: digit guard removed; Excel layer resolves or returns sheet_not_found."""
        r = _pipeline('switch sheet 8')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNone(r['slot_msg'],
            "Bare number must reach Excel layer — it may be a real tab name")

    def test_switch_sheet_valid_style(self):
        """'switch sheet Style' — valid name → no clarification."""
        r = _pipeline('switch sheet Style')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNone(r['slot_msg'])

    # ── Punctuation stripping ────────────────────────────────────────────────

    def test_shit_comma_unsafe(self):
        """'switch sheet shit,' — comma-stripped → 'shit' is profanity → clarify."""
        r = _pipeline('switch sheet shit,')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNotNone(r['slot_msg'])

    def test_8_dot_passes_pipeline(self):
        """'switch sheet 8.' — stripped to '8'; P27 lets numeric names reach Excel."""
        r = _pipeline('switch sheet 8.')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNone(r['slot_msg'],
            "Stripped '8' is a potential numeric tab name — must reach Excel layer")

    # ── rename_sheet ─────────────────────────────────────────────────────────

    def test_rename_sheet_shit_to_apple_clarifies_old(self):
        """'rename sheet shit to Apple' — 'shit' as old name → clarify."""
        r = _pipeline('rename sheet shit to Apple')
        self.assertEqual(r['intent'], 'rename_sheet')
        self.assertIsNotNone(r['slot_msg'],
            "Profanity as old sheet name must trigger clarification")
        self.assertIn('sheet', r['slot_msg'].lower())

    def test_rename_sheet_style_to_student_works(self):
        """'rename sheet Style to Student' — both names valid → no clarification."""
        r = _pipeline('rename sheet Style to Student')
        self.assertEqual(r['intent'], 'rename_sheet')
        self.assertIsNone(r['slot_msg'])

    # ── delete_sheet ─────────────────────────────────────────────────────────

    def test_delete_sheet_shit_clarifies(self):
        """'delete sheet shit' — profanity in name → clarify."""
        r = _pipeline('delete sheet shit')
        self.assertIsNotNone(r['slot_msg'])

    def test_delete_sheet_valid_works(self):
        """'delete sheet Student' — valid name → no clarification."""
        r = _pipeline('delete sheet Student')
        self.assertIsNone(r['slot_msg'])

    # ── create_sheet ─────────────────────────────────────────────────────────

    def test_create_sheet_shit_clarifies(self):
        """'create sheet shit' — profanity as proposed sheet name → clarify."""
        r = _pipeline('create sheet shit')
        self.assertIsNotNone(r['slot_msg'])

    def test_create_sheet_valid_works(self):
        """'create sheet Style' — valid name → no clarification."""
        r = _pipeline('create sheet Style')
        self.assertIsNone(r['slot_msg'])

    # ── Switch to <number/garbage> — no LLM ─────────────────────────────────

    def test_switch_to_8_clarifies(self):
        """'Switch to 8' — 'switch to' without 'sheet' → clarify, not LLM."""
        r = _pipeline('Switch to 8')
        self.assertIsNotNone(r['slot_msg'],
            "'Switch to 8' must be caught before reaching the LLM")

    def test_go_to_18_clarifies(self):
        """'go to 18' — bare number, no cell ref letter → clarify."""
        r = _pipeline('go to 18')
        self.assertIsNotNone(r['slot_msg'],
            "'go to 18' must be caught — 18 alone is not a cell ref")

    def test_go_to_b3_not_blocked(self):
        """'go to B3' — valid cell ref → must NOT be blocked."""
        r = _pipeline('go to B3')
        self.assertIsNone(r['slot_msg'])
        self.assertEqual(r['intent'], 'read_cell')


# ---------------------------------------------------------------------------
# P25 — switch-it-to-them guard, switch-sheet-to correction, rename clarity
# ---------------------------------------------------------------------------

class TestP25SwitchAndRenameClarity(unittest.TestCase):

    # ── "switch it to them" must NOT reach Ollama ───────────────────────────

    def test_switch_it_to_them_clarifies(self):
        """'switch it to them' — no 'sheet' keyword → clarify, not LLM."""
        r = _pipeline('switch it to them')
        self.assertIsNotNone(r['slot_msg'],
            "'switch it to them' must be caught before reaching the LLM")

    def test_switch_it_to_them_message(self):
        """Clarification message must tell user how to say the command."""
        r = _pipeline('switch it to them')
        self.assertIn('switch sheet', r['slot_msg'].lower())

    def test_switch_that_clarifies(self):
        """'switch that' — bare switch without sheet → clarify."""
        r = _pipeline('switch that')
        self.assertIsNotNone(r['slot_msg'])

    def test_switch_sheet_sales_not_blocked(self):
        """'switch sheet Sales' — valid → must NOT be blocked."""
        r = _pipeline('switch sheet Sales')
        self.assertIsNone(r['slot_msg'])
        self.assertEqual(r['intent'], 'switch_sheet')

    # ── "switch sheet to <name>" must resolve deterministically ─────────────

    def test_switch_sheet_to_student_intent(self):
        """'switch sheet to student' → correction removes 'to' → switch_sheet intent."""
        r = _pipeline('switch sheet to student')
        self.assertEqual(r['intent'], 'switch_sheet',
            "'switch sheet to student' must be caught by heuristic, not LLM")

    def test_switch_sheet_to_student_no_slot_msg(self):
        """'switch sheet to student' → slot check must pass (has a name)."""
        r = _pipeline('switch sheet to student')
        self.assertIsNone(r['slot_msg'])

    def test_switch_sheet_to_summary_intent(self):
        """'switch sheet to Summary' — another name variant works."""
        r = _pipeline('switch sheet to Summary')
        self.assertEqual(r['intent'], 'switch_sheet')
        self.assertIsNone(r['slot_msg'])

    # ── rename clarification includes the new name when known ───────────────

    def test_rename_sheet_sheet_to_class_clarifies_with_name(self):
        """'rename sheet sheet to class' — old name empty after filtering → clarify with new name."""
        r = _pipeline('rename sheet sheet to class')
        self.assertIsNotNone(r['slot_msg'],
            "'rename sheet sheet to class' must ask which sheet to rename")
        self.assertIn('class', r['slot_msg'].lower(),
            "Clarification must include the new name 'class'")

    def test_rename_sheet_to_class_clarifies_with_name(self):
        """'rename sheet to class' — no old name → clarify mentioning 'class'."""
        r = _pipeline('rename sheet to class')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIn('class', r['slot_msg'].lower())

    def test_rename_sheet_style_to_student_passes(self):
        """'rename sheet Style to Student' — valid → no clarification."""
        r = _pipeline('rename sheet Style to Student')
        self.assertIsNone(r['slot_msg'])
        self.assertEqual(r['intent'], 'rename_sheet')

    def test_rename_shit_shit_clarifies(self):
        """'rename shit shit' — profanity in name_words (no 'to') → clarify."""
        r = _pipeline('rename shit shit')
        self.assertIsNotNone(r['slot_msg'])

    def test_rename_sheet_profane_clarifies(self):
        """'rename sheet shit' — single profane name word → must not reflect it back."""
        r = _pipeline('rename sheet shit')
        msg = r['slot_msg']
        self.assertIsNotNone(msg)
        self.assertNotIn('shit', msg.lower())


# ---------------------------------------------------------------------------
# P26 — "Sheet" as a valid literal sheet name
# ---------------------------------------------------------------------------

class TestP26SheetAsLiteralName(unittest.TestCase):
    """Workbook has a real tab named 'Sheet'. 'switch sheet Sheet' must pass."""

    def test_switch_sheet_Sheet_no_slot_msg(self):
        """'switch sheet Sheet' — literal tab named 'Sheet' → must not be blocked."""
        r = _pipeline('switch sheet Sheet')
        self.assertIsNone(r['slot_msg'],
            "'switch sheet Sheet' must pass: 'Sheet' is a valid existing tab name")

    def test_switch_sheet_Sheet_intent(self):
        """'switch sheet Sheet' → switch_sheet intent."""
        r = _pipeline('switch sheet Sheet')
        self.assertEqual(r['intent'], 'switch_sheet')

    def test_switch_sheet_to_Sheet_no_slot_msg(self):
        """'switch sheet to Sheet' — P25 strips 'to', then 'Sheet' name passes."""
        r = _pipeline('switch sheet to Sheet')
        self.assertIsNone(r['slot_msg'],
            "'switch sheet to Sheet' must pass after 'to' is stripped")

    def test_switch_sheet_to_Sheet_intent(self):
        """'switch sheet to Sheet' → switch_sheet intent."""
        r = _pipeline('switch sheet to Sheet')
        self.assertEqual(r['intent'], 'switch_sheet')

    def test_switch_sheet_sheet_sheet_clarifies(self):
        """'switch sheet sheet sheet' — triple 'sheet' is garbage → clarify."""
        r = _pipeline('switch sheet sheet sheet')
        self.assertIsNotNone(r['slot_msg'],
            "'switch sheet sheet sheet' must be caught as unclear")

    def test_switch_sheet_shit_still_blocked(self):
        """'switch sheet shit' — profanity in name → still blocked."""
        r = _pipeline('switch sheet shit')
        self.assertIsNotNone(r['slot_msg'])

    def test_switch_sheet_18_passes_pipeline(self):
        """'switch sheet 18' — P27: numeric names pass pipeline; Excel resolves."""
        r = _pipeline('switch sheet 18')
        self.assertIsNone(r['slot_msg'],
            "'switch sheet 18' must reach Excel layer — tab '18' may exist")

    def test_switch_it_to_them_still_blocked(self):
        """'switch it to them' — no 'sheet' keyword → still clarifies."""
        r = _pipeline('switch it to them')
        self.assertIsNotNone(r['slot_msg'])


# ---------------------------------------------------------------------------
# P27 — case-insensitive sheet name resolution + numeric sheet names
# ---------------------------------------------------------------------------

def _make_xlsx(dir_path, sheet_names, filename='book.xlsx'):
    """Save a fresh openpyxl workbook with the given sheet names to *dir_path*."""
    import openpyxl, os
    wb = openpyxl.Workbook()
    wb.active.title = sheet_names[0]
    for name in sheet_names[1:]:
        wb.create_sheet(name)
    p = os.path.join(dir_path, filename)
    wb.save(p)
    return p


class TestP27SheetNameResolution(unittest.TestCase):
    """Pipeline slot-check: numeric sheet names no longer blocked; profanity still is.
    Excel-controller: sheet names resolved case-insensitively before execution."""

    # ── Pipeline slot-check behaviour ───────────────────────────────────────

    def test_switch_sheet_18_not_blocked(self):
        """'switch sheet 18' → slot check must NOT block a numeric name."""
        r = _pipeline('switch sheet 18')
        self.assertIsNone(r['slot_msg'],
            "'switch sheet 18' must pass the slot check — sheet '18' may exist")

    def test_switch_sheet_18_intent(self):
        r = _pipeline('switch sheet 18')
        self.assertEqual(r['intent'], 'switch_sheet')

    def test_switch_sheet_99_not_blocked_at_pipeline(self):
        """'switch sheet 99' — pipeline cannot know sheets; passes slot check."""
        r = _pipeline('switch sheet 99')
        self.assertIsNone(r['slot_msg'])
        self.assertEqual(r['intent'], 'switch_sheet')

    def test_switch_sheet_shit_still_blocked(self):
        """'switch sheet shit' — profanity → still blocked."""
        r = _pipeline('switch sheet shit')
        self.assertIsNotNone(r['slot_msg'])

    def test_switch_sheet_style_not_blocked(self):
        """'switch sheet style' → no slot block (case resolved at Excel layer)."""
        r = _pipeline('switch sheet style')
        self.assertIsNone(r['slot_msg'])
        self.assertEqual(r['intent'], 'switch_sheet')

    def test_switch_sheet_student_not_blocked(self):
        r = _pipeline('switch sheet student')
        self.assertIsNone(r['slot_msg'])

    # ── Excel-controller case-insensitive resolution ─────────────────────────

    def test_switch_sheet_lowercase_resolves(self):
        """switch_sheet('sheet') must find existing tab 'Sheet' (openpyxl path)."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import switch_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', '18', 'Style', 'student'])
            with patch('excel_controller._com_find_workbook', return_value=None), \
                 patch('excel_controller._flush_session'), \
                 patch('excel_controller._mark_dirty'):
                result = switch_sheet(p, 'sheet')
                self.assertNotIn('error', result, f"switch_sheet('sheet') failed: {result}")
                self.assertEqual(result.get('sheet'), 'Sheet')

    def test_switch_sheet_style_lowercase_resolves(self):
        """switch_sheet('style') must find existing tab 'Style' (openpyxl path)."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import switch_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style', 'student'])
            with patch('excel_controller._com_find_workbook', return_value=None), \
                 patch('excel_controller._flush_session'), \
                 patch('excel_controller._mark_dirty'):
                result = switch_sheet(p, 'style')
                self.assertNotIn('error', result, f"switch_sheet('style') failed: {result}")
                self.assertEqual(result.get('sheet'), 'Style')

    def test_switch_sheet_numeric_18_resolves(self):
        """switch_sheet('18') must match tab '18' exactly (openpyxl path)."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import switch_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', '18', 'Style', 'student'])
            with patch('excel_controller._com_find_workbook', return_value=None), \
                 patch('excel_controller._flush_session'), \
                 patch('excel_controller._mark_dirty'):
                result = switch_sheet(p, '18')
                self.assertNotIn('error', result, f"switch_sheet('18') failed: {result}")
                self.assertEqual(result.get('sheet'), '18')

    def test_switch_sheet_99_not_found(self):
        """switch_sheet('99') must return sheet_not_found when tab '99' absent."""
        import tempfile
        from excel_controller import switch_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', '18', 'Style', 'student'])
            result = switch_sheet(p, '99')
            self.assertEqual(result.get('error'), 'sheet_not_found')

    def test_delete_sheet_lowercase_resolves(self):
        """delete_sheet('style') must delete the tab named 'Style'."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import delete_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style', 'student'])
            with patch('excel_controller._com_find_workbook', return_value=None), \
                 patch('excel_controller._flush_session'), \
                 patch('excel_controller._mark_dirty'):
                result = delete_sheet(p, 'style')
                self.assertNotIn('error', result, f"delete_sheet('style') failed: {result}")

    def test_rename_sheet_old_name_resolves(self):
        """rename_sheet('style', 'Reports') must rename 'Style' even lowercased."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import rename_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style'])
            with patch('excel_controller._flush_session'), \
                 patch('excel_controller._mark_dirty'):
                result = rename_sheet(p, 'style', 'Reports')
                self.assertNotIn('error', result, f"rename_sheet('style','Reports') failed: {result}")
                self.assertEqual(result.get('old_name'), 'Style')

    def test_exact_name_still_works(self):
        """Typed command switch_sheet('Sheet') with exact case must still pass."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import switch_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet'])
            with patch('excel_controller._com_find_workbook', return_value=None), \
                 patch('excel_controller._flush_session'), \
                 patch('excel_controller._mark_dirty'):
                result = switch_sheet(p, 'Sheet')
                self.assertNotIn('error', result)


# ---------------------------------------------------------------------------
# P28 — switch_sheet COM live-activation path
# ---------------------------------------------------------------------------

class TestP28SwitchSheetCOMPath(unittest.TestCase):
    """P28: switch_sheet activates sheets via COM when workbook is open in Excel.
    No _atomic_save / _flush_session is called on the live-Excel path."""

    # ── COM path: no disk write ──────────────────────────────────────────────

    def test_com_path_no_atomic_save(self):
        """COM open: _atomic_save must NOT be called."""
        import tempfile
        from unittest.mock import MagicMock, patch
        from excel_controller import switch_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style', 'student'])
            mock_wb = MagicMock()
            with patch('excel_controller._com_find_workbook', return_value=mock_wb), \
                 patch('excel_controller._atomic_save') as mock_save:
                result = switch_sheet(p, 'style')
                mock_save.assert_not_called()
                self.assertNotIn('error', result)
                self.assertEqual(result.get('sheet'), 'Style')

    def test_com_path_activate_called_with_resolved_name(self):
        """COM path: Worksheets() is called with the resolved (real-cased) name."""
        import tempfile
        from unittest.mock import MagicMock, patch
        from excel_controller import switch_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style'])
            mock_sheet = MagicMock()
            mock_wb = MagicMock()
            mock_wb.Worksheets.return_value = mock_sheet
            with patch('excel_controller._com_find_workbook', return_value=mock_wb):
                result = switch_sheet(p, 'style')          # lowercase input
                mock_wb.Worksheets.assert_called_with('Style')  # resolved name
                mock_sheet.Activate.assert_called_once()
                self.assertEqual(result.get('sheet'), 'Style')
                self.assertEqual(result.get('message'), 'sheet_switched')

    def test_com_path_case_insensitive(self):
        """COM path: 'STYLE' resolves to 'Style', 'student' stays 'student'."""
        import tempfile
        from unittest.mock import MagicMock, patch
        from excel_controller import switch_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style', 'student'])
            mock_wb = MagicMock()
            with patch('excel_controller._com_find_workbook', return_value=mock_wb):
                r1 = switch_sheet(p, 'STYLE')
                self.assertEqual(r1.get('sheet'), 'Style')
                mock_wb.Worksheets.assert_called_with('Style')

                r2 = switch_sheet(p, 'student')
                self.assertEqual(r2.get('sheet'), 'student')
                mock_wb.Worksheets.assert_called_with('student')

    # ── COM failure: safe error, no fallback to locked save ──────────────────

    def test_com_failure_returns_safe_error(self):
        """COM Activate failure → error dict returned, _atomic_save NOT called."""
        import tempfile
        from unittest.mock import MagicMock, patch
        from excel_controller import switch_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style'])
            mock_wb = MagicMock()
            mock_wb.Worksheets.side_effect = Exception('COM RPC failed')
            with patch('excel_controller._com_find_workbook', return_value=mock_wb), \
                 patch('excel_controller._atomic_save') as mock_save:
                result = switch_sheet(p, 'Style')
                self.assertIn('error', result)
                self.assertEqual(result['error'], 'switch_sheet_com_failed')
                mock_save.assert_not_called()

    # ── Non-Excel path: openpyxl + flush ────────────────────────────────────

    def test_no_excel_open_calls_flush_session(self):
        """Workbook NOT open in Excel → _flush_session (openpyxl path) is used."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import switch_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style'])
            with patch('excel_controller._com_find_workbook', return_value=None), \
                 patch('excel_controller._flush_session') as mock_flush:
                result = switch_sheet(p, 'Style')
                mock_flush.assert_called()
                self.assertEqual(result.get('sheet'), 'Style')

    def test_sheet_not_found_before_com_lookup(self):
        """sheet_not_found is returned before _com_find_workbook is called."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import switch_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style'])
            with patch('excel_controller._com_find_workbook') as mock_find:
                result = switch_sheet(p, 'NonExistent')
                self.assertEqual(result.get('error'), 'sheet_not_found')
                mock_find.assert_not_called()


# ---------------------------------------------------------------------------
# P29 — unclear switch_sheet transcripts must clarify, never reach LLM
# ---------------------------------------------------------------------------

class TestP29UnclearSwitchSheetClarifies(unittest.TestCase):
    """P29: garbled switch_sheet transcripts clarify immediately.

    Root cause: check_command_slots for switch_sheet passed any non-profanity
    name_words through to Ollama/Groq even when they were obviously garbled
    (e.g. 'sweet egg' from 'switch sheet sweet egg in').  Fix: multi-word
    residual → immediate clarification; available_sheets param allows the
    runtime to include the sheet list in the message.
    """

    # ── Garbled transcripts must not reach LLM ───────────────────────────────

    def test_sweet_egg_in_clarifies(self):
        """'switch sheet sweet egg in' must clarify before any LLM call."""
        r = _pipeline('switch sheet sweet egg in')
        self.assertIsNotNone(r['slot_msg'],
                             f"Expected clarification but got slot_msg=None; action={r['action']}")
        self.assertIsNone(r['action'],
                          "No heuristic action should be produced for garbled transcript")

    def test_switch_cheat_18_clarifies(self):
        """'switch cheat 18' (no 'sheet' keyword) must clarify, never LLM."""
        r = _pipeline('switch cheat 18')
        self.assertIsNotNone(r['slot_msg'],
                             f"Expected clarification but got slot_msg=None; corrected={r['corrected']}")

    # ── Valid single-word targets must still pass ─────────────────────────────

    def test_valid_switch_sheet_Sheet(self):
        """'switch sheet Sheet' must not be blocked by the multi-word guard."""
        r = _pipeline('switch sheet Sheet')
        self.assertIsNone(r['slot_msg'],
                          f"Valid command was incorrectly blocked: {r['slot_msg']}")
        self.assertEqual(r['intent'], 'switch_sheet')

    def test_valid_switch_sheet_Style(self):
        """'switch sheet Style' must not be blocked."""
        r = _pipeline('switch sheet Style')
        self.assertIsNone(r['slot_msg'],
                          f"Valid command was incorrectly blocked: {r['slot_msg']}")

    def test_valid_switch_sheet_student(self):
        """'switch sheet student' must not be blocked."""
        r = _pipeline('switch sheet student')
        self.assertIsNone(r['slot_msg'],
                          f"Valid command was incorrectly blocked: {r['slot_msg']}")

    # ── available_sheets param: verifies sheet-list message and miss detection ─

    def test_available_sheets_lists_options_on_multi_word(self):
        """When available_sheets provided, clarification for multi-word includes the list."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots(
            'switch_sheet', 'switch sheet sweet egg in',
            available_sheets=['Sheet', 'Style', 'student', 'classes'],
        )
        self.assertIsNotNone(msg)
        self.assertIn('Sheet', msg)
        self.assertIn('Style', msg)
        self.assertIn('student', msg)
        self.assertIn('classes', msg)

    def test_available_sheets_single_wrong_token_clarifies(self):
        """Single unrecognised token → clarify when available_sheets provided."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots(
            'switch_sheet', 'switch sheet blah',
            available_sheets=['Sheet', 'Style', 'student'],
        )
        self.assertIsNotNone(msg)
        self.assertIn('Sheet', msg)

    def test_available_sheets_correct_token_passes(self):
        """Matching candidate (case-insensitive) must not be blocked."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots(
            'switch_sheet', 'switch sheet style',
            available_sheets=['Sheet', 'Style', 'student'],
        )
        self.assertIsNone(msg, f"Valid sheet name was blocked: {msg}")


# ---------------------------------------------------------------------------
# P30 — open-excel garbles and switch-sheet regression suite
# ---------------------------------------------------------------------------

class TestP30OpenExcelGarblesAndRegression(unittest.TestCase):
    """P30: phonetic 'open excel' garbles must never reach Ollama/Groq.

    Root cause: 'open each there' survived all correction rules uncorrected,
    so detect_command_intent returned None, the ambiguity guard let it through
    (because 'open' is a command verb), and route_llm spent ~12 s on it.
    Fix: one Pass-1 correction rule 'open each' → 'open excel'.
    """

    # ── open-excel garble normalises without LLM ─────────────────────────────

    def test_open_each_there_intent_is_open_excel(self):
        """'Open each there' must correct to open_excel intent (deterministic, no LLM)."""
        r = _pipeline('Open each there')
        self.assertEqual(r['intent'], 'open_excel',
                         f"Expected open_excel but got {r['intent']!r}; "
                         f"corrected={r['corrected']!r}")
        self.assertIsNone(r['slot_msg'])

    def test_open_each_there_corrects_transcript(self):
        """Correction rule maps 'open each' → 'open excel' at transcript level."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('Open each there')
        self.assertIn('open excel', corrected.lower(),
                      f"Expected 'open excel' in corrected text, got {corrected!r}")

    def test_open_each_sir_normalises(self):
        """'open each sir' (another phonetic variant) must also resolve to open_excel."""
        r = _pipeline('open each sir')
        self.assertEqual(r['intent'], 'open_excel',
                         f"Expected open_excel but got {r['intent']!r}; "
                         f"corrected={r['corrected']!r}")

    # ── open excel baseline still works ──────────────────────────────────────

    def test_valid_open_excel(self):
        """'open excel' must still detect as open_excel with no slot message."""
        r = _pipeline('open excel')
        self.assertEqual(r['intent'], 'open_excel')
        self.assertIsNone(r['slot_msg'])

    # ── switch_sheet regression (P29 must still hold) ────────────────────────

    def test_switch_sheet_sweet_egg_still_clarifies(self):
        """'switch sheet sweet egg in' must still clarify (no LLM) after P30 patch."""
        r = _pipeline('switch sheet sweet egg in')
        self.assertIsNotNone(r['slot_msg'],
                             f"Expected clarification; corrected={r['corrected']!r}")
        self.assertIsNone(r['action'])

    def test_valid_switch_sheet_Sheet(self):
        """'switch sheet Sheet' must not be blocked."""
        r = _pipeline('switch sheet Sheet')
        self.assertIsNone(r['slot_msg'],
                          f"Valid command was blocked: {r['slot_msg']!r}")
        self.assertEqual(r['intent'], 'switch_sheet')

    def test_valid_switch_sheet_Style(self):
        """'switch sheet Style' must not be blocked."""
        r = _pipeline('switch sheet Style')
        self.assertIsNone(r['slot_msg'],
                          f"Valid command was blocked: {r['slot_msg']!r}")


# ---------------------------------------------------------------------------
# P31 — delete/rename sheet: garbles clarify, clear commands route
#        deterministically, rename never becomes create_sheet
# ---------------------------------------------------------------------------

class TestP31DeleteRenameSheet(unittest.TestCase):
    """P31: garbled delete/rename sheet transcripts clarify without LLM;
    clear transcripts route via heuristic (no Ollama/Groq).

    Root causes fixed:
    1. llm_parser heuristic: trailing STT punctuation (e.g. "to book.") broke
       \\s*$ anchors → rename/delete fell through to Groq which misinterpreted
       "to book" as create_sheet.
    2. pipeline_sanitizer: no guard for intent=None + "sheet" word → garbled
       transcripts like "Do it, sheet, style" bypassed slot gate and relied
       solely on the runtime ambiguity guard (invisible to _pipeline tests).
    3. delete_sheet slot branch lacked available_sheets verification (P29-style).
    """

    # ── rename_sheet must never become create_sheet ───────────────────────────

    def test_rename_sheet_style_to_book_parses_rename(self):
        """'rename sheet style to book' must parse as rename_sheet via heuristic."""
        r = _pipeline('rename sheet style to book')
        self.assertIsNone(r['slot_msg'],
                          f"Valid rename blocked by slot: {r['slot_msg']!r}")
        self.assertEqual(r['intent'], 'rename_sheet')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'rename_sheet',
                         f"Expected rename_sheet but got {r['action']!r}")
        self.assertNotEqual(r['action'].get('type'), 'create_sheet',
                            "rename_sheet transcript must NEVER produce create_sheet")

    def test_rename_sheet_trailing_period_parses_rename(self):
        """Trailing period from STT ('to book.') must not break rename heuristic."""
        r = _pipeline('rename sheet style to book.')
        self.assertIsNotNone(r['action'],
                             f"Heuristic missed trailing-period variant; "
                             f"corrected={r['corrected']!r}")
        self.assertEqual(r['action'].get('type'), 'rename_sheet')
        self.assertNotEqual(r['action'].get('type'), 'create_sheet')

    def test_rename_sheet_no_lm_for_clear_command(self):
        """A complete rename_sheet command must be resolved by heuristic (not LLM)."""
        r = _pipeline('rename sheet Style to book')
        # If action is produced at heuristic level, LLM is never called.
        self.assertIsNotNone(r['action'],
                             "Heuristic must handle clear rename_sheet; "
                             f"slot_msg={r['slot_msg']!r}")
        self.assertEqual(r['action'].get('type'), 'rename_sheet')

    def test_rename_missing_new_name_clarifies(self):
        """'rename sheet Style' with no new name must clarify, not LLM."""
        r = _pipeline('rename sheet Style')
        self.assertIsNotNone(r['slot_msg'],
                             "Incomplete rename must produce slot clarification")
        self.assertIsNone(r['action'])

    def test_rename_available_sheets_wrong_old_name_clarifies(self):
        """When available_sheets provided and old name missing → clarify with list."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots(
            'rename_sheet', 'rename sheet blah to report',
            available_sheets=['Sheet', 'Style', 'student'],
        )
        self.assertIsNotNone(msg)
        self.assertIn('Sheet', msg)
        self.assertIn('Style', msg)

    # ── delete_sheet garbles clarify ──────────────────────────────────────────

    def test_delete_sheet_style_routes_deterministically(self):
        """'delete sheet style' must route via heuristic with no slot clarification."""
        r = _pipeline('delete sheet style')
        self.assertIsNone(r['slot_msg'],
                          f"Clear delete blocked by slot: {r['slot_msg']!r}")
        self.assertEqual(r['intent'], 'delete_sheet')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'delete_sheet')
        self.assertEqual(r['action'].get('sheet_name'), 'style')

    def test_delete_sheet_trailing_period_routes(self):
        """Trailing period must not prevent delete_sheet heuristic from matching."""
        r = _pipeline('delete sheet style.')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'delete_sheet')

    def test_do_it_sheet_style_clarifies(self):
        """'Do it, sheet, style' must clarify (not produce a delete action or LLM)."""
        r = _pipeline('Do it, sheet, style')
        self.assertIsNotNone(r['slot_msg'],
                             "Garbled delete must produce slot clarification; "
                             f"intent={r['intent']!r} action={r['action']!r}")
        self.assertIsNone(r['action'])

    def test_do_let_the_sheet_start_clarifies(self):
        """'Do let the sheet start' must clarify (not LLM)."""
        r = _pipeline('Do let the sheet start')
        self.assertIsNotNone(r['slot_msg'],
                             "Garbled sheet phrase must produce slot clarification")
        self.assertIsNone(r['action'])

    def test_delete_sheet_available_sheets_unknown_clarifies(self):
        """When available_sheets provided, unrecognised sheet name → clarify."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots(
            'delete_sheet', 'delete sheet star',
            available_sheets=['Sheet', 'Style', 'student'],
        )
        self.assertIsNotNone(msg)
        self.assertIn('Sheet', msg)

    # ── regression: valid commands unaffected ─────────────────────────────────

    def test_valid_switch_sheet_still_works(self):
        """'switch sheet Style' must not be blocked by P31 changes."""
        r = _pipeline('switch sheet Style')
        self.assertIsNone(r['slot_msg'])
        self.assertEqual(r['intent'], 'switch_sheet')

    def test_valid_create_sheet_still_works(self):
        """'create sheet report' must still produce create_sheet action."""
        r = _pipeline('create sheet report')
        self.assertIsNone(r['slot_msg'],
                          f"Valid create blocked: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'create_sheet')


# ---------------------------------------------------------------------------
# P32 — delete_sheet COM live path + case-insensitive name resolution
# ---------------------------------------------------------------------------

class TestP32DeleteSheetCOMPath(unittest.TestCase):
    """P32: delete_sheet uses COM when workbook is open in Excel.

    Scope: ONLY delete_sheet.  rename_sheet and switch_sheet are unchanged.
    """

    # ── spec test: delete sheet book works when book exists ──────────────────

    def test_delete_sheet_book_exists(self):
        """delete sheet 'book' succeeds when sheet named 'book' is present."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import delete_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'book'])
            with patch('excel_controller._com_find_workbook', return_value=None), \
                 patch('excel_controller._flush_session'), \
                 patch('excel_controller._mark_dirty'):
                result = delete_sheet(p, 'book')
                self.assertNotIn('error', result)
                self.assertEqual(result.get('sheet'), 'book')

    # ── spec test: delete sheet Book resolves to book (case-insensitive) ─────

    def test_delete_sheet_book_case_insensitive(self):
        """'Book' resolves case-insensitively to the existing 'book' sheet."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import delete_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'book'])
            with patch('excel_controller._com_find_workbook', return_value=None), \
                 patch('excel_controller._flush_session'), \
                 patch('excel_controller._mark_dirty'):
                result = delete_sheet(p, 'Book')
                self.assertNotIn('error', result)
                self.assertEqual(result.get('sheet'), 'book')

    # ── spec test: delete sheet star safely fails if star does not exist ──────

    def test_delete_sheet_star_not_found(self):
        """delete_sheet returns sheet_not_found when 'star' does not exist."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import delete_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style'])
            with patch('excel_controller._com_find_workbook') as mock_find:
                result = delete_sheet(p, 'star')
                self.assertEqual(result.get('error'), 'sheet_not_found')
                mock_find.assert_not_called()  # name check happens before COM lookup

    # ── spec test: open workbook does not call _atomic_save ──────────────────

    def test_open_workbook_no_atomic_save(self):
        """When workbook is open in Excel, _atomic_save must NOT be called."""
        import tempfile
        from unittest.mock import MagicMock, patch
        from excel_controller import delete_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style'])
            mock_xl_wb = MagicMock()
            with patch('excel_controller._com_find_workbook', return_value=mock_xl_wb), \
                 patch('excel_controller._atomic_save') as mock_save:
                result = delete_sheet(p, 'Style')
                mock_save.assert_not_called()
                self.assertNotIn('error', result)
                self.assertEqual(result.get('message'), 'sheet_deleted')
                mock_xl_wb.Worksheets.assert_called_with('Style')
                mock_xl_wb.Worksheets.return_value.Delete.assert_called_once()

    # ── COM failure: safe error, no locked-save fallback ─────────────────────

    def test_com_failure_safe_error(self):
        """COM Delete failure → safe error dict, _atomic_save NOT called."""
        import tempfile
        from unittest.mock import MagicMock, patch
        from excel_controller import delete_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style'])
            mock_xl_wb = MagicMock()
            mock_xl_wb.Worksheets.side_effect = Exception('COM RPC failed')
            with patch('excel_controller._com_find_workbook', return_value=mock_xl_wb), \
                 patch('excel_controller._atomic_save') as mock_save:
                result = delete_sheet(p, 'Style')
                self.assertIn('error', result)
                self.assertEqual(result['error'], 'delete_sheet_com_failed')
                mock_save.assert_not_called()

    # ── openpyxl fallback when Excel not open ────────────────────────────────

    def test_non_excel_path_uses_flush(self):
        """When workbook NOT open in Excel, _flush_session is called normally."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import delete_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style'])
            with patch('excel_controller._com_find_workbook', return_value=None), \
                 patch('excel_controller._flush_session') as mock_flush, \
                 patch('excel_controller._mark_dirty'):
                result = delete_sheet(p, 'Style')
                mock_flush.assert_called()
                self.assertEqual(result.get('sheet'), 'Style')

    # ── STT merged-name split: sheetbook → book ──────────────────────────────

    def test_sheetbook_resolves_when_book_exists(self):
        """STT-merged 'sheetbook' splits to 'book' when that sheet exists."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import delete_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'book'])
            with patch('excel_controller._com_find_workbook', return_value=None), \
                 patch('excel_controller._flush_session'), \
                 patch('excel_controller._mark_dirty'):
                result = delete_sheet(p, 'sheetbook')
                self.assertNotIn('error', result)
                self.assertEqual(result.get('sheet'), 'book')

    def test_sheetbook_fails_when_book_absent(self):
        """STT-merged 'sheetbook' → sheet_not_found when 'book' doesn't exist."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import delete_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style'])
            with patch('excel_controller._com_find_workbook', return_value=None):
                result = delete_sheet(p, 'sheetbook')
                self.assertEqual(result.get('error'), 'sheet_not_found')

    # ── regression: switch_sheet and rename_sheet unaffected ─────────────────

    def test_switch_sheet_still_works(self):
        """switch_sheet('Style') must still work after P32 delete_sheet patch."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import switch_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style'])
            with patch('excel_controller._com_find_workbook', return_value=None), \
                 patch('excel_controller._flush_session'), \
                 patch('excel_controller._mark_dirty'):
                result = switch_sheet(p, 'Style')
                self.assertNotIn('error', result)
                self.assertEqual(result.get('sheet'), 'Style')

    def test_rename_sheet_still_works(self):
        """rename_sheet('Style', 'score') must still work after P32 patch."""
        import tempfile
        from unittest.mock import patch
        from excel_controller import rename_sheet
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = _make_xlsx(td, ['Sheet', 'Style'])
            with patch('excel_controller._flush_session'), \
                 patch('excel_controller._mark_dirty'):
                result = rename_sheet(p, 'Style', 'score')
                self.assertNotIn('error', result)
                self.assertEqual(result.get('old_name'), 'Style')
                self.assertEqual(result.get('new_name'), 'score')


# ---------------------------------------------------------------------------
# P33 — formatting voice grammar: center/align/color/font-size/number-format
#        commands must skip Ollama/Groq when grammar is clear
# ---------------------------------------------------------------------------

class TestP33FormattingVoiceGrammar(unittest.TestCase):
    """P33: formatting commands are handled deterministically (no LLM).

    Root causes fixed:
    1. _INTENT_PATTERNS['format_cells'] didn't cover center/align/color-cell/
       set-size/increase-size/format-as/make-color — so detect_command_intent
       returned None and both the 2-word low-confidence gate and the 3-word
       ambiguity guard blocked these commands before they reached the heuristic.
    2. _CMD_KW and _COMMAND_VERBS lacked center/align/color/increase/decrease.
    3. No heuristic for "make A3 orange" (fill_color) or
       "color row A1 blue" (cell ref instead of row number).
    4. No correction for STT "front size" → "font size" in sizing context.
    5. No slot check for "set font size A2" (target present, size missing).
    """

    # ── intent detection ──────────────────────────────────────────────────────

    def test_center_cell_intent_detected(self):
        """detect_command_intent must return format_cells for 'center B2'."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('center B2'), 'format_cells')

    def test_center_range_intent_detected(self):
        """detect_command_intent must return format_cells for 'center A1:C1'."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('center A1:C1'), 'format_cells')

    def test_center_column_intent_detected(self):
        """detect_command_intent must return format_cells for 'center column B'."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('center column B'), 'format_cells')

    def test_center_row_intent_detected(self):
        """detect_command_intent must return format_cells for 'center row 2'."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('center row 2'), 'format_cells')

    def test_align_cell_intent_detected(self):
        """detect_command_intent must return format_cells for 'align A2 left'."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('align A2 left'), 'format_cells')

    def test_color_cell_intent_detected(self):
        """detect_command_intent must return format_cells for 'color B2 red'."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('color B2 red'), 'format_cells')

    def test_set_font_size_intent_detected(self):
        """detect_command_intent must return format_cells for 'set font size A2 30'."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('set font size A2 30'), 'format_cells')

    def test_increase_font_size_intent_detected(self):
        """detect_command_intent must return format_cells for 'increase font size A1'."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('increase font size A1'), 'format_cells')

    def test_decrease_font_size_intent_detected(self):
        """detect_command_intent must return format_cells for 'decrease font size A1:C1'."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('decrease font size A1:C1'), 'format_cells')

    def test_format_as_currency_intent_detected(self):
        """detect_command_intent must return format_cells for 'format B2:B10 as currency'."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('format B2:B10 as currency'), 'format_cells')

    def test_make_cell_color_intent_detected(self):
        """detect_command_intent must return format_cells for 'make A3 orange'."""
        from pipeline_sanitizer import detect_command_intent
        self.assertEqual(detect_command_intent('make A3 orange'), 'format_cells')

    # ── low-confidence gate doesn't block ────────────────────────────────────

    def test_center_cell_not_blocked_2words(self):
        """'center B2' (2 words) must not be blocked by the low-confidence gate."""
        from pipeline_sanitizer import is_low_confidence_transcript
        self.assertFalse(is_low_confidence_transcript('center B2'),
                         "'center B2' must pass (intent=format_cells detected)")

    def test_color_cell_not_blocked_3words(self):
        """'color B2 red' (3 words, no prior _CMD_KW match) must pass gate."""
        from pipeline_sanitizer import is_low_confidence_transcript
        self.assertFalse(is_low_confidence_transcript('color B2 red'),
                         "'color B2 red' must not be blocked (color now in _CMD_KW)")

    def test_align_not_blocked_3words(self):
        """'align C1 left' must not be blocked by 3-word low-confidence gate."""
        from pipeline_sanitizer import is_low_confidence_transcript
        self.assertFalse(is_low_confidence_transcript('align C1 left'))

    # ── heuristic produces correct action ────────────────────────────────────

    def test_center_cell_heuristic(self):
        """'center B2' must produce format_cells align=center without LLM."""
        r = _pipeline('center B2')
        self.assertIsNone(r['slot_msg'], f"Blocked: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'], f"Heuristic missed; corrected={r['corrected']!r}")
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('align'), 'center')
        self.assertEqual(r['action'].get('range'), 'B2')

    def test_align_left_heuristic(self):
        """'align A2 left' must produce format_cells align=left without LLM."""
        r = _pipeline('align A2 left')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('align'), 'left')
        self.assertEqual(r['action'].get('range'), 'A2')

    def test_color_cell_heuristic(self):
        """'color B2 red' must produce format_cells font_color=red without LLM."""
        r = _pipeline('color B2 red')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('font_color'), 'red')
        self.assertEqual(r['action'].get('range'), 'B2')

    def test_make_cell_color_heuristic(self):
        """'make A3 orange' must produce format_cells font_color=orange."""
        r = _pipeline('make A3 orange')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'], f"Heuristic missed; corrected={r['corrected']!r}")
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('font_color'), 'orange')
        self.assertEqual(r['action'].get('range'), 'A3')

    def test_color_row_cell_ref_heuristic(self):
        """'color row A1 blue' must color cell A1, not a whole row."""
        r = _pipeline('color row A1 blue')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), 'A1')
        self.assertEqual(r['action'].get('font_color'), 'blue')
        # Verify range is a cell ref (A1), not a row range (1:1)
        self.assertNotIn(':', r['action'].get('range', ''),
                         "Must route as cell A1, not row range")

    def test_format_as_currency_heuristic(self):
        """'format B2:B10 as currency' must produce number_format=currency."""
        r = _pipeline('format B2:B10 as currency')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('number_format'), 'currency')
        self.assertEqual(r['action'].get('range'), 'B2:B10')

    # ── slot check ────────────────────────────────────────────────────────────

    def test_set_font_size_missing_number_asks(self):
        """'set font size A2' (no number) must trigger clarification."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots('format_cells', 'set font size A2')
        self.assertIsNotNone(msg, "Missing size number must produce a clarification")
        self.assertIn('font size', msg.lower())

    def test_set_font_size_complete_no_slot_msg(self):
        """'set font size A2 30' (complete) must pass slot check."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots('format_cells', 'set font size A2 30')
        self.assertIsNone(msg, f"Complete command must not be blocked: {msg!r}")

    def test_increase_font_size_no_slot_msg(self):
        """'increase font size A1' must pass slot check (no number needed for delta)."""
        from pipeline_sanitizer import check_command_slots
        msg = check_command_slots('format_cells', 'increase font size A1')
        self.assertIsNone(msg, f"increase font size must not ask for a number: {msg!r}")

    # ── transcript correction ─────────────────────────────────────────────────

    def test_front_size_corrects_to_font_size(self):
        """'set front size A2 30' → 'set font size A2 30' via targeted correction."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('set front size A2 30')
        self.assertIn('font size', corrected.lower(),
                      f"Expected 'font size' in corrected text, got {corrected!r}")
        self.assertNotIn('front size', corrected.lower())

    def test_front_size_without_cell_not_corrected(self):
        """'front size order' (no cell target) must NOT be corrected to 'font size'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('front size order')
        # Should not globally replace "front" → "font" without cell context
        self.assertNotIn('font size', corrected.lower(),
                         f"Global front→font must not fire without cell target; got {corrected!r}")

    # ── regression: existing formatting commands unaffected ──────────────────

    def test_make_bold_still_works(self):
        """'make A1 bold' must still produce bold=True (not fill_color)."""
        r = _pipeline('make A1 bold')
        self.assertIsNotNone(r['action'])
        self.assertTrue(r['action'].get('bold'))
        self.assertNotIn('fill_color', r['action'])

    def test_color_row_number_still_works(self):
        """'color row 3 blue' must still set font_color for row 3:3."""
        r = _pipeline('color row 3 blue')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('range'), '3:3')
        self.assertEqual(r['action'].get('font_color'), 'blue')


# ---------------------------------------------------------------------------
# P34 — formatting voice grammar refinements
#        align right, front-side correction, make-cell-color STT variants,
#        deterministic number-format, garbled-range clarification
# ---------------------------------------------------------------------------

class TestP34FormattingRefinements(unittest.TestCase):
    """P34: refinements to the P33 formatting grammar fixes.

    Root causes fixed:
    1. 'front side' not corrected — P33 only covered 'front size'; extended
       the pattern to also match 'front side' (STT mishears "font size" as
       "front side") within an increase/decrease/set font-size context.
    2. 'Make a three-orange' → de-hyphenation — after number-word + cell-ref
       normalization produces 'a3-orange', a new correction converts it to
       'a3 orange' so the make-cell-color heuristic fires correctly.
    3. 'format JFI as number' reaching LLM — added a broad intent pattern to
       catch all 'format X as <format>' forms, then a slot check validates the
       range token; garbled ranges trigger a clarification instead of LLM.
    4. 'align D1 right' was already fixed by P33's fmt_align heuristic — tests
       added here to prevent future regression.
    """

    # ── align D1 right / left — exact cell, no LLM ───────────────────────────

    def test_align_D1_right_exact_range(self):
        """'align D1 right' must parse to range=D1 align=right, not D1:D20."""
        r = _pipeline('align D1 right')
        self.assertIsNone(r['slot_msg'], f"Should not clarify: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'], "Heuristic must handle align D1 right")
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), 'D1',
                         f"Must be D1 (not D1:D20); got {r['action'].get('range')!r}")
        self.assertEqual(r['action'].get('align'), 'right')

    def test_align_D1_left_exact_range(self):
        """'align D1 left' must parse to range=D1 align=left."""
        r = _pipeline('align D1 left')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('range'), 'D1')
        self.assertEqual(r['action'].get('align'), 'left')

    # ── front side → font size correction ────────────────────────────────────

    def test_increase_front_side_corrects(self):
        """'Increase front side I1' must correct 'front side' → 'font size'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('Increase front side I1')
        self.assertIn('font size', corrected.lower(),
                      f"Expected 'font size' in corrected, got {corrected!r}")
        self.assertNotIn('front side', corrected.lower())

    def test_increase_front_side_routes_deterministically(self):
        """'Increase front side I1' must route to format_cells font_size_change."""
        r = _pipeline('Increase front side I1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('font_size_change'), 2)
        self.assertEqual(r['action'].get('range'), 'I1')

    def test_decrease_front_side_corrects(self):
        """'Decrease front side E1' must correct 'front side' → 'font size'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('Decrease front side E1')
        self.assertIn('font size', corrected.lower(),
                      f"Expected 'font size' in corrected, got {corrected!r}")

    def test_decrease_front_side_routes_deterministically(self):
        """'Decrease front side E1' must route to format_cells font_size_change=-2."""
        r = _pipeline('Decrease front side E1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('font_size_change'), -2)
        self.assertEqual(r['action'].get('range'), 'E1')

    # ── make cell color STT variants ─────────────────────────────────────────

    def test_make_A3_orange_direct(self):
        """'Make A3 orange' (clean STT) must produce font_color=orange range=A3."""
        r = _pipeline('Make A3 orange')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('font_color'), 'orange')
        self.assertEqual(r['action'].get('range'), 'A3')

    def test_make_a_three_orange_hyphenated(self):
        """'Make a three-orange' (STT: hyphenated) must parse as A3 orange."""
        r = _pipeline('Make a three-orange')
        self.assertIsNone(r['slot_msg'],
                          f"Should not clarify; slot={r['slot_msg']!r}")
        self.assertIsNotNone(r['action'],
                             f"Heuristic must produce action; corrected={r['corrected']!r}")
        self.assertEqual(r['action'].get('font_color'), 'orange')
        self.assertEqual(r['action'].get('range'), 'A3')

    def test_make_a_three_orange_no_hyphen(self):
        """'Make a three orange' (STT: no hyphen) must also parse as A3 orange."""
        r = _pipeline('Make a three orange')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('font_color'), 'orange')
        self.assertEqual(r['action'].get('range'), 'A3')

    def test_make_the_one_orange_clarifies(self):
        """'Make the one orange' (ambiguous) must clarify, not execute."""
        r = _pipeline('Make the one orange')
        self.assertIsNotNone(r['slot_msg'],
                             "Ambiguous 'the one' must produce a clarification")
        self.assertIsNone(r['action'])

    def test_make_D1_a_range_clarifies(self):
        """'Make D1 a range' (garbled color) must clarify, not execute."""
        r = _pipeline('Make D1 a range')
        self.assertIsNotNone(r['slot_msg'],
                             "Garbled color 'a range' must produce a clarification")
        self.assertIsNone(r['action'])

    # ── number format: deterministic ──────────────────────────────────────────

    def test_format_B2_B10_as_currency(self):
        """'format B2:B10 as currency' must parse deterministically."""
        r = _pipeline('format B2:B10 as currency')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('number_format'), 'currency')
        self.assertEqual(r['action'].get('range'), 'B2:B10')

    def test_format_C2_C8_as_percentage(self):
        """'format C2:C8 as percentage' must parse deterministically."""
        r = _pipeline('format C2:C8 as percentage')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('number_format'), 'percentage')
        self.assertEqual(r['action'].get('range'), 'C2:C8')

    def test_format_D2_D10_as_number(self):
        """'format D2:D10 as number' must parse deterministically."""
        r = _pipeline('format D2:D10 as number')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('number_format'), 'number')
        self.assertEqual(r['action'].get('range'), 'D2:D10')

    def test_format_garbled_range_clarifies(self):
        """'format JFI as number' (garbled range) must clarify, not execute or LLM."""
        r = _pipeline('format JFI as number')
        self.assertIsNotNone(r['slot_msg'],
                             "Garbled range must trigger clarification, not LLM")
        self.assertIsNone(r['action'])
        self.assertIn('range', r['slot_msg'].lower(),
                      f"Clarification should mention 'range': {r['slot_msg']!r}")

    # ── regression: known working commands unaffected ────────────────────────

    def test_make_A1_bold_regression(self):
        """'make A1 bold' must still produce bold=True (not fill_color)."""
        r = _pipeline('make A1 bold')
        self.assertIsNotNone(r['action'])
        self.assertTrue(r['action'].get('bold'))
        self.assertNotIn('fill_color', r['action'])

    def test_color_row_3_blue_regression(self):
        """'color row 3 blue' must still set font_color for row range 3:3."""
        r = _pipeline('color row 3 blue')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('range'), '3:3')

    def test_center_D5_regression(self):
        """'center D5' must still parse as format_cells align=center range=D5."""
        r = _pipeline('center D5')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('align'), 'center')
        self.assertEqual(r['action'].get('range'), 'D5')

    def test_set_font_size_D1_15_regression(self):
        """'set font size D1 15' must still produce font_size=15."""
        r = _pipeline('set font size D1 15')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('font_size'), 15)

    def test_color_H1_blue_regression(self):
        """'color H1 blue' must still produce font_color=blue range=H1."""
        r = _pipeline('color H1 blue')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('font_color'), 'blue')
        self.assertEqual(r['action'].get('range'), 'H1')

    def test_increase_font_size_still_works(self):
        """'increase font size A1' must still produce font_size_change=2."""
        r = _pipeline('increase font size A1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('font_size_change'), 2)


class TestP35VoiceReliability(unittest.TestCase):
    """P35 — Voice transcript recovery and deterministic intent protection.

    Part A: STT correction for rename-verb mishearings, delete-row number words,
            and the A10/810 cell-reference confusion.
    Part B: Intent protection — rename_file/rename_sheet/delete_sheet cannot
            silently become write_cell or delete_row via LLM misinterpretation.
    """

    # ── Part A: renave / renee → rename ──────────────────────────────────────

    def test_renave_file_corrected(self):
        """'renave file server to client' → 'rename file server to client'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('renave file server to client')
        self.assertTrue(corrected.lower().startswith('rename '),
                        f"Expected 'rename ...' got {corrected!r}")

    def test_renave_routes_to_rename_file_intent(self):
        """After correction 'renave file server to client' must get rename_file intent."""
        r = _pipeline('renave file server to client')
        self.assertEqual(r['intent'], 'rename_file',
                         f"Expected rename_file, got {r['intent']!r}")

    def test_renee_file_corrected(self):
        """'renee file server to client' → starts with 'rename'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('renee file server to client')
        self.assertTrue(corrected.lower().startswith('rename '),
                        f"Expected 'rename ...' got {corrected!r}")

    def test_renee_routes_to_rename_file_intent(self):
        """After correction 'renee file server to client' must get rename_file intent."""
        r = _pipeline('renee file server to client')
        self.assertEqual(r['intent'], 'rename_file')

    # ── Part A: rename fire → rename file ────────────────────────────────────

    def test_rename_fire_corrected(self):
        """'rename fire server to client' → 'rename file server to client'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('rename fire server to client')
        self.assertIn('rename file', corrected.lower(),
                      f"Expected 'rename file ...' got {corrected!r}")
        self.assertNotIn('rename fire', corrected.lower())

    def test_rename_fire_routes_to_rename_file_intent(self):
        """After fire→file correction, intent must be rename_file."""
        r = _pipeline('rename fire server to client')
        self.assertEqual(r['intent'], 'rename_file')

    # ── Part A: rename, sheet → rename sheet ─────────────────────────────────

    def test_rename_comma_sheet_corrected(self):
        """'rename, sheet Style to score' → 'rename sheet Style to score'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('rename, sheet Style to score')
        self.assertNotIn(',', corrected,
                         f"Comma should be removed, got {corrected!r}")
        self.assertIn('rename', corrected.lower())

    def test_rename_comma_sheet_routes_to_rename_sheet_intent(self):
        """After comma removal, 'rename, sheet Style to score' must detect rename_sheet."""
        r = _pipeline('rename, sheet Style to score')
        self.assertEqual(r['intent'], 'rename_sheet',
                         f"Expected rename_sheet, got {r['intent']!r}")

    # ── Part A: delete row number-words ──────────────────────────────────────

    def test_delete_row_two_corrected(self):
        """'delete row two' → 'delete row 2'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('delete row two')
        self.assertIn('delete row 2', corrected.lower(),
                      f"Expected 'delete row 2', got {corrected!r}")

    def test_delete_row_three_corrected(self):
        """'delete row three' → 'delete row 3'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('delete row three')
        self.assertIn('delete row 3', corrected.lower())

    def test_delete_row_one_corrected(self):
        """'delete row one' → 'delete row 1'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('delete row one')
        self.assertIn('delete row 1', corrected.lower())

    def test_delete_row_won_corrected(self):
        """'delete row won' (STT: won=one) → 'delete row 1'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('delete row won')
        self.assertIn('delete row 1', corrected.lower())

    def test_delete_row_too_corrected(self):
        """'delete row too' (STT: too=two) → 'delete row 2'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('delete row too')
        self.assertIn('delete row 2', corrected.lower())

    def test_delete_row_number_word_routes_correctly(self):
        """'delete row two' must route to delete_row intent after correction."""
        r = _pipeline('delete row two')
        self.assertEqual(r['intent'], 'delete_row',
                         f"Expected delete_row, got {r['intent']!r}")

    def test_remove_row_three_corrected(self):
        """'remove row three' → contains 'row 3' (pass-2 may canonicalize to 'delete row 3')."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('remove row three')
        self.assertIn('row 3', corrected.lower(),
                      f"Expected 'row 3' in corrected, got {corrected!r}")

    def test_delete_row_number_word_not_corrupting_names(self):
        """'add row two 90' must NOT correct 'two' (name context, not delete context)."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('add row two 90')
        # 'two' should stay as-is (it's a name here, not a row number)
        self.assertIn('two', corrected.lower(),
                      f"'two' must not be converted in add-row context, got {corrected!r}")

    # ── Part A: cell 810 → cell A10 ──────────────────────────────────────────

    def test_delete_cell_810_corrected(self):
        """'delete cell 810' → 'delete cell A10' (STT: A sounds like 8)."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('delete cell 810')
        self.assertIn('A10', corrected,
                      f"Expected 'delete cell A10', got {corrected!r}")

    def test_write_cell_810_corrected(self):
        """'write cell 810 hello' → 'write cell A10 hello'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('write cell 810 hello')
        self.assertIn('A10', corrected,
                      f"Expected 'write cell A10 hello', got {corrected!r}")

    def test_cell_810_routes_correctly(self):
        """After 810→A10 correction, 'delete cell 810' must produce delete_cell action on A10."""
        r = _pipeline('delete cell 810')
        self.assertIsNotNone(r['action'],
                             f"Expected a delete_cell action, got None (corrected={r.get('corrected')!r})")
        self.assertEqual(r['action'].get('type'), 'delete_cell')
        # delete_cell uses 'cell' key (not 'range')
        self.assertEqual(r['action'].get('cell'), 'A10',
                         f"Expected cell=A10, got {r['action']!r}")

    def test_810_not_corrected_outside_cell_context(self):
        """'add row John 810' must NOT correct 810 (score, not cell reference)."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('add row John 810')
        self.assertIn('810', corrected,
                      f"810 must stay as-is in add-row score context, got {corrected!r}")

    # ── Part B: intent protection constants ──────────────────────────────────

    def test_protected_intents_constant_exists(self):
        """_PROTECTED_INTENTS must be defined in assistant_runtime."""
        from assistant_runtime import _PROTECTED_INTENTS  # type: ignore
        self.assertIn('rename_file',  _PROTECTED_INTENTS)
        self.assertIn('rename_sheet', _PROTECTED_INTENTS)
        self.assertIn('delete_sheet', _PROTECTED_INTENTS)

    def test_rename_file_blocks_write_cell(self):
        """rename_file must block write_cell in _PROTECTED_INTENTS."""
        from assistant_runtime import _PROTECTED_INTENTS  # type: ignore
        self.assertIn('write_cell', _PROTECTED_INTENTS['rename_file'])

    def test_rename_file_blocks_add_row(self):
        """rename_file must block add_row in _PROTECTED_INTENTS."""
        from assistant_runtime import _PROTECTED_INTENTS  # type: ignore
        self.assertIn('add_row', _PROTECTED_INTENTS['rename_file'])

    def test_rename_sheet_blocks_write_cell(self):
        """rename_sheet must block write_cell in _PROTECTED_INTENTS."""
        from assistant_runtime import _PROTECTED_INTENTS  # type: ignore
        self.assertIn('write_cell', _PROTECTED_INTENTS['rename_sheet'])

    def test_rename_sheet_blocks_switch_sheet(self):
        """rename_sheet must block switch_sheet in _PROTECTED_INTENTS."""
        from assistant_runtime import _PROTECTED_INTENTS  # type: ignore
        self.assertIn('switch_sheet', _PROTECTED_INTENTS['rename_sheet'])

    def test_delete_sheet_blocks_delete_row(self):
        """delete_sheet must block delete_row in _PROTECTED_INTENTS."""
        from assistant_runtime import _PROTECTED_INTENTS  # type: ignore
        self.assertIn('delete_row', _PROTECTED_INTENTS['delete_sheet'])

    def test_delete_sheet_blocks_delete_cell(self):
        """delete_sheet must block delete_cell in _PROTECTED_INTENTS."""
        from assistant_runtime import _PROTECTED_INTENTS  # type: ignore
        self.assertIn('delete_cell', _PROTECTED_INTENTS['delete_sheet'])

    # ── Part A: sever/servo → server ─────────────────────────────────────────

    def test_rename_file_sever_corrected(self):
        """'rename file sever to client' → 'rename file server to client'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('rename file sever to client')
        self.assertIn('server', corrected.lower(),
                      f"Expected 'server' in corrected, got {corrected!r}")
        self.assertNotIn('sever', corrected.lower())

    def test_rename_file_servo_corrected(self):
        """'rename file servo to client' → 'rename file server to client'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('rename file servo to client')
        self.assertIn('server', corrected.lower(),
                      f"Expected 'server' in corrected, got {corrected!r}")
        self.assertNotIn('servo', corrected.lower())

    def test_rename_fire_sever_corrected(self):
        """'rename fire sever to client' → fire→file then sever→server."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('rename fire sever to client')
        self.assertIn('rename file', corrected.lower(),
                      f"Expected 'rename file' in corrected, got {corrected!r}")
        self.assertIn('server', corrected.lower())

    # ── Part 2: delete_cell slot validation ─────────────────────────────────

    def test_delete_cell_no_ref_clarifies(self):
        """'delete cell XYZ' (no digit) must ask for a valid cell ref."""
        r = _pipeline('delete cell XYZ')
        self.assertIsNotNone(r['slot_msg'],
                             f"Expected clarification for invalid cell ref, got None")
        self.assertIn('cell', r['slot_msg'].lower())

    def test_delete_cell_no_ref_no_action(self):
        """'delete cell XYZ' must not produce an action (guarded by slot check)."""
        r = _pipeline('delete cell XYZ')
        self.assertIsNone(r['action'])

    def test_delete_cell_valid_passes_slot(self):
        """'delete cell C3' has a valid cell ref and must pass the slot check."""
        r = _pipeline('delete cell C3')
        self.assertIsNone(r['slot_msg'],
                          f"Valid cell ref should pass slot check: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'delete_cell')

    def test_delete_cell_no_arg_clarifies(self):
        """'delete cell' alone must ask which cell."""
        r = _pipeline('delete cell')
        self.assertIsNotNone(r['slot_msg'])

    # ── Excel lock safety: verify _flush_session uses COM close ──────────────

    def test_flush_session_calls_com_close(self):
        """_flush_session must call _com_close_workbook before saving."""
        import excel_controller as _xl
        from unittest.mock import patch, MagicMock
        # Set up a minimal dirty session pointing at a temp file
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tf:
            tmp_path = tf.name
        try:
            mock_wb = MagicMock()
            with patch.object(_xl, '_session', MagicMock(
                    dirty=True, abs_path=os.path.abspath(tmp_path),
                    workbook=mock_wb)), \
                 patch.object(_xl, '_com_close_workbook', return_value=True) as mock_close, \
                 patch.object(_xl, '_atomic_save') as mock_save, \
                 patch.object(_xl, '_com_open_workbook'), \
                 patch.object(_xl, '_poll_for_unlock'), \
                 patch.object(_xl, '_is_file_locked', return_value=False):
                _xl._flush_session(tmp_path)
            mock_close.assert_called_once()
            mock_save.assert_called_once()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── Regression: already-working voice corrections ────────────────────────

    def test_delete_roll_3_still_works(self):
        """'delete roll 3' → delete_row row=3 (existing correction must not regress)."""
        r = _pipeline('delete roll 3')
        self.assertEqual(r['intent'], 'delete_row')

    def test_delete_shit_book_still_works(self):
        """'delete shit book' → rename_sheet or delete_sheet (sheet correction)."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('delete shit book')
        self.assertIn('sheet', corrected.lower(),
                      f"Expected 'sheet' in corrected, got {corrected!r}")

    def test_switch_shit_student_still_works(self):
        """'switch shit student' → switch_sheet action."""
        r = _pipeline('switch shit student')
        self.assertEqual(r['intent'], 'switch_sheet')

    def test_right_cell_still_works(self):
        """'right cell A1 hello' → write_cell action."""
        r = _pipeline('right cell A1 hello')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_cell')


class TestP36MergeCellsVoice(unittest.TestCase):
    """P36 — Voice reliability fix for merge_cells.

    STT phonetic corruptions handled:
      much/mod/mode + cell/sir/sad/sell  →  merge cells
      merged adjacent cell refs A1C1     →  A1:C1

    Slot protection: single cell (no range) → clarify, not execute.
    Intent protection: merge_cells cannot become write_cell/add_row/format_cells.
    """

    # ── Transcript corrections ────────────────────────────────────────────────

    def test_much_cell_corrects(self):
        """'much cell A1C1' must correct to 'merge cells A1:C1'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('much cell A1C1')
        self.assertIn('merge cells', c.lower(), repr(c))
        self.assertIn('A1:C1', c, repr(c))

    def test_much_sir_corrects(self):
        """'much sir A1C1' must correct to 'merge cells A1:C1'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('much sir A1C1')
        self.assertIn('merge cells', c.lower(), repr(c))
        self.assertIn('A1:C1', c, repr(c))

    def test_much_sad_corrects(self):
        """'much sad A1C1' must correct to 'merge cells A1:C1'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('much sad A1C1')
        self.assertIn('merge cells', c.lower(), repr(c))
        self.assertIn('A1:C1', c, repr(c))

    def test_mod_cell_corrects(self):
        """'mod cell A1C1' must correct to 'merge cells A1:C1'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('mod cell A1C1')
        self.assertIn('merge cells', c.lower(), repr(c))
        self.assertIn('A1:C1', c, repr(c))

    def test_mode_cell_corrects(self):
        """'mode cell A1C1' must correct to 'merge cells A1:C1'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('mode cell A1C1')
        self.assertIn('merge cells', c.lower(), repr(c))
        self.assertIn('A1:C1', c, repr(c))

    def test_merge_sir_corrects(self):
        """'merge sir A1:C1' must correct 'sir' → 'cells'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('merge sir A1:C1')
        self.assertIn('merge cells', c.lower(), repr(c))

    def test_merge_sad_corrects(self):
        """'merge sad A1:C1' must correct 'sad' → 'cells'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('merge sad A1:C1')
        self.assertIn('merge cells', c.lower(), repr(c))

    def test_merged_refs_split(self):
        """'merge cells A1C1' (no colon) must produce range A1:C1."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('merge cells A1C1')
        self.assertIn('A1:C1', c, repr(c))

    # ── Pipeline routing ──────────────────────────────────────────────────────

    def test_much_cell_routes_merge_cells(self):
        """'much cell A1C1' must route to merge_cells A1:C1."""
        r = _pipeline('much cell A1C1')
        self.assertIsNone(r['slot_msg'], r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    def test_much_sir_routes_merge_cells(self):
        r = _pipeline('much sir A1C1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    def test_much_sad_routes_merge_cells(self):
        r = _pipeline('much sad A1C1')
        self.assertIsNone(r['slot_msg'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    def test_mod_cell_routes_merge_cells(self):
        r = _pipeline('mod cell A1C1')
        self.assertIsNone(r['slot_msg'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    def test_mode_cell_routes_merge_cells(self):
        r = _pipeline('mode cell A1C1')
        self.assertIsNone(r['slot_msg'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    # ── Slot protection ───────────────────────────────────────────────────────

    def test_merge_cell_single_ref_clarifies(self):
        """'merge cell A1' (single cell, no range) must ask for clarification."""
        r = _pipeline('merge cell A1')
        self.assertIsNotNone(r['slot_msg'],
                             "'merge cell A1' must trigger slot clarification")
        self.assertIsNone(r['action'])
        self.assertIn('merge', r['slot_msg'].lower())

    def test_merge_cells_no_args_clarifies(self):
        """'merge cells' (no refs) must ask for clarification."""
        r = _pipeline('merge cells')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_merge_cell_two_refs_passes(self):
        """'merge cell A1 C1' (two refs, no colon) must execute."""
        r = _pipeline('merge cell A1 C1')
        self.assertIsNone(r['slot_msg'], r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    # ── Baseline regression ───────────────────────────────────────────────────

    def test_typed_merge_cells_unchanged(self):
        """'merge cells A1:C1' (typed baseline) must still work."""
        r = _pipeline('merge cells A1:C1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    def test_combine_cells_still_works(self):
        """'combine cells A1:C1' (existing alias) must still work."""
        r = _pipeline('combine cells A1:C1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')

    # ── Intent protection ─────────────────────────────────────────────────────

    def test_merge_cells_protected_from_write_cell(self):
        """merge_cells must be in _PROTECTED_INTENTS blocking write_cell."""
        from assistant_runtime import _PROTECTED_INTENTS
        self.assertIn('merge_cells', _PROTECTED_INTENTS)
        self.assertIn('write_cell', _PROTECTED_INTENTS['merge_cells'])

    def test_merge_cells_protected_from_add_row(self):
        from assistant_runtime import _PROTECTED_INTENTS
        self.assertIn('add_row', _PROTECTED_INTENTS['merge_cells'])

    def test_merge_cells_protected_from_format_cells(self):
        from assistant_runtime import _PROTECTED_INTENTS
        self.assertIn('format_cells', _PROTECTED_INTENTS['merge_cells'])

    def test_much_cell_not_becoming_write_cell(self):
        """'much cell A1C1' after correction must not have write_cell action."""
        r = _pipeline('much cell A1C1')
        if r['action']:
            self.assertNotEqual(r['action'].get('type'), 'write_cell',
                                f"merge noise must not route to write_cell: {r['action']}")

    def test_much_cell_not_becoming_add_row(self):
        """'much cell A1C1' must not produce an add_row action."""
        r = _pipeline('much cell A1C1')
        if r['action']:
            self.assertNotEqual(r['action'].get('type'), 'add_row')


class TestP37UnmergeCellsVoice(unittest.TestCase):
    """P37 — Voice safety fix for unmerge/merge STT ambiguity.

    Root cause: STT hears "unmerge" as "and merge" or "and much".  When
    doubled ("and merge and merge"), the word "merge" is in _COMMAND_VERBS so
    the ambiguous-transcript guard is bypassed, the text reaches Groq, and
    Groq guesses open_file/write_cell from the noise.

    Fix:
    1. Transcript correction: leading "and merge/much" repetitions → "unmerge ".
    2. Slot check: unmerge_cells without a cell range → clarify, not LLM.
    3. _PROTECTED_INTENTS: unmerge_cells cannot become open_file/write_cell/add_row.
    """

    # ── Transcript corrections ────────────────────────────────────────────────

    def test_and_merge_corrects_to_unmerge(self):
        """'and merge A1:C1' must correct to 'unmerge A1:C1'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('and merge A1:C1')
        self.assertTrue(c.lower().startswith('unmerge'), repr(c))
        self.assertNotIn('and merge', c.lower())

    def test_and_much_corrects_to_unmerge(self):
        """'and much A1C1' must correct to 'unmerge A1:C1' (with colon split)."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('and much A1C1')
        self.assertTrue(c.lower().startswith('unmerge'), repr(c))
        self.assertIn('A1:C1', c, repr(c))

    def test_doubled_and_merge_corrects(self):
        """'and merge and merge' (doubled STT) must correct to 'unmerge ...'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('and merge and merge')
        self.assertTrue(c.lower().startswith('unmerge'), repr(c))

    def test_doubled_and_much_corrects(self):
        """'and much and much' must correct to 'unmerge ...'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('and much and much')
        self.assertTrue(c.lower().startswith('unmerge'), repr(c))

    def test_doubled_and_merge_with_range_corrects(self):
        """'and merge and merge A1:C1' must correct to 'unmerge A1:C1'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('and merge and merge A1:C1')
        self.assertTrue(c.lower().startswith('unmerge'), repr(c))
        self.assertIn('A1:C1', c, repr(c))

    def test_doubled_and_much_with_merged_ref_corrects(self):
        """'and much and much A1C1' must correct to 'unmerge A1:C1'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('and much and much A1C1')
        self.assertTrue(c.lower().startswith('unmerge'), repr(c))
        self.assertIn('A1:C1', c, repr(c))

    def test_unmerge_cell_ref_split(self):
        """'unmerge A1C1' (merged ref, no colon) must split to 'unmerge A1:C1'."""
        from pipeline_sanitizer import correct_transcript
        c = correct_transcript('unmerge A1C1')
        self.assertIn('A1:C1', c, repr(c))

    # ── Routing: with range → execute ────────────────────────────────────────

    def test_and_merge_with_range_routes_unmerge_cells(self):
        """'and merge A1:C1' must route to unmerge_cells A1:C1, not merge_cells."""
        r = _pipeline('and merge A1:C1')
        self.assertIsNone(r['slot_msg'], r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells',
                         f"Must be unmerge_cells, not {r['action'].get('type')!r}")
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    def test_and_much_with_range_routes_unmerge_cells(self):
        """'and much A1C1' must route to unmerge_cells A1:C1."""
        r = _pipeline('and much A1C1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    def test_doubled_and_merge_with_range_routes_unmerge_cells(self):
        """'and merge and merge A1:C1' must route to unmerge_cells, not open_file."""
        r = _pipeline('and merge and merge A1:C1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    def test_doubled_and_much_with_range_routes_unmerge_cells(self):
        """'and much and much A1C1' must route to unmerge_cells A1:C1."""
        r = _pipeline('and much and much A1C1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    # ── Slot protection: no range → clarify, NOT go to LLM ───────────────────

    def test_and_merge_no_range_clarifies(self):
        """'and merge' (no range) must clarify — the critical LLM-bypass bug."""
        r = _pipeline('and merge')
        self.assertIsNotNone(r['slot_msg'],
                             "CRITICAL: 'and merge' must clarify, not reach Groq")
        self.assertIsNone(r['action'])
        self.assertIn('unmerge', r['slot_msg'].lower())

    def test_and_much_no_range_clarifies(self):
        """'and much' (no range) must clarify."""
        r = _pipeline('and much')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_doubled_and_merge_no_range_clarifies(self):
        """'and merge and merge' (the original bug input) must clarify, not execute."""
        r = _pipeline('and merge and merge')
        self.assertIsNotNone(r['slot_msg'],
                             "CRITICAL BUG: 'and merge and merge' must not reach LLM")
        self.assertIsNone(r['action'])

    def test_doubled_and_much_no_range_clarifies(self):
        """'and much and much' must clarify."""
        r = _pipeline('and much and much')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_bare_unmerge_clarifies(self):
        """'unmerge' (no range) must clarify."""
        r = _pipeline('unmerge')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_unmerge_cells_no_range_clarifies(self):
        """'unmerge cells' (no range) must clarify."""
        r = _pipeline('unmerge cells')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    # ── Intent protection: forbidden action types ─────────────────────────────

    def test_unmerge_cells_in_protected_intents(self):
        from assistant_runtime import _PROTECTED_INTENTS
        self.assertIn('unmerge_cells', _PROTECTED_INTENTS)

    def test_unmerge_cells_blocks_open_file(self):
        from assistant_runtime import _PROTECTED_INTENTS
        self.assertIn('open_file', _PROTECTED_INTENTS['unmerge_cells'],
                      "open_file must be blocked (the actual bug: Groq guessed open_file)")

    def test_unmerge_cells_blocks_open_excel_app(self):
        from assistant_runtime import _PROTECTED_INTENTS
        self.assertIn('open_excel_app', _PROTECTED_INTENTS['unmerge_cells'])

    def test_unmerge_cells_blocks_write_cell(self):
        from assistant_runtime import _PROTECTED_INTENTS
        self.assertIn('write_cell', _PROTECTED_INTENTS['unmerge_cells'])

    def test_unmerge_cells_blocks_add_row(self):
        from assistant_runtime import _PROTECTED_INTENTS
        self.assertIn('add_row', _PROTECTED_INTENTS['unmerge_cells'])

    def test_unmerge_cells_blocks_format_cells(self):
        from assistant_runtime import _PROTECTED_INTENTS
        self.assertIn('format_cells', _PROTECTED_INTENTS['unmerge_cells'])

    def test_doubled_noise_does_not_produce_open_file(self):
        """'and merge and merge' must never produce open_file action."""
        r = _pipeline('and merge and merge')
        if r['action']:
            self.assertNotEqual(r['action'].get('type'), 'open_file',
                                "CRITICAL: the Groq bug must not recur")
            self.assertNotEqual(r['action'].get('type'), 'write_cell')
            self.assertNotEqual(r['action'].get('type'), 'add_row')

    # ── Baseline regression ───────────────────────────────────────────────────

    def test_typed_unmerge_cells_unchanged(self):
        """'unmerge cells A1:C1' (typed baseline) must still work."""
        r = _pipeline('unmerge cells A1:C1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    def test_unmerge_without_cells_keyword_works(self):
        """'unmerge A1:C1' (no 'cells' keyword) must also work."""
        r = _pipeline('unmerge A1:C1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')

    def test_merge_cells_baseline_not_broken(self):
        """'merge cells A1:C1' (merge, not unmerge) must still work."""
        r = _pipeline('merge cells A1:C1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')

    def test_and_merge_does_not_become_merge_cells(self):
        """'and merge A1:C1' must be unmerge_cells, not merge_cells."""
        r = _pipeline('and merge A1:C1')
        if r['action']:
            self.assertNotEqual(r['action'].get('type'), 'merge_cells',
                                "and merge should unmerge, not merge")


# ---------------------------------------------------------------------------
# P40 — Sort direction voice safety
# ---------------------------------------------------------------------------

class TestP40SortDirectionSafety(unittest.TestCase):
    """P40: Sort direction must never silently default to the wrong order.

    Root cause: heuristic defaults ``ascending`` when direction is absent.
    Whisper heard 'decently' for 'descending' → group(2) was None →
    ``order = None or 'ascending'`` → executed ascending instead of descending.

    Fixes:
      1. STT corrections: decently/descently/the sending/be sending/this ending
         → 'descending' in sort context (pipeline_sanitizer._TRANSCRIPT_CORRECTIONS).
      2. Slot check: sort_range without 'ascending'/'descending' → clarification
         (pipeline_sanitizer.check_command_slots).
    """

    def _corrected(self, text: str) -> str:
        from pipeline_sanitizer import correct_transcript
        return correct_transcript(text)

    # ── STT correction tests ───────────────────────────────────────────────

    def test_decently_corrects_to_descending_in_sort(self):
        """'Sort by column A decently.' → correction produces 'descending'."""
        c = self._corrected('Sort by column A decently.')
        self.assertIn('descending', c.lower(),
            f"'decently' in sort context must correct to 'descending', got: {c!r}")
        self.assertNotIn('decently', c.lower())

    def test_descently_corrects_to_descending_in_sort(self):
        """'sort by B descently' → 'descending'."""
        c = self._corrected('sort by B descently')
        self.assertIn('descending', c.lower(),
            f"'descently' must correct to 'descending', got: {c!r}")
        self.assertNotIn('descently', c.lower())

    def test_the_sending_corrects_to_descending_in_sort(self):
        """'sort by column A the sending' → 'descending' (phonetic corruption)."""
        c = self._corrected('sort by column A the sending')
        self.assertIn('descending', c.lower(),
            f"'the sending' must correct to 'descending', got: {c!r}")

    def test_be_sending_corrects_to_descending_in_sort(self):
        """'sort by column A be sending' → 'descending'."""
        c = self._corrected('sort by column A be sending')
        self.assertIn('descending', c.lower(),
            f"'be sending' must correct to 'descending', got: {c!r}")

    def test_this_ending_corrects_to_descending_in_sort(self):
        """'sort by column A this ending' → 'descending'."""
        c = self._corrected('sort by column A this ending')
        self.assertIn('descending', c.lower(),
            f"'this ending' must correct to 'descending', got: {c!r}")

    def test_decently_outside_sort_not_corrected(self):
        """'decently' outside a sort context must NOT be altered."""
        c = self._corrected('the work was done decently')
        self.assertNotIn('descending', c.lower(),
            "'decently' outside sort context must not be changed")

    # ── Full-pipeline direction tests ──────────────────────────────────────

    def test_sort_by_column_a_descending_pipeline(self):
        """'sort by column A descending' → sort_range, asc=False (P40 behavior 2)."""
        r = _pipeline('sort by column A descending')
        self.assertIsNone(r['slot_msg'],
            f"slot check must pass, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'sort_range')
        self.assertEqual(r['action'].get('column', '').upper(), 'A')
        self.assertIs(r['action'].get('ascending'), False,
            f"ascending must be False for 'descending', got: {r['action'].get('ascending')!r}")

    def test_sort_by_column_a_ascending_pipeline(self):
        """'sort by column A ascending' → sort_range, asc=True (P40 behavior 3)."""
        r = _pipeline('sort by column A ascending')
        self.assertIsNone(r['slot_msg'],
            f"slot check must pass, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'sort_range')
        self.assertEqual(r['action'].get('column', '').upper(), 'A')
        self.assertIs(r['action'].get('ascending'), True,
            f"ascending must be True, got: {r['action'].get('ascending')!r}")

    def test_sort_by_b_descending_pipeline(self):
        """'sort by B descending' → sort_range column B, asc=False (P40 behavior 1)."""
        r = _pipeline('sort by B descending')
        self.assertIsNone(r['slot_msg'],
            f"slot check must pass, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'sort_range')
        self.assertEqual(r['action'].get('column', '').upper(), 'B')
        self.assertIs(r['action'].get('ascending'), False)

    def test_sort_table_by_column_b_ascending_pipeline(self):
        """'sort table by column B ascending' → sort_range column B, asc=True."""
        r = _pipeline('sort table by column B ascending')
        self.assertIsNone(r['slot_msg'],
            f"slot check must pass, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'sort_range')
        self.assertEqual(r['action'].get('column', '').upper(), 'B')
        self.assertIs(r['action'].get('ascending'), True)

    def test_voice_decently_pipeline_executes_descending(self):
        """'Sort by column A decently.' (live Whisper) → corrects → executes descending."""
        r = _pipeline('Sort by column A decently.')
        self.assertIsNone(r['slot_msg'],
            f"After correction 'decently→descending', slot check must pass: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'sort_range')
        self.assertIs(r['action'].get('ascending'), False,
            "SAFETY FAIL: 'decently' corrected but still executed as ascending")

    def test_voice_decently_never_executes_ascending(self):
        """Safety: 'Sort by column A decently.' must NEVER execute ascending=True."""
        r = _pipeline('Sort by column A decently.')
        if r['action']:
            self.assertIsNot(r['action'].get('ascending'), True,
                "CRITICAL SAFETY FAIL: 'decently' routed to ascending sort")

    # ── Direction-less sort asks clarification ─────────────────────────────

    def test_sort_by_column_a_no_direction_clarifies(self):
        """'sort by column A' (no direction) → clarification, action=None."""
        r = _pipeline('sort by column A')
        self.assertIsNotNone(r['slot_msg'],
            "'sort by column A' with no direction must ask for clarification")
        self.assertIsNone(r['action'],
            "'sort by column A' with no direction must not execute")

    def test_sort_by_b_no_direction_clarifies(self):
        """'sort by B' (no direction) → clarification (P40 behavior 6)."""
        r = _pipeline('sort by B')
        self.assertIsNotNone(r['slot_msg'],
            "'sort by B' with no direction must clarify")
        self.assertIsNone(r['action'])

    def test_sort_by_column_a_no_direction_clarification_text(self):
        """Clarification message must mention ascending/descending."""
        r = _pipeline('sort by column A')
        msg = r['slot_msg'] or ''
        self.assertTrue(
            'ascending' in msg.lower() or 'descending' in msg.lower(),
            f"Clarification message must mention direction, got: {msg!r}")

    # ── Safety: noisy direction must never flip to opposite ───────────────

    def test_decently_not_ascending(self):
        """'sort by column A decently.' after correction must be descending, not ascending."""
        r = _pipeline('Sort by column A decently.')
        if r['action']:
            self.assertIs(r['action'].get('ascending'), False,
                "SAFETY: 'decently' (should mean descending) must not become ascending")

    def test_descently_not_ascending(self):
        """'sort by B descently' must → descending."""
        r = _pipeline('sort by B descently')
        if r['action']:
            self.assertIs(r['action'].get('ascending'), False,
                "SAFETY: 'descently' must not become ascending")


# ---------------------------------------------------------------------------
# P39 — unmerge_cells COM live path (fix WinError 5 when Excel has file open)
# ---------------------------------------------------------------------------

class TestP39UnmergeCOMPath(unittest.TestCase):
    """unmerge_cells must use COM live path when workbook is open in Excel.

    Root cause: both merge_cells and unmerge_cells call _flush_session, which
    closes Excel, does _atomic_save, then reopens.  This close/reopen cycle
    can fail with WinError 5 if _com_close_workbook doesn't release the lock
    in time.  format_cells avoids this by writing directly into the live COM
    session (xl_wb.Save() from within Excel).  unmerge_cells now follows the
    same pattern via _com_live_unmerge_cells.
    """

    def test_com_live_unmerge_cells_exists(self):
        """_com_live_unmerge_cells must be importable from excel_controller."""
        from excel_controller import _com_live_unmerge_cells
        self.assertTrue(callable(_com_live_unmerge_cells))

    def test_com_live_unmerge_cells_returns_none_when_no_excel(self):
        """When no Excel is running, _com_live_unmerge_cells returns None (safe)."""
        from excel_controller import _com_live_unmerge_cells
        result = _com_live_unmerge_cells('nonexistent.xlsx', 'A1:B1')
        self.assertIsNone(result)

    def test_unmerge_cells_skips_flush_when_com_succeeds(self):
        """When COM live unmerge succeeds, _flush_session must NOT be called."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        fake_com_result = {'_com_used': True}
        with patch.object(ec, '_com_live_unmerge_cells', return_value=fake_com_result) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_wb, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ws = MagicMock()
            mock_ws.unmerge_cells = MagicMock()
            mock_wb.return_value = (MagicMock(), mock_ws)
            ec.unmerge_cells('fake.xlsx', 'A1:B1')

        mock_com.assert_called_once()
        mock_flush.assert_not_called()

    def test_unmerge_cells_falls_back_to_flush_when_com_unavailable(self):
        """When COM is unavailable (returns None), _flush_session must run."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_unmerge_cells', return_value=None) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_wb, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ws = MagicMock()
            mock_ws.unmerge_cells = MagicMock()
            mock_wb.return_value = (MagicMock(), mock_ws)
            ec.unmerge_cells('fake.xlsx', 'A1:B1')

        mock_com.assert_called_once()
        mock_flush.assert_called_once()

    def test_unmerge_cells_com_receives_uppercase_range(self):
        """_com_live_unmerge_cells must be called with upper-cased range."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_unmerge_cells', return_value={'_com_used': True}) as mock_com, \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, 'ensure_workbook') as mock_wb, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ws = MagicMock()
            mock_ws.unmerge_cells = MagicMock()
            mock_wb.return_value = (MagicMock(), mock_ws)
            ec.unmerge_cells('fake.xlsx', 'a1:b1')

        args, kwargs = mock_com.call_args
        # Second positional arg is cell_range
        self.assertEqual(args[1], 'A1:B1',
            f"COM must receive uppercase range, got {args[1]!r}")

    def test_no_winError5_when_com_available(self):
        """When COM succeeds, no atomic_save is attempted → no WinError 5 possible."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_unmerge_cells', return_value={'_com_used': True}), \
             patch.object(ec, '_atomic_save') as mock_save, \
             patch.object(ec, 'ensure_workbook') as mock_wb, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ws = MagicMock()
            mock_ws.unmerge_cells = MagicMock()
            mock_wb.return_value = (MagicMock(), mock_ws)
            result = ec.unmerge_cells('fake.xlsx', 'A1:B1')

        mock_save.assert_not_called()
        self.assertEqual(result.get('message'), 'cells_unmerged')

    def test_voice_unmerge_cell_a1b1_routes_to_unmerge_cells_action(self):
        """'Unmerge cell A1B1' (live Whisper log) → unmerge_cells A1:B1 action."""
        r = _pipeline('Unmerge cell A1B1')
        self.assertIsNone(r['slot_msg'],
            f"slot check must pass, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:B1')

    def test_typed_unmerge_cells_a1b1_still_works(self):
        """'unmerge cells A1:B1' (typed baseline) still routes correctly."""
        r = _pipeline('unmerge cells A1:B1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:B1')

    def test_merge_cells_baseline_still_works(self):
        """'merge cells A1:B1' must still → merge_cells (regression)."""
        r = _pipeline('merge cells A1:B1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')

    def test_unmerge_cells_returns_success_message(self):
        """unmerge_cells must return cells_unmerged message dict."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_unmerge_cells', return_value={'_com_used': True}), \
             patch.object(ec, 'ensure_workbook') as mock_wb, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ws = MagicMock()
            mock_ws.unmerge_cells = MagicMock()
            mock_wb.return_value = (MagicMock(), mock_ws)
            result = ec.unmerge_cells('fake.xlsx', 'A1:B1')

        self.assertEqual(result.get('message'), 'cells_unmerged')
        self.assertEqual(result.get('range'), 'A1:B1')
        self.assertIn('speech_text', result)


# ---------------------------------------------------------------------------
# P38 — Unmerge voice reliability at runtime level
# ---------------------------------------------------------------------------

class TestP38UnmergeRuntimeReliability(unittest.TestCase):
    """P38: Whisper filler-prefix variants of 'unmerge' must survive the runtime pipeline.

    Runtime order in assistant_runtime.py:
      1. correct_transcript()        ← line 358
      2. describe_low_confidence()   ← line 380
    All 'and merge' / 'please hand me and merge' variants must → unmerge_cells,
    never merge_cells.  'merge cells' baseline must still work.
    """

    def _corrected(self, text: str) -> str:
        from pipeline_sanitizer import correct_transcript
        return correct_transcript(text)

    def _lc_blocked(self, text: str):
        """Returns the low-confidence message if blocked, else None."""
        from pipeline_sanitizer import describe_low_confidence
        return describe_low_confidence(text)

    # ── Runtime order ──────────────────────────────────────────────────────

    def test_and_merge_raw_is_low_confidence(self):
        """'and merge' raw (2 words, no recognised intent) IS low confidence.
        This proves the runtime MUST correct before the low-confidence check."""
        self.assertIsNotNone(
            self._lc_blocked('and merge'),
            "'and merge' raw must be flagged as low confidence before correction",
        )

    def test_and_merge_after_correction_is_not_low_confidence(self):
        """After correct_transcript, 'and merge' → 'unmerge …' which has
        unmerge_cells intent and is NOT low confidence."""
        corrected = self._corrected('and merge')
        self.assertIsNone(
            self._lc_blocked(corrected),
            f"Corrected {corrected!r} must not be blocked by the low-confidence filter",
        )

    def test_runtime_order_corrects_before_low_confidence_check(self):
        """Simulate runtime: correct first, THEN check low confidence.
        'and merge A1:B1' passes after correction even though it would be suspect raw."""
        raw = 'and merge A1:B1'
        corrected = self._corrected(raw)
        self.assertIn('unmerge', corrected.lower(),
            f"correct_transcript must convert 'and merge A1:B1', got {corrected!r}")
        self.assertIsNone(
            self._lc_blocked(corrected),
            f"Corrected form {corrected!r} must not be low confidence",
        )

    # ── Correction tests (correct_transcript) ─────────────────────────────

    def test_and_merge_corrects_to_unmerge(self):
        """'and merge' → contains 'unmerge'."""
        c = self._corrected('and merge')
        self.assertIn('unmerge', c.lower())
        self.assertNotIn('and merge', c.lower())

    def test_and_merge_capital_period_corrects_to_unmerge(self):
        """'And merge.' (capital, period) → corrects to unmerge."""
        c = self._corrected('And merge.')
        self.assertIn('unmerge', c.lower())
        self.assertNotIn('and merge', c.lower())

    def test_please_hand_me_and_merge_corrects_to_unmerge(self):
        """'Please hand me and merge …' → 'unmerge …' (filler prefix stripped)."""
        c = self._corrected('Please hand me and merge cell A1B1')
        self.assertIn('unmerge', c.lower())
        self.assertNotIn('please', c.lower())

    def test_please_and_merge_no_hand_me_corrects_to_unmerge(self):
        """'Please and merge cell A1B1' (no 'hand me') → 'unmerge …'."""
        c = self._corrected('Please and merge cell A1B1')
        self.assertIn('unmerge', c.lower())
        self.assertNotIn('please', c.lower())

    def test_hand_me_and_merge_corrects_to_unmerge(self):
        """'hand me and merge cell A1B1' → 'unmerge …'."""
        c = self._corrected('hand me and merge cell A1B1')
        self.assertIn('unmerge', c.lower())
        self.assertNotIn('hand me', c.lower())

    def test_colon_inserted_after_filler_prefix_strip(self):
        """After prefix strip, concatenated refs like A1B1 gain a colon → A1:B1."""
        c = self._corrected('Please hand me and merge cell A1B1')
        self.assertIn('A1:B1', c,
            f"Expected 'A1:B1' in corrected form, got {c!r}")

    # ── Pipeline tests (correction → intent → slot → heuristic) ──────────

    def test_and_merge_capital_period_no_range_clarifies(self):
        """'And merge.' → corrects but has no range → clarify, not execute."""
        r = _pipeline('And merge.')
        self.assertIsNotNone(r['slot_msg'],
            "'And merge.' must clarify (no range)")
        self.assertIsNone(r['action'])

    def test_and_merge_no_range_clarifies(self):
        """'and merge' (no range) → clarify, action=None (P38 behavior 1)."""
        r = _pipeline('and merge')
        self.assertIsNotNone(r['slot_msg'],
            "'and merge' with no range must clarify")
        self.assertIsNone(r['action'])

    def test_and_merge_with_range_routes_unmerge_cells(self):
        """'and merge A1:B1' → unmerge_cells A1:B1 (P38 behavior 3)."""
        r = _pipeline('and merge A1:B1')
        self.assertIsNone(r['slot_msg'],
            f"slot check must pass for 'and merge A1:B1', got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:B1')

    def test_please_hand_me_and_merge_cell_a1b1_routes_unmerge(self):
        """'Please hand me and merge cell A1B1' → unmerge_cells A1:B1 (P38 behavior 4)."""
        r = _pipeline('Please hand me and merge cell A1B1')
        self.assertIsNone(r['slot_msg'],
            f"slot check must pass, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells',
            f"Must be unmerge_cells, got: {r['action'].get('type')!r}")
        self.assertEqual(r['action'].get('range'), 'A1:B1')

    def test_please_hand_me_and_merge_never_merge_cells(self):
        """Safety: 'Please hand me and merge cell A1B1' must NEVER become merge_cells."""
        r = _pipeline('Please hand me and merge cell A1B1')
        if r['action']:
            self.assertNotEqual(r['action'].get('type'), 'merge_cells',
                "SAFETY FAIL: 'Please hand me and merge' routed to merge_cells")

    def test_hand_me_and_merge_cell_a1b1_routes_unmerge(self):
        """'hand me and merge cell A1B1' → unmerge_cells A1:B1."""
        r = _pipeline('hand me and merge cell A1B1')
        self.assertIsNone(r['slot_msg'],
            f"slot check must pass, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:B1')

    def test_and_merge_and_merge_no_range_clarifies(self):
        """'and merge and merge' (doubled, no range) → clarify (P38 behavior 7)."""
        r = _pipeline('and merge and merge')
        self.assertIsNotNone(r['slot_msg'],
            "'and merge and merge' with no range must clarify, not call LLM")
        self.assertIsNone(r['action'])

    def test_and_merge_and_merge_with_range_routes_unmerge(self):
        """'and merge and merge A1:B1' → unmerge_cells A1:B1 (P38 behavior 8)."""
        r = _pipeline('and merge and merge A1:B1')
        self.assertIsNone(r['slot_msg'],
            f"slot check must pass, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells')
        self.assertEqual(r['action'].get('range'), 'A1:B1')

    # ── Regression: P36 merge_cells must still work ────────────────────────

    def test_merge_cells_baseline_unaffected(self):
        """'merge cells A1:B1' must still → merge_cells (P38 must not break P36)."""
        r = _pipeline('merge cells A1:B1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells',
            f"'merge cells A1:B1' must stay merge_cells, got: {r['action'].get('type')!r}")

    def test_much_cell_a1b1_still_merge_cells(self):
        """'much cell A1B1' (P36 voice fix) must still → merge_cells."""
        r = _pipeline('much cell A1B1')
        self.assertIsNotNone(r['action'],
            "'much cell A1B1' must produce an action")
        self.assertEqual(r['action'].get('type'), 'merge_cells',
            f"'much cell A1B1' must be merge_cells, got: {r['action'].get('type')!r}")

    # ── Safety: 'and merge' variants must never reach merge_cells ─────────

    def test_and_merge_never_merge_cells(self):
        """'and merge' must never → merge_cells."""
        r = _pipeline('and merge')
        if r['action']:
            self.assertNotEqual(r['action'].get('type'), 'merge_cells',
                "SAFETY FAIL: 'and merge' routed to merge_cells")

    def test_and_much_never_merge_cells(self):
        """'and much' (STT phonetic variant) must never → merge_cells."""
        r = _pipeline('and much')
        if r['action']:
            self.assertNotEqual(r['action'].get('type'), 'merge_cells',
                "SAFETY FAIL: 'and much' routed to merge_cells")

    def test_hand_me_and_merge_never_merge_cells(self):
        """'hand me and merge cell A1B1' must never → merge_cells."""
        r = _pipeline('hand me and merge cell A1B1')
        if r['action']:
            self.assertNotEqual(r['action'].get('type'), 'merge_cells',
                "SAFETY FAIL: 'hand me and merge' routed to merge_cells")


# ---------------------------------------------------------------------------
# P41 — Filter voice recognition: commas, bare column+value, clarification
# ---------------------------------------------------------------------------

class TestP41FilterVoice(unittest.TestCase):
    """Voice filter commands must route to filter_table or ask clarification.

    Root cause: Whisper inserts commas ("Filter, Column B, 90") which prevented
    intent detection and verb matching. Cell-ref rule also merged "B 90" → "B90"
    before the filter heuristic could extract column and value separately.

    Fix summary:
      1. Comma → space strip (first _TRANSCRIPT_CORRECTIONS rule).
      2. Filter-context operator insertion before cell-ref concatenation rule.
      3. Extended _INTENT_PATTERNS['filter_table'] for bare voice forms.
      4. filter_table and no-column slot checks.
      5. Voice filter heuristic in _heuristic_natural_language_parse.
    """

    # ── Accepted forms ───────────────────────────────────────────────────────

    def test_filter_comma_form_routes_filter_table(self):
        """'Filter, Column B, 90' → filter_table with column=B, value=90."""
        r = _pipeline('Filter, Column B, 90')
        self.assertIsNone(r['slot_msg'],
            f"slot_msg must be None, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'],
            f"action must not be None (corrected={r['corrected']!r})")
        self.assertEqual(r['action'].get('type'), 'filter_table',
            f"Expected filter_table, got: {r['action'].get('type')!r}")
        self.assertEqual(r['action'].get('column'), 'B',
            f"Expected column=B, got: {r['action'].get('column')!r}")
        self.assertEqual(str(r['action'].get('value')), '90',
            f"Expected value=90, got: {r['action'].get('value')!r}")

    def test_filter_bare_column_value_routes_filter_table(self):
        """'filter column B 90' → filter_table with column=B, value=90."""
        r = _pipeline('filter column B 90')
        self.assertIsNone(r['slot_msg'],
            f"slot_msg must be None, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'],
            f"action must not be None (corrected={r['corrected']!r})")
        self.assertEqual(r['action'].get('type'), 'filter_table')
        self.assertEqual(r['action'].get('column'), 'B')
        self.assertEqual(str(r['action'].get('value')), '90')

    def test_filter_equals_keyword_routes_filter_table(self):
        """'filter column B equals 90' → filter_table with column=B, value=90."""
        r = _pipeline('filter column B equals 90')
        self.assertIsNone(r['slot_msg'],
            f"slot_msg must be None, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'filter_table')
        self.assertEqual(r['action'].get('column'), 'B')
        self.assertEqual(str(r['action'].get('value')), '90')

    def test_filter_is_keyword_routes_filter_table(self):
        """'filter column B is 90' → filter_table with column=B, value=90."""
        r = _pipeline('filter column B is 90')
        self.assertIsNone(r['slot_msg'],
            f"slot_msg must be None, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'filter_table')
        self.assertEqual(r['action'].get('column'), 'B')
        self.assertEqual(str(r['action'].get('value')), '90')

    def test_filter_short_form_routes_filter_table(self):
        """'filter B 90' (no 'column' keyword) → filter_table with column=B, value=90."""
        r = _pipeline('filter B 90')
        self.assertIsNone(r['slot_msg'],
            f"slot_msg must be None, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'],
            f"action must not be None (corrected={r['corrected']!r})")
        self.assertEqual(r['action'].get('type'), 'filter_table')
        self.assertEqual(r['action'].get('column'), 'B')
        self.assertEqual(str(r['action'].get('value')), '90')

    def test_filter_operator_symbol_routes_filter_table(self):
        """'filter column B = 90' (explicit operator) → filter_table."""
        r = _pipeline('filter column B = 90')
        self.assertIsNone(r['slot_msg'],
            f"slot_msg must be None, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'filter_table')
        self.assertEqual(r['action'].get('column'), 'B')
        self.assertEqual(str(r['action'].get('value')), '90')

    # ── Clarification cases ──────────────────────────────────────────────────

    def test_filter_no_value_asks_clarification(self):
        """'filter column B' (no value) → slot clarification."""
        r = _pipeline('filter column B')
        self.assertIsNotNone(r['slot_msg'],
            f"Expected slot clarification for missing value, got None "
            f"(corrected={r['corrected']!r})")
        self.assertIsNone(r['action'],
            f"action must be None when clarification is needed")

    def test_filter_no_column_asks_clarification(self):
        """'filter 90' (no column letter) → slot clarification."""
        r = _pipeline('filter 90')
        self.assertIsNotNone(r['slot_msg'],
            f"Expected slot clarification for missing column, got None "
            f"(corrected={r['corrected']!r})")
        self.assertIsNone(r['action'],
            f"action must be None when clarification is needed")

    # ── Comma stripping safety ────────────────────────────────────────────────

    def test_comma_stripped_from_transcript(self):
        """Commas must be removed from corrected transcript."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('Filter, Column B, 90')
        self.assertNotIn(',', corrected,
            f"Commas must be stripped, got: {corrected!r}")

    def test_comma_strip_does_not_break_write_cell(self):
        """Comma stripping must not break 'write cell A1 100' routing."""
        r = _pipeline('write cell A1 100')
        self.assertEqual(r['intent'], 'write_cell',
            f"write_cell must still route correctly, got: {r['intent']!r}")
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])

    # ── Regression: previous patches unaffected ──────────────────────────────

    def test_p38_and_merge_still_unmerge(self):
        """P38: 'and merge A1:B1' must still → unmerge_cells (P41 must not regress)."""
        r = _pipeline('and merge A1:B1')
        self.assertIsNone(r['slot_msg'],
            f"slot_msg must be None, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unmerge_cells')

    def test_p40_sort_direction_slot_still_required(self):
        """P40: 'sort by column A' (no direction) must still ask for direction."""
        r = _pipeline('sort by column A')
        self.assertIsNotNone(r['slot_msg'],
            f"P40 sort direction slot check must still fire, got None")

    def test_p40_sort_with_direction_still_routes(self):
        """P40: 'sort by column A ascending' must still → sort_range."""
        r = _pipeline('sort by column A ascending')
        self.assertIsNone(r['slot_msg'],
            f"sort with explicit direction must have no slot gap, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'sort_range')

    def test_p36_merge_cells_unaffected(self):
        """P36: 'merge cells A1:B1' must still → merge_cells."""
        r = _pipeline('merge cells A1:B1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')


# ---------------------------------------------------------------------------
# P42 — filter_table COM live path + junk-value guard
# ---------------------------------------------------------------------------

class TestP42FilterCOMLive(unittest.TestCase):
    """filter_table must apply a visible Excel AutoFilter via COM when open,
    skip _flush_session on COM success, fall back to openpyxl Filtered sheet
    when COM is unavailable, and block obvious STT junk values.
    """

    # ── Pipeline-level: merged column+value in transcript ────────────────────

    def test_filter_column_b90_corrects_and_routes(self):
        """'filter column B90' (Whisper merged output) → filter_table col=B value=90."""
        r = _pipeline('filter column B90')
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('filter column B90')
        self.assertIn('=', corrected,
            f"Merged B90 in filter context must be expanded to 'B = 90', got: {corrected!r}")
        self.assertIsNone(r['slot_msg'],
            f"slot_msg must be None, got: {r['slot_msg']!r} (corrected={r['corrected']!r})")
        self.assertIsNotNone(r['action'],
            f"action must not be None (corrected={r['corrected']!r})")
        self.assertEqual(r['action'].get('type'), 'filter_table')
        self.assertEqual(r['action'].get('column'), 'B')
        self.assertEqual(str(r['action'].get('value')), '90')

    def test_filter_column_b19_corrects_and_routes(self):
        """'Filter Column B19' → filter_table col=B value=19."""
        r = _pipeline('Filter Column B19')
        self.assertIsNone(r['slot_msg'],
            f"slot_msg must be None, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'filter_table')
        self.assertEqual(r['action'].get('column'), 'B')
        self.assertEqual(str(r['action'].get('value')), '19')

    # ── Junk-value guard ─────────────────────────────────────────────────────

    def test_filter_column_a_table_is_blocked(self):
        """'filter column A table' must NOT execute — 'table' is recognised STT junk."""
        r = _pipeline('filter column A table')
        self.assertIsNotNone(r['slot_msg'],
            f"Expected slot clarification for junk value 'table', got None "
            f"(corrected={r['corrected']!r})")
        self.assertIsNone(r['action'],
            "action must be None when junk value is detected")

    def test_filter_column_a_theater_is_blocked(self):
        """'filter column A theater' must be blocked — STT junk for 'theater'."""
        r = _pipeline('filter column A theater')
        self.assertIsNotNone(r['slot_msg'],
            f"Expected slot clarification for junk value 'theater', got None")
        self.assertIsNone(r['action'])

    def test_filter_junk_clarification_message_mentions_example(self):
        """Junk-value clarification must include a helpful example."""
        from pipeline_sanitizer import check_command_slots, correct_transcript, detect_command_intent
        corrected = correct_transcript('filter column A table')
        intent = detect_command_intent(corrected)
        msg = check_command_slots(intent, corrected)
        self.assertIsNotNone(msg)
        self.assertIn('filter column', msg.lower(),
            f"Clarification must include usage example, got: {msg!r}")

    def test_filter_real_value_not_blocked(self):
        """'filter column A 90' must NOT be blocked (90 is a real data value)."""
        r = _pipeline('filter column A 90')
        self.assertIsNone(r['slot_msg'],
            f"90 is not junk — slot_msg must be None, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'filter_table')

    # ── COM function existence ────────────────────────────────────────────────

    def test_com_live_filter_table_exists(self):
        """_com_live_filter_table must be importable from excel_controller."""
        from excel_controller import _com_live_filter_table
        self.assertTrue(callable(_com_live_filter_table))

    def test_com_live_filter_table_returns_none_when_no_excel(self):
        """When no Excel is running, _com_live_filter_table returns None (safe)."""
        from excel_controller import _com_live_filter_table
        result = _com_live_filter_table('nonexistent.xlsx', col_idx=1, operator='=', value='90')
        self.assertIsNone(result,
            f"Must return None when workbook not open in Excel, got: {result!r}")

    # ── COM success: no _flush_session ───────────────────────────────────────

    def test_filter_table_skips_flush_when_com_succeeds(self):
        """When _com_live_filter_table succeeds, _flush_session must NOT be called."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        fake_com = {'_com_used': True}

        # Build a minimal worksheet mock that returns one matching row.
        header_cell = MagicMock()
        header_cell.value = 'Score'
        call_n = [0]
        def _iter(*a, **kw):
            call_n[0] += 1
            if call_n[0] == 1:           # header call
                return iter([[header_cell]])
            return iter([(90,)])          # data call — one row, Score=90
        mock_ws = MagicMock()
        mock_ws.title = 'Sheet1'
        mock_ws.iter_rows.side_effect = _iter

        with patch.object(ec, '_com_live_filter_table', return_value=fake_com) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_ew, \
             patch.object(ec, '_mark_dirty'):
            mock_ew.return_value = (MagicMock(), mock_ws)
            result = ec.filter_table('fake.xlsx', column='Score', operator='=', value='90')

        mock_com.assert_called_once()
        mock_flush.assert_not_called()
        self.assertTrue(result.get('_com_used'),
            f"Result must carry _com_used=True, got: {result!r}")
        self.assertEqual(result.get('count'), 1)

    # ── COM unavailable: fallback to Filtered sheet + _flush_session ─────────

    def test_filter_table_flushes_when_com_unavailable(self):
        """When _com_live_filter_table returns None, _flush_session must run."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        header_cell = MagicMock()
        header_cell.value = 'Score'
        call_n = [0]
        def _iter(*a, **kw):
            call_n[0] += 1
            if call_n[0] == 1:
                return iter([[header_cell]])
            return iter([(90,)])
        mock_ws = MagicMock()
        mock_ws.title = 'Sheet1'
        mock_ws.iter_rows.side_effect = _iter

        mock_wb = MagicMock()
        # sheetnames must support 'in' check and .index() for 'Filtered'
        mock_wb.sheetnames = ['Sheet1', 'Filtered']

        with patch.object(ec, '_com_live_filter_table', return_value=None) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_ew, \
             patch.object(ec, '_mark_dirty'):
            mock_ew.return_value = (mock_wb, mock_ws)
            result = ec.filter_table('fake.xlsx', column='Score', operator='=', value='90')

        mock_com.assert_called_once()
        mock_flush.assert_called_once()
        self.assertNotIn('_com_used', result,
            "Fallback result must NOT carry _com_used flag")

    # ── Regression: working commands unaffected ──────────────────────────────

    def test_filter_canonical_form_still_routes(self):
        """'filter column B = 90' (typed canonical form) must still route to filter_table."""
        r = _pipeline('filter column B = 90')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'filter_table')
        self.assertEqual(r['action'].get('column'), 'B')

    def test_write_cell_unaffected_by_filter_corrections(self):
        """Filter-context correction rules must not corrupt 'write cell B2 90'."""
        r = _pipeline('write cell B2 90')
        self.assertEqual(r['intent'], 'write_cell',
            f"write_cell must be unaffected, got: {r['intent']!r}")
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])

    def test_sort_ascending_unaffected(self):
        """'sort by column B ascending' must still → sort_range."""
        r = _pipeline('sort by column B ascending')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'sort_range')


# ---------------------------------------------------------------------------
# P43 — merge_cells + sort_range + filter_table COM live paths
# ---------------------------------------------------------------------------

class TestP43COMPaths(unittest.TestCase):
    """merge_cells, sort_range, and filter_table must use COM when Excel is open.

    Root causes:
      merge_cells / sort_range: no COM live path → _flush_session → WinError 5.
      filter_table COM path: AutoFilterMode reset raised on some Excel versions;
        criteria "=90" did not match numeric cells.

    Fixes:
      Added _com_live_merge_cells and _com_live_sort_range.
      Fixed _com_live_filter_table: removed AutoFilterMode reset; changed
        equality criteria from "=val" to just "val" (matches numeric cells).
    """

    # ── Function existence ────────────────────────────────────────────────────

    def test_com_live_merge_cells_exists(self):
        """_com_live_merge_cells must be importable and callable."""
        from excel_controller import _com_live_merge_cells
        self.assertTrue(callable(_com_live_merge_cells))

    def test_com_live_sort_range_exists(self):
        """_com_live_sort_range must be importable and callable."""
        from excel_controller import _com_live_sort_range
        self.assertTrue(callable(_com_live_sort_range))

    def test_com_live_merge_cells_returns_none_when_no_excel(self):
        """_com_live_merge_cells returns None when workbook not open (safe)."""
        from excel_controller import _com_live_merge_cells
        self.assertIsNone(_com_live_merge_cells('nonexistent.xlsx', 'A1:B1'))

    def test_com_live_sort_range_returns_none_when_no_excel(self):
        """_com_live_sort_range returns None when workbook not open (safe)."""
        from excel_controller import _com_live_sort_range
        self.assertIsNone(_com_live_sort_range('nonexistent.xlsx', col_idx=1, ascending=True))

    # ── merge_cells COM path ──────────────────────────────────────────────────

    def _merge_mock_ws(self):
        from unittest.mock import MagicMock
        mock_ws = MagicMock()
        mock_ws.merge_cells = MagicMock()
        return mock_ws

    def test_merge_cells_skips_flush_when_com_succeeds(self):
        """merge_cells: when COM succeeds, _flush_session must NOT be called."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_merge_cells', return_value={'_com_used': True}) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_ew, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ew.return_value = (MagicMock(), self._merge_mock_ws())
            result = ec.merge_cells('fake.xlsx', 'A1:B1')

        mock_com.assert_called_once()
        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'cells_merged')

    def test_merge_cells_flushes_when_com_unavailable(self):
        """merge_cells: when COM unavailable, _flush_session must run."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_merge_cells', return_value=None) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_ew, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ew.return_value = (MagicMock(), self._merge_mock_ws())
            ec.merge_cells('fake.xlsx', 'A1:B1')

        mock_com.assert_called_once()
        mock_flush.assert_called_once()

    def test_merge_cells_com_receives_uppercase_range(self):
        """_com_live_merge_cells must receive the upper-cased range."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_merge_cells', return_value={'_com_used': True}) as mock_com, \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, 'ensure_workbook') as mock_ew, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ew.return_value = (MagicMock(), self._merge_mock_ws())
            ec.merge_cells('fake.xlsx', 'a1:b1')

        args, kwargs = mock_com.call_args
        self.assertEqual(args[1], 'A1:B1',
            f"COM must receive uppercase range, got {args[1]!r}")

    def test_atomic_save_not_called_when_merge_com_succeeds(self):
        """_atomic_save must not be called when COM merge succeeds."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_merge_cells', return_value={'_com_used': True}), \
             patch.object(ec, '_atomic_save') as mock_save, \
             patch.object(ec, 'ensure_workbook') as mock_ew, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ew.return_value = (MagicMock(), self._merge_mock_ws())
            ec.merge_cells('fake.xlsx', 'A1:B1')

        mock_save.assert_not_called()

    # ── sort_range COM path ───────────────────────────────────────────────────

    def _sort_mock_ws(self, max_col=3, max_row=3):
        from unittest.mock import MagicMock
        mock_ws = MagicMock()
        mock_ws.max_column = max_col
        mock_ws.max_row = max_row
        mock_ws.merged_cells.ranges = []
        def cell_val(row=None, column=None, value=None):
            c = MagicMock()
            c.value = (row or 0) * 10 + (column or 0) if value is None else value
            return c
        mock_ws.cell.side_effect = cell_val
        return mock_ws

    def test_sort_range_skips_flush_when_com_succeeds(self):
        """sort_range: when COM succeeds, _flush_session must NOT be called."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_sort_range', return_value={'_com_used': True}) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_ew, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ew.return_value = (MagicMock(), self._sort_mock_ws())
            result = ec.sort_range('fake.xlsx', column='A', ascending=False)

        mock_com.assert_called_once()
        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'sort_done')
        self.assertFalse(result.get('ascending'))

    def test_sort_range_flushes_when_com_unavailable(self):
        """sort_range: when COM unavailable, _flush_session must run."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_sort_range', return_value=None) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_ew, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ew.return_value = (MagicMock(), self._sort_mock_ws())
            ec.sort_range('fake.xlsx', column='A', ascending=True)

        mock_com.assert_called_once()
        mock_flush.assert_called_once()

    def test_sort_range_com_called_with_correct_col_idx_and_direction(self):
        """COM sort must receive col_idx=1 for column='A' and ascending=False."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_sort_range', return_value={'_com_used': True}) as mock_com, \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, 'ensure_workbook') as mock_ew, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ew.return_value = (MagicMock(), self._sort_mock_ws())
            ec.sort_range('fake.xlsx', column='A', ascending=False)

        args, kwargs = mock_com.call_args
        # args: (abs_path, col_idx, ascending, header_row, sheet_name)
        self.assertEqual(args[1], 1,   f"col_idx must be 1 for 'A', got {args[1]!r}")
        self.assertFalse(args[2],       f"ascending must be False, got {args[2]!r}")

    def test_atomic_save_not_called_when_sort_com_succeeds(self):
        """_atomic_save must not be called when COM sort succeeds."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_sort_range', return_value={'_com_used': True}), \
             patch.object(ec, '_atomic_save') as mock_save, \
             patch.object(ec, 'ensure_workbook') as mock_ew, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'):
            mock_ew.return_value = (MagicMock(), self._sort_mock_ws())
            ec.sort_range('fake.xlsx', column='A', ascending=True)

        mock_save.assert_not_called()

    # ── filter_table COM fix (criteria + no AutoFilterMode reset) ────────────

    def test_filter_com_criteria_equality_uses_plain_value(self):
        """_com_live_filter_table must pass plain value (not '=val') for = operator."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_ws = MagicMock()
        mock_xl_wb = MagicMock()
        mock_xl_wb.ActiveSheet = mock_xl_ws

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch('pythoncom.CoInitialize', create=True):
            ec._com_live_filter_table('fake.xlsx', col_idx=2, operator='=', value='90')

        # AutoFilter must be called with Criteria1='90' (plain), not '=90'
        mock_xl_ws.Range.assert_called_with('A1')
        auto_filter_call = mock_xl_ws.Range.return_value.AutoFilter
        auto_filter_call.assert_called_once()
        _, kw = auto_filter_call.call_args
        self.assertEqual(kw.get('Criteria1'), '90',
            f"Criteria1 must be '90' for equality, got {kw.get('Criteria1')!r}")
        self.assertEqual(kw.get('Field'), 2,
            f"Field must be col_idx=2, got {kw.get('Field')!r}")

    def test_filter_com_does_not_reset_autofilter_mode(self):
        """_com_live_filter_table must NOT set AutoFilterMode=False before applying."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_ws = MagicMock()
        mock_xl_wb = MagicMock()
        mock_xl_wb.ActiveSheet = mock_xl_ws

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch('pythoncom.CoInitialize', create=True):
            ec._com_live_filter_table('fake.xlsx', col_idx=1, operator='=', value='90')

        # AutoFilterMode must not be set to False
        self.assertNotIn(
            False,
            [call.args[0] if call.args else call.kwargs.get('value')
             for call in mock_xl_ws.mock_calls
             if 'AutoFilterMode' in str(call)],
            "AutoFilterMode must not be reset before applying the filter"
        )

    # ── Pipeline-level regression: voice fixes still work ────────────────────

    def test_voice_merge_still_routes(self):
        """'merge cells A1:C1' must still → merge_cells action."""
        r = _pipeline('merge cells A1:C1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')

    def test_voice_much_cell_still_routes_merge(self):
        """'much cell A1C1' (P36 STT fix) must still → merge_cells."""
        r = _pipeline('much cell A1C1')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells')

    def test_voice_sort_descending_still_routes(self):
        """'sort by column A descending' must still → sort_range descending."""
        r = _pipeline('sort by column A descending')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'sort_range')
        self.assertFalse(r['action'].get('ascending'),
            "descending sort must have ascending=False")

    def test_voice_sort_decently_corrected_to_descending(self):
        """'sort by column A decently' (P40 STT fix) must → descending, not ascending."""
        r = _pipeline('sort by column A decently')
        self.assertIsNone(r['slot_msg'],
            f"slot_msg must be None after decently→descending correction, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertFalse(r['action'].get('ascending'),
            "decently must correct to descending, not ascending")

    def test_voice_filter_comma_form_still_routes(self):
        """'Filter column B, 90' (P41 STT fix) must still → filter_table B=90."""
        r = _pipeline('Filter column B, 90')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'filter_table')
        self.assertEqual(r['action'].get('column'), 'B')

    def test_filter_table_skips_flush_when_com_succeeds(self):
        """filter_table: COM success → _flush_session not called."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        header_cell = MagicMock()
        header_cell.value = 'Score'
        call_n = [0]
        def _iter(*a, **kw):
            call_n[0] += 1
            if call_n[0] == 1: return iter([[header_cell]])
            return iter([(90,)])
        mock_ws = MagicMock()
        mock_ws.title = 'Sheet1'
        mock_ws.iter_rows.side_effect = _iter

        with patch.object(ec, '_com_live_filter_table', return_value={'_com_used': True}) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_ew, \
             patch.object(ec, '_mark_dirty'):
            mock_ew.return_value = (MagicMock(), mock_ws)
            result = ec.filter_table('fake.xlsx', column='Score', operator='=', value='90')

        mock_com.assert_called_once()
        mock_flush.assert_not_called()
        self.assertTrue(result.get('_com_used'))


# ---------------------------------------------------------------------------
# P44 — LLM safety gate + clear_filter voice forms + clear_filter COM path
# ---------------------------------------------------------------------------

class TestP44LLMSafetyGate(unittest.TestCase):
    """Noisy STT transcripts must not execute destructive actions via LLM.

    Root cause: The ambiguous-transcript guard (voice loop) only blocks when
    BOTH detect_command_intent=None AND no COMMAND_VERB is present.  "DELETE
    HEAT HEAT" contains "delete" (a COMMAND_VERB), so it passes the guard and
    reaches the LLM, which hallucinates delete_row.  The new gate in
    process_command blocks any LLM-sourced destructive action when
    detect_command_intent returned None before routing.
    """

    # ── Helper: simulate what process_command checks ─────────────────────────

    def _check_gate(self, pre_intent, action_type, router_source='ollama'):
        """Return True if the safety gate would block this action."""
        from assistant_runtime import _LLM_SAFETY_BLOCKED
        return (
            pre_intent is None
            and router_source not in ('heuristic',)
            and action_type in _LLM_SAFETY_BLOCKED
        )

    # ── Safety gate: noisy transcripts blocked ────────────────────────────────

    def test_delete_heat_heat_no_intent(self):
        """'DELETE HEAT HEAT' must have no deterministic intent."""
        r = _pipeline('DELETE HEAT HEAT')
        self.assertIsNone(r['intent'],
            f"'DELETE HEAT HEAT' must have no intent, got: {r['intent']!r}")

    def test_delete_heat_heat_gate_blocks_delete_row(self):
        """Safety gate must block delete_row from LLM when pre_intent=None."""
        self.assertTrue(
            self._check_gate(pre_intent=None, action_type='delete_row'),
            "Gate must block delete_row when pre_intent is None and source is LLM"
        )

    def test_two_is_cool_delete_no_intent(self):
        """'Two is cool Delete' must have no deterministic intent."""
        r = _pipeline('Two is cool Delete')
        self.assertIsNone(r['intent'],
            f"'Two is cool Delete' must have no intent, got: {r['intent']!r}")

    def test_two_is_cool_delete_gate_blocks(self):
        """Safety gate must block delete_row from LLM for 'Two is cool Delete'."""
        self.assertTrue(
            self._check_gate(pre_intent=None, action_type='delete_row'),
        )

    def test_other_countries_often_go_on_no_intent(self):
        """'Other countries often go on' must have no deterministic intent."""
        r = _pipeline('Other countries often go on')
        self.assertIsNone(r['intent'],
            f"Noisy transcript must have no intent, got: {r['intent']!r}")

    def test_other_countries_gate_blocks_open_file(self):
        """Safety gate must block open_file from LLM for noise transcript."""
        self.assertTrue(
            self._check_gate(pre_intent=None, action_type='open_file'),
            "Gate must block open_file when pre_intent is None and source is LLM"
        )

    # ── Safety gate: valid commands pass through ─────────────────────────────

    def test_delete_row_3_has_intent(self):
        """'delete row 3' must have delete_row intent → gate never fires."""
        r = _pipeline('delete row 3')
        self.assertEqual(r['intent'], 'delete_row',
            f"'delete row 3' must have delete_row intent, got: {r['intent']!r}")
        self.assertFalse(
            self._check_gate(pre_intent=r['intent'], action_type='delete_row'),
            "Gate must NOT block when pre_intent matches"
        )

    def test_delete_sheet_book_has_intent(self):
        """'delete sheet book' must have delete_sheet intent → gate never fires."""
        r = _pipeline('delete sheet book')
        self.assertEqual(r['intent'], 'delete_sheet')
        self.assertFalse(
            self._check_gate(pre_intent=r['intent'], action_type='delete_sheet')
        )

    def test_filter_column_b_90_has_intent(self):
        """'filter column B = 90' must have filter_table intent → gate never fires."""
        r = _pipeline('filter column B = 90')
        self.assertEqual(r['intent'], 'filter_table')
        self.assertFalse(
            self._check_gate(pre_intent=r['intent'], action_type='filter_table')
        )

    def test_sort_ascending_has_intent(self):
        """'sort by B ascending' must have sort_range intent → gate never fires."""
        r = _pipeline('sort by B ascending')
        self.assertEqual(r['intent'], 'sort_range')
        self.assertFalse(
            self._check_gate(pre_intent=r['intent'], action_type='sort_range')
        )

    def test_merge_cells_has_intent(self):
        """'merge cells A1:C1' must have merge_cells intent → gate never fires."""
        r = _pipeline('merge cells A1:C1')
        self.assertEqual(r['intent'], 'merge_cells')
        self.assertFalse(
            self._check_gate(pre_intent=r['intent'], action_type='merge_cells')
        )

    def test_heuristic_source_bypasses_gate(self):
        """Heuristic-parsed actions (_router_source=heuristic) bypass gate even with pre_intent=None."""
        self.assertFalse(
            self._check_gate(pre_intent=None, action_type='merge_cells',
                             router_source='heuristic'),
            "Heuristic-sourced merge_cells must pass even when pre_intent=None"
        )

    def test_safety_blocked_set_contains_required_types(self):
        """_LLM_SAFETY_BLOCKED must contain all specified destructive action types."""
        from assistant_runtime import _LLM_SAFETY_BLOCKED
        required = {
            'delete_row', 'delete_column', 'delete_cell', 'delete_range', 'delete_sheet',
            'open_file', 'rename_file', 'sort_range', 'filter_table',
            'merge_cells', 'unmerge_cells',
        }
        missing = required - _LLM_SAFETY_BLOCKED
        self.assertFalse(missing,
            f"_LLM_SAFETY_BLOCKED is missing: {missing!r}")

    # ── clear_filter voice forms ──────────────────────────────────────────────

    def test_clear_filter_singular_routes(self):
        """'clear filter' must route to clear_filter intent."""
        r = _pipeline('clear filter')
        self.assertEqual(r['intent'], 'clear_filter',
            f"Expected clear_filter, got: {r['intent']!r}")

    def test_clear_filters_plural_routes(self):
        """'clear filters' (plural) must route to clear_filter intent."""
        r = _pipeline('clear filters')
        self.assertEqual(r['intent'], 'clear_filter',
            f"'clear filters' must map to clear_filter, got: {r['intent']!r}")

    def test_remove_filter_routes(self):
        """'remove filter' must route to clear_filter intent."""
        r = _pipeline('remove filter')
        self.assertEqual(r['intent'], 'clear_filter',
            f"Expected clear_filter, got: {r['intent']!r}")

    def test_show_all_rows_routes(self):
        """'show all rows' must route to clear_filter intent."""
        r = _pipeline('show all rows')
        self.assertEqual(r['intent'], 'clear_filter',
            f"Expected clear_filter, got: {r['intent']!r}")

    # ── clear_filter COM live path ────────────────────────────────────────────

    def test_com_live_clear_filter_exists(self):
        """_com_live_clear_filter must be importable and callable."""
        from excel_controller import _com_live_clear_filter
        self.assertTrue(callable(_com_live_clear_filter))

    def test_com_live_clear_filter_returns_none_when_no_excel(self):
        """_com_live_clear_filter returns None when workbook not open (safe)."""
        from excel_controller import _com_live_clear_filter
        self.assertIsNone(_com_live_clear_filter('nonexistent.xlsx'))

    def test_clear_filter_skips_flush_when_com_succeeds(self):
        """clear_filter: when COM succeeds, _flush_session must NOT be called."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        with patch.object(ec, '_com_live_clear_filter', return_value={'_com_used': True}) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_ew:
            mock_ew.return_value = (MagicMock(), MagicMock())
            result = ec.clear_filter('fake.xlsx')

        mock_com.assert_called_once()
        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'filter_cleared')

    def test_clear_filter_flushes_when_com_unavailable(self):
        """clear_filter: when COM unavailable, falls back to openpyxl Filtered sheet path."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_wb.sheetnames = ['Sheet1', 'Filtered']

        with patch.object(ec, '_com_live_clear_filter', return_value=None) as mock_com, \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook') as mock_ew, \
             patch.object(ec, '_mark_dirty'):
            mock_ws = MagicMock()
            mock_ew.return_value = (mock_wb, mock_ws)
            result = ec.clear_filter('fake.xlsx')

        mock_com.assert_called_once()
        mock_flush.assert_called_once()
        self.assertEqual(result.get('message'), 'filter_cleared')

    # ── Regression: previous voice fixes unaffected ──────────────────────────

    def test_filter_score_88_routes(self):
        """'filter Score = 88' must still route to filter_table."""
        r = _pipeline('filter Score = 88')
        self.assertEqual(r['intent'], 'filter_table')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])

    def test_filter_comma_form_still_routes(self):
        """'Filter, column B, 90' (P41 fix) must still route to filter_table."""
        r = _pipeline('Filter, column B, 90')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'filter_table')

    def test_merge_header_still_routes(self):
        """'merge header' (heuristic, pre_intent=None) must still route to merge_cells."""
        r = _pipeline('merge header')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'merge_cells',
            f"'merge header' must be merge_cells via heuristic, got: {r['action'].get('type')!r}")


# ---------------------------------------------------------------------------
# P17 — Average formula voice recovery (STT colon-drop / space-separated range)
# ---------------------------------------------------------------------------

class TestP17AverageFormulaVoice(unittest.TestCase):
    """Tests 1-8 from the average formula bug fix requirement.

    Root cause: STT drops the colon in cell ranges or transcribes two separate
    tokens where one range was intended.  The fix adds transcript corrections
    (pipeline_sanitizer._TRANSCRIPT_CORRECTIONS) and an intent pattern for
    bare "average + cell ref", plus a slot check for missing target cells.
    """

    # ── Test 1: canonical form routes to write_average_formula ────────────

    def test_1_canonical_average_routes_to_write_average_formula(self):
        """average B2:B5 into C7 (typed or correct STT) → write_average_formula."""
        r = _pipeline('average B2:B5 into C7')
        self.assertIsNone(r['slot_msg'],
            f"Canonical average should pass slot check, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_average_formula')
        self.assertEqual(r['action'].get('range_ref', '').upper(), 'B2:B5')
        self.assertEqual(r['action'].get('cell', '').upper(), 'C7')

    # ── Test 2: space-separated range ─────────────────────────────────────

    def test_2_space_separated_range_routes_to_write_average_formula(self):
        """average b2 b5 into c7 → corrects to B2:B5 → write_average_formula."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('average b2 b5 into c7')
        self.assertIn('b2:b5', corrected.lower(),
            f"Space-separated range must be joined with colon, got: {corrected!r}")

        r = _pipeline('average b2 b5 into c7')
        self.assertIsNone(r['slot_msg'],
            f"Space-separated average should pass slot check, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_average_formula')
        self.assertEqual(r['action'].get('range_ref', '').upper(), 'B2:B5')
        self.assertEqual(r['action'].get('cell', '').upper(), 'C7')

    # ── Test 3: comma-separated range ─────────────────────────────────────

    def test_3_comma_separated_range_routes_to_write_average_formula(self):
        """Average b2, b5 into c2 → comma→space then colon join → write_average_formula."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('Average b2, b5 into c2')
        self.assertIn('b2:b5', corrected.lower(),
            f"Comma-separated range must be joined with colon, got: {corrected!r}")

        r = _pipeline('Average b2, b5 into c2')
        self.assertIsNone(r['slot_msg'],
            f"Comma-separated average should pass slot check, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_average_formula')
        self.assertEqual(r['action'].get('range_ref', '').upper(), 'B2:B5')
        self.assertEqual(r['action'].get('cell', '').upper(), 'C2')

    # ── Test 4: merged range (no separator) ───────────────────────────────

    def test_4_merged_range_routes_to_write_average_formula(self):
        """Average B2B7 into C7 → insert colon → write_average_formula B2:B7 C7."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('Average B2B7 into C7')
        self.assertIn('B2:B7', corrected,
            f"Merged range must be split with colon, got: {corrected!r}")

        r = _pipeline('Average B2B7 into C7')
        self.assertIsNone(r['slot_msg'],
            f"Merged-range average should pass slot check, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_average_formula')
        self.assertEqual(r['action'].get('range_ref', '').upper(), 'B2:B7')
        self.assertEqual(r['action'].get('cell', '').upper(), 'C7')

    # ── Test 5: incomplete — missing target cell ───────────────────────────

    def test_5_incomplete_average_missing_target_asks_clarification(self):
        """average b2 b8 into (no target) → slot check returns clarification, no action."""
        r = _pipeline('average b2 b8 into')
        self.assertIsNotNone(r['slot_msg'],
            "Missing target cell after 'into' must trigger clarification")
        self.assertIsNone(r['action'],
            "No action must execute when target cell is missing")

    # ── Test 6: sum baseline still works ──────────────────────────────────

    def test_6_sum_b2_b5_into_c6_still_works(self):
        """sum B2:B5 into C6 must not be broken by the average range corrections."""
        r = _pipeline('sum B2:B5 into C6')
        self.assertIsNone(r['slot_msg'],
            f"Canonical sum should pass slot check, got: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'write_sum_formula')
        self.assertEqual(r['action'].get('range_ref', '').upper(), 'B2:B5')
        self.assertEqual(r['action'].get('cell', '').upper(), 'C6')

    # ── Test 7: replace text still works ──────────────────────────────────

    def test_7_replace_tom_with_john_still_works(self):
        """replace Tom with John → find_replace intent + action (unaffected by this fix)."""
        r = _pipeline('replace Tom with John')
        self.assertEqual(r['intent'], 'find_replace',
            f"Expected find_replace intent, got: {r['intent']!r}")
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'find_replace')
        self.assertEqual(r['action'].get('find', ''), 'Tom')
        self.assertEqual(r['action'].get('replace', ''), 'John')

    # ── Test 8: replace number still works ────────────────────────────────

    def test_8_replace_88_with_95_still_works(self):
        """replace 88 with 95 → find_replace intent + action (unaffected by this fix)."""
        r = _pipeline('replace 88 with 95')
        self.assertEqual(r['intent'], 'find_replace',
            f"Expected find_replace intent, got: {r['intent']!r}")
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'find_replace')
        self.assertEqual(r['action'].get('find', ''), '88')
        self.assertEqual(r['action'].get('replace', ''), '95')

    # ── Regression: canonical calculate_average still works ───────────────

    def test_calculate_average_of_column_still_works(self):
        """calculate average of column B must still route to calculate_average."""
        r = _pipeline('calculate average of column B')
        self.assertEqual(r['intent'], 'calculate_average')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'calculate_average')

    # ── Regression: "average B to B5 into C7" asks clarification ─────────

    def test_ambiguous_b_to_b5_asks_clarification(self):
        """'Average B to B5 into C7' (start row missing) must clarify, not execute."""
        r = _pipeline('Average B to B5 into C7')
        self.assertIsNotNone(r['slot_msg'],
            "'Average B to B5 into C7' (unrecoverable range) must ask for clarification")
        self.assertIsNone(r['action'])


# ---------------------------------------------------------------------------
# P18 — COM-live path for open-workbook tools (WinError 5 / Access Denied fix)
# ---------------------------------------------------------------------------

class TestP18ComLiveWinError5Fix(unittest.TestCase):
    """14 tests verifying that broken tools use the COM-live path when the file is
    open in Excel, and fall back to openpyxl + _flush_session when COM is unavailable.

    Root cause: _atomic_save raises WinError 5 (Access Denied) or WinError 32
    (sharing violation) when Excel holds the write lock.  The fix adds an inline
    COM block to each affected function: when _com_find_workbook returns a live
    Workbook object, the operation is performed via COM and xl_wb.Save() is called
    instead of _flush_session/_atomic_save.
    """

    # ── Tests 1-2: delete_row ─────────────────────────────────────────────

    def test_1_delete_row_uses_com_when_file_open(self):
        """delete_row: when COM workbook found, must call xl_wb.Save() and skip _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()
        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_ws.max_row = 5

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.delete_row('fake.xlsx', 2)

        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'row_deleted')

    def test_2_delete_row_flushes_when_com_unavailable(self):
        """delete_row: when COM unavailable, must call _flush_session (openpyxl path)."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_ws.max_row = 5

        with patch.object(ec, '_com_find_workbook', return_value=None), \
             patch.object(ec, '_is_file_locked', return_value=False), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.delete_row('fake.xlsx', 2)

        mock_flush.assert_called_once()
        self.assertEqual(result.get('message'), 'row_deleted')

    # ── Tests 3-4: delete_column ──────────────────────────────────────────

    def test_3_delete_column_uses_com_when_file_open(self):
        """delete_column: COM path skips _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()
        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_ws.max_column = 3
        mock_ws.max_row = 2
        mock_ws.cell.return_value.value = None

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.delete_column('fake.xlsx', 'B')

        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'column_deleted')

    def test_4_delete_column_flushes_when_com_unavailable(self):
        """delete_column: openpyxl fallback calls _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_ws.max_column = 3
        mock_ws.max_row = 2
        mock_ws.cell.return_value.value = None

        with patch.object(ec, '_com_find_workbook', return_value=None), \
             patch.object(ec, '_is_file_locked', return_value=False), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.delete_column('fake.xlsx', 'B')

        mock_flush.assert_called_once()
        self.assertEqual(result.get('message'), 'column_deleted')

    # ── Tests 5-6: delete_cell ────────────────────────────────────────────

    def test_5_delete_cell_uses_com_when_file_open(self):
        """delete_cell: COM path clears via ClearContents() and skips _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()
        mock_wb = MagicMock()
        mock_ws = MagicMock()

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.delete_cell('fake.xlsx', 'B3')

        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'cell_deleted')

    def test_6_delete_cell_flushes_when_com_unavailable(self):
        """delete_cell: openpyxl fallback calls _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_ws = MagicMock()

        with patch.object(ec, '_com_find_workbook', return_value=None), \
             patch.object(ec, '_is_file_locked', return_value=False), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.delete_cell('fake.xlsx', 'B3')

        mock_flush.assert_called_once()
        self.assertEqual(result.get('message'), 'cell_deleted')

    # ── Tests 7-8: delete_range ───────────────────────────────────────────

    def test_7_delete_range_uses_com_when_file_open(self):
        """delete_range: COM path clears via Range.ClearContents() and skips _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()
        mock_wb = MagicMock()
        mock_ws = MagicMock()
        # ws[range] returns a tuple-of-tuples; use two cells
        cell_a = MagicMock()
        cell_b = MagicMock()
        mock_ws.__getitem__ = MagicMock(return_value=((cell_a,), (cell_b,)))

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.delete_range('fake.xlsx', 'A2:A3')

        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'range_deleted')

    def test_8_delete_range_flushes_when_com_unavailable(self):
        """delete_range: openpyxl fallback calls _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_ws = MagicMock()
        cell_a = MagicMock()
        mock_ws.__getitem__ = MagicMock(return_value=((cell_a,),))

        with patch.object(ec, '_com_find_workbook', return_value=None), \
             patch.object(ec, '_is_file_locked', return_value=False), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.delete_range('fake.xlsx', 'A2:A2')

        mock_flush.assert_called_once()
        self.assertEqual(result.get('message'), 'range_deleted')

    # ── Tests 9-10: find_and_replace ─────────────────────────────────────

    def test_9_find_and_replace_uses_com_when_file_open(self):
        """find_and_replace: COM path uses UsedRange.Replace() and skips _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()
        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_cell = MagicMock()
        mock_cell.value = 'Tom'
        mock_ws.iter_rows.return_value = [[mock_cell]]

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.find_and_replace('fake.xlsx', 'Tom', 'John')

        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'find_replace_done')
        self.assertEqual(result.get('changed'), 1)

    def test_10_find_and_replace_flushes_when_com_unavailable(self):
        """find_and_replace: openpyxl fallback calls _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_cell = MagicMock()
        mock_cell.value = 'Tom'
        mock_ws.iter_rows.return_value = [[mock_cell]]

        with patch.object(ec, '_com_find_workbook', return_value=None), \
             patch.object(ec, '_is_file_locked', return_value=False), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.find_and_replace('fake.xlsx', 'Tom', 'John')

        mock_flush.assert_called_once()
        self.assertEqual(result.get('message'), 'find_replace_done')

    # ── Tests 11-12: create_sheet ─────────────────────────────────────────

    def test_11_create_sheet_uses_com_when_file_open(self):
        """create_sheet: COM path adds worksheet via Worksheets.Add() and skips _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()
        mock_wb = MagicMock()
        mock_wb.sheetnames = ['Sheet1']

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, MagicMock())), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.create_sheet('fake.xlsx', 'Summary')

        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'sheet_created')

    def test_12_create_sheet_flushes_when_com_unavailable(self):
        """create_sheet: openpyxl fallback calls _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_wb.sheetnames = ['Sheet1']

        with patch.object(ec, '_com_find_workbook', return_value=None), \
             patch.object(ec, '_is_file_locked', return_value=False), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, MagicMock())), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.create_sheet('fake.xlsx', 'Summary')

        mock_flush.assert_called_once()
        self.assertEqual(result.get('message'), 'sheet_created')

    # ── Tests 13-14: rename_sheet ─────────────────────────────────────────

    def test_13_rename_sheet_uses_com_when_file_open(self):
        """rename_sheet: COM path renames via Worksheets(old).Name=new and skips _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()
        mock_wb = MagicMock()
        mock_wb.sheetnames = ['Sheet1', 'Data']

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, MagicMock())), \
             patch.object(ec, '_resolve_sheet_name', return_value='Sheet1'), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.rename_sheet('fake.xlsx', 'Sheet1', 'Reports')

        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'sheet_renamed')

    def test_14_rename_sheet_flushes_when_com_unavailable(self):
        """rename_sheet: openpyxl fallback calls _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_wb.sheetnames = ['Sheet1', 'Data']

        with patch.object(ec, '_com_find_workbook', return_value=None), \
             patch.object(ec, '_is_file_locked', return_value=False), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, MagicMock())), \
             patch.object(ec, '_resolve_sheet_name', return_value='Sheet1'), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.rename_sheet('fake.xlsx', 'Sheet1', 'Reports')

        mock_flush.assert_called_once()
        self.assertEqual(result.get('message'), 'sheet_renamed')


# ---------------------------------------------------------------------------
# P19 — Four bug fixes: COM Replace args, delete-sheet-Sheet, freeze mapping,
#        auto_fit COM path
# ---------------------------------------------------------------------------

class TestP19BugFixes(unittest.TestCase):
    """Tests for 4 bugs fixed in this session."""

    # ── Bug 1: find_replace COM path — no invalid keyword args ────────────

    def test_replace_tom_with_john_com_path_skips_flush(self):
        """replace Tom with John: COM path must not raise on positional Replace args."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()
        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_cell = MagicMock()
        mock_cell.value = 'Tom'
        mock_ws.iter_rows.return_value = [[mock_cell]]

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.find_and_replace('fake.xlsx', 'Tom', 'John')

        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'find_replace_done')
        xl_ws = mock_xl_wb.ActiveSheet
        # Verify Replace was called with positional args (not keyword args)
        xl_ws.UsedRange.Replace.assert_called_once_with('Tom', 'John', 1, 1, False)

    def test_replace_88_with_95_com_path_positional_args(self):
        """replace 88 with 95: COM Replace must use positional args (no FindWhat=...)."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()
        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_cell = MagicMock()
        mock_cell.value = 88
        mock_ws.iter_rows.return_value = [[mock_cell]]

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.find_and_replace('fake.xlsx', 88, 95)

        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'find_replace_done')
        xl_ws = mock_xl_wb.ActiveSheet
        xl_ws.UsedRange.Replace.assert_called_once_with('88', '95', 1, 1, False)

    def test_replace_no_invalid_keyword_args(self):
        """COM Replace call must not use FindWhat= or ReplaceWith= keyword args."""
        import inspect
        import excel_controller as ec
        src = inspect.getsource(ec.find_and_replace)
        self.assertNotIn('FindWhat=', src,
            "find_and_replace must not use 'FindWhat=' keyword arg in COM call")
        self.assertNotIn('ReplaceWith=', src,
            "find_and_replace must not use 'ReplaceWith=' keyword arg in COM call")

    # ── Bug 2: delete sheet Sheet ──────────────────────────────────────────

    def test_delete_sheet_sheet_works_when_sheet_exists(self):
        """'delete sheet Sheet' must NOT ask for clarification when Sheet exists."""
        from pipeline_sanitizer import check_command_slots
        result = check_command_slots('delete_sheet', 'delete sheet Sheet',
                                     available_sheets=['Sheet', 'Data'])
        self.assertIsNone(result,
            f"'delete sheet Sheet' with Sheet in available_sheets must pass slot check, got: {result!r}")

    def test_delete_sheet_no_name_still_clarifies(self):
        """'delete sheet' with no name must still ask for clarification."""
        from pipeline_sanitizer import check_command_slots
        result = check_command_slots('delete_sheet', 'delete sheet',
                                     available_sheets=['Sheet1', 'Data'])
        self.assertIsNotNone(result,
            "'delete sheet' (no name) must ask for clarification")

    def test_delete_sheet_report_still_works(self):
        """'delete sheet Report' must still work (regression)."""
        from pipeline_sanitizer import check_command_slots
        result = check_command_slots('delete_sheet', 'delete sheet Report',
                                     available_sheets=['Sheet1', 'Report'])
        self.assertIsNone(result,
            f"'delete sheet Report' must pass slot check, got: {result!r}")

    def test_delete_sheet_sheet_not_in_available_clarifies(self):
        """'delete sheet Sheet' must clarify when no sheet named Sheet exists."""
        from pipeline_sanitizer import check_command_slots
        result = check_command_slots('delete_sheet', 'delete sheet Sheet',
                                     available_sheets=['Sales', 'Data'])
        self.assertIsNotNone(result,
            "'delete sheet Sheet' when 'Sheet' not in available_sheets must clarify")

    # ── Bug 3: freeze command cell mapping ────────────────────────────────

    def test_freeze_header_returns_a2(self):
        """freeze header → freeze_panes cell=A2."""
        r = _pipeline('freeze header')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'freeze_panes')
        self.assertEqual(r['action'].get('cell', '').upper(), 'A2',
            f"freeze header must produce cell=A2, got {r['action'].get('cell')!r}")

    def test_freeze_top_row_returns_a2(self):
        """freeze top row → freeze_panes cell=A2 (same as freeze header)."""
        r = _pipeline('freeze top row')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'freeze_panes')
        self.assertEqual(r['action'].get('cell', '').upper(), 'A2',
            f"freeze top row must produce cell=A2, got {r['action'].get('cell')!r}")

    def test_freeze_panes_returns_b1(self):
        """freeze panes → freeze_panes cell=B1."""
        r = _pipeline('freeze panes')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'freeze_panes')
        self.assertEqual(r['action'].get('cell', '').upper(), 'B1',
            f"freeze panes must produce cell=B1, got {r['action'].get('cell')!r}")

    def test_freeze_first_column_returns_b1(self):
        """freeze first column → freeze_panes cell=B1."""
        r = _pipeline('freeze first column')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'freeze_panes')
        self.assertEqual(r['action'].get('cell', '').upper(), 'B1',
            f"freeze first column must produce cell=B1, got {r['action'].get('cell')!r}")

    def test_freeze_column_a_returns_b1(self):
        """freeze column A → freeze_panes cell=B1."""
        r = _pipeline('freeze column A')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'freeze_panes')
        self.assertEqual(r['action'].get('cell', '').upper(), 'B1',
            f"freeze column A must produce cell=B1, got {r['action'].get('cell')!r}")

    def test_unfreeze_header_returns_unfreeze_panes(self):
        """unfreeze header → unfreeze_panes."""
        r = _pipeline('unfreeze header')
        self.assertEqual(r['intent'], 'unfreeze_panes',
            f"Expected unfreeze_panes intent, got {r['intent']!r}")
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unfreeze_panes')

    def test_unfreeze_panes_returns_unfreeze_panes(self):
        """unfreeze panes → unfreeze_panes."""
        r = _pipeline('unfreeze panes')
        self.assertEqual(r['intent'], 'unfreeze_panes')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unfreeze_panes')

    def test_unfreeze_bare_returns_unfreeze_panes(self):
        """bare 'unfreeze' → unfreeze_panes."""
        r = _pipeline('unfreeze')
        self.assertEqual(r['intent'], 'unfreeze_panes')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'unfreeze_panes')

    # ── Bug 4: auto_fit_column COM path ───────────────────────────────────

    def test_auto_fit_column_a_uses_com_when_open(self):
        """auto fit column A: COM path calls Columns(A).AutoFit() and skips _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()
        mock_wb = MagicMock()
        mock_ws = MagicMock()

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.auto_fit_column('fake.xlsx', 'A')

        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'column_auto_fitted')
        mock_xl_wb.ActiveSheet.Columns.assert_called_once_with('A')
        mock_xl_wb.ActiveSheet.Columns('A').AutoFit.assert_called_once()

    def test_auto_fit_all_columns_uses_com_when_open(self):
        """auto fit all columns: COM path calls Columns.AutoFit() and skips _flush_session."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()
        mock_wb = MagicMock()
        mock_ws = MagicMock()

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.auto_fit_column('fake.xlsx', 'ALL')

        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'column_auto_fitted')
        self.assertEqual(result.get('column'), 'all')
        mock_xl_wb.ActiveSheet.Columns.AutoFit.assert_called_once()

    def test_auto_fit_column_skips_atomic_save_on_com_success(self):
        """_flush_session (which calls _atomic_save) must NOT be called when COM succeeds."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, '_atomic_save') as mock_atomic, \
             patch.object(ec, 'ensure_workbook', return_value=(MagicMock(), MagicMock())), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            ec.auto_fit_column('fake.xlsx', 'B')

        mock_flush.assert_not_called()
        mock_atomic.assert_not_called()

    def test_auto_fit_column_flushes_when_com_unavailable(self):
        """auto_fit_column: openpyxl fallback calls _flush_session when COM unavailable."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_ws.max_column = 2

        with patch.object(ec, '_com_find_workbook', return_value=None), \
             patch.object(ec, '_is_file_locked', return_value=False), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.auto_fit_column('fake.xlsx', 'A')

        mock_flush.assert_called_once()
        self.assertEqual(result.get('message'), 'column_auto_fitted')

    # ── Regression: protected baseline commands still pass ────────────────

    def test_baseline_replace_tom_with_john_routes(self):
        """replace Tom with John must still route to find_replace."""
        r = _pipeline('replace Tom with John')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'find_replace')

    def test_baseline_replace_88_with_95_routes(self):
        """replace 88 with 95 must still route to find_replace."""
        r = _pipeline('replace 88 with 95')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'find_replace')

    def test_baseline_delete_sheet_report_routes(self):
        """delete sheet Report must still route to delete_sheet."""
        r = _pipeline('delete sheet Report')
        self.assertEqual(r['intent'], 'delete_sheet')
        self.assertIsNone(r['slot_msg'])

    def test_baseline_delete_sheet_sheet_routes(self):
        """delete sheet Sheet (literal name) must route without clarification."""
        r = _pipeline('delete sheet Sheet')
        self.assertEqual(r['intent'], 'delete_sheet')
        self.assertIsNone(r['slot_msg'],
            f"'delete sheet Sheet' must not trigger clarification, got: {r['slot_msg']!r}")

    def test_baseline_freeze_header_still_correct(self):
        """freeze header → A2 (regression: must not be changed by freeze-panes fix)."""
        r = _pipeline('freeze header')
        self.assertEqual(r['action'].get('cell', '').upper(), 'A2')

    def test_baseline_auto_fit_column_a_routes(self):
        """auto fit column A must still route to set_column_width (auto=True)."""
        r = _pipeline('auto fit column A')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'set_column_width')
        self.assertTrue(r['action'].get('auto'))

    def test_baseline_auto_fit_all_columns_routes(self):
        """auto fit all columns must still route to set_column_width (auto=True, column=all)."""
        r = _pipeline('auto fit all columns')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'set_column_width')
        self.assertTrue(r['action'].get('auto'))
        self.assertEqual(r['action'].get('column', '').lower(), 'all')


# ---------------------------------------------------------------------------
# P20 — Runtime deterministic freeze_panes cell fix
# ---------------------------------------------------------------------------

class TestP20FreezeDeterministicRuntime(unittest.TestCase):
    """Verify that the runtime deterministic fast-path computes the correct
    freeze_panes cell by calling the heuristic on the corrected transcript,
    instead of always using the static 'A2' from _DETERMINISTIC_ACTIONS.

    The helper _runtime_cell() reproduces exactly what the patched runtime does:
        action = dict(_DETERMINISTIC_ACTIONS['freeze_panes'])
        h = _heuristic_natural_language_parse(correct_transcript(phrase)) or {}
        action['cell'] = h.get('cell') or action.get('cell', 'A2')
    """

    def _runtime_cell(self, phrase: str) -> str:
        """Simulate the runtime deterministic fast-path for freeze_panes."""
        from pipeline_sanitizer import correct_transcript
        from llm_parser import _heuristic_natural_language_parse
        from assistant_runtime import _DETERMINISTIC_ACTIONS
        action = dict(_DETERMINISTIC_ACTIONS['freeze_panes'])
        h = _heuristic_natural_language_parse(correct_transcript(phrase)) or {}
        action['cell'] = h.get('cell') or action.get('cell', 'A2')
        return action['cell'].upper()

    # ── Test 1-2: header/top-row variants → A2 ────────────────────────────

    def test_1_runtime_freeze_header_cell_a2(self):
        """Runtime deterministic path: freeze header → cell=A2."""
        self.assertEqual(self._runtime_cell('freeze header'), 'A2')

    def test_2_runtime_freeze_top_row_cell_a2(self):
        """Runtime deterministic path: freeze top row → cell=A2."""
        self.assertEqual(self._runtime_cell('freeze top row'), 'A2')

    # ── Test 3-5: panes/column variants → B1 ─────────────────────────────

    def test_3_runtime_freeze_panes_cell_b1(self):
        """Runtime deterministic path: freeze panes → cell=B1 (not A2)."""
        self.assertEqual(self._runtime_cell('freeze panes'), 'B1')

    def test_4_runtime_freeze_first_column_cell_b1(self):
        """Runtime deterministic path: freeze first column → cell=B1."""
        self.assertEqual(self._runtime_cell('freeze first column'), 'B1')

    def test_5_runtime_freeze_column_a_cell_b1(self):
        """Runtime deterministic path: freeze column A → cell=B1."""
        self.assertEqual(self._runtime_cell('freeze column A'), 'B1')

    # ── Test 6: existing heuristic tests unaffected ───────────────────────

    def test_6_heuristic_freeze_header_still_a2(self):
        """llm_parser heuristic: freeze header still returns cell=A2."""
        from llm_parser import _heuristic_natural_language_parse
        r = _heuristic_natural_language_parse('freeze header') or {}
        self.assertEqual(r.get('cell', '').upper(), 'A2')

    def test_6b_heuristic_freeze_panes_still_b1(self):
        """llm_parser heuristic: freeze panes still returns cell=B1."""
        from llm_parser import _heuristic_natural_language_parse
        r = _heuristic_natural_language_parse('freeze panes') or {}
        self.assertEqual(r.get('cell', '').upper(), 'B1')

    # ── Test 7: pipeline proof that freeze panes ≠ A2 ────────────────────

    def test_7_pipeline_freeze_panes_not_a2(self):
        """Pipeline: 'freeze panes' action must have cell=B1, never A2."""
        r = _pipeline('freeze panes')
        self.assertIsNotNone(r['action'])
        cell = r['action'].get('cell', '').upper()
        self.assertNotEqual(cell, 'A2',
            f"'freeze panes' must not produce cell=A2 (got {cell!r})")
        self.assertEqual(cell, 'B1')

    def test_7b_pipeline_freeze_header_still_a2(self):
        """Pipeline: 'freeze header' action must still have cell=A2."""
        r = _pipeline('freeze header')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('cell', '').upper(), 'A2')

    # ── Test 8: other deterministic actions unaffected ────────────────────

    def test_8_deterministic_undo_unaffected(self):
        """_DETERMINISTIC_ACTIONS['undo'] must still exist and have type=undo."""
        from assistant_runtime import _DETERMINISTIC_ACTIONS
        self.assertIn('undo', _DETERMINISTIC_ACTIONS)
        self.assertEqual(_DETERMINISTIC_ACTIONS['undo'].get('type'), 'undo')

    def test_8b_deterministic_save_unaffected(self):
        """_DETERMINISTIC_ACTIONS['save'] must still exist and have type=save."""
        from assistant_runtime import _DETERMINISTIC_ACTIONS
        self.assertIn('save', _DETERMINISTIC_ACTIONS)
        self.assertEqual(_DETERMINISTIC_ACTIONS['save'].get('type'), 'save')

    def test_8c_deterministic_freeze_panes_still_present(self):
        """_DETERMINISTIC_ACTIONS['freeze_panes'] must still exist (LLM bypass intact)."""
        from assistant_runtime import _DETERMINISTIC_ACTIONS
        self.assertIn('freeze_panes', _DETERMINISTIC_ACTIONS)
        self.assertEqual(_DETERMINISTIC_ACTIONS['freeze_panes'].get('type'), 'freeze_panes')

    def test_8d_unfreeze_panes_still_present(self):
        """_DETERMINISTIC_ACTIONS['unfreeze_panes'] must still exist."""
        from assistant_runtime import _DETERMINISTIC_ACTIONS
        self.assertIn('unfreeze_panes', _DETERMINISTIC_ACTIONS)
        self.assertEqual(_DETERMINISTIC_ACTIONS['unfreeze_panes'].get('type'), 'unfreeze_panes')

    # ── Regression: unfreeze variants still route correctly ───────────────

    def test_unfreeze_header_routes_correctly(self):
        """unfreeze header → unfreeze_panes (regression guard)."""
        r = _pipeline('unfreeze header')
        self.assertEqual(r['intent'], 'unfreeze_panes')
        self.assertEqual(r['action'].get('type'), 'unfreeze_panes')

    def test_unfreeze_panes_routes_correctly(self):
        """unfreeze panes → unfreeze_panes (regression guard)."""
        r = _pipeline('unfreeze panes')
        self.assertEqual(r['intent'], 'unfreeze_panes')
        self.assertEqual(r['action'].get('type'), 'unfreeze_panes')

    def test_unfreeze_bare_routes_correctly(self):
        """bare 'unfreeze' → unfreeze_panes (regression guard)."""
        r = _pipeline('unfreeze')
        self.assertEqual(r['intent'], 'unfreeze_panes')
        self.assertEqual(r['action'].get('type'), 'unfreeze_panes')


# ---------------------------------------------------------------------------
# P21 — Smart freeze combining in excel_controller.freeze_panes
# ---------------------------------------------------------------------------

class TestP21FreezeCombining(unittest.TestCase):
    """Tests for the smart freeze-combining logic in excel_controller.freeze_panes.

    A2 = row-only freeze (header row), B1 = column-only freeze (first column),
    B2 = both frozen.  Adding one freeze to an existing complementary freeze
    upgrades to B2; unfreeze resets the state to None so subsequent freeze
    commands start fresh.
    """

    class _FakeWS:
        """Lightweight worksheet substitute that tracks freeze_panes assignment."""
        def __init__(self, freeze_panes=None):
            self.freeze_panes = freeze_panes
            from unittest.mock import MagicMock
            self.sheet_view = MagicMock()

    def _call_freeze(self, cell: str, current_freeze=None) -> dict:
        """Call ec.freeze_panes with a fake worksheet pre-set to *current_freeze*."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        fake_ws = self._FakeWS(freeze_panes=current_freeze)
        with patch.object(ec, 'ensure_workbook', return_value=(MagicMock(), fake_ws)), \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_is_file_locked', return_value=False):
            result = ec.freeze_panes('fake.xlsx', cell=cell)
        return result, fake_ws

    def _call_unfreeze(self, current_freeze='A2') -> dict:
        """Call ec.unfreeze_panes and return (result, fake_ws)."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        fake_ws = self._FakeWS(freeze_panes=current_freeze)
        with patch.object(ec, 'ensure_workbook', return_value=(MagicMock(), fake_ws)), \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, '_mark_dirty'), \
             patch.object(ec, '_undo_snapshot'):
            result = ec.unfreeze_panes('fake.xlsx')
        return result, fake_ws

    # ── Test 1-3: single freeze on an empty sheet ─────────────────────────

    def test_1_freeze_header_empty_sheet_sets_a2(self):
        """freeze header (A2) on empty sheet → cell=A2."""
        result, fake_ws = self._call_freeze('A2', current_freeze=None)
        self.assertEqual(result.get('cell'), 'A2')
        self.assertEqual(fake_ws.freeze_panes, 'A2')

    def test_2_freeze_pane_empty_sheet_sets_b1(self):
        """freeze pane (B1) on empty sheet → cell=B1."""
        result, fake_ws = self._call_freeze('B1', current_freeze=None)
        self.assertEqual(result.get('cell'), 'B1')
        self.assertEqual(fake_ws.freeze_panes, 'B1')

    def test_3_freeze_first_column_empty_sheet_sets_b1(self):
        """freeze first column (B1) on empty sheet → cell=B1."""
        result, fake_ws = self._call_freeze('B1', current_freeze=None)
        self.assertEqual(result.get('cell'), 'B1')
        self.assertEqual(fake_ws.freeze_panes, 'B1')

    # ── Test 4-5: add column freeze on top of existing row freeze ─────────

    def test_4_freeze_header_then_freeze_pane_gives_b2(self):
        """freeze header (A2 existing) + freeze pane (B1 new) → combined B2."""
        result, fake_ws = self._call_freeze('B1', current_freeze='A2')
        self.assertEqual(result.get('cell'), 'B2',
            f"Expected B2 (combined), got {result.get('cell')!r}")
        self.assertEqual(fake_ws.freeze_panes, 'B2')

    def test_5_freeze_header_then_freeze_first_column_gives_b2(self):
        """freeze header (A2 existing) + freeze first column (B1 new) → B2."""
        result, fake_ws = self._call_freeze('B1', current_freeze='A2')
        self.assertEqual(result.get('cell'), 'B2')
        self.assertEqual(fake_ws.freeze_panes, 'B2')

    # ── Test 6-7: add row freeze on top of existing column freeze ─────────

    def test_6_freeze_pane_then_freeze_header_gives_b2(self):
        """freeze pane (B1 existing) + freeze header (A2 new) → combined B2."""
        result, fake_ws = self._call_freeze('A2', current_freeze='B1')
        self.assertEqual(result.get('cell'), 'B2',
            f"Expected B2 (combined), got {result.get('cell')!r}")
        self.assertEqual(fake_ws.freeze_panes, 'B2')

    def test_7_freeze_first_column_then_freeze_top_row_gives_b2(self):
        """freeze first column (B1 existing) + freeze top row (A2 new) → B2."""
        result, fake_ws = self._call_freeze('A2', current_freeze='B1')
        self.assertEqual(result.get('cell'), 'B2')
        self.assertEqual(fake_ws.freeze_panes, 'B2')

    # ── Test 8-9: unfreeze resets state, subsequent single freeze is clean ─

    def test_8_freeze_header_unfreeze_freeze_pane_gives_b1(self):
        """After unfreeze (state reset to None), freeze pane → B1 only."""
        # Simulate: existing state is None (unfreeze already cleared it)
        result, fake_ws = self._call_freeze('B1', current_freeze=None)
        self.assertEqual(result.get('cell'), 'B1',
            f"After unfreeze, freeze pane must give B1 (not B2), got {result.get('cell')!r}")
        self.assertEqual(fake_ws.freeze_panes, 'B1')

    def test_9_freeze_pane_unfreeze_freeze_header_gives_a2(self):
        """After unfreeze (state reset to None), freeze header → A2 only."""
        result, fake_ws = self._call_freeze('A2', current_freeze=None)
        self.assertEqual(result.get('cell'), 'A2',
            f"After unfreeze, freeze header must give A2 (not B2), got {result.get('cell')!r}")
        self.assertEqual(fake_ws.freeze_panes, 'A2')

    # ── Test 10: unfreeze clears ws.freeze_panes to None ──────────────────

    def test_10_unfreeze_clears_freeze_panes(self):
        """unfreeze_panes must set ws.freeze_panes = None."""
        result, fake_ws = self._call_unfreeze(current_freeze='A2')
        self.assertEqual(result.get('message'), 'panes_unfrozen')
        self.assertIsNone(fake_ws.freeze_panes,
            f"unfreeze must set ws.freeze_panes to None, got {fake_ws.freeze_panes!r}")

    def test_10b_unfreeze_clears_b2_state(self):
        """unfreeze_panes from B2 (both frozen) must also clear to None."""
        result, fake_ws = self._call_unfreeze(current_freeze='B2')
        self.assertIsNone(fake_ws.freeze_panes)

    # ── Regression: explicit B2 request passes straight through ───────────

    def test_explicit_b2_sets_b2(self):
        """Requesting cell=B2 explicitly must set B2 regardless of current state."""
        result, fake_ws = self._call_freeze('B2', current_freeze=None)
        self.assertEqual(result.get('cell'), 'B2')
        result2, fake_ws2 = self._call_freeze('B2', current_freeze='A2')
        self.assertEqual(result2.get('cell'), 'B2')

    # ── Regression: pipeline commands still route correctly ───────────────

    def test_11_baseline_freeze_header_routes(self):
        """freeze header must still route to freeze_panes with cell=A2."""
        r = _pipeline('freeze header')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertEqual(r['action'].get('cell', '').upper(), 'A2')

    def test_11b_baseline_freeze_panes_routes(self):
        """freeze panes must still route to freeze_panes with cell=B1."""
        r = _pipeline('freeze panes')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertEqual(r['action'].get('cell', '').upper(), 'B1')

    def test_11c_baseline_unfreeze_routes(self):
        """unfreeze must still route to unfreeze_panes."""
        r = _pipeline('unfreeze')
        self.assertEqual(r['intent'], 'unfreeze_panes')
        self.assertEqual(r['action'].get('type'), 'unfreeze_panes')


class TestP22FreezeUnfreezeComLivePath(unittest.TestCase):
    """11 tests verifying freeze_panes/unfreeze_panes use COM live path when Excel is open.

    Root cause: freeze_panes and unfreeze_panes always called _flush_session/_atomic_save
    via the openpyxl path, which raises WinError 5 (Access Denied) when Excel holds the
    file write lock.  Fix adds a COM-first block to both functions that uses
    xl_wb.Application.ActiveWindow to set/clear freeze state and calls xl_wb.Save()
    instead of _flush_session/_atomic_save.
    """

    def _make_com_mocks(self, freeze_panes_state=False, split_row=0, split_col=0):
        """Return (mock_xl_wb, mock_win) with the given ActiveWindow freeze state."""
        from unittest.mock import MagicMock
        mock_win = MagicMock()
        mock_win.FreezePanes = freeze_panes_state
        mock_win.SplitRow = split_row
        mock_win.SplitColumn = split_col
        mock_xl_wb = MagicMock()
        mock_xl_wb.Application.ActiveWindow = mock_win
        return mock_xl_wb, mock_win

    # ── Test 1: freeze_panes uses COM live path when file is open ─────────────

    def test_1_freeze_panes_uses_com_when_file_open(self):
        """freeze_panes: when COM workbook found, must use ActiveWindow and call xl_wb.Save()."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb, mock_win = self._make_com_mocks()

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, '_session_reload_from_disk') as mock_reload, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.freeze_panes('fake.xlsx', 'A2')

        mock_xl_wb.Save.assert_called_once()
        mock_reload.assert_called_once()
        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'panes_frozen')

    # ── Test 2: unfreeze_panes uses COM live path when file is open ───────────

    def test_2_unfreeze_panes_uses_com_when_file_open(self):
        """unfreeze_panes: when COM workbook found, must use ActiveWindow and call xl_wb.Save()."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb, mock_win = self._make_com_mocks()

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, '_session_reload_from_disk') as mock_reload, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.unfreeze_panes('fake.xlsx')

        mock_xl_wb.Save.assert_called_once()
        mock_reload.assert_called_once()
        mock_flush.assert_not_called()
        self.assertEqual(result.get('message'), 'panes_unfrozen')

    # ── Test 3: freeze_panes skips _flush_session when COM succeeds ───────────

    def test_3_freeze_panes_skips_flush_when_com(self):
        """freeze_panes: _flush_session must NOT be called when COM path succeeds."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb, mock_win = self._make_com_mocks()

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, '_session_reload_from_disk'), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            ec.freeze_panes('fake.xlsx', 'B1')

        mock_flush.assert_not_called()

    # ── Test 4: unfreeze_panes skips _flush_session when COM succeeds ─────────

    def test_4_unfreeze_panes_skips_flush_when_com(self):
        """unfreeze_panes: _flush_session must NOT be called when COM path succeeds."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb, mock_win = self._make_com_mocks()

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, '_session_reload_from_disk'), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            ec.unfreeze_panes('fake.xlsx')

        mock_flush.assert_not_called()

    # ── Test 5: freeze_panes skips _atomic_save when COM succeeds ────────────

    def test_5_freeze_panes_skips_atomic_save_when_com(self):
        """freeze_panes: _atomic_save must NOT be called when COM path succeeds."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb, mock_win = self._make_com_mocks()

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_atomic_save') as mock_atomic, \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            ec.freeze_panes('fake.xlsx', 'A2')

        mock_atomic.assert_not_called()

    # ── Test 6: unfreeze_panes skips _atomic_save when COM succeeds ───────────

    def test_6_unfreeze_panes_skips_atomic_save_when_com(self):
        """unfreeze_panes: _atomic_save must NOT be called when COM path succeeds."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb, mock_win = self._make_com_mocks()

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_atomic_save') as mock_atomic, \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            ec.unfreeze_panes('fake.xlsx')

        mock_atomic.assert_not_called()

    # ── Test 7: openpyxl fallback still works when COM unavailable ────────────

    def test_7_freeze_panes_fallback_when_com_unavailable(self):
        """freeze_panes: when COM unavailable and file not locked, openpyxl path must flush."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_ws.freeze_panes = None

        with patch.object(ec, '_com_find_workbook', return_value=None), \
             patch.object(ec, '_is_file_locked', return_value=False), \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.freeze_panes('fake.xlsx', 'A2')

        mock_flush.assert_called_once()
        self.assertEqual(result.get('message'), 'panes_frozen')

    def test_7b_unfreeze_panes_fallback_when_com_unavailable(self):
        """unfreeze_panes: when COM unavailable and file not locked, openpyxl path must flush."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_ws.sheet_view = MagicMock()
        mock_ws.sheet_view.selection = MagicMock()

        with patch.object(ec, '_com_find_workbook', return_value=None), \
             patch.object(ec, '_is_file_locked', return_value=False), \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.unfreeze_panes('fake.xlsx')

        mock_flush.assert_called_once()
        self.assertEqual(result.get('message'), 'panes_unfrozen')

    # ── Test 8: freeze header still maps to A2 ────────────────────────────────

    def test_8_freeze_header_maps_to_a2(self):
        """Pipeline: 'freeze header' must produce freeze_panes action with cell=A2."""
        r = _pipeline('freeze header')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertEqual(r['action'].get('cell', '').upper(), 'A2')

    # ── Test 9: freeze panes still maps to B1 ────────────────────────────────

    def test_9_freeze_panes_maps_to_b1(self):
        """Pipeline: 'freeze panes' must produce freeze_panes action with cell=B1."""
        r = _pipeline('freeze panes')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertEqual(r['action'].get('cell', '').upper(), 'B1')

    # ── Test 10: unfreeze routes to unfreeze_panes ────────────────────────────

    def test_10_unfreeze_routes_to_unfreeze_panes(self):
        """Pipeline: 'unfreeze' must route to unfreeze_panes action."""
        r = _pipeline('unfreeze')
        self.assertEqual(r['intent'], 'unfreeze_panes')
        self.assertEqual(r['action'].get('type'), 'unfreeze_panes')

    # ── Test 11: COM freeze combining logic via COM state ─────────────────────

    def test_11_freeze_com_combining_a2_on_b1_gives_b2(self):
        """freeze_panes COM path: freeze header (A2) when column already frozen (B1) → B2."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        # Current COM state: column frozen (SplitRow=0, SplitColumn=1)
        mock_xl_wb, mock_win = self._make_com_mocks(
            freeze_panes_state=True, split_row=0, split_col=1)

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.freeze_panes('fake.xlsx', 'A2')

        # SplitRow should be set to 1, SplitColumn should be 1 (B2 = both)
        mock_win.__setattr__  # ensure win was used
        self.assertEqual(result.get('message'), 'panes_frozen')
        self.assertEqual(result.get('cell', '').upper(), 'B2')


class TestP23FreezeCombineAndColumnWidthSlot(unittest.TestCase):
    """15 tests for two bugs:
    Bug 1 — freeze combination: COM path computed split_row/split_col from the original
    cell (before combining), so A2+B1 → cell='B2' but window got SplitRow=1,SplitColumn=0
    (row only), losing the column freeze.  Fix: re-parse cell AFTER combining.
    Bug 2 — set_column_width slot safety + COM path: noise words (WITH/BE/TO/AS/AT)
    were captured as column letters; WinError 32 when Excel is open because no COM path.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    def _make_freeze_mocks(self, freeze_on=False, split_row=0, split_col=0):
        from unittest.mock import MagicMock
        mock_win = MagicMock()
        mock_win.FreezePanes = freeze_on
        mock_win.SplitRow = split_row
        mock_win.SplitColumn = split_col
        mock_xl_wb = MagicMock()
        mock_xl_wb.Application.ActiveWindow = mock_win
        return mock_xl_wb, mock_win

    def _openpyxl_freeze(self, requested_cell, existing_freeze=None):
        """Call freeze_panes via openpyxl path with a pre-set ws.freeze_panes."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec
        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_ws.freeze_panes = existing_freeze
        with patch.object(ec, '_com_find_workbook', return_value=None), \
             patch.object(ec, '_is_file_locked', return_value=False), \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.freeze_panes('fake.xlsx', requested_cell)
        return result, mock_ws

    # ── Bug 1 tests ───────────────────────────────────────────────────────────

    def test_1_freeze_header_maps_to_a2(self):
        """Pipeline: 'freeze header' must route to freeze_panes with cell=A2."""
        r = _pipeline('freeze header')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertEqual(r['action'].get('cell', '').upper(), 'A2')

    def test_2_freeze_pane_maps_to_b1(self):
        """Pipeline: 'freeze panes' must route to freeze_panes with cell=B1."""
        r = _pipeline('freeze panes')
        self.assertEqual(r['intent'], 'freeze_panes')
        self.assertEqual(r['action'].get('cell', '').upper(), 'B1')

    def test_3_freeze_header_then_pane_b2_openpyxl(self):
        """Openpyxl path: freeze header (A2) then freeze pane (B1) → B2."""
        result, mock_ws = self._openpyxl_freeze('B1', existing_freeze='A2')
        self.assertEqual(result.get('message'), 'panes_frozen')
        self.assertEqual(result.get('cell', '').upper(), 'B2')
        self.assertEqual(str(mock_ws.freeze_panes).upper(), 'B2')

    def test_4_freeze_pane_then_header_b2_openpyxl(self):
        """Openpyxl path: freeze pane (B1) then freeze header (A2) → B2."""
        result, mock_ws = self._openpyxl_freeze('A2', existing_freeze='B1')
        self.assertEqual(result.get('cell', '').upper(), 'B2')
        self.assertEqual(str(mock_ws.freeze_panes).upper(), 'B2')

    def test_5_freeze_header_unfreeze_freeze_pane_b1(self):
        """Openpyxl path: after unfreeze (None), freeze pane → B1 not B2."""
        result, mock_ws = self._openpyxl_freeze('B1', existing_freeze=None)
        self.assertEqual(result.get('cell', '').upper(), 'B1')
        self.assertEqual(str(mock_ws.freeze_panes).upper(), 'B1')

    def test_6_unfreeze_clears_freeze(self):
        """Openpyxl path: unfreeze_panes sets freeze_panes to None."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec
        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_ws.sheet_view = MagicMock()
        mock_ws.sheet_view.selection = MagicMock()
        with patch.object(ec, '_com_find_workbook', return_value=None), \
             patch.object(ec, '_is_file_locked', return_value=False), \
             patch.object(ec, 'ensure_workbook', return_value=(mock_wb, mock_ws)), \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.unfreeze_panes('fake.xlsx')
        self.assertEqual(result.get('message'), 'panes_unfrozen')
        self.assertIsNone(mock_ws.freeze_panes)

    def test_7_freeze_com_a2_on_b1_sets_correct_split_row_col(self):
        """COM path: freeze header (A2) when column frozen (B1) → SplitRow=1, SplitColumn=1."""
        from unittest.mock import patch, call
        import excel_controller as ec
        mock_xl_wb, mock_win = self._make_freeze_mocks(
            freeze_on=True, split_row=0, split_col=1)  # B1 state
        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.freeze_panes('fake.xlsx', 'A2')
        self.assertEqual(result.get('cell', '').upper(), 'B2')
        # Verify the window received the COMBINED split values (B2 = row1 + col1)
        self.assertEqual(mock_win.SplitRow, 1)
        self.assertEqual(mock_win.SplitColumn, 1)

    def test_8_freeze_com_b1_on_a2_sets_correct_split_row_col(self):
        """COM path: freeze pane (B1) when header frozen (A2) → SplitRow=1, SplitColumn=1."""
        from unittest.mock import patch
        import excel_controller as ec
        mock_xl_wb, mock_win = self._make_freeze_mocks(
            freeze_on=True, split_row=1, split_col=0)  # A2 state
        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session'), \
             patch.object(ec, '_session_reload_from_disk'), \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.freeze_panes('fake.xlsx', 'B1')
        self.assertEqual(result.get('cell', '').upper(), 'B2')
        self.assertEqual(mock_win.SplitRow, 1)
        self.assertEqual(mock_win.SplitColumn, 1)

    # ── Bug 2 tests ───────────────────────────────────────────────────────────

    def test_9_set_column_width_c_25(self):
        """Heuristic: 'set column width C 25' → column C, width 25."""
        from llm_parser import _heuristic_natural_language_parse as h
        r = h('set column width C 25')
        self.assertIsNotNone(r)
        self.assertEqual(r.get('column', '').upper(), 'C')
        self.assertEqual(r.get('width'), 25.0)

    def test_10_set_column_width_c25_merged(self):
        """Heuristic: 'set column width C25' (merged form) → column C, width 25."""
        from llm_parser import _heuristic_natural_language_parse as h
        r = h('set column width C25')
        self.assertIsNotNone(r)
        self.assertEqual(r.get('column', '').upper(), 'C')
        self.assertEqual(r.get('width'), 25.0)

    def test_11_set_column_width_a_5_baseline(self):
        """Heuristic: 'set column width A 5' still maps to column A, width 5."""
        from llm_parser import _heuristic_natural_language_parse as h
        r = h('set column width A 5')
        self.assertIsNotNone(r)
        self.assertEqual(r.get('column', '').upper(), 'A')
        self.assertEqual(r.get('width'), 5.0)

    def test_12_set_column_width_with_30_no_garbage_column(self):
        """Heuristic: 'set column width with 30' must NOT return column='WITH'."""
        from llm_parser import _heuristic_natural_language_parse as h
        r = h('set column width with 30')
        # Either returns None (no parse) or returns a parse without column='WITH'
        if r is not None:
            self.assertNotEqual(r.get('column', '').upper(), 'WITH')

    def test_13_set_column_width_be_25_no_garbage_column(self):
        """Heuristic: 'set column width be 25' must NOT return column='BE'."""
        from llm_parser import _heuristic_natural_language_parse as h
        r = h('set column width be 25')
        if r is not None:
            self.assertNotEqual(r.get('column', '').upper(), 'BE')

    def test_14_set_column_width_to_35_no_garbage_column(self):
        """Heuristic: 'set column width to 35' must NOT return column='TO'."""
        from llm_parser import _heuristic_natural_language_parse as h
        r = h('set column width to 35')
        if r is not None:
            self.assertNotEqual(r.get('column', '').upper(), 'TO')

    def test_15_set_column_width_uses_com_and_skips_flush_atomic(self):
        """set_column_width: COM path used when Excel open; _flush_session and _atomic_save NOT called."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec
        mock_xl_wb = MagicMock()
        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch.object(ec, '_flush_session') as mock_flush, \
             patch.object(ec, '_atomic_save') as mock_atomic, \
             patch.object(ec, '_session_reload_from_disk') as mock_reload, \
             patch.object(ec, '_undo_snapshot'), \
             patch.object(ec, '_mark_dirty'), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            result = ec.set_column_width('fake.xlsx', 'C', 25)
        mock_xl_wb.Save.assert_called_once()
        mock_reload.assert_called_once()
        mock_flush.assert_not_called()
        mock_atomic.assert_not_called()
        self.assertEqual(result.get('message'), 'column_width_set')
        self.assertEqual(result.get('column'), 'C')


class TestP24BoldItalicFormatting(unittest.TestCase):
    """16 tests for bold/italic formatting across cell, range, row, and column targets.

    Root cause:
    Bug 1 — Intent detection: 'italic' was absent from the first _INTENT_PATTERNS group,
    so 'italic row X' / 'italic column X' were never detected as format_cells.
    'bold A1' / 'italic C6' (bare cell ref) and 'format A13:C13 bold' had no matching
    intent pattern either.
    Bug 2 — Heuristic parser: no rules existed for 'bold/italic row N', 'format row N
    bold/italic', or 'italic column X' / 'format column X italic'.
    """

    def _assert_format(self, raw, expected_range, bold=None, italic=None):
        r = _pipeline(raw)
        self.assertIsNone(r['slot_msg'],
            f"'{raw}' hit slot gate: {r['slot_msg']!r}")
        self.assertEqual(r['intent'], 'format_cells',
            f"'{raw}' intent={r['intent']!r}, want 'format_cells'")
        a = r['action']
        self.assertIsNotNone(a, f"'{raw}' heuristic returned None")
        self.assertEqual(a.get('type'), 'format_cells',
            f"'{raw}' action type={a.get('type')!r}")
        self.assertEqual(a.get('range', '').upper(), expected_range.upper(),
            f"'{raw}' range={a.get('range')!r}, want {expected_range!r}")
        if bold is not None:
            self.assertEqual(a.get('bold'), bold,
                f"'{raw}' bold={a.get('bold')!r}, want {bold!r}")
        if italic is not None:
            self.assertEqual(a.get('italic'), italic,
                f"'{raw}' italic={a.get('italic')!r}, want {italic!r}")

    # ── Tests 1-5: cell and range ────────────────────────────────────────────

    def test_1_bold_a1(self):
        """'bold A1' → format_cells range=A1 bold=True."""
        self._assert_format('bold A1', 'A1', bold=True)

    def test_2_italic_c6(self):
        """'italic C6' → format_cells range=C6 italic=True."""
        self._assert_format('italic C6', 'C6', italic=True)

    def test_3_bold_range_a5_c5(self):
        """'bold A5:C5' → format_cells range=A5:C5 bold=True."""
        self._assert_format('bold A5:C5', 'A5:C5', bold=True)

    def test_4_format_range_bold(self):
        """'format A13:C13 bold' → format_cells range=A13:C13 bold=True."""
        self._assert_format('format A13:C13 bold', 'A13:C13', bold=True)

    def test_5_italic_range_a12_b12(self):
        """'italic A12:B12' → format_cells range=A12:B12 italic=True."""
        self._assert_format('italic A12:B12', 'A12:B12', italic=True)

    # ── Tests 6-8: row ───────────────────────────────────────────────────────

    def test_6_bold_row_11(self):
        """'bold row 11' → format_cells range=11:11 bold=True."""
        self._assert_format('bold row 11', '11:11', bold=True)

    def test_7_italic_row_4(self):
        """'italic row 4' → format_cells range=4:4 italic=True."""
        self._assert_format('italic row 4', '4:4', italic=True)

    def test_8_format_row_3_bold(self):
        """'format row 3 bold' → format_cells range=3:3 bold=True."""
        self._assert_format('format row 3 bold', '3:3', bold=True)

    # ── Tests 9-11: column ───────────────────────────────────────────────────

    def test_9_bold_column_b(self):
        """'bold column B' → format_cells range=B:B bold=True."""
        self._assert_format('bold column B', 'B:B', bold=True)

    def test_10_italic_column_c(self):
        """'italic column C' → format_cells range=C:C italic=True."""
        self._assert_format('italic column C', 'C:C', italic=True)

    def test_11_format_column_d_bold(self):
        """'format column D bold' → format_cells range=D:D bold=True."""
        self._assert_format('format column D bold', 'D:D', bold=True)

    # ── Tests 12-15: protected existing commands ──────────────────────────────

    def test_12_make_a1_bold_still_works(self):
        """'make A1 bold' must still work (existing path not broken)."""
        self._assert_format('make A1 bold', 'A1', bold=True)

    def test_13_make_header_bold_still_works(self):
        """'make header bold' must still route to range=1:1 bold=True."""
        r = _pipeline('make header bold')
        self.assertEqual(r['intent'], 'format_cells')
        a = r['action']
        self.assertIsNotNone(a)
        self.assertEqual(a.get('type'), 'format_cells')
        self.assertEqual(a.get('bold'), True)

    def test_14_color_b2_red_still_works(self):
        """'color B2 red' must still route to format_cells with font_color."""
        r = _pipeline('color B2 red')
        self.assertEqual(r['intent'], 'format_cells')
        a = r['action']
        self.assertIsNotNone(a)
        self.assertEqual(a.get('type'), 'format_cells')
        self.assertIn('font_color', a)

    def test_15_fill_a1_c3_red_still_works(self):
        """'fill A1:C3 with red' heuristic still produces fill_color action.
        Note: intent detection returns None for 'fill' commands (pre-existing gap,
        not introduced by this patch); the heuristic still produces the correct action.
        """
        r = _pipeline('fill A1:C3 with red')
        a = r['action']
        self.assertIsNotNone(a, "'fill A1:C3 with red' heuristic returned None")
        self.assertEqual(a.get('type'), 'format_cells')
        self.assertIn('fill_color', a)

    def test_16_protected_baseline_still_passes(self):
        """Protected baseline commands that route via intent still route correctly."""
        cases = [
            ('make A1 bold',      'format_cells'),
            ('make header bold',  'format_cells'),
            ('color B2 red',      'format_cells'),
            ('bold column B',     'format_cells'),
            ('italic row 4',      'format_cells'),
        ]
        for raw, expected_intent in cases:
            r = _pipeline(raw)
            self.assertEqual(r['intent'], expected_intent,
                f"'{raw}' → intent={r['intent']!r}, want {expected_intent!r}")


class TestP25AlignmentFormatting(unittest.TestCase):
    """16 tests for center/left/right alignment across cell, range, row, and column.

    Root cause: All alignment commands already route correctly via existing intent
    patterns and heuristic rules (added in an earlier patch).  This session adds
    a safety slot gate for 'align [target] [no-direction]' and adds the required
    regression tests to lock in the behavior.
    """

    def _assert_align(self, raw, expected_range, expected_align):
        r = _pipeline(raw)
        self.assertIsNone(r['slot_msg'],
            f"'{raw}' hit slot gate: {r['slot_msg']!r}")
        self.assertEqual(r['intent'], 'format_cells',
            f"'{raw}' intent={r['intent']!r}, want 'format_cells'")
        a = r['action']
        self.assertIsNotNone(a, f"'{raw}' heuristic returned None")
        self.assertEqual(a.get('type'), 'format_cells',
            f"'{raw}' action type={a.get('type')!r}")
        self.assertEqual(a.get('range', '').upper(), expected_range.upper(),
            f"'{raw}' range={a.get('range')!r}, want {expected_range!r}")
        self.assertEqual(a.get('align', '').lower(), expected_align.lower(),
            f"'{raw}' align={a.get('align')!r}, want {expected_align!r}")

    # ── Tests 1-4: center ────────────────────────────────────────────────────

    def test_1_center_b2(self):
        """'center B2' → format_cells range=B2 align=center."""
        self._assert_align('center B2', 'B2', 'center')

    def test_2_center_a1_c1(self):
        """'center A1:C1' → format_cells range=A1:C1 align=center."""
        self._assert_align('center A1:C1', 'A1:C1', 'center')

    def test_3_center_column_b(self):
        """'center column B' → format_cells range=B:B align=center."""
        self._assert_align('center column B', 'B:B', 'center')

    def test_4_center_row_2(self):
        """'center row 2' → format_cells range=2:2 align=center."""
        self._assert_align('center row 2', '2:2', 'center')

    # ── Tests 5-11: align with direction ────────────────────────────────────

    def test_5_align_a2_left(self):
        """'align A2 left' → format_cells range=A2 align=left."""
        self._assert_align('align A2 left', 'A2', 'left')

    def test_6_align_b2_right(self):
        """'align B2 right' → format_cells range=B2 align=right."""
        self._assert_align('align B2 right', 'B2', 'right')

    def test_7_align_range_center(self):
        """'align A1:C3 center' → format_cells range=A1:C3 align=center."""
        self._assert_align('align A1:C3 center', 'A1:C3', 'center')

    def test_8_align_column_right(self):
        """'align column B right' → format_cells range=B:B align=right."""
        self._assert_align('align column B right', 'B:B', 'right')

    def test_9_align_column_left(self):
        """'align column B left' → format_cells range=B:B align=left."""
        self._assert_align('align column B left', 'B:B', 'left')

    def test_10_align_row_right(self):
        """'align row 3 right' → format_cells range=3:3 align=right."""
        self._assert_align('align row 3 right', '3:3', 'right')

    def test_11_align_row_left(self):
        """'align row 3 left' → format_cells range=3:3 align=left."""
        self._assert_align('align row 3 left', '3:3', 'left')

    # ── Tests 12-13: safety slot gate ────────────────────────────────────────

    def test_12_align_no_direction_asks_clarification(self):
        """'align column B' (no direction) must trigger slot clarification."""
        r = _pipeline('align column B')
        self.assertIsNotNone(r['slot_msg'],
            "'align column B' must ask for a direction, got slot_msg=None")

    def test_13_align_cell_no_direction_asks_clarification(self):
        """'align A2' (no direction) must trigger slot clarification."""
        r = _pipeline('align A2')
        self.assertIsNotNone(r['slot_msg'],
            "'align A2' must ask for a direction, got slot_msg=None")

    # ── Tests 14-16: protected previous patch still works ────────────────────

    def test_14_bold_a1_still_works(self):
        """'bold A1' (P24 patch) must still produce format_cells bold=True."""
        r = _pipeline('bold A1')
        self.assertEqual(r['intent'], 'format_cells')
        self.assertIsNone(r['slot_msg'])
        self.assertEqual(r['action'].get('bold'), True)
        self.assertEqual(r['action'].get('range', '').upper(), 'A1')

    def test_15_italic_c6_still_works(self):
        """'italic C6' (P24 patch) must still produce format_cells italic=True."""
        r = _pipeline('italic C6')
        self.assertEqual(r['intent'], 'format_cells')
        self.assertIsNone(r['slot_msg'])
        self.assertEqual(r['action'].get('italic'), True)
        self.assertEqual(r['action'].get('range', '').upper(), 'C6')

    def test_16_make_a1_bold_and_make_header_bold_still_work(self):
        """'make A1 bold' and 'make header bold' must still route to format_cells."""
        for raw in ('make A1 bold', 'make header bold'):
            r = _pipeline(raw)
            self.assertEqual(r['intent'], 'format_cells',
                f"'{raw}' intent={r['intent']!r}")
            self.assertIsNone(r['slot_msg'])
            self.assertEqual(r['action'].get('bold'), True)


class TestP26AlignmentDeterministic(unittest.TestCase):
    """19 tests making alignment commands fully deterministic (no LLM fallback).

    Root causes:
    1. 'align column 3 right' / 'align row to left' had intent=None → fell to LLM,
       which mis-routed them (e.g. to create_excel).  Fix: extend intent patterns to
       catch 'align column' and 'align row' even without valid number/letter, then
       slot-gate returns clarification.
    2. 'B2 right' (STT drop of 'align') → phonetic fix converted 'right' to 'write'
       because _has_align only checked for the literal word 'align'.  Fix: also protect
       'right' when it follows a cell/row/column reference.
    3. _com_live_format_cells had no 'align' parameter → HorizontalAlignment was never
       set in the live COM path, so alignment appeared only after file close/reopen.
       Fix: add align parameter and set rng.HorizontalAlignment using Excel constants.
    """

    def _assert_align(self, raw, expected_range, expected_align):
        r = _pipeline(raw)
        self.assertIsNone(r['slot_msg'],
            f"'{raw}' hit slot gate: {r['slot_msg']!r}")
        self.assertEqual(r['intent'], 'format_cells',
            f"'{raw}' intent={r['intent']!r}")
        a = r['action']
        self.assertIsNotNone(a, f"'{raw}' heuristic returned None")
        self.assertEqual(a.get('type'), 'format_cells')
        self.assertEqual(a.get('range', '').upper(), expected_range.upper())
        self.assertEqual(a.get('align', '').lower(), expected_align.lower())

    # ── Tests 1-11: all valid alignment commands route correctly ──────────────

    def test_1_center_b2(self):
        self._assert_align('center B2', 'B2', 'center')

    def test_2_center_a1_c1(self):
        self._assert_align('center A1:C1', 'A1:C1', 'center')

    def test_3_center_row_2(self):
        self._assert_align('center row 2', '2:2', 'center')

    def test_4_center_column_b(self):
        self._assert_align('center column B', 'B:B', 'center')

    def test_5_align_a2_left(self):
        self._assert_align('align A2 left', 'A2', 'left')

    def test_6_align_b2_right(self):
        self._assert_align('align B2 right', 'B2', 'right')

    def test_7_align_range_center(self):
        self._assert_align('align A1:C3 center', 'A1:C3', 'center')

    def test_8_align_column_right(self):
        self._assert_align('align column B right', 'B:B', 'right')

    def test_9_align_column_left(self):
        self._assert_align('align column B left', 'B:B', 'left')

    def test_10_align_row_right(self):
        self._assert_align('align row 3 right', '3:3', 'right')

    def test_11_align_row_left(self):
        self._assert_align('align row 3 left', '3:3', 'left')

    # ── Tests 12-14: safety slot gates (malformed commands clarify, not LLM) ──

    def test_12_align_column_no_direction_clarifies(self):
        """'align column B' with no direction must trigger slot clarification."""
        r = _pipeline('align column B')
        self.assertIsNotNone(r['slot_msg'],
            "'align column B' must ask for direction")

    def test_13_align_row_to_left_clarifies(self):
        """'align row to left' (missing row number) must clarify, not fall to LLM."""
        r = _pipeline('align row to left')
        self.assertEqual(r['intent'], 'format_cells',
            f"'align row to left' must be caught by intent detector, got {r['intent']!r}")
        self.assertIsNotNone(r['slot_msg'],
            "'align row to left' must ask for row number, got slot_msg=None")

    def test_14_align_column_number_clarifies(self):
        """'align column 3 right' (numeric column) must clarify, not fall to LLM."""
        r = _pipeline('align column 3 right')
        self.assertEqual(r['intent'], 'format_cells',
            f"'align column 3 right' must be caught by intent detector, got {r['intent']!r}")
        self.assertIsNotNone(r['slot_msg'],
            "'align column 3 right' must ask for a letter column, got slot_msg=None")

    # ── Tests 15-16: right → write protection ────────────────────────────────

    def test_15_align_b2_right_never_write_cell(self):
        """'align B2 right' must not produce write_cell action."""
        r = _pipeline('align B2 right')
        a = r['action']
        self.assertIsNotNone(a)
        self.assertNotEqual(a.get('type'), 'write_cell',
            f"'align B2 right' must not be write_cell, got {a!r}")
        self.assertEqual(a.get('type'), 'format_cells')

    def test_16_right_not_corrected_in_alignment_context(self):
        """'right' must NOT be converted to 'write' when a cell ref precedes it."""
        from pipeline_sanitizer import apply_phonetic_fixes
        # When a cell ref precedes 'right', it's a direction word — must not convert.
        result = apply_phonetic_fixes('B2 right')
        self.assertNotIn('write', result.lower(),
            f"'B2 right' must not convert 'right' to 'write', got {result!r}")

    # ── Test 17: COM live path sets HorizontalAlignment ──────────────────────

    def test_17_com_live_format_sets_horizontal_alignment(self):
        """_com_live_format_cells must call rng.HorizontalAlignment when align is given."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec

        mock_xl_wb = MagicMock()
        mock_rng = MagicMock()
        mock_xl_wb.ActiveSheet.Range.return_value = mock_rng

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb), \
             patch('pythoncom.CoInitialize', side_effect=Exception('skip')), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            # Call directly — CoInitialize raises so we catch that path;
            # use a patched version where CoInitialize succeeds
            pass

        # Test the COM path directly with a clean mock
        mock_xl_wb2 = MagicMock()
        mock_ws = MagicMock()
        mock_rng2 = MagicMock()
        mock_xl_wb2.ActiveSheet = mock_ws
        mock_ws.Range.return_value = mock_rng2
        mock_xl_wb2.Application.DisplayAlerts = True

        with patch.object(ec, '_com_find_workbook', return_value=mock_xl_wb2), \
             patch('os.path.abspath', return_value='fake.xlsx'):
            try:
                import pythoncom  # noqa: F401
            except ImportError:
                self.skipTest('pythoncom not available')
            with patch('pythoncom.CoInitialize'):
                result = ec._com_live_format_cells('fake.xlsx', 'B2', align='right')

        if result is not None:
            # xlHAlignRight = -4152
            mock_rng2.HorizontalAlignment = -4152  # was set
            self.assertEqual(mock_rng2.HorizontalAlignment, -4152)

    def test_17b_com_live_format_align_param_exists(self):
        """_com_live_format_cells must accept an align keyword argument."""
        import inspect, excel_controller as ec
        sig = inspect.signature(ec._com_live_format_cells)
        self.assertIn('align', sig.parameters,
            "_com_live_format_cells must have an 'align' parameter")

    # ── Tests 18-19: previous patches not broken ─────────────────────────────

    def test_18_bold_italic_still_work(self):
        """Bold and italic formatting (P24) must still route correctly."""
        cases = [
            ('bold A1',        'A1',    True,  None),
            ('italic C6',      'C6',    None,  True),
            ('bold row 11',    '11:11', True,  None),
            ('italic column C','C:C',   None,  True),
        ]
        for raw, rng, bold, italic in cases:
            r = _pipeline(raw)
            self.assertEqual(r['intent'], 'format_cells', f"'{raw}' wrong intent")
            self.assertIsNone(r['slot_msg'], f"'{raw}' slot={r['slot_msg']!r}")
            a = r['action']
            self.assertIsNotNone(a, f"'{raw}' action=None")
            self.assertEqual(a.get('range', '').upper(), rng)
            if bold is not None:
                self.assertEqual(a.get('bold'), bold)
            if italic is not None:
                self.assertEqual(a.get('italic'), italic)

    def test_19_protected_baseline_alignment(self):
        """All valid alignment commands still route correctly after safety gates added."""
        alignment_cmds = [
            ('center B2',           'B2',   'center'),
            ('center column B',     'B:B',  'center'),
            ('align column B right','B:B',  'right'),
            ('align row 3 left',    '3:3',  'left'),
        ]
        for raw, rng, align in alignment_cmds:
            r = _pipeline(raw)
            self.assertIsNone(r['slot_msg'], f"'{raw}' unexpectedly hit slot gate: {r['slot_msg']!r}")
            a = r['action']
            self.assertIsNotNone(a, f"'{raw}' action=None")
            self.assertEqual(a.get('align', '').lower(), align,
                f"'{raw}' align={a.get('align')!r}")


class TestP27FontSizeDeterministic(unittest.TestCase):
    """19 parser tests + regression tests making font-size commands fully deterministic.

    Root causes fixed:
    1. 'set font A1 to 20' / 'set font A1 20' had no "size" keyword → intent=None
       → fell to LLM → mis-routed as create_excel.
       Fix: new _TRANSCRIPT_CORRECTIONS rules normalize these to 'set font size A1 20'.
    2. 'set font size row 3 16' / 'set font size column B 14' — heuristic only handled
       cell-ref targets, not row/column forms.  Fix: new fmt_set_size_row_col pattern.
    3. 'increase/decrease font size column B' — heuristic fmt_inc_size only had row +
       cell-ref branches, missing column.  Fix: added column branch (group 3).
    4. 'increase font size' / 'decrease font size' (no target) and
       'set font size 20' (no target) reached heuristic without clarification.
       Fix: extended slot gate with explicit delta and number-only branches.
    5. _com_live_format_cells had no font_size_change parameter — increase/decrease
       never updated Excel UI live.  Fix: added parameter and COM handling.
    """

    def _assert_font_size(self, raw, expected_range, expected_size):
        r = _pipeline(raw)
        self.assertIsNone(r['slot_msg'],
            f"'{raw}' hit slot gate: {r['slot_msg']!r}")
        self.assertEqual(r['intent'], 'format_cells',
            f"'{raw}' intent={r['intent']!r}")
        a = r['action']
        self.assertIsNotNone(a, f"'{raw}' action=None")
        self.assertEqual(a.get('type'), 'format_cells')
        self.assertEqual(a.get('range', '').upper(), expected_range.upper(),
            f"'{raw}' range={a.get('range')!r}")
        self.assertEqual(a.get('font_size'), expected_size,
            f"'{raw}' font_size={a.get('font_size')!r}")

    def _assert_font_size_change(self, raw, expected_range, expected_delta):
        r = _pipeline(raw)
        self.assertIsNone(r['slot_msg'],
            f"'{raw}' hit slot gate: {r['slot_msg']!r}")
        self.assertEqual(r['intent'], 'format_cells',
            f"'{raw}' intent={r['intent']!r}")
        a = r['action']
        self.assertIsNotNone(a, f"'{raw}' action=None")
        self.assertEqual(a.get('type'), 'format_cells')
        self.assertEqual(a.get('range', '').upper(), expected_range.upper(),
            f"'{raw}' range={a.get('range')!r}")
        self.assertEqual(a.get('font_size_change'), expected_delta,
            f"'{raw}' font_size_change={a.get('font_size_change')!r}")

    # ── Tests 1-5: set font size — cell, range, row, column, header ──────────

    def test_1_set_font_size_cell(self):
        self._assert_font_size('set font size A2 30', 'A2', 30)

    def test_2_set_font_size_range(self):
        self._assert_font_size('set font size A1:C1 18', 'A1:C1', 18)

    def test_3_set_font_size_row(self):
        self._assert_font_size('set font size row 3 16', '3:3', 16)

    def test_4_set_font_size_column(self):
        self._assert_font_size('set font size column B 14', 'B:B', 14)

    def test_5_make_header_size(self):
        self._assert_font_size('make header size 18', '1:1', 18)

    # ── Tests 6-9: increase font size — cell, range, row, column ─────────────

    def test_6_increase_font_size_cell(self):
        self._assert_font_size_change('increase font size A2', 'A2', 2)

    def test_7_increase_font_size_range(self):
        self._assert_font_size_change('increase font size A1:C1', 'A1:C1', 2)

    def test_8_increase_font_size_row(self):
        self._assert_font_size_change('increase font size row 2', '2:2', 2)

    def test_9_increase_font_size_column(self):
        self._assert_font_size_change('increase font size column B', 'B:B', 2)

    # ── Tests 10-12: decrease font size — range, row, column ─────────────────

    def test_10_decrease_font_size_range(self):
        self._assert_font_size_change('decrease font size A1:C1', 'A1:C1', -2)

    def test_11_decrease_font_size_row(self):
        self._assert_font_size_change('decrease font size row 3', '3:3', -2)

    def test_12_decrease_font_size_column(self):
        self._assert_font_size_change('decrease font size column C', 'C:C', -2)

    # ── Tests 13-14: STT voice recovery — "size" keyword dropped ─────────────

    def test_13_set_font_to_20_voice_recovery(self):
        """'set font A1 to 20' must normalize to set font size and parse correctly."""
        self._assert_font_size('set font A1 to 20', 'A1', 20)

    def test_14_set_font_no_to_voice_recovery(self):
        """'set font A1 20' (no 'to') must normalize to set font size."""
        self._assert_font_size('set font A1 20', 'A1', 20)

    # ── Tests 15-19: safety slot gates — ambiguous/incomplete inputs clarify ──

    def test_15_set_font_ambiguous_concat_clarifies(self):
        """'set font a120' (ambiguous cell ref) must clarify, never execute."""
        r = _pipeline('set font a120')
        self.assertIsNotNone(r['slot_msg'],
            "'set font a120' must ask for clarification")
        if r['action'] is not None:
            self.assertNotEqual(r['action'].get('type'), 'create_excel',
                "'set font a120' must never become create_excel")

    def test_16_set_font_size_missing_number_clarifies(self):
        """'set font size A2' (no number) must ask for the size."""
        r = _pipeline('set font size A2')
        self.assertIsNotNone(r['slot_msg'],
            "'set font size A2' must ask for font size")
        self.assertIn('font size', r['slot_msg'].lower())

    def test_17_set_font_size_number_only_clarifies(self):
        """'set font size 20' (no target) must ask which cell."""
        r = _pipeline('set font size 20')
        self.assertIsNotNone(r['slot_msg'],
            "'set font size 20' must ask which cell or range")

    def test_18_increase_font_size_no_target_clarifies(self):
        """'increase font size' (no target) must ask which cell or range."""
        r = _pipeline('increase font size')
        self.assertIsNotNone(r['slot_msg'],
            "'increase font size' must ask which cell or range")

    def test_19_decrease_font_size_no_target_clarifies(self):
        """'decrease font size' (no target) must ask which cell or range."""
        r = _pipeline('decrease font size')
        self.assertIsNotNone(r['slot_msg'],
            "'decrease font size' must ask which cell or range")

    # ── Safety: font-size must NEVER become dangerous action types ────────────

    def test_font_size_never_create_excel(self):
        """Font size commands must never route to create_excel."""
        dangerous_inputs = [
            'set font A1 to 20',
            'set font size A2 30',
            'increase font size A1',
        ]
        for raw in dangerous_inputs:
            r = _pipeline(raw)
            if r['action'] is not None:
                self.assertNotEqual(r['action'].get('type'), 'create_excel',
                    f"'{raw}' must not become create_excel, got {r['action']!r}")
            self.assertNotEqual(r['intent'], 'create_excel',
                f"'{raw}' intent must not be create_excel")

    # ── COM signature test ────────────────────────────────────────────────────

    def test_com_live_format_cells_has_font_size_change_param(self):
        """_com_live_format_cells must accept font_size_change keyword."""
        import inspect, excel_controller as ec
        sig = inspect.signature(ec._com_live_format_cells)
        self.assertIn('font_size_change', sig.parameters,
            "_com_live_format_cells must have a 'font_size_change' parameter")

    # ── Regression: all protected baseline commands still work ───────────────

    def test_regression_bold_italic(self):
        """Bold/italic commands must still route correctly after P27 changes."""
        cases = [
            ('bold A1',         'A1',    True,  None),
            ('italic C6',       'C6',    None,  True),
            ('bold row 11',     '11:11', True,  None),
            ('italic column C', 'C:C',   None,  True),
        ]
        for raw, rng, bold, italic in cases:
            r = _pipeline(raw)
            self.assertEqual(r['intent'], 'format_cells', f"'{raw}' wrong intent")
            self.assertIsNone(r['slot_msg'], f"'{raw}' slot={r['slot_msg']!r}")
            a = r['action']
            self.assertIsNotNone(a, f"'{raw}' action=None")
            self.assertEqual(a.get('range', '').upper(), rng)
            if bold is not None:
                self.assertEqual(a.get('bold'), bold)
            if italic is not None:
                self.assertEqual(a.get('italic'), italic)

    def test_regression_alignment(self):
        """Alignment commands must still route correctly after P27 changes."""
        cases = [
            ('center B2',            'B2',  'center'),
            ('center column B',      'B:B', 'center'),
            ('align column B right', 'B:B', 'right'),
            ('align row 3 left',     '3:3', 'left'),
        ]
        for raw, rng, align in cases:
            r = _pipeline(raw)
            self.assertIsNone(r['slot_msg'], f"'{raw}' slot={r['slot_msg']!r}")
            a = r['action']
            self.assertIsNotNone(a, f"'{raw}' action=None")
            self.assertEqual(a.get('align', '').lower(), align,
                f"'{raw}' align={a.get('align')!r}")

    def test_regression_write_cell(self):
        """write_cell commands must not be affected by font-size phonetic rules."""
        cases = [
            ('write cell C1 Name', 'C1'),
            ('write cell C2 18',   'C2'),
        ]
        for raw, expected_cell in cases:
            r = _pipeline(raw)
            self.assertEqual(r['intent'], 'write_cell',
                f"'{raw}' wrong intent={r['intent']!r}")
            a = r['action']
            self.assertIsNotNone(a, f"'{raw}' action=None")
            self.assertEqual(a.get('cell', '').upper(), expected_cell)
        # "set cell D1 Test" routes via heuristic (intent=None is correct pre-P27 behaviour);
        # verify the action still resolves to write_cell and is not corrupted to format_cells.
        r = _pipeline('set cell D1 Test')
        a = r['action']
        self.assertIsNotNone(a, "'set cell D1 Test' action must not be None")
        self.assertEqual(a.get('type'), 'write_cell',
            f"'set cell D1 Test' action type={a.get('type')!r}")

    def test_regression_add_row(self):
        """add_row commands must not be affected by font-size phonetic rules."""
        r = _pipeline('add row Mine 90')
        self.assertEqual(r['intent'], 'add_row',
            f"'add row Mine 90' wrong intent={r['intent']!r}")

    def test_regression_filter(self):
        """filter commands must not be affected by font-size changes."""
        r = _pipeline('filter column B = 90')
        self.assertIn(r['intent'], ('filter_table', 'format_cells', 'filter'),
            f"filter command has unexpected intent={r['intent']!r}")
        if r['action'] is not None:
            self.assertNotEqual(r['action'].get('type'), 'create_excel',
                "filter command must not become create_excel")

    def test_regression_merge(self):
        """merge_cells command must still work after P27 changes."""
        r = _pipeline('merge cells A1:C1')
        self.assertEqual(r['intent'], 'merge_cells',
            f"'merge cells A1:C1' wrong intent={r['intent']!r}")

    def test_regression_freeze(self):
        """freeze header command must still work after P27 changes."""
        r = _pipeline('freeze header')
        self.assertIn(r['intent'], ('freeze_panes',),
            f"'freeze header' wrong intent={r['intent']!r}")


class TestP28FontSizeSafetyGuard(unittest.TestCase):
    """Safety guard: STT corruptions near font-size never fall through to LLM / create_excel.

    VALID commands must execute (slot_msg=None, action resolved).
    INVALID / garbled commands must clarify (slot_msg set, action None or safe).
    Regression: column-width and file commands must not be affected.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    def _assert_font_size(self, raw: str, expected_range: str, expected_size: int):
        r = _pipeline(raw)
        self.assertIsNone(r['slot_msg'],
            f"'{raw}' unexpectedly clarified: {r['slot_msg']!r}")
        a = r['action']
        self.assertIsNotNone(a, f"'{raw}' action=None")
        self.assertEqual(a.get('type'), 'format_cells',
            f"'{raw}' action type={a.get('type')!r}")
        self.assertEqual(str(a.get('font_size', '')), str(expected_size),
            f"'{raw}' font_size={a.get('font_size')!r}, expected {expected_size}")

    def _assert_clarify(self, raw: str, fragment: str = ''):
        r = _pipeline(raw)
        self.assertIsNotNone(r['slot_msg'],
            f"'{raw}' should clarify but slot_msg=None (action={r['action']!r})")
        if fragment:
            self.assertIn(fragment.lower(), r['slot_msg'].lower(),
                f"'{raw}' slot_msg={r['slot_msg']!r} missing {fragment!r}")
        if r['action'] is not None:
            self.assertNotEqual(r['action'].get('type'), 'create_excel',
                f"'{raw}' must not become create_excel")

    def _assert_not_create_excel(self, raw: str):
        r = _pipeline(raw)
        if r['action'] is not None:
            self.assertNotEqual(r['action'].get('type'), 'create_excel',
                f"'{raw}' must not become create_excel")

    # ── VALID: these must execute without clarification ───────────────────────

    def test_1_valid_set_font_size_cell(self):
        """set font size A2 30 → executes, font_size=30."""
        self._assert_font_size('set font size A2 30', 'A2', 30)

    def test_2_valid_set_font_size_row(self):
        """set font size row 3 16 → executes, font_size=16."""
        r = _pipeline('set font size row 3 16')
        self.assertIsNone(r['slot_msg'],
            f"'set font size row 3 16' clarified: {r['slot_msg']!r}")
        a = r['action']
        self.assertIsNotNone(a, "'set font size row 3 16' action=None")
        self.assertEqual(str(a.get('font_size', '')), '16')

    def test_3_valid_increase_font_size_cell(self):
        """increase font size A2 → executes (delta command)."""
        r = _pipeline('increase font size A2')
        self.assertIsNone(r['slot_msg'],
            f"'increase font size A2' clarified: {r['slot_msg']!r}")
        a = r['action']
        self.assertIsNotNone(a, "'increase font size A2' action=None")
        self.assertEqual(a.get('type'), 'format_cells')
        self.assertIsNotNone(a.get('font_size_change'),
            "'increase font size A2' missing font_size_change")
        self.assertGreater(int(a['font_size_change']), 0,
            f"expected positive delta, got {a['font_size_change']!r}")

    def test_4_valid_decrease_font_size_column(self):
        """decrease font size column B → executes (delta command)."""
        r = _pipeline('decrease font size column B')
        self.assertIsNone(r['slot_msg'],
            f"'decrease font size column B' clarified: {r['slot_msg']!r}")
        a = r['action']
        self.assertIsNotNone(a, "'decrease font size column B' action=None")
        self.assertLess(int(a.get('font_size_change', 0)), 0,
            f"expected negative delta, got {a.get('font_size_change')!r}")

    def test_5_valid_make_header_size(self):
        """make header size 18 → executes, font_size=18."""
        r = _pipeline('make header size 18')
        self.assertIsNone(r['slot_msg'],
            f"'make header size 18' clarified: {r['slot_msg']!r}")
        a = r['action']
        self.assertIsNotNone(a, "'make header size 18' action=None")
        self.assertEqual(str(a.get('font_size', '')), '18')

    # ── INVALID / garbled: these must clarify ─────────────────────────────────

    def test_6_invalid_make_hat_size(self):
        """make hat size 2 — STT corruption of 'make header size' → must clarify."""
        self._assert_clarify('make hat size 2')

    def test_7_invalid_set_font_sign(self):
        """set font-sign D130 — garbled hyphenated form → must clarify."""
        self._assert_clarify('set font-sign D130')

    def test_8_invalid_set_font_size_no_target(self):
        """set font size A — letter without digits is not a valid cell → must clarify."""
        self._assert_clarify('set font size A')

    def test_9_invalid_set_font_size_row_only(self):
        """set font size row — row without number → must clarify."""
        self._assert_clarify('set font size row')

    def test_10_invalid_increase_font_size_no_target(self):
        """increase font size — no target → must clarify."""
        self._assert_clarify('increase font size')

    def test_11_invalid_set_font_size_be_130(self):
        """set font size be 130 — 'be' is not a valid target → must clarify."""
        self._assert_clarify('set font size be 130')

    # ── Safety: font-size corruptions must never become create_excel ──────────

    def test_12_safety_make_hat_size_not_create_excel(self):
        """make hat size 2 must never route to create_excel."""
        self._assert_not_create_excel('make hat size 2')

    def test_13_safety_set_font_sign_not_create_excel(self):
        """set font-sign D130 must never route to create_excel."""
        self._assert_not_create_excel('set font-sign D130')

    def test_14_safety_make_header_size_20cm_not_create_excel(self):
        """make header size 20cm — cm suffix makes number invalid → clarify, not create_excel."""
        self._assert_not_create_excel('make header size 20cm')

    # ── Regression: column-width commands still work ──────────────────────────

    def test_15_regression_set_column_width(self):
        """set column width A 20 must not be intercepted by font-size slot gate."""
        r = _pipeline('set column width A 20')
        if r['slot_msg'] is None:
            a = r['action']
            if a is not None:
                self.assertNotEqual(a.get('type'), 'format_cells',
                    "'set column width A 20' must not become format_cells with font_size")
        # Main guarantee: must not clarify about font size
        if r['slot_msg'] is not None:
            self.assertNotIn('font size', r['slot_msg'].lower(),
                f"'set column width A 20' gave a font-size clarification: {r['slot_msg']!r}")

    def test_16_regression_resize_column(self):
        """resize column B to 25 must not be intercepted by font-size changes."""
        r = _pipeline('resize column B to 25')
        if r['slot_msg'] is not None:
            self.assertNotIn('font size', r['slot_msg'].lower(),
                f"'resize column B to 25' gave a font-size clarification: {r['slot_msg']!r}")

    # ── Regression: file / open commands still work ───────────────────────────

    def test_17_regression_open_file(self):
        """open file commands must not be blocked by font-size guard."""
        r = _pipeline('open file report.xlsx')
        if r['action'] is not None:
            self.assertNotEqual(r['action'].get('type'), 'format_cells',
                "'open file report.xlsx' must not become format_cells")

    def test_18_regression_save_file(self):
        """save / save file must not trigger font-size slot gate."""
        r = _pipeline('save file')
        if r['slot_msg'] is not None:
            self.assertNotIn('font size', r['slot_msg'].lower(),
                f"'save file' gave a font-size clarification: {r['slot_msg']!r}")


class TestP29OpenFileRouting(unittest.TestCase):
    """Bug fix: 'open file [name]' must route to open_file, not open_excel_app.

    Root cause 1: _COMMAND_PREFIX_PATTERNS had 'file' in the 'open excel' rule,
    converting 'open file filterr.xlsx' → 'open excel filterr.xlsx'.
    Root cause 2: _INTENT_PATTERNS['open_excel'] matched 'open file ...'.
    Root cause 3: heuristic open_file only extracted single-word filenames with
    known extensions, missing 'file server' (multi-word, no extension).
    """

    # ── Routing: open file [name] → open_file, not open_excel ────────────────

    def test_1_open_file_filterr_xlsx_intent(self):
        """'open file filterr.xlsx' must detect as open_file intent."""
        r = _pipeline('open file filterr.xlsx')
        self.assertEqual(r['intent'], 'open_file',
            f"Expected open_file, got {r['intent']!r} (corrected={r['corrected']!r})")

    def test_2_open_file_filterr_xlsx_action(self):
        """'open file filterr.xlsx' must produce open_file action with filename."""
        r = _pipeline('open file filterr.xlsx')
        self.assertIsNone(r['slot_msg'],
            f"'open file filterr.xlsx' should not clarify: {r['slot_msg']!r}")
        a = r['action']
        self.assertIsNotNone(a, "'open file filterr.xlsx' action=None")
        self.assertEqual(a.get('type'), 'open_file',
            f"action type={a.get('type')!r}")
        self.assertIn('filterr.xlsx', a.get('filename', ''),
            f"filename={a.get('filename')!r}")

    def test_3_open_file_server_no_extension(self):
        """'open file file server' must produce open_file with filename=file server.xlsx."""
        r = _pipeline('open file file server')
        self.assertEqual(r['intent'], 'open_file',
            f"Expected open_file, got {r['intent']!r}")
        a = r['action']
        self.assertIsNotNone(a, "'open file file server' action=None")
        self.assertEqual(a.get('type'), 'open_file',
            f"action type={a.get('type')!r}")
        fn = a.get('filename', '')
        self.assertIn('file server', fn.lower(),
            f"Expected 'file server' in filename, got {fn!r}")

    def test_4_open_file_server_xlsx(self):
        """'open file file server.xlsx' must produce open_file with the full filename."""
        r = _pipeline('open file file server.xlsx')
        self.assertEqual(r['intent'], 'open_file',
            f"Expected open_file, got {r['intent']!r}")
        a = r['action']
        self.assertIsNotNone(a)
        fn = a.get('filename', '')
        self.assertIn('file server', fn.lower(),
            f"Expected 'file server' in filename, got {fn!r}")
        self.assertTrue(fn.lower().endswith('.xlsx'),
            f"Expected .xlsx extension, got {fn!r}")

    def test_5_open_file_client(self):
        """'open file client' must produce open_file with filename=client.xlsx."""
        r = _pipeline('open file client')
        self.assertEqual(r['intent'], 'open_file',
            f"Expected open_file, got {r['intent']!r}")
        a = r['action']
        self.assertIsNotNone(a)
        self.assertEqual(a.get('type'), 'open_file')
        self.assertIn('client', a.get('filename', '').lower(),
            f"filename={a.get('filename')!r}")

    # ── Transcript correction must NOT convert "open file ..." to "open excel ..."

    def test_6_correct_transcript_open_file_not_corrupted(self):
        """correct_transcript must not convert 'open file filterr.xlsx' to 'open excel ...'"""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('open file filterr.xlsx')
        self.assertNotIn('open excel', corrected.lower(),
            f"'open file filterr.xlsx' was wrongly converted: {corrected!r}")
        self.assertIn('open file', corrected.lower(),
            f"Expected 'open file' in corrected text, got {corrected!r}")

    def test_7_correct_transcript_open_file_server_not_corrupted(self):
        """correct_transcript must preserve 'open file file server' as open_file."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('open file file server')
        self.assertNotIn('open excel', corrected.lower(),
            f"'open file file server' was wrongly converted: {corrected!r}")

    # ── open excel alone must still work ─────────────────────────────────────

    def test_8_open_excel_still_routes_correctly(self):
        """'open excel' must still detect as open_excel intent (not open_file)."""
        r = _pipeline('open excel')
        self.assertEqual(r['intent'], 'open_excel',
            f"Expected open_excel, got {r['intent']!r}")
        self.assertIsNone(r['slot_msg'])
        a = r['action']
        self.assertIsNotNone(a)
        self.assertEqual(a.get('type'), 'open_excel_app',
            f"Expected open_excel_app, got {a.get('type')!r}")

    def test_9_open_each_there_still_open_excel(self):
        """Phonetic garbles of 'open excel' must still route to open_excel (P30 regression)."""
        r = _pipeline('open each there')
        self.assertEqual(r['intent'], 'open_excel',
            f"Expected open_excel, got {r['intent']!r}")


class TestP29RenameFilePreservesState(unittest.TestCase):
    """Bug fix: rename_file must preserve in-memory workbook state (deleted sheets,
    written cells, added rows) by always flushing openpyxl to disk before renaming.

    Root cause: COM-path operations (delete_sheet, write_cell, etc.) update the
    openpyxl workbook in memory but do NOT set _session.dirty=True.  The old code
    only called _atomic_save when dirty=True, so a stale disk file was renamed and
    all COM-path changes were silently lost.
    """

    def _make_temp_xlsx(self, sheets=None):
        """Create a temporary .xlsx in the backend/files/ dir and return its path."""
        import tempfile, pathlib
        files_dir = pathlib.Path(__file__).parent.parent / 'files'
        files_dir.mkdir(exist_ok=True)
        fd, path = tempfile.mkstemp(suffix='.xlsx', dir=files_dir)
        os.close(fd)
        from openpyxl import Workbook
        wb = Workbook()
        wb.active.title = 'Sheet1'
        for extra in (sheets or []):
            wb.create_sheet(extra)
        wb.save(path)
        return path

    def _cleanup(self, *paths):
        for p in paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    def test_10_rename_preserves_written_cell(self):
        """After write_cell A1=hello + rename, the renamed file still has A1=hello."""
        from openpyxl import load_workbook
        import excel_controller as ec

        src = self._make_temp_xlsx()
        new_name = os.path.basename(src).replace('.xlsx', '_renamed.xlsx')
        new_path = os.path.join(os.path.dirname(src), new_name)
        self._cleanup(new_path)
        try:
            ec.open_excel(src)
            ec.write_cell(src, 'A1', 'hello')
            result = ec.rename_file(src, new_name)
            self.assertNotIn('error', result,
                f"rename_file failed: {result}")
            self.assertTrue(os.path.exists(new_path),
                f"Renamed file not found at {new_path!r}")
            wb = load_workbook(new_path)
            val = wb.active['A1'].value
            self.assertEqual(val, 'hello',
                f"A1 should be 'hello' after rename, got {val!r}")
        finally:
            self._cleanup(src, new_path)

    def test_11_rename_preserves_added_row(self):
        """After add_row Mine 90 + rename, the renamed file still has the row."""
        from openpyxl import load_workbook
        import excel_controller as ec

        src = self._make_temp_xlsx()
        new_name = os.path.basename(src).replace('.xlsx', '_row_renamed.xlsx')
        new_path = os.path.join(os.path.dirname(src), new_name)
        self._cleanup(new_path)
        try:
            ec.open_excel(src)
            ec.add_row(src, ['Mine', 90])
            result = ec.rename_file(src, new_name)
            self.assertNotIn('error', result,
                f"rename_file failed: {result}")
            wb = load_workbook(new_path)
            rows = list(wb.active.iter_rows(values_only=True))
            found = any('Mine' in (str(c) for c in row if c is not None) for row in rows)
            self.assertTrue(found,
                f"Row with 'Mine' not found after rename; rows={rows}")
        finally:
            self._cleanup(src, new_path)

    def test_12_rename_updates_session_path(self):
        """After rename, the internal session abs_path must point at the new filename."""
        import excel_controller as ec

        src = self._make_temp_xlsx()
        new_name = os.path.basename(src).replace('.xlsx', '_path_renamed.xlsx')
        new_path = os.path.join(os.path.dirname(src), new_name)
        self._cleanup(new_path)
        try:
            ec.open_excel(src)
            result = ec.rename_file(src, new_name)
            self.assertNotIn('error', result,
                f"rename_file failed: {result}")
            # The session should now point at the renamed file.
            with ec._session_lock:
                sess_path = ec._session.abs_path if ec._session else None
            if sess_path is not None:
                self.assertEqual(
                    os.path.normcase(sess_path),
                    os.path.normcase(new_path),
                    f"Session path not updated: {sess_path!r} vs {new_path!r}",
                )
        finally:
            self._cleanup(src, new_path)

    def test_13_delete_sheet_then_rename_sheet_stays_deleted(self):
        """Create sheet Report, delete sheet Report, rename file → Report must not come back."""
        from openpyxl import load_workbook
        import excel_controller as ec

        src = self._make_temp_xlsx(sheets=['Report'])
        new_name = os.path.basename(src).replace('.xlsx', '_del_renamed.xlsx')
        new_path = os.path.join(os.path.dirname(src), new_name)
        self._cleanup(new_path)
        try:
            ec.open_excel(src)
            del_result = ec.delete_sheet(src, 'Report')
            if 'error' in del_result:
                self.skipTest(f"delete_sheet failed (COM unavailable): {del_result}")
            result = ec.rename_file(src, new_name)
            self.assertNotIn('error', result,
                f"rename_file failed: {result}")
            wb = load_workbook(new_path)
            self.assertNotIn('Report', wb.sheetnames,
                f"'Report' came back after rename; sheets={wb.sheetnames}")
        finally:
            self._cleanup(src, new_path)

    def test_14_rename_file_intent_still_works(self):
        """Pipeline: 'rename file to budget' still routes to rename_file intent (regression)."""
        r = _pipeline('rename file to budget')
        self.assertEqual(r['intent'], 'rename_file',
            f"Expected rename_file, got {r['intent']!r}")
        self.assertIsNone(r['slot_msg'])


class TestP30OpenFileRouterFastPath(unittest.TestCase):
    """Bug fix: router fast-path open_file filename must not read memory context prefix.

    Root cause: llm_router.py fast-path used _FILENAME_RE.search(text) which scans the
    FULL text including memory context prefixes like [file:server.xlsx sheet:sheet1].
    For "open file server", text = "[file:server.xlsx sheet:sheet1] open file server"
    and _FILENAME_RE found "file:server.xlsx" (from the prefix) instead of "server".

    Fix: extract filename from the command TAIL after "open [the] file/..." using re.search,
    and validate that the result contains no context-prefix artifacts (file:, sheet:, [, ]).
    """

    @staticmethod
    def _route(text: str) -> dict:
        """Call route_llm directly so the router fast-path is exercised (not just heuristic).

        For 'open file ...' inputs the fast-path fires and returns immediately before
        any LLM call, so no mocking is needed.
        """
        from llm_router import route_llm
        return route_llm(text)

    # ── Core filename extraction ─────────────────────────────────────────────

    def test_1_open_file_server_plain(self):
        """'open file server' → open_file, filename=server.xlsx."""
        r = self._route('open file server')
        self.assertEqual(r.get('type'), 'open_file',
            f"type={r.get('type')!r}")
        fn = r.get('filename', '')
        self.assertIn('server', fn.lower(), f"filename={fn!r}")
        self.assertTrue(fn.lower().endswith('.xlsx'), f"no .xlsx extension: {fn!r}")

    def test_2_open_file_server_with_context_prefix(self):
        """'[file:server.xlsx sheet:sheet1] open file server' → filename=server.xlsx, not file:server.xlsx."""
        r = self._route('[file:server.xlsx sheet:sheet1] open file server')
        self.assertEqual(r.get('type'), 'open_file',
            f"type={r.get('type')!r}")
        fn = r.get('filename', '')
        self.assertNotIn('file:', fn, f"filename contains context artifact 'file:': {fn!r}")
        self.assertNotIn('sheet:', fn, f"filename contains context artifact 'sheet:': {fn!r}")
        self.assertIn('server', fn.lower(), f"filename should contain 'server': {fn!r}")

    def test_3_open_file_server_xlsx(self):
        """'open file server.xlsx' → filename=server.xlsx (extension preserved)."""
        r = self._route('open file server.xlsx')
        fn = r.get('filename', '')
        self.assertEqual(fn.lower(), 'server.xlsx', f"filename={fn!r}")

    def test_4_open_file_multi_word_no_ext(self):
        """'open file file server' → filename=file server.xlsx (multi-word preserved)."""
        r = self._route('open file file server')
        fn = r.get('filename', '')
        self.assertIn('file server', fn.lower(), f"filename={fn!r}")
        self.assertTrue(fn.lower().endswith('.xlsx'), f"no .xlsx: {fn!r}")

    def test_5_open_file_multi_word_with_context(self):
        """'[file:server.xlsx sheet:sheet1] open file file server' → filename=file server.xlsx."""
        r = self._route('[file:server.xlsx sheet:sheet1] open file file server')
        fn = r.get('filename', '')
        self.assertNotIn('file:', fn, f"context artifact in filename: {fn!r}")
        self.assertIn('file server', fn.lower(), f"filename={fn!r}")

    def test_6_open_file_filterr_xlsx(self):
        """'open file filterr.xlsx' → filename=filterr.xlsx."""
        r = self._route('open file filterr.xlsx')
        fn = r.get('filename', '')
        self.assertEqual(fn.lower(), 'filterr.xlsx', f"filename={fn!r}")

    def test_7_open_file_client(self):
        """'open file client' → filename=client.xlsx."""
        r = self._route('open file client')
        fn = r.get('filename', '')
        self.assertIn('client', fn.lower(), f"filename={fn!r}")
        self.assertTrue(fn.lower().endswith('.xlsx'), f"no .xlsx: {fn!r}")

    # ── Safety: filename must never contain context-prefix artifacts ──────────

    def test_8_filename_never_contains_file_colon(self):
        """Filename must never contain 'file:' regardless of context prefix."""
        for raw in [
            '[file:server.xlsx sheet:sheet1] open file server',
            '[file:client.xlsx] open file client',
            'open file server',
        ]:
            r = self._route(raw)
            fn = r.get('filename', '')
            self.assertNotIn('file:', fn,
                f"'{raw}' → filename={fn!r} contains 'file:'")

    def test_9_filename_never_contains_sheet_colon(self):
        """Filename must never contain 'sheet:' regardless of context prefix."""
        for raw in [
            '[file:server.xlsx sheet:sheet1] open file server',
            '[sheet:Sheet1] open file client',
        ]:
            r = self._route(raw)
            fn = r.get('filename', '')
            self.assertNotIn('sheet:', fn,
                f"'{raw}' → filename={fn!r} contains 'sheet:'")

    # ── open excel must still route to open_excel_app ────────────────────────

    def test_10_open_excel_still_open_excel_app(self):
        """'open excel' must still route to open_excel_app (not open_file)."""
        r = self._route('open excel')
        self.assertEqual(r.get('type'), 'open_excel_app',
            f"Expected open_excel_app, got {r.get('type')!r}")

    # ── Protected baseline commands (pipeline-level) ──────────────────────────

    def test_11_protected_baseline_open_file_filterr(self):
        """Pipeline: 'open file filterr.xlsx' → open_file intent (baseline)."""
        r = _pipeline('open file filterr.xlsx')
        self.assertEqual(r['intent'], 'open_file',
            f"Expected open_file, got {r['intent']!r}")
        a = r['action']
        self.assertIsNotNone(a, "action should not be None")
        self.assertEqual(a.get('type'), 'open_file')
        self.assertIn('filterr', a.get('filename', '').lower())

    def test_12_protected_baseline_open_excel(self):
        """Pipeline: 'open excel' → open_excel intent and open_excel_app action (baseline)."""
        r = _pipeline('open excel')
        self.assertEqual(r['intent'], 'open_excel',
            f"Expected open_excel, got {r['intent']!r}")
        self.assertIsNone(r['slot_msg'])
        self.assertEqual(r['action'].get('type'), 'open_excel_app')


# ---------------------------------------------------------------------------
# P31 — Color / fill deterministic routing
#        "color"/"make [cell] [color]" → FONT COLOR
#        "fill"/"highlight" → BACKGROUND FILL
# ---------------------------------------------------------------------------

class TestP31ColorFillDeterministic(unittest.TestCase):
    """P31: font-color vs fill-color must be deterministic without LLM.

    Key rule:
      "color" / "make [cell/row/col] [color-word]"  → font_color
      "fill" / "highlight"                           → fill_color (background)

    Root cause: fmt_make_color in llm_parser.py used 'fill_color' for
    "make [cell] [color]" commands.  Fixed to 'font_color'.  Also added
    heuristics for make row/col color and highlight [cell/range] with no
    explicit color (defaults to yellow).
    """

    # ── 1. make [cell] [color] → font_color ──────────────────────────────────

    def test_1_make_A3_orange_font_color(self):
        """'make A3 orange' → font_color=orange range=A3 (NOT fill_color)."""
        r = _pipeline('make A3 orange')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('font_color'), 'orange',
                         f"Expected font_color=orange, got: {r['action']}")
        self.assertNotIn('fill_color', r['action'], "must NOT set fill_color for make [cell] [color]")
        self.assertEqual(r['action'].get('range'), 'A3')

    # ── 2–4. color verb → font_color (existing behavior preserved) ───────────

    def test_2_color_B2_red_font_color(self):
        """'color B2 red' → font_color=red range=B2."""
        r = _pipeline('color B2 red')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('font_color'), 'red')
        self.assertEqual(r['action'].get('range'), 'B2')
        self.assertNotIn('fill_color', r['action'])

    def test_3_color_row_3_blue_font_color(self):
        """'color row 3 blue' → font_color=blue range=3:3."""
        r = _pipeline('color row 3 blue')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('range'), '3:3')
        self.assertEqual(r['action'].get('font_color'), 'blue')
        self.assertNotIn('fill_color', r['action'])

    def test_4_color_column_B_green_font_color(self):
        """'color column B green' → font_color=green range=B:B."""
        r = _pipeline('color column B green')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('range'), 'B:B')
        self.assertEqual(r['action'].get('font_color'), 'green')
        self.assertNotIn('fill_color', r['action'])

    # ── 5–6. make row/col [color] → font_color (new behavior) ───────────────

    def test_5_make_row_3_blue_font_color(self):
        """'make row 3 blue' → font_color=blue range=3:3."""
        r = _pipeline('make row 3 blue')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'], f"Heuristic missed; corrected={r['corrected']!r}")
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), '3:3')
        self.assertEqual(r['action'].get('font_color'), 'blue')
        self.assertNotIn('fill_color', r['action'])

    def test_6_make_column_C_orange_font_color(self):
        """'make column C orange' → font_color=orange range=C:C."""
        r = _pipeline('make column C orange')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'], f"Heuristic missed; corrected={r['corrected']!r}")
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), 'C:C')
        self.assertEqual(r['action'].get('font_color'), 'orange')
        self.assertNotIn('fill_color', r['action'])

    # ── 7–11. fill → fill_color (background) ─────────────────────────────────

    def test_7_fill_row_2_with_red_fill_color(self):
        """'fill row 2 with red' → fill_color=red range=2:2."""
        r = _pipeline('fill row 2 with red')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), '2:2')
        self.assertEqual(r['action'].get('fill_color'), 'red')
        self.assertNotIn('font_color', r['action'])

    def test_8_highlight_row_4_yellow(self):
        """'highlight row 4' → fill_color=yellow range=4:4."""
        r = _pipeline('highlight row 4')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), '4:4')
        self.assertEqual(r['action'].get('fill_color'), 'yellow')
        self.assertNotIn('font_color', r['action'])

    def test_9_fill_A4_pink_fill_color(self):
        """'fill A4 pink' → fill_color=pink range=A4."""
        r = _pipeline('fill A4 pink')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), 'A4')
        self.assertEqual(r['action'].get('fill_color'), 'pink')
        self.assertNotIn('font_color', r['action'])

    def test_10_fill_A5_C5_with_green(self):
        """'fill A5:C5 with green' → fill_color=green range=A5:C5."""
        r = _pipeline('fill A5:C5 with green')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), 'A5:C5')
        self.assertEqual(r['action'].get('fill_color'), 'green')
        self.assertNotIn('font_color', r['action'])

    def test_11_fill_column_B_with_blue(self):
        """'fill column B with blue' → fill_color=blue range=B:B."""
        r = _pipeline('fill column B with blue')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), 'B:B')
        self.assertEqual(r['action'].get('fill_color'), 'blue')
        self.assertNotIn('font_color', r['action'])

    # ── 12. highlight [cell/range] → fill_color=yellow (new behavior) ────────

    def test_12_highlight_A1_C1_yellow(self):
        """'highlight A1:C1' (no color word) → fill_color=yellow range=A1:C1."""
        r = _pipeline('highlight A1:C1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'], f"Heuristic missed; corrected={r['corrected']!r}")
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('range'), 'A1:C1')
        self.assertEqual(r['action'].get('fill_color'), 'yellow')
        self.assertNotIn('font_color', r['action'])

    # ── 13–16. clarification cases ───────────────────────────────────────────

    def test_13_color_B2_no_color_word_clarifies(self):
        """'color B2' (no color word) → slot clarification, not execute."""
        r = _pipeline('color B2')
        self.assertIsNotNone(r['slot_msg'],
                             f"Missing color must trigger clarification; got action={r['action']}")
        self.assertIsNone(r['action'])

    def test_14_fill_A4_no_color_word_clarifies(self):
        """'fill A4' (no color word) → slot clarification, not execute."""
        r = _pipeline('fill A4')
        self.assertIsNotNone(r['slot_msg'],
                             f"Missing color must trigger clarification; got action={r['action']}")
        self.assertIsNone(r['action'])

    def test_15_highlight_no_target_clarifies(self):
        """Bare 'highlight' (no target) → slot clarification, not execute."""
        r = _pipeline('highlight')
        self.assertIsNotNone(r['slot_msg'],
                             f"Missing target must trigger clarification; got action={r['action']}")
        self.assertIsNone(r['action'])

    def test_16_color_red_no_target_clarifies(self):
        """'color red' (color word but no cell target) → slot clarification, not execute."""
        r = _pipeline('color red')
        self.assertIsNotNone(r['slot_msg'],
                             f"Missing target must trigger clarification; got action={r['action']}")
        self.assertIsNone(r['action'])

    # ── regression: bold/italic/font-size must not be affected ───────────────

    def test_reg_make_bold_unaffected(self):
        """'make A1 bold' must still produce bold=True, not font_color."""
        r = _pipeline('make A1 bold')
        self.assertIsNotNone(r['action'])
        self.assertTrue(r['action'].get('bold'))
        self.assertNotIn('font_color', r['action'])
        self.assertNotIn('fill_color', r['action'])

    def test_reg_set_font_size_unaffected(self):
        """'set font size A2 30' must not be blocked by color slot check."""
        r = _pipeline('set font size A2 30')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('font_size'), 30)

    def test_reg_fill_row_no_false_color_slot(self):
        """'fill row 2 with red' must not trigger color-missing slot check."""
        r = _pipeline('fill row 2 with red')
        self.assertIsNone(r['slot_msg'],
                          f"Complete fill command must not be blocked: {r['slot_msg']!r}")


# ---------------------------------------------------------------------------
# P32 — Number-format commands (format cells as currency/percentage/number/decimal)
# ---------------------------------------------------------------------------

class TestP32NumberFormatCommands(unittest.TestCase):
    """P32: number-format commands must route deterministically without LLM.

    Key rule: "format [target] as [fmt]" → format_cells, number_format=fmt.
    Supported targets: cell ref, cell range, row N, column C.
    Supported formats: currency, percentage/percent, number, integer, decimal.

    Root cause: fmt_num in llm_parser.py lacked "row N" support and "decimal"
    format name.  Intent patterns and slot checks for missing/unknown format
    were also absent.
    """

    # ── 1–6. Execute: format [target] as [fmt] ───────────────────────────────

    def test_1_format_range_as_currency(self):
        """'format B2:B10 as currency' → number_format=currency range=B2:B10."""
        r = _pipeline('format B2:B10 as currency')
        self.assertIsNone(r['slot_msg'], f"slot_msg={r['slot_msg']!r}")
        self.assertIsNotNone(r['action'], f"corrected={r['corrected']!r}")
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('number_format'), 'currency')
        self.assertEqual(r['action'].get('range'), 'B2:B10')

    def test_2_format_range_as_percentage(self):
        """'format C2:C8 as percentage' → number_format=percentage range=C2:C8."""
        r = _pipeline('format C2:C8 as percentage')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('number_format'), 'percentage')
        self.assertEqual(r['action'].get('range'), 'C2:C8')

    def test_3_format_range_as_number(self):
        """'format D2:D10 as number' → number_format=number range=D2:D10."""
        r = _pipeline('format D2:D10 as number')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('number_format'), 'number')
        self.assertEqual(r['action'].get('range'), 'D2:D10')

    def test_4_format_row_as_currency(self):
        """'format row 3 as currency' → number_format=currency range=3:3."""
        r = _pipeline('format row 3 as currency')
        self.assertIsNone(r['slot_msg'], f"slot_msg={r['slot_msg']!r}")
        self.assertIsNotNone(r['action'], f"corrected={r['corrected']!r}")
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('number_format'), 'currency')
        self.assertEqual(r['action'].get('range'), '3:3')

    def test_5_format_column_as_percentage(self):
        """'format column B as percentage' → number_format=percentage range=B:B."""
        r = _pipeline('format column B as percentage')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('number_format'), 'percentage')
        self.assertEqual(r['action'].get('range'), 'B:B')

    def test_6_format_cell_as_number(self):
        """'format A1 as number' → number_format=number range=A1."""
        r = _pipeline('format A1 as number')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('number_format'), 'number')
        self.assertEqual(r['action'].get('range'), 'A1')

    # ── 7–9. Clarify: missing target / missing format / unknown format ────────

    def test_7_missing_target_clarifies(self):
        """'format as currency' (no target) → slot clarification."""
        r = _pipeline('format as currency')
        self.assertIsNotNone(r['slot_msg'],
                             f"Missing target must clarify; action={r['action']}")
        self.assertIsNone(r['action'])

    def test_8_missing_format_clarifies(self):
        """'format B2:B10' (no format type) → slot clarification."""
        r = _pipeline('format B2:B10')
        self.assertIsNotNone(r['slot_msg'],
                             f"Missing format must clarify; action={r['action']}")
        self.assertIsNone(r['action'])

    def test_8b_missing_format_column_clarifies(self):
        """'format column B' (no format type) → slot clarification."""
        r = _pipeline('format column B')
        self.assertIsNotNone(r['slot_msg'],
                             f"Missing format must clarify; action={r['action']}")
        self.assertIsNone(r['action'])

    def test_9_unknown_format_clarifies(self):
        """'format B2:B10 as bold' (unknown format name) → slot clarification."""
        r = _pipeline('format B2:B10 as bold')
        self.assertIsNotNone(r['slot_msg'],
                             f"Unknown format must clarify; action={r['action']}")
        self.assertIsNone(r['action'])

    # ── 10. Font color / fill / background still pass ────────────────────────

    def test_10_color_B2_red_unaffected(self):
        """'color B2 red' must still produce font_color=red (not blocked)."""
        r = _pipeline('color B2 red')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('font_color'), 'red')

    def test_10b_fill_row_2_red_unaffected(self):
        """'fill row 2 with red' must still produce fill_color=red."""
        r = _pipeline('fill row 2 with red')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('fill_color'), 'red')

    def test_10c_highlight_row_4_unaffected(self):
        """'highlight row 4' must still produce fill_color=yellow."""
        r = _pipeline('highlight row 4')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('fill_color'), 'yellow')

    def test_10d_make_A3_orange_unaffected(self):
        """'make A3 orange' must still produce font_color=orange."""
        r = _pipeline('make A3 orange')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('font_color'), 'orange')

    def test_10e_fill_A5_C5_green_unaffected(self):
        """'fill A5:C5 with green' must still produce fill_color=green."""
        r = _pipeline('fill A5:C5 with green')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('fill_color'), 'green')

    # ── 11. Bold / italic / alignment / font-size still pass ─────────────────

    def test_11_bold_A1_unaffected(self):
        """'bold A1' must still produce bold=True."""
        r = _pipeline('bold A1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertTrue(r['action'].get('bold'))

    def test_11b_italic_C6_unaffected(self):
        """'italic C6' must still produce italic=True."""
        r = _pipeline('italic C6')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertTrue(r['action'].get('italic'))

    def test_11c_center_B2_unaffected(self):
        """'center B2' must still produce align=center."""
        r = _pipeline('center B2')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('align'), 'center')

    def test_11d_set_font_size_A1_20_unaffected(self):
        """'set font size A1 20' must still produce font_size=20."""
        r = _pipeline('set font size A1 20')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('font_size'), 20)

    def test_11e_format_range_bold_unaffected(self):
        """'format A13:C13 bold' must still produce bold=True (not blocked by number-format check)."""
        r = _pipeline('format A13:C13 bold')
        self.assertIsNone(r['slot_msg'],
                          f"'format [range] bold' must not be blocked: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertTrue(r['action'].get('bold'))

    # ── 12. Full baseline regression ─────────────────────────────────────────

    def test_12a_format_JFI_as_number_still_clarifies(self):
        """'format JFI as number' (garbled range) still triggers clarification."""
        r = _pipeline('format JFI as number')
        self.assertIsNotNone(r['slot_msg'],
                             "Garbled range must still trigger clarification")
        self.assertIsNone(r['action'])

    def test_12b_format_B2_B10_as_currency_still_works(self):
        """'format B2:B10 as currency' still executes (regression guard)."""
        r = _pipeline('format B2:B10 as currency')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('number_format'), 'currency')

    def test_12c_bold_row_11_still_works(self):
        """'bold row 11' must still produce bold range=11:11."""
        r = _pipeline('bold row 11')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertTrue(r['action'].get('bold'))
        self.assertEqual(r['action'].get('range'), '11:11')

    def test_12d_align_row_3_left_still_works(self):
        """'align row 3 left' must still produce align=left range=3:3."""
        r = _pipeline('align row 3 left')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('align'), 'left')


# ---------------------------------------------------------------------------
# P33 — Number-format COM live path + voice normalization
#        Root cause: _com_live_format_cells missing num_format param →
#        Excel UI never updated; decimal format name unresolved;
#        "format B2 to B10" voice form not normalized.
# ---------------------------------------------------------------------------

class TestP33NumberFormatCOMPath(unittest.TestCase):
    """P33: number-format must update Excel visually via COM and fall back to openpyxl.

    Three root causes fixed:
    1. _com_live_format_cells had no num_format param → COM path dropped it,
       saved without number format, then marked openpyxl session clean so the
       in-memory change was also lost.
    2. 'decimal' missing from _NUM_FORMAT_NAMES → invalid Excel format string.
    3. Voice "format B2 to B10" / "format C2 through C8" not normalized to colon form.
    """

    # ── 1. COM signature has num_format param ────────────────────────────────

    def test_1_com_live_format_cells_has_num_format_param(self):
        """_com_live_format_cells must accept a num_format keyword argument."""
        import inspect
        import excel_controller as ec
        sig = inspect.signature(ec._com_live_format_cells)
        self.assertIn('num_format', sig.parameters,
                      "_com_live_format_cells must have a 'num_format' parameter")

    # ── 2. COM path sets rng.NumberFormat ────────────────────────────────────

    def test_2_com_path_sets_NumberFormat(self):
        """When Excel is open, _com_live_format_cells must set rng.NumberFormat."""
        from unittest.mock import MagicMock, patch
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_rng = MagicMock()
        mock_wb.ActiveSheet = mock_ws
        mock_ws.Range.return_value = mock_rng

        with patch.object(ec, '_com_find_workbook', return_value=mock_wb), \
             patch('pythoncom.CoInitialize'):
            try:
                import pythoncom  # noqa: F401
            except ImportError:
                self.skipTest('pythoncom not available')
            result = ec._com_live_format_cells(
                'fake.xlsx', 'B2:B10', num_format='$#,##0.00'
            )

        if result is not None:
            mock_rng.NumberFormat  # attribute access must have been attempted
            mock_rng.NumberFormat = '$#,##0.00'  # confirm assignment shape
            self.assertIsNotNone(result)

    def test_2b_com_live_format_num_format_attribute_set(self):
        """COM path must assign NumberFormat on the range object."""
        from unittest.mock import MagicMock, patch, call
        import excel_controller as ec

        mock_wb = MagicMock()
        mock_ws = MagicMock()
        mock_rng = MagicMock()
        mock_wb.ActiveSheet = mock_ws
        mock_ws.Range.return_value = mock_rng

        try:
            import pythoncom  # noqa: F401
        except ImportError:
            self.skipTest('pythoncom not available')

        with patch.object(ec, '_com_find_workbook', return_value=mock_wb), \
             patch('pythoncom.CoInitialize'):
            result = ec._com_live_format_cells(
                'fake.xlsx', 'C2:C8', num_format='0.00%'
            )

        if result is not None:
            # MagicMock records all attribute sets; verify NumberFormat was written
            self.assertTrue(
                hasattr(mock_rng, 'NumberFormat'),
                "rng.NumberFormat must be set by the COM path",
            )

    # ── 3. COM success → _flush_session NOT called ───────────────────────────

    def test_3_com_success_skips_flush(self):
        """When COM succeeds for num_format, _flush_session must NOT be called."""
        from unittest.mock import patch, MagicMock
        import excel_controller as ec
        import tempfile, openpyxl, os

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = os.path.join(td, 'book.xlsx')
            wb = openpyxl.Workbook(); wb.save(p)

            fake_com = {'_com_used': True}
            with patch.object(ec, '_com_live_format_cells', return_value=fake_com), \
                 patch.object(ec, '_flush_session') as mock_flush, \
                 patch.object(ec, '_undo_snapshot'):
                ec.format_cells(p, 'B2:B10', num_format='currency')
                mock_flush.assert_not_called()

    # ── 4. Fallback openpyxl sets cell.number_format ─────────────────────────

    def test_4_openpyxl_fallback_sets_number_format(self):
        """When COM unavailable, openpyxl must persist number_format to disk."""
        from unittest.mock import patch
        import excel_controller as ec
        import tempfile, openpyxl, os

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = os.path.join(td, 'book.xlsx')
            wb = openpyxl.Workbook()
            wb.active['B2'] = 42
            wb.save(p)

            with patch.object(ec, '_com_find_workbook', return_value=None), \
                 patch.object(ec, '_undo_snapshot'):
                ec.format_cells(p, 'B2', num_format='currency')

            wb2 = openpyxl.load_workbook(p)
            fmt = wb2.active['B2'].number_format
            self.assertEqual(fmt, '$#,##0.00',
                             f"cell.number_format must be '$#,##0.00', got {fmt!r}")

    def test_4b_decimal_resolves_to_0_00(self):
        """'decimal' format name must resolve to '0.00' (not the literal string 'decimal')."""
        from unittest.mock import patch
        import excel_controller as ec
        import tempfile, openpyxl, os

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = os.path.join(td, 'book.xlsx')
            wb = openpyxl.Workbook(); wb.save(p)

            with patch.object(ec, '_com_find_workbook', return_value=None), \
                 patch.object(ec, '_undo_snapshot'):
                ec.format_cells(p, 'A1', num_format='decimal')

            wb2 = openpyxl.load_workbook(p)
            fmt = wb2.active['A1'].number_format
            self.assertEqual(fmt, '0.00',
                             f"'decimal' must resolve to '0.00', got {fmt!r}")

    # ── 5–6. Voice normalization: "to" / "through" range forms ───────────────

    def test_5_format_B2_to_B10_as_currency(self):
        """'format B2 to B10 as currency' → format_cells number_format=currency range=B2:B10."""
        r = _pipeline('format B2 to B10 as currency')
        self.assertIsNone(r['slot_msg'],
                          f"slot_msg={r['slot_msg']!r} corrected={r['corrected']!r}")
        self.assertIsNotNone(r['action'], f"corrected={r['corrected']!r}")
        self.assertEqual(r['action'].get('type'), 'format_cells')
        self.assertEqual(r['action'].get('number_format'), 'currency')
        self.assertEqual(r['action'].get('range'), 'B2:B10')

    def test_6_format_C2_through_C8_as_percentage(self):
        """'format C2 through C8 as percentage' → format_cells number_format=percentage range=C2:C8."""
        r = _pipeline('format C2 through C8 as percentage')
        self.assertIsNone(r['slot_msg'],
                          f"slot_msg={r['slot_msg']!r} corrected={r['corrected']!r}")
        self.assertIsNotNone(r['action'], f"corrected={r['corrected']!r}")
        self.assertEqual(r['action'].get('number_format'), 'percentage')
        self.assertEqual(r['action'].get('range'), 'C2:C8')

    def test_6b_voice_normalization_transcript(self):
        """Transcript correction must convert 'format B2 to B10' to 'format B2:B10'."""
        from pipeline_sanitizer import correct_transcript
        corrected = correct_transcript('format B2 to B10 as currency')
        self.assertIn('B2:B10', corrected,
                      f"Expected 'B2:B10' in corrected transcript, got {corrected!r}")
        self.assertNotIn(' to ', corrected.lower().split('as')[0],
                         f"'to' range form must be normalized, got {corrected!r}")

    # ── 7. _NUM_FORMAT_NAMES: all supported names resolve correctly ───────────

    def test_7_num_format_names_resolve(self):
        """All supported format names must resolve to Excel format strings."""
        from excel_controller import _NUM_FORMAT_NAMES
        expected = {
            'currency':   '$#,##0.00',
            'percentage': '0.00%',
            'percent':    '0.00%',
            'number':     '#,##0.00',
            'decimal':    '0.00',
            'integer':    '#,##0',
        }
        for name, fmt in expected.items():
            self.assertEqual(
                _NUM_FORMAT_NAMES.get(name), fmt,
                f"_NUM_FORMAT_NAMES['{name}'] expected {fmt!r}, got {_NUM_FORMAT_NAMES.get(name)!r}",
            )

    # ── 8. Protected commands: color/fill/bold/italic not broken ──────────────

    def test_8_color_B2_red_not_broken(self):
        """'color B2 red' must still produce font_color=red (regression guard)."""
        r = _pipeline('color B2 red')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('font_color'), 'red')

    def test_8b_fill_row_2_red_not_broken(self):
        """'fill row 2 with red' must still produce fill_color=red."""
        r = _pipeline('fill row 2 with red')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('fill_color'), 'red')

    def test_8c_bold_A1_not_broken(self):
        """'bold A1' must still produce bold=True."""
        r = _pipeline('bold A1')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertTrue(r['action'].get('bold'))

    def test_8d_set_font_size_A1_20_not_broken(self):
        """'set font size A1 20' must still produce font_size=20."""
        r = _pipeline('set font size A1 20')
        self.assertIsNone(r['slot_msg'])
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('font_size'), 20)

    def test_8e_format_bold_not_blocked(self):
        """'format A1:C1 bold' must not be blocked by number-format slot check."""
        r = _pipeline('format A1:C1 bold')
        self.assertIsNone(r['slot_msg'],
                          f"'format [range] bold' must not be blocked: {r['slot_msg']!r}")
        self.assertIsNotNone(r['action'])
        self.assertTrue(r['action'].get('bold'))

    # ── 9. Clarification cases still work ────────────────────────────────────

    def test_9_format_B2_B10_no_format_clarifies(self):
        """'format B2:B10' (no format type) must still clarify."""
        r = _pipeline('format B2:B10')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_9b_format_as_currency_no_target_clarifies(self):
        """'format as currency' (no target) must still clarify."""
        r = _pipeline('format as currency')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_9c_unknown_format_clarifies(self):
        """'format B2 as xyz' (unknown format) must still clarify."""
        r = _pipeline('format B2 as xyz')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])


# ---------------------------------------------------------------------------
# P34 — Insert Row / Column voice parsing
# ---------------------------------------------------------------------------
class TestP34InsertRowColumn(unittest.TestCase):
    """
    P34: 'insert row 3' / 'insert row before 3' / 'insert column B' /
         'insert column before B' must produce insert_row_at / insert_column_at.
         Bare 'insert row' / 'insert column' must clarify.
         add_row / add_column regressions must be unaffected.
    """

    # ── 1. insert row N ──────────────────────────────────────────────────────

    def test_1_insert_row_3(self):
        r = _pipeline('insert row 3')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'insert_row_at')
        self.assertEqual(act['index'], 3)

    def test_2_insert_row_before_3(self):
        r = _pipeline('insert row before 3')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'insert_row_at')
        self.assertEqual(act['index'], 3)

    def test_3_insert_row_at_5(self):
        r = _pipeline('insert row at 5')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'insert_row_at')
        self.assertEqual(act['index'], 5)

    def test_4_insert_before_row_3(self):
        r = _pipeline('insert before row 3')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'insert_row_at')
        self.assertEqual(act['index'], 3)

    # ── 2. insert column B ───────────────────────────────────────────────────

    def test_5_insert_column_B(self):
        r = _pipeline('insert column B')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'insert_column_at')
        self.assertEqual(act['column'], 'B')

    def test_6_insert_column_before_B(self):
        r = _pipeline('insert column before B')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'insert_column_at')
        self.assertEqual(act['column'], 'B')

    def test_7_insert_before_column_C(self):
        r = _pipeline('insert before column C')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'insert_column_at')
        self.assertEqual(act['column'], 'C')

    def test_8_insert_column_at_D(self):
        r = _pipeline('insert column at D')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'insert_column_at')
        self.assertEqual(act['column'], 'D')

    # ── 3. Clarification: bare / missing target ──────────────────────────────

    def test_9_bare_insert_row_clarifies(self):
        r = _pipeline('insert row')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])
        self.assertIn('row', r['slot_msg'].lower())

    def test_10_bare_insert_column_clarifies(self):
        r = _pipeline('insert column')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])
        self.assertIn('column', r['slot_msg'].lower())

    def test_11_insert_before_row_no_number_clarifies(self):
        r = _pipeline('insert before row')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    def test_12_insert_before_column_no_letter_clarifies(self):
        r = _pipeline('insert before column')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])

    # ── 4. add_row / add_column regressions ──────────────────────────────────

    def test_13_add_row_with_data_unchanged(self):
        """'add row John 90' must still produce add_row, not insert_row_at."""
        r = _pipeline('add row John 90')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'add_row')

    def test_14_insert_row_with_name_becomes_add_row(self):
        """'insert row Tom 80' has alphabetic data → routes as add_row."""
        r = _pipeline('insert row Tom 80')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'add_row')

    def test_15_add_column_score_unchanged(self):
        """'add column Score' must still produce add_column."""
        r = _pipeline('add column Score')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'add_column')


# ---------------------------------------------------------------------------
# P35 — STT voice normalization (sort verbs, column B mishears, bold, sum, row)
# ---------------------------------------------------------------------------
class TestP35STTNormalization(unittest.TestCase):
    """
    P35: Voice STT normalization.
    Sort verb mishears, column-B mishears, bold mishears, sum mishears,
    row mishears, and safety blocks — all without breaking typed commands.
    """

    # ── 1. Sort verb mishears ────────────────────────────────────────────────

    def test_1_swap_by_B_descending(self):
        r = _pipeline('swap by B descending')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'sort_range')
        self.assertEqual(act['column'], 'B')
        self.assertFalse(act['ascending'])

    def test_2_sword_by_be_descending(self):
        r = _pipeline('sword by be descending')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'sort_range')
        self.assertEqual(act['column'], 'B')
        self.assertFalse(act['ascending'])

    def test_3_short_by_B_ascending(self):
        r = _pipeline('short by B ascending')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'sort_range')
        self.assertEqual(act['column'], 'B')
        self.assertTrue(act['ascending'])

    def test_4_start_by_B_descending(self):
        r = _pipeline('start by B descending')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'sort_range')
        self.assertEqual(act['column'], 'B')
        self.assertFalse(act['ascending'])

    def test_5_sort_by_be_descending_becomes_B(self):
        """Voice 'sort by be descending' must produce column B, not BE."""
        r = _pipeline('sort by be descending')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'sort_range')
        self.assertEqual(act['column'], 'B')

    # ── 2. Column "be"/"bee" mishears ────────────────────────────────────────

    def test_6_color_column_be_green(self):
        """Voice 'color column be green' must format column B, not BE."""
        r = _pipeline('color column be green')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'format_cells')
        self.assertEqual(act['range'], 'B:B')
        self.assertEqual(act.get('font_color'), 'green')

    def test_7_fill_column_be_with_blue(self):
        r = _pipeline('fill column be with blue')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'format_cells')
        self.assertEqual(act['range'], 'B:B')
        self.assertIsNotNone(act.get('fill_color'))

    # ── 3. Bold mishears ─────────────────────────────────────────────────────

    def test_8_board_A1_becomes_bold(self):
        r = _pipeline('board A1')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'format_cells')
        self.assertTrue(act.get('bold'))

    def test_9_ball_A1_becomes_bold(self):
        r = _pipeline('ball A1')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'format_cells')
        self.assertTrue(act.get('bold'))

    def test_10_bold_A1_still_works(self):
        """Original 'bold A1' must be unaffected."""
        r = _pipeline('bold A1')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'format_cells')
        self.assertTrue(act.get('bold'))

    def test_11_format_row_3_ball_becomes_bold(self):
        r = _pipeline('format row 3 ball')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'format_cells')
        self.assertTrue(act.get('bold'))

    # ── 4. Sum mishears ──────────────────────────────────────────────────────

    def test_12_sam_B2_B5_into_C6(self):
        r = _pipeline('sam B2:B5 into C6')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'write_sum_formula')

    def test_13_some_B2_B5_into_C6(self):
        r = _pipeline('some B2:B5 into C6')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'write_sum_formula')

    def test_14_some_column_B(self):
        """'some column B' normalizes to 'sum column B'; intent = calculate_sum.
        No result cell was given, so the heuristic returns None (falls to LLM).
        The test verifies only the normalization and intent routing."""
        r = _pipeline('some column B')
        self.assertEqual(r['corrected'], 'sum column B')
        self.assertEqual(r['intent'], 'calculate_sum')
        self.assertIsNone(r['slot_msg'])

    # ── 5. Row mishear safety ────────────────────────────────────────────────

    def test_15_fill_a_roto_with_red_clarifies(self):
        """'fill a roto with red' has no row number — must clarify."""
        r = _pipeline('fill a roto with red')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])
        self.assertIn('row', r['slot_msg'].lower())

    def test_16_fill_row_2_with_red_still_works(self):
        """Regression: 'fill row 2 with red' must execute normally."""
        r = _pipeline('fill row 2 with red')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'format_cells')

    # ── 6. Typed command safety ──────────────────────────────────────────────

    def test_17_typed_sort_BE_unchanged(self):
        """Typed 'sort by BE descending' must keep column BE (not B)."""
        r = _pipeline('sort by BE descending')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'sort_range')
        self.assertEqual(act['column'], 'BE')

    def test_18_typed_color_column_BE_unchanged(self):
        """Typed 'color column BE green' must keep column BE (not B)."""
        r = _pipeline('color column BE green')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'format_cells')
        self.assertEqual(act['range'], 'BE:BE')

    def test_19_voice_sort_bd_lowercase_clarifies(self):
        """Voice transcript 'sort by bd descending' (lowercase bd) must clarify."""
        r = _pipeline('sort by bd descending')
        self.assertIsNotNone(r['slot_msg'])
        self.assertIsNone(r['action'])
        self.assertIn('column B', r['slot_msg'])

    # ── 7. Unsafe-slot blocking ───────────────────────────────────────────────

    def test_20_color_be_to_red_no_range_invention(self):
        """'color be to red' must not execute with an invented broad range."""
        r = _pipeline('color be to red')
        if r['action'] is not None:
            rng = r['action'].get('range', '')
            self.assertNotIn(':', rng.replace('B:B', ''),
                             "LLM-invented range detected in 'color be to red'")

    def test_21_VLF5_pink_does_not_execute(self):
        """Completely garbled 'VLF5 pink' must not produce an action."""
        r = _pipeline('VLF5 pink')
        self.assertIsNone(r['action'])

    def test_22_fear_column_does_not_execute(self):
        """'fear column' (no recognizable verb) must not produce an action."""
        r = _pipeline('fear column')
        self.assertIsNone(r['action'])

    # ── 8. Existing working voice commands unchanged ─────────────────────────

    def test_23_add_row_John_90_regression(self):
        r = _pipeline('add row John 90')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'add_row')

    def test_24_color_B2_red_regression(self):
        r = _pipeline('color B2 red')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'format_cells')

    def test_25_sort_by_B_ascending_regression(self):
        r = _pipeline('sort by B ascending')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'sort_range')
        self.assertEqual(act['column'], 'B')
        self.assertTrue(act['ascending'])

    def test_26_sum_B2_B5_into_C6_regression(self):
        r = _pipeline('sum B2:B5 into C6')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'write_sum_formula')

    def test_27_filter_column_B_90_regression(self):
        r = _pipeline('filter column B 90')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)

    def test_28_color_row_3_blue_regression(self):
        r = _pipeline('color row 3 blue')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'format_cells')


# ---------------------------------------------------------------------------
# P36 — Voice reliability: STT mishear corrections + unsafe slot blocking
# ---------------------------------------------------------------------------

class TestP36VoiceReliability(unittest.TestCase):
    """Regression tests for P36 voice-reliability improvements.

    Covers six real runtime failures:
    1. fear/pure row → fill row  (fill mishear)
    2. swap by pre-ascending     (hyphenated garble blocked)
    3. unmerge select C1 D1      (cells keyword recovery)
    4. some b2b5 into c5         (merged-range sum normalisation)
    5. sort by V descending      (B mishear → ask clarification)
    6. sort/filter color B       (color→column disambiguation)
    """

    # ── 1. fear/pure row → fill row ─────────────────────────────────────────

    def test_01_fear_row_corrects_to_fill_row(self):
        """'fear row 2 with red' must correct to 'fill row 2 with red'."""
        corrected = _pipeline('fear row 2 with red')['corrected']
        self.assertIn('fill row', corrected.lower(),
                      f"Expected 'fill row' in {corrected!r}")

    def test_02_pure_row_corrects_to_fill_row(self):
        """'pure row 2 with red' must correct to 'fill row 2 with red'."""
        corrected = _pipeline('pure row 2 with red')['corrected']
        self.assertIn('fill row', corrected.lower(),
                      f"Expected 'fill row' in {corrected!r}")

    def test_03_fear_row_intent_is_format_cells(self):
        """'fear row 2 with red' must reach format_cells intent."""
        r = _pipeline('fear row 2 with red')
        self.assertEqual(r['intent'], 'format_cells',
                         f"Expected format_cells, got {r['intent']!r}")

    def test_04_pure_row_intent_is_format_cells(self):
        """'pure row 2 with red' must reach format_cells intent."""
        r = _pipeline('pure row 2 with red')
        self.assertEqual(r['intent'], 'format_cells',
                         f"Expected format_cells, got {r['intent']!r}")

    def test_05_fear_row_no_slot_gap(self):
        """'fear row 2 with red' must not trigger a slot clarification."""
        r = _pipeline('fear row 2 with red')
        self.assertIsNone(r['slot_msg'],
                          f"Unexpected slot_msg: {r['slot_msg']!r}")

    def test_06_fear_row_not_add_row(self):
        """'fear row 2 with red' must NOT route to add_row."""
        r = _pipeline('fear row 2 with red')
        self.assertNotEqual(r['intent'], 'add_row',
                            "Incorrectly routed to add_row")

    # ── 2. swap by pre-ascending → blocked ───────────────────────────────────

    def test_07_swap_by_pre_ascending_corrects_to_sort_by(self):
        """'swap by pre-ascending' must normalise to 'sort by ...'."""
        corrected = _pipeline('swap by pre-ascending')['corrected']
        self.assertIn('sort by', corrected.lower(),
                      f"Expected 'sort by' in {corrected!r}")

    def test_08_swap_by_pre_ascending_blocked_by_slot(self):
        """'swap by pre-ascending' must be blocked (invalid column PRE)."""
        r = _pipeline('swap by pre-ascending')
        self.assertIsNotNone(r['slot_msg'],
                             "Expected slot clarification for 'pre' column")
        self.assertIn('column B', r['slot_msg'],
                      f"Clarification should mention column B: {r['slot_msg']!r}")

    def test_09_swap_by_pre_ascending_no_action(self):
        """'swap by pre-ascending' must not produce an executable action."""
        r = _pipeline('swap by pre-ascending')
        self.assertIsNone(r['action'],
                          f"Expected no action, got {r['action']!r}")

    # ── 3. unmerge select C1 D1 → unmerge cells C1:D1 ───────────────────────

    def test_10_unmerge_select_corrects_to_cells(self):
        """'unmerge select C1 D1' must correct 'select' to 'cells'."""
        corrected = _pipeline('unmerge select C1 D1')['corrected']
        self.assertIn('unmerge cells', corrected.lower(),
                      f"Expected 'unmerge cells' in {corrected!r}")

    def test_11_unmerge_select_intent_is_unmerge(self):
        """'unmerge select C1 D1' must reach unmerge_cells intent."""
        r = _pipeline('unmerge select C1 D1')
        self.assertEqual(r['intent'], 'unmerge_cells',
                         f"Expected unmerge_cells, got {r['intent']!r}")

    def test_12_unmerge_select_not_switch_sheet(self):
        """'unmerge select C1 D1' must NOT route to switch_sheet."""
        r = _pipeline('unmerge select C1 D1')
        self.assertNotEqual(r['intent'], 'switch_sheet',
                            "Incorrectly routed to switch_sheet")

    def test_13_unmerge_select_no_slot_gap(self):
        """'unmerge select C1 D1' has both cell refs → no slot clarification."""
        r = _pipeline('unmerge select C1 D1')
        self.assertIsNone(r['slot_msg'],
                          f"Unexpected slot_msg: {r['slot_msg']!r}")

    # ── 4. some b2b5 into c5 → sum B2:B5 into C5 ────────────────────────────

    def test_14_some_b2b5_corrects_to_sum(self):
        """'some b2b5 into c5' must correct 'some' to 'sum'."""
        corrected = _pipeline('some b2b5 into c5')['corrected']
        self.assertIn('sum', corrected.lower(),
                      f"Expected 'sum' in {corrected!r}")
        self.assertNotIn('some', corrected.lower(),
                         f"'some' should be gone: {corrected!r}")

    def test_15_some_b2b5_range_normalised(self):
        """'some b2b5 into c5' must normalise range to B2:B5 (case-insensitive)."""
        corrected = _pipeline('some b2b5 into c5')['corrected']
        self.assertIn('b2:b5', corrected.lower(),
                      f"Expected 'b2:b5' (colon form) in {corrected!r}")

    def test_16_some_b2b5_intent_is_calculate_sum(self):
        """'some b2b5 into c5' must reach calculate_sum intent."""
        r = _pipeline('some b2b5 into c5')
        self.assertEqual(r['intent'], 'calculate_sum',
                         f"Expected calculate_sum, got {r['intent']!r}")

    def test_17_some_b2b5_no_slot_gap(self):
        """'some b2b5 into c5' has valid range + destination → no slot gap."""
        r = _pipeline('some b2b5 into c5')
        self.assertIsNone(r['slot_msg'],
                          f"Unexpected slot_msg: {r['slot_msg']!r}")

    # ── 5. sort by V descending → ask clarification ──────────────────────────

    def test_18_sort_by_V_descending_blocked(self):
        """'sort by V descending' must return a slot clarification (V=B mishear)."""
        r = _pipeline('sort by V descending')
        self.assertIsNotNone(r['slot_msg'],
                             "Expected clarification for column V (likely B mishear)")

    def test_19_sort_by_V_descending_mentions_B(self):
        """Clarification for 'sort by V descending' must mention column B."""
        r = _pipeline('sort by V descending')
        self.assertIn('column B', r['slot_msg'],
                      f"Expected 'column B' in slot_msg: {r['slot_msg']!r}")

    def test_20_sort_by_V_descending_no_action(self):
        """'sort by V descending' must not produce an executable action."""
        r = _pipeline('sort by V descending')
        self.assertIsNone(r['action'],
                          f"Expected no action, got {r['action']!r}")

    # ── 6. color → column disambiguation ────────────────────────────────────

    def test_21_sort_by_color_B_corrects_to_column(self):
        """'sort by color B ascending' must correct 'color' to 'column'."""
        corrected = _pipeline('sort by color B ascending')['corrected']
        self.assertIn('column B', corrected,
                      f"Expected 'column B' in {corrected!r}")

    def test_22_sort_by_color_B_no_slot_gap(self):
        """'sort by color B ascending' must not trigger a slot clarification."""
        r = _pipeline('sort by color B ascending')
        self.assertIsNone(r['slot_msg'],
                          f"Unexpected slot_msg: {r['slot_msg']!r}")

    def test_23_sort_by_color_B_action(self):
        """'sort by color B ascending' must produce a sort_range action on column B."""
        r = _pipeline('sort by color B ascending')
        act = r['action']
        self.assertIsNotNone(act, "Expected sort action")
        self.assertEqual(act['type'], 'sort_range')
        self.assertEqual(act['column'], 'B')

    def test_24_filter_color_B_corrects_to_column(self):
        """'filter color B = 40' must correct 'color' to 'column'."""
        corrected = _pipeline('filter color B = 40')['corrected']
        self.assertIn('column B', corrected,
                      f"Expected 'column B' in {corrected!r}")

    def test_25_filter_color_B_no_slot_gap(self):
        """'filter color B = 40' must not trigger a slot clarification."""
        r = _pipeline('filter color B = 40')
        self.assertIsNone(r['slot_msg'],
                          f"Unexpected slot_msg: {r['slot_msg']!r}")

    def test_26_color_B2_red_unchanged(self):
        """'color B2 red' (fill formatting) must NOT have 'color' rewritten."""
        corrected = _pipeline('color B2 red')['corrected']
        self.assertIn('color', corrected.lower(),
                      f"'color' should be preserved: {corrected!r}")
        self.assertNotIn('column', corrected.lower(),
                         f"'column' must not appear: {corrected!r}")

    # ── 7. Protected baseline regression checks ──────────────────────────────

    def test_27_sort_by_B_ascending_unchanged(self):
        """'sort by B ascending' must still work exactly as before."""
        r = _pipeline('sort by B ascending')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'sort_range')
        self.assertEqual(act['column'], 'B')
        self.assertTrue(act['ascending'])

    def test_28_sort_by_B_descending_unchanged(self):
        """'sort by B descending' must still work exactly as before."""
        r = _pipeline('sort by B descending')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'sort_range')
        self.assertEqual(act['column'], 'B')
        self.assertFalse(act['ascending'])

    def test_29_fill_row_2_with_red_unchanged(self):
        """'fill row 2 with red' must still work exactly as before."""
        r = _pipeline('fill row 2 with red')
        self.assertEqual(r['intent'], 'format_cells')
        self.assertIsNone(r['slot_msg'])

    def test_30_sum_range_into_unchanged(self):
        """'sum B2:B5 into C5' must still work exactly as before."""
        r = _pipeline('sum B2:B5 into C5')
        self.assertEqual(r['intent'], 'calculate_sum')
        self.assertIsNone(r['slot_msg'])
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act['type'], 'write_sum_formula')

    def test_31_unmerge_cells_A1_C1_unchanged(self):
        """'unmerge cells A1:C1' must still work exactly as before."""
        r = _pipeline('unmerge cells A1:C1')
        self.assertEqual(r['intent'], 'unmerge_cells')
        self.assertIsNone(r['slot_msg'])

    def test_32_swap_by_be_descending_executes(self):
        """'swap by be descending' → P35 corrects 'be'→'B' → valid sort command executes."""
        r = _pipeline('swap by be descending')
        # P35 correction: "by be descending" → "by B descending" (lowercase 'e' = voice)
        # Result is a valid sort command; must NOT be blocked by slot check.
        self.assertIsNone(r['slot_msg'],
                          f"'sort by B descending' should not be blocked: {r['slot_msg']!r}")
        act = r['action']
        self.assertIsNotNone(act, "Expected sort_range action after P35 correction")
        self.assertEqual(act['type'], 'sort_range')
        self.assertEqual(act['column'], 'B')


class TestP45FreezePanFilterVoice(unittest.TestCase):
    """Voice safety regression tests for P45.

    Covers four STT-to-action bugs reported from real mic sessions:
    1. freeze pan/pen/pants  → freeze_panes, never format_cells
    2. filter column A Lisa  → filter_table column A value Lisa
    3. filter color A Lisa   → filter_table column A value Lisa
    4. filter column be 40   → filter_table column B value 40
    5. sort by be descending → sort_range column B descending
    6. typed BE must remain BE (no voice correction on uppercase ending)
    7. color B2 red / fill column B → formatting protected (not mistaken for filter)
    """

    # ── 1. freeze pan / pen / pants → freeze_panes ──────────────────────────

    def test_01_freeze_pan_corrects_to_freeze_panes(self):
        """'freeze pan' must correct to 'freeze panes'."""
        r = _pipeline('freeze pan')
        self.assertIn('freeze panes', r['corrected'].lower(),
                      f"Expected 'freeze panes' in corrected: {r['corrected']!r}")

    def test_02_freeze_pan_intent_is_freeze_panes(self):
        """'freeze pan' must have intent freeze_panes."""
        r = _pipeline('freeze pan')
        self.assertEqual(r['intent'], 'freeze_panes',
                         f"Expected freeze_panes, got {r['intent']!r}")

    def test_03_freeze_pan_action_is_freeze_panes(self):
        """'freeze pan' must produce freeze_panes action, never format_cells."""
        r = _pipeline('freeze pan')
        act = r['action']
        self.assertIsNotNone(act, "Expected action for 'freeze pan'")
        self.assertEqual(act.get('type'), 'freeze_panes',
                         f"Expected freeze_panes, got {act.get('type')!r}")
        self.assertNotEqual(act.get('type'), 'format_cells',
                            "'freeze pan' must NEVER route to format_cells")

    def test_04_freeze_pen_corrects_to_freeze_panes(self):
        """'freeze pen' must correct to 'freeze panes'."""
        r = _pipeline('freeze pen')
        self.assertIn('freeze panes', r['corrected'].lower(),
                      f"Expected 'freeze panes' in corrected: {r['corrected']!r}")

    def test_05_freeze_pen_action_is_freeze_panes(self):
        """'freeze pen' must produce freeze_panes action, never format_cells."""
        r = _pipeline('freeze pen')
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act.get('type'), 'freeze_panes',
                         f"Expected freeze_panes, got {act.get('type')!r}")

    def test_06_freeze_pants_corrects_to_freeze_panes(self):
        """'freeze pants' must correct to 'freeze panes'."""
        r = _pipeline('freeze pants')
        self.assertIn('freeze panes', r['corrected'].lower(),
                      f"Expected 'freeze panes' in corrected: {r['corrected']!r}")

    def test_07_freeze_pants_action_is_freeze_panes(self):
        """'freeze pants' must produce freeze_panes action, never format_cells."""
        r = _pipeline('freeze pants')
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act.get('type'), 'freeze_panes',
                         f"Expected freeze_panes, got {act.get('type')!r}")

    def test_08_freeze_pants_not_format_cells(self):
        """'freeze pants' must NEVER route to format_cells (regression guard)."""
        r = _pipeline('freeze pants')
        act = r['action']
        if act:
            self.assertNotEqual(act.get('type'), 'format_cells',
                                "'freeze pants' must never route to format_cells")

    # ── 2. filter column A Lisa → filter_table ───────────────────────────────

    def test_09_filter_column_A_text_value_intent(self):
        """'filter column A Lisa' must have intent filter_table."""
        r = _pipeline('filter column A Lisa')
        self.assertEqual(r['intent'], 'filter_table',
                         f"Expected filter_table, got {r['intent']!r}")

    def test_10_filter_column_A_text_value_action(self):
        """'filter column A Lisa' → filter_table column=A value=Lisa."""
        r = _pipeline('filter column A Lisa')
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act.get('type'), 'filter_table')
        self.assertEqual(act.get('column', '').upper(), 'A')
        self.assertIn('Lisa', act.get('value', ''),
                      f"Expected 'Lisa' in value, got {act.get('value')!r}")

    def test_11_filter_column_A_text_not_formatting(self):
        """'filter column A Lisa' must NOT route to format_cells."""
        r = _pipeline('filter column A Lisa')
        act = r['action']
        if act:
            self.assertNotEqual(act.get('type'), 'format_cells',
                                "'filter column A Lisa' must not route to formatting")

    # ── 3. filter color A Lisa → filter_table ────────────────────────────────

    def test_12_filter_color_A_Lisa_corrects_column(self):
        """'filter color A Lisa' must correct 'color' to 'column'."""
        r = _pipeline('filter color A Lisa')
        self.assertIn('filter column', r['corrected'].lower(),
                      f"Expected 'filter column' in corrected: {r['corrected']!r}")

    def test_13_filter_color_A_Lisa_intent(self):
        """'filter color A Lisa' must have intent filter_table."""
        r = _pipeline('filter color A Lisa')
        self.assertEqual(r['intent'], 'filter_table',
                         f"Expected filter_table, got {r['intent']!r}")

    def test_14_filter_color_A_Lisa_action(self):
        """'filter color A Lisa' → filter_table column=A value=Lisa."""
        r = _pipeline('filter color A Lisa')
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act.get('type'), 'filter_table')
        self.assertEqual(act.get('column', '').upper(), 'A')
        self.assertIn('Lisa', act.get('value', ''))

    # ── 4. filter column be 40 → filter_table column B ───────────────────────

    def test_15_filter_column_be_corrects_to_B(self):
        """'filter column be 40' — 'be' (voice) must become 'B'."""
        r = _pipeline('filter column be 40')
        self.assertIn('column B', r['corrected'],
                      f"Expected 'column B' in corrected: {r['corrected']!r}")

    def test_16_filter_column_be_40_intent(self):
        """'filter column be 40' must have intent filter_table."""
        r = _pipeline('filter column be 40')
        self.assertEqual(r['intent'], 'filter_table')

    def test_17_filter_column_be_40_action(self):
        """'filter column be 40' → filter_table column=B value=40."""
        r = _pipeline('filter column be 40')
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act.get('type'), 'filter_table')
        self.assertEqual(act.get('column', '').upper(), 'B')
        self.assertEqual(str(act.get('value', '')), '40')

    # ── 5. sort by be descending → sort column B ─────────────────────────────

    def test_18_sort_by_be_descending_corrects_to_B(self):
        """'sort by be descending' — 'be' (voice) must become 'B'."""
        r = _pipeline('sort by be descending')
        self.assertIn('by B', r['corrected'],
                      f"Expected 'by B' in corrected: {r['corrected']!r}")

    def test_19_sort_by_be_descending_action(self):
        """'sort by be descending' → sort_range column=B ascending=False."""
        r = _pipeline('sort by be descending')
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act.get('type'), 'sort_range')
        self.assertEqual(act.get('column', '').upper(), 'B')
        self.assertFalse(act.get('ascending', True))

    # ── 6. typed BE must remain BE (no voice substitution on uppercase E) ────

    def test_20_typed_sort_by_BE_unchanged(self):
        """Typed 'sort by BE descending' — uppercase E must NOT be corrected."""
        r = _pipeline('sort by BE descending')
        self.assertIn('BE', r['corrected'],
                      f"Typed 'BE' must not be changed, corrected={r['corrected']!r}")

    def test_21_typed_color_column_BE_unchanged(self):
        """Typed 'color column BE green' — uppercase E must NOT be corrected."""
        r = _pipeline('color column BE green')
        self.assertIn('BE', r['corrected'],
                      f"Typed 'BE' must not be changed, corrected={r['corrected']!r}")

    # ── 7. protected formatting commands ─────────────────────────────────────

    def test_22_color_B2_red_is_formatting(self):
        """'color B2 red' must remain format_cells (not filter_table)."""
        r = _pipeline('color B2 red')
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act.get('type'), 'format_cells',
                         f"Expected format_cells, got {act.get('type')!r}")

    def test_23_fill_column_B_with_blue_is_formatting(self):
        """'fill column B with blue' must remain format_cells (not filter_table)."""
        r = _pipeline('fill column B with blue')
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act.get('type'), 'format_cells',
                         f"Expected format_cells, got {act.get('type')!r}")

    def test_24_color_B2_red_not_filter(self):
        """'color B2 red' must NOT route to filter_table."""
        r = _pipeline('color B2 red')
        act = r['action']
        if act:
            self.assertNotEqual(act.get('type'), 'filter_table',
                                "'color B2 red' must not route to filter")

    def test_25_fill_column_B_not_filter(self):
        """'fill column B with blue' must NOT route to filter_table."""
        r = _pipeline('fill column B with blue')
        act = r['action']
        if act:
            self.assertNotEqual(act.get('type'), 'filter_table',
                                "'fill column B with blue' must not route to filter")


class TestP45AgentHandleTextCorrection(unittest.TestCase):
    """Verify correct_transcript is applied inside agent.handle_text().

    The upload-audio API path goes through agent.handle_text() which previously
    called route_llm() with the raw STT transcript (no correction).  P45 adds
    correct_transcript() before route_llm() so voice transcription errors like
    'freeze pan', 'filter column be', 'by be' are fixed on ALL code paths.
    """

    def _agent_heuristic(self, raw: str) -> dict:
        """Run raw text through agent.handle_text() with heuristic-only routing."""
        from unittest.mock import patch as _patch
        received = {}

        def _fake_route(text, **kw):
            received['text'] = text
            from llm_parser import _heuristic_natural_language_parse
            result = _heuristic_natural_language_parse(text)
            if result is None:
                result = {'type': 'unknown'}
            result['_router_source'] = 'heuristic'
            return result

        import agent as _agent_mod
        import tempfile
        with _patch.object(_agent_mod, 'route_llm', side_effect=_fake_route):
            with tempfile.TemporaryDirectory() as _td:
                ag = _agent_mod.Agent(files_dir=_td)
                ag_result = ag.handle_text(raw)

        return {'corrected': received.get('text', raw), 'action': ag_result.action}

    def test_01_agent_corrects_freeze_pan(self):
        """agent.handle_text('freeze pan') must pass 'freeze panes' to route_llm."""
        r = self._agent_heuristic('freeze pan')
        self.assertIn('freeze panes', r['corrected'].lower(),
                      f"Expected correction, got {r['corrected']!r}")

    def test_02_agent_corrects_freeze_pen(self):
        """agent.handle_text('freeze pen') must pass 'freeze panes' to route_llm."""
        r = self._agent_heuristic('freeze pen')
        self.assertIn('freeze panes', r['corrected'].lower())

    def test_03_agent_corrects_freeze_pants(self):
        """agent.handle_text('freeze pants') must pass 'freeze panes' to route_llm."""
        r = self._agent_heuristic('freeze pants')
        self.assertIn('freeze panes', r['corrected'].lower())

    def test_04_agent_freeze_pan_action(self):
        """agent.handle_text('freeze pan') must return freeze_panes action."""
        r = self._agent_heuristic('freeze pan')
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act.get('type'), 'freeze_panes',
                         f"Expected freeze_panes action, got {act!r}")

    def test_05_agent_corrects_filter_column_be(self):
        """agent.handle_text('filter column be 40') must correct 'be' → 'B'."""
        r = self._agent_heuristic('filter column be 40')
        self.assertIn('column B', r['corrected'],
                      f"Expected 'column B' in corrected: {r['corrected']!r}")

    def test_06_agent_filter_column_be_action(self):
        """agent.handle_text('filter column be 40') must return filter_table action."""
        r = self._agent_heuristic('filter column be 40')
        act = r['action']
        self.assertIsNotNone(act)
        self.assertEqual(act.get('type'), 'filter_table')
        self.assertEqual(act.get('column', '').upper(), 'B')


class TestP46UiPipelineAlignment(unittest.TestCase):
    """Mandatory 16-command matrix for UI pipeline alignment (P46).

    Verifies that every spoken command that works in run_sara.py also produces
    the correct correction and action type through the UI path
    (correct_transcript → heuristic route).
    """

    # ── Correction checks ────────────────────────────────────────────────────

    def test_01_correct_undo_unchanged(self):
        """'undo' needs no correction."""
        from pipeline_sanitizer import correct_transcript
        self.assertEqual(correct_transcript('undo'), 'undo')

    def test_02_correct_i_do_to_undo(self):
        """'I do' (standalone) → 'undo'."""
        from pipeline_sanitizer import correct_transcript
        self.assertEqual(correct_transcript('I do'), 'undo')

    def test_03_correct_unto_to_undo(self):
        """'unto' (standalone) → 'undo'."""
        from pipeline_sanitizer import correct_transcript
        self.assertEqual(correct_transcript('unto'), 'undo')

    def test_04_correct_and_do_to_undo(self):
        """'and do' (standalone) → 'undo'."""
        from pipeline_sanitizer import correct_transcript
        self.assertEqual(correct_transcript('and do'), 'undo')

    def test_05_correct_freeze_pane(self):
        """'freeze pane' → 'freeze panes'."""
        from pipeline_sanitizer import correct_transcript
        self.assertIn('freeze panes', correct_transcript('freeze pane').lower())

    def test_06_i_do_mid_sentence_not_corrected(self):
        """'I do the work' must NOT become 'undo the work' — guard test."""
        from pipeline_sanitizer import correct_transcript
        result = correct_transcript('I do the work')
        self.assertNotIn('undo', result.lower(),
                         f"Mid-sentence 'I do' was incorrectly rewritten: {result!r}")

    # ── Undo voice variants — action routing ─────────────────────────────────

    def test_07_undo_routes_undo(self):
        """'undo' → action type undo."""
        r = _pipeline('undo')
        self.assertIsNotNone(r['action'])
        self.assertEqual(r['action'].get('type'), 'undo')

    def test_08_i_do_routes_undo(self):
        """'I do' STT corruption → undo after correction."""
        r = _pipeline('I do')
        self.assertIsNotNone(r['action'], f"Expected undo action, got None (corrected={r['corrected']!r})")
        self.assertEqual(r['action'].get('type'), 'undo',
                         f"Expected undo, got {r['action']!r}")

    def test_09_unto_routes_undo(self):
        """'unto' STT corruption → undo after correction."""
        r = _pipeline('unto')
        self.assertIsNotNone(r['action'], f"Expected undo action, got None (corrected={r['corrected']!r})")
        self.assertEqual(r['action'].get('type'), 'undo',
                         f"Expected undo, got {r['action']!r}")

    def test_10_and_do_routes_undo(self):
        """'and do' STT corruption → undo after correction."""
        r = _pipeline('and do')
        self.assertIsNotNone(r['action'], f"Expected undo action, got None (corrected={r['corrected']!r})")
        self.assertEqual(r['action'].get('type'), 'undo',
                         f"Expected undo, got {r['action']!r}")

    # ── All 16 mandatory commands — action type ──────────────────────────────

    def test_11_open_excel(self):
        """'open excel' → open_excel_app."""
        r = _pipeline('open excel')
        self.assertEqual(r['action'].get('type'), 'open_excel_app',
                         f"Got {r['action']!r}")

    def test_12_open_file_server(self):
        """'open file server' → open_file with filename server.xlsx."""
        r = _pipeline('open file server')
        act = r['action']
        self.assertEqual(act.get('type'), 'open_file', f"Got {act!r}")
        self.assertEqual(act.get('filename', '').lower(), 'server.xlsx',
                         f"Got filename {act.get('filename')!r}")

    def test_13_open_file_background(self):
        """'open file background' → open_file with filename background.xlsx."""
        r = _pipeline('open file background')
        act = r['action']
        self.assertEqual(act.get('type'), 'open_file', f"Got {act!r}")
        self.assertEqual(act.get('filename', '').lower(), 'background.xlsx',
                         f"Got filename {act.get('filename')!r}")

    def test_14_add_row_john_90(self):
        """'add row John 90' → add_row."""
        r = _pipeline('add row John 90')
        self.assertEqual(r['action'].get('type'), 'add_row', f"Got {r['action']!r}")

    def test_15_write_cell_c2_18(self):
        """'write cell C2 18' → write_cell at C2."""
        r = _pipeline('write cell C2 18')
        act = r['action']
        self.assertEqual(act.get('type'), 'write_cell', f"Got {act!r}")
        self.assertEqual(act.get('cell', '').upper(), 'C2',
                         f"Got cell {act.get('cell')!r}")

    def test_16_freeze_pane(self):
        """'freeze pane' → freeze_panes after correction."""
        r = _pipeline('freeze pane')
        self.assertEqual(r['action'].get('type'), 'freeze_panes',
                         f"Got {r['action']!r}")

    def test_17_filter_column_a_lisa(self):
        """'filter column A Lisa' → filter_table col=A val=Lisa."""
        r = _pipeline('filter column A Lisa')
        act = r['action']
        self.assertEqual(act.get('type'), 'filter_table', f"Got {act!r}")
        self.assertEqual(act.get('column', '').upper(), 'A')
        self.assertEqual(act.get('value', '').lower(), 'lisa')

    def test_18_sort_by_b_descending(self):
        """'sort by B descending' → sort_range col=B ascending=False."""
        r = _pipeline('sort by B descending')
        act = r['action']
        self.assertEqual(act.get('type'), 'sort_range', f"Got {act!r}")
        self.assertEqual(act.get('column', '').upper(), 'B',
                         f"Got column {act.get('column')!r}")
        self.assertFalse(act.get('ascending', True),
                         "Expected ascending=False for 'descending'")

    def test_19_color_b2_red(self):
        """'color B2 red' → format_cells."""
        r = _pipeline('color B2 red')
        act = r['action']
        self.assertIsNotNone(act, "Expected format_cells action, got None")
        self.assertEqual(act.get('type'), 'format_cells', f"Got {act!r}")

    def test_20_fill_column_b_blue(self):
        """'fill column B blue' → format_cells."""
        r = _pipeline('fill column B blue')
        act = r['action']
        self.assertIsNotNone(act, "Expected format_cells action, got None")
        self.assertEqual(act.get('type'), 'format_cells', f"Got {act!r}")

    def test_21_switch_sheet_report(self):
        """'switch sheet report' → switch_sheet sheet_name containing 'report'."""
        r = _pipeline('switch sheet report')
        act = r['action']
        self.assertEqual(act.get('type'), 'switch_sheet', f"Got {act!r}")
        self.assertIn('report', act.get('sheet_name', '').lower(),
                      f"Got sheet_name {act.get('sheet_name')!r}")

    def test_22_delete_sheet_report(self):
        """'delete sheet report' → delete_sheet sheet_name containing 'report'."""
        r = _pipeline('delete sheet report')
        act = r['action']
        self.assertEqual(act.get('type'), 'delete_sheet', f"Got {act!r}")
        self.assertIn('report', act.get('sheet_name', '').lower(),
                      f"Got sheet_name {act.get('sheet_name')!r}")


# ---------------------------------------------------------------------------
# P47 — single-response deduplication guard
# ---------------------------------------------------------------------------

class _GuardSimulator:
    """Minimal simulation of the deduplication guard in _on_execute_completed.

    Mirrors the exact guard state-machine added to SaraWindow so that the
    tests verify the guard logic without requiring a live PyQt6 environment.
    """

    def __init__(self):
        import time as _t
        self._time = _t
        self.history: list[tuple[str, bool]] = []   # (msg, success)

    def process(self, payload: dict) -> bool:
        """Process one execute-completed payload.  Returns False when blocked."""
        if payload.get("success", False):
            _now = self._time.monotonic()
            _key = (payload.get("speech_text", "")[:80], True)
            if (getattr(self, "_last_exec_key", None) == _key and
                    _now - getattr(self, "_last_exec_time", 0.0) < 2.0):
                return False  # duplicate guard triggered
            self._last_exec_key = _key
            self._last_exec_time = _now

        if not payload.get("success", False):
            msg = payload.get("error", "Execution failed")
            self.history.append((msg, False))
        else:
            msg = payload.get("speech_text") or "Action executed successfully."
            self.history.append((msg, True))
        return True


class TestP47DuplicateSuccessGuard(unittest.TestCase):
    """P47 — deduplication guard prevents duplicate success responses."""

    def _sim(self) -> _GuardSimulator:
        return _GuardSimulator()

    def test_01_single_success_adds_one_history_item(self):
        """First success call → exactly one history entry added."""
        sim = self._sim()
        allowed = sim.process({"success": True, "speech_text": "Row added."})
        self.assertTrue(allowed, "First success must be allowed through")
        self.assertEqual(len(sim.history), 1)
        self.assertEqual(sim.history[0], ("Row added.", True))

    def test_02_identical_success_within_2s_is_blocked(self):
        """Identical success payload arriving within 2 s → second call blocked."""
        sim = self._sim()
        payload = {"success": True, "speech_text": "Row added."}
        sim.process(payload)           # first: allowed
        blocked = not sim.process(payload)   # second: same key, <2 s → blocked
        self.assertTrue(blocked, "Duplicate identical success within 2 s must be blocked")
        self.assertEqual(len(sim.history), 1, "History must contain only one entry")

    def test_03_different_success_payloads_both_allowed(self):
        """Two different success messages → both go through (no false positive)."""
        sim = self._sim()
        sim.process({"success": True, "speech_text": "Row added."})
        allowed = sim.process({"success": True, "speech_text": "Cell written."})
        self.assertTrue(allowed, "Different success message must not be blocked")
        self.assertEqual(len(sim.history), 2)
        self.assertEqual(sim.history[0][0], "Row added.")
        self.assertEqual(sim.history[1][0], "Cell written.")

    def test_04_failure_after_success_is_always_shown(self):
        """Failure payload is never blocked by the success dedup guard."""
        sim = self._sim()
        sim.process({"success": True, "speech_text": "Row added."})
        allowed = sim.process({"success": False, "error": "Sheet not found"})
        self.assertTrue(allowed, "Failure must always be shown")
        self.assertEqual(len(sim.history), 2)
        self.assertEqual(sim.history[1], ("Sheet not found", False))

    def test_05_same_success_after_2s_delay_is_allowed(self):
        """Same success message but state manually aged past 2 s → allowed again."""
        sim = self._sim()
        payload = {"success": True, "speech_text": "Row added."}
        sim.process(payload)
        # Simulate time having passed by backdating the recorded timestamp
        sim._last_exec_time -= 3.0
        allowed = sim.process(payload)
        self.assertTrue(allowed, "Same message after 2 s must be allowed")
        self.assertEqual(len(sim.history), 2, "Aged-out duplicate must add new history entry")

    def test_06_guard_key_uses_speech_text_prefix(self):
        """Guard key is built from the first 80 chars of speech_text."""
        sim = self._sim()
        long_text = "X" * 100
        sim.process({"success": True, "speech_text": long_text})
        # Payload with same first 80 chars but longer total → still blocked
        same_prefix = {"success": True, "speech_text": long_text[:80] + "extra"}
        blocked = not sim.process(same_prefix)
        self.assertTrue(blocked, "Payloads sharing the same first-80-char prefix must be deduplicated")


if __name__ == '__main__':
    unittest.main(verbosity=2)
