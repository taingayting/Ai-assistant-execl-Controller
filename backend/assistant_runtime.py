"""Continuous runtime loop for the Sara voice assistant.

This module connects the existing subsystems into a single assistant loop:

    wake word
    -> show floating UI
    -> record speech
    -> transcribe speech
    -> route through the LLM router
    -> execute with the agent
    -> speak result
    -> return to idle

The runtime is intentionally UI-agnostic. If a UI object is passed in, the
runtime will update common fields when they exist. This works well with the
existing desktop windows without forcing a direct PyQt dependency here.
"""
from __future__ import annotations

import itertools
import logging
import os
import re
import sys
import threading
import time
import uuid


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from agent import Agent, AgentResult
from llm_router import route_llm, ConversationBuffer
from metrics import metrics as _metrics
from pipeline_sanitizer import (
    sanitize_speech,
    sanitize_memory_value,
    correct_transcript,
    detect_command_intent,
    is_low_confidence_transcript,
    check_command_slots,
    describe_low_confidence,
    is_suspicious_transcript,
)
import excel_controller as _xl_ctrl
from speech_recognition import listen_once
import tts_engine
from wakeword.wake_detector import WakeDetector


log = logging.getLogger('assistant_runtime')

FILES_DIR = os.path.join(CURRENT_DIR, 'files')
DEFAULT_LANGUAGE = os.environ.get('SARA_LANGUAGE', 'en-US')
WAKE_COOLDOWN_SECONDS = float(os.environ.get('SARA_WAKE_COOLDOWN', '2.0'))

# Phrases that end the conversation loop and return to idle wake-word mode.
EXIT_PHRASES = frozenset({
    'stop', 'exit', 'quit', 'done',
    'thank you', 'thanks', 'thank you sara',
    'goodbye', 'bye', 'bye bye', 'bye sara',
    "that's all", 'never mind',
})
MAX_CONSECUTIVE_ERRORS = 3
MAX_CLARIFICATION_ATTEMPTS = 1  # ask once, then return to idle

# Words that signal genuine command intent.  Used together with
# detect_command_intent() to block ambiguous transcripts like
# "Andrew and the 10" or "why does he want" before they reach the LLM.
# The check requires BOTH conditions to fail before rejecting:
#   (1) detect_command_intent returns None  — no known phonetic command pattern
#   (2) no word from this set present       — no command-indicating verb at all
# Either condition alone passing means the transcript goes through.
_COMMAND_VERBS: frozenset[str] = frozenset({
    'add', 'create', 'delete', 'remove', 'write', 'read', 'save', 'open',
    'close', 'switch', 'rename', 'calculate', 'calc', 'format', 'sort',
    'find', 'freeze', 'unfreeze', 'merge', 'unmerge', 'undo', 'insert',
    'filter', 'clear', 'list', 'show', 'set', 'make', 'build', 'replace',
    'search', 'sum', 'average', 'go',
    'center', 'align', 'color', 'colour', 'increase', 'decrease',
})

# How long to wait before Sara speaks a "thinking" audio hint during a slow
# LLM call (Ollama cold-start can take 5–30 s).  Set to 0 to disable.
THINKING_HINT_DELAY = float(os.environ.get('SARA_THINKING_HINT_DELAY', '1.8'))

# Silence gap inserted between TTS finishing and the microphone opening.
# Prevents the tail of Sara's spoken response from being captured by the mic
# and sent to STT as a fake command (laptop speaker echo).
# pyttsx3 runAndWait() returns when bytes are flushed to the audio driver,
# not when the speaker stops vibrating — this gap covers the residual echo.
# Tune via SARA_POST_TTS_DELAY in .env (seconds, float).  Set to 0 to disable.
POST_TTS_DELAY = float(os.environ.get('SARA_POST_TTS_DELAY', '0.4'))

# Greeting lines cycled on each wake-word trigger.
# Kept short so TTS finishes quickly and Sara is ready to listen.
_GREETINGS = itertools.cycle([
    "Yes?",
    "I'm listening.",
    "How can I help?",
    "Go ahead.",
])

# Thinking hints spoken when the LLM call is taking longer than expected.
_THINKING_HINTS = itertools.cycle([
    "Give me a moment.",
    "Just a second.",
    "Working on it.",
    "Let me think.",
])

# High-stakes intents that must never silently mutate into a different action
# type due to STT noise or LLM misinterpretation.  If the pre-route intent is
# one of these keys and the LLM returns one of the listed forbidden action
# types, the action is rejected and the user is asked to rephrase.
_PROTECTED_INTENTS: dict[str, frozenset[str]] = {
    'rename_file':  frozenset({'write_cell', 'write_formula', 'add_row'}),
    'rename_sheet': frozenset({'write_cell', 'switch_sheet'}),
    'delete_sheet': frozenset({'delete_row', 'delete_cell'}),
    'merge_cells':   frozenset({'write_cell', 'write_formula', 'add_row', 'format_cells'}),
    'unmerge_cells': frozenset({'open_file', 'open_excel_app', 'write_cell',
                                'write_formula', 'add_row', 'format_cells'}),
}

# Action types that the LLM must NOT execute when no deterministic intent was
# matched before routing.  If Ollama/Groq returns one of these types but
# detect_command_intent() returned None, the action is blocked and the user
# is asked to repeat more clearly.  This prevents noisy STT transcripts like
# "DELETE HEAT HEAT" from executing destructive actions via LLM hallucination.
# Heuristic-sourced actions (_router_source='heuristic') are always allowed
# even when pre_intent is None — e.g. "merge header" has no intent pattern but
# is correctly parsed by the deterministic heuristic regex.
_LLM_SAFETY_BLOCKED: frozenset[str] = frozenset({
    'delete_row', 'delete_column', 'delete_cell', 'delete_range', 'delete_sheet',
    'open_file', 'rename_file',
    'sort_range', 'filter_table',
    'merge_cells', 'unmerge_cells',
})

# Deterministic intent → action mapping.  These commands need no LLM
# interpretation; bypassing Ollama/Groq makes them instant and reliable.
_DETERMINISTIC_ACTIONS: dict[str, dict] = {
    'undo':           {'type': 'undo'},
    'save':           {'type': 'save'},
    'open_excel':     {'type': 'open_excel_app'},
    'close_excel':    {'type': 'close_excel_app'},
    'freeze_panes':   {'type': 'freeze_panes', 'cell': 'A2'},
    'unfreeze_panes': {'type': 'unfreeze_panes'},
}


def _is_safe_add_row_values(values: list) -> bool:
    """Return True only when values look like exactly [name, number].

    In voice mode, add_row must have one name-like alpha token and one numeric
    token.  Rejects garbled payloads such as:
      ['one.']            — no number (e.g. "Insert row one")
      ['tool.']           — no number (e.g. "Insert row tool")
      ['eight', 'nine', 10] — too many tokens (e.g. "at row eight nine 10")
    """
    if not isinstance(values, list) or len(values) != 2:
        return False
    str_vals = [str(v).strip().rstrip('.').strip() for v in values]
    _alpha = re.compile(r'^[A-Za-z]{2,}$')
    _num   = re.compile(r'^\d+(?:\.\d+)?$')
    return any(_alpha.match(s) for s in str_vals) and any(_num.match(s) for s in str_vals)


class SaraRuntime:
    """Run Sara as a continuous voice assistant.

    Parameters:
        files_dir: Directory for Excel files used by the agent.
        ui: Optional UI object. When present, common text/status widgets are
            updated if they exist.
        agent: Optional preconfigured Agent instance.
        wake_detector: Optional preconfigured WakeDetector instance.
        language: Language passed to speech recognition.
        hide_ui_when_idle: Hide the UI after every command cycle.
        cooldown_seconds: Minimum gap between wake detections.
    """

    def __init__(
        self,
        *,
        files_dir: str | None = None,
        ui: object | None = None,
        agent: Agent | None = None,
        wake_detector: WakeDetector | None = None,
        language: str = DEFAULT_LANGUAGE,
        hide_ui_when_idle: bool = True,
        cooldown_seconds: float = WAKE_COOLDOWN_SECONDS,
    ):
        self.files_dir = files_dir or FILES_DIR
        os.makedirs(self.files_dir, exist_ok=True)

        self.ui = ui
        self.agent = agent or Agent(self.files_dir)
        self.wake_detector = wake_detector or WakeDetector()
        self.language = language
        self.hide_ui_when_idle = hide_ui_when_idle
        self.cooldown_seconds = max(0.0, cooldown_seconds)

        self._running = False
        self._stop_event = threading.Event()
        self._handling_wake = threading.Lock()
        # Session and TTS timing (set per-conversation)
        self._current_session_id: str | None = None
        self._tts_start: float = 0.0
        # Confirmation layer: action awaiting voice yes/no (cleared after each cycle)
        self._pending_confirmation_action: dict | None = None
        # Improvement 2: per-session conversation buffer for follow-up commands
        self._conv_buffer = ConversationBuffer(max_turns=5)

    def start(self) -> None:
        """Start wake-word detection and keep the runtime alive."""
        log.info('[Sara] Starting assistant runtime')
        tts_engine.initialize_engine()
        self._running = True
        self._stop_event.clear()
        self._set_idle_state()
        log.info('[WakeWord] Starting background wake detector')
        self.wake_detector.start(self.handle_wake_word)

        try:
            while self._running:
                time.sleep(0.5)
                # Watchdog: restart wake detector thread if it died unexpectedly
                t = getattr(self.wake_detector, '_thread', None)
                if (self._running
                        and t is not None
                        and not t.is_alive()
                        and not self._handling_wake.locked()):
                    log.warning('[Sara] Wake detector thread died — restarting')
                    self.wake_detector.start(self.handle_wake_word)
        except KeyboardInterrupt:
            log.info('[Sara] Runtime stopped by user')
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the runtime loop and release audio resources."""
        self._running = False
        self._stop_event.set()
        try:
            self.wake_detector.stop()
        except Exception:
            pass
        self._set_idle_state()
        log.info('[Sara] Assistant runtime stopped')

    def handle_wake_word(self) -> str:
        """Show the UI and enter the multi-turn conversation loop.

        The loop keeps listening for follow-up commands until the user says
        a stop phrase (e.g. "stop", "thank you", "goodbye") or the runtime
        is stopped externally.
        """
        if not self._handling_wake.acquire(blocking=False):
            log.info('[Sara] Wake word ignored because a command is already in progress')
            return 'Command already in progress.'

        session_id = uuid.uuid4().hex[:8]
        self._current_session_id = session_id
        _metrics.start_session(session_id, wake_source='wake_word')
        # Improvement 2: clear history at the start of each new wake-word session
        # so context from a previous conversation does not bleed into the next one.
        self._conv_buffer.clear()

        log.info('[WakeWord] Wake word detected')
        log.info('[Sara] Wake word detected')
        log.info('[UI] Showing assistant window')
        self._show_ui()

        # Cycle through short greetings so Sara doesn't sound like a recording
        greeting = next(_GREETINGS)
        self._call_ui_hook('set_listening')
        self._set_ui_status('Listening', state='listening')
        log.info('[TTS] Speaking greeting: %r', greeting)
        tts_engine.safe_speak(greeting)
        tts_engine.wait_until_done()

        try:
            return self._conversation_loop()
        except Exception as exc:
            log.exception('[Sara] Runtime cycle failed: %s', exc)
            message = "I encountered an unexpected error. Please try again."
            self.respond(message)
            return message
        finally:
            _metrics.end_session(session_id)
            self._current_session_id = None
            log.info('[UI] Hiding assistant window')
            self._set_idle_state()
            log.info('[Sara] Returning to idle')
            self._handling_wake.release()

    # ------------------------------------------------------------------ #
    # Conversation loop
    # ------------------------------------------------------------------ #

    def _conversation_loop(self) -> str:
        """Multi-turn conversation loop.

        Keeps listening for commands after a single wake word until the user
        says a stop phrase (e.g. "stop", "thank you") or too many consecutive
        errors occur.  The microphone is only open during each listen_once()
        call — it is released for LLM / execution / TTS so Sara can speak
        without feedback.
        """
        consecutive_errors = 0
        clarification_attempts = 0   # resets after each successful command
        retry_count = 0  # auto-retry for low confidence
        last_message = ''

        while self._running:
            # Post-TTS echo gap: let speaker resonance and the Windows audio
            # driver buffer drain before opening the mic.  Without this, the
            # tail of Sara's greeting or command response is captured by the
            # microphone and sent to STT as a spurious command.
            if POST_TTS_DELAY > 0:
                time.sleep(POST_TTS_DELAY)

            # --- Listen ---------------------------------------------------
            self._call_ui_hook('set_listening')
            self._set_ui_status('Listening', state='listening')
            self._set_ui_transcript('Listening...')

            log.info('[Speech] Starting listen-once capture')
            _t_stt = time.perf_counter()
            transcript = listen_once(language=self.language)
            _stt_ms = (time.perf_counter() - _t_stt) * 1000
            _stt_ok = bool(transcript and not transcript.startswith('[transcription_error]'))
            _metrics.record_pipeline_stage(
                'stt', latency_ms=_stt_ms, success=_stt_ok,
                session_id=self._current_session_id,
            )

            # --- No voice detected: user went quiet → exit conversation ----
            if transcript == '[no_voice]':
                log.info('[Sara] No voice detected — ending conversation and returning to idle')
                last_message = ''
                break

            # --- Handle transcription failure -----------------------------
            if not transcript or transcript.startswith('[transcription_error]'):
                if retry_count < 1:
                    retry_count += 1
                    continue
                message = self._transcription_error_message(transcript)
                log.info('[Sara] Transcription error (%d/%d)',
                         consecutive_errors + 1, MAX_CONSECUTIVE_ERRORS)
                self.respond(message)
                tts_engine.wait_until_done()
                self._record_tts_latency()
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log.info('[Sara] Too many consecutive errors — returning to idle')
                    last_message = message
                    break
                # Keep listening so user can try again
                continue

            consecutive_errors = 0

            # --- Transcript correction (STT post-processing) --------------
            # Fix common misrecognitions before anything else sees the text,
            # e.g. "at Rojo 90" → "add row 90", "rise at C1" → "write cell C1".
            corrected = correct_transcript(transcript)
            if corrected != transcript:
                log.info('[Sara] Transcript corrected: %r → %r',
                         transcript[:120], corrected[:120])
            transcript = corrected

            # --- Suspicious-transcript silent retry -----------------------
            # If the corrected transcript still contains known garble fragments
            # (e.g. "adro John 90" not covered by any correction rule), do one
            # silent retry before giving feedback — improves recovery without
            # adding latency on well-formed commands.
            if is_suspicious_transcript(transcript) and retry_count < 1:
                retry_count += 1
                log.info('[Sara] Suspicious transcript %r — silent retry', transcript[:80])
                continue

            # --- Low-confidence filter ------------------------------------
            # Drop single noise words and unrecognisable 1-2 word fragments
            # before they reach the router.  Examples: "taste", "oh", "18".
            # describe_low_confidence() gives a reason-specific message:
            #   bare number → "I heard a number only…"
            #   noise word  → "I didn't catch that clearly…"
            _lc_msg = describe_low_confidence(transcript)
            if _lc_msg is not None:
                if retry_count < 1:
                    retry_count += 1
                    continue
                log.info('[Sara] Low-confidence transcript filtered: %r → %r',
                         transcript[:80], _lc_msg[:80])
                if clarification_attempts >= MAX_CLARIFICATION_ATTEMPTS:
                    log.info('[Sara] Clarification budget exhausted — returning to idle')
                    break
                self.respond(_lc_msg)
                tts_engine.wait_until_done()
                self._record_tts_latency()
                clarification_attempts += 1
                continue

            # --- Ambiguous transcript guard (3+ word transcripts) -----------
            # Reject when BOTH conditions fail: no known phonetic intent pattern
            # AND no command verb present.  Either condition passing means the
            # transcript reaches the router as normal.
            # Examples blocked: "Andrew and the 10", "why does he want"
            # Examples allowed: "at row Alice 90" (intent=add_row), "make a score table" (verb=make)
            _words = transcript.lower().split()
            if (len(_words) >= 3
                    and detect_command_intent(transcript) is None
                    and not any(w in _COMMAND_VERBS for w in _words)):
                # Check for partial recognition: name + number → suggest add row
                match = re.search(r'\b([a-z]{2,})\s+(\d+)\b', transcript.lower())
                if match:
                    name, number = match.groups()
                    suggestion = f"I heard '{name} {number}'. Did you mean 'add row {name} {number}'?"
                    log.info('[Sara] Partial recognition: %r → suggesting %r',
                             transcript[:80], suggestion)
                    self.respond(suggestion)
                    tts_engine.wait_until_done()
                    self._record_tts_latency()
                    clarification_attempts += 1
                    if clarification_attempts >= MAX_CLARIFICATION_ATTEMPTS:
                        log.info('[Sara] Clarification budget exhausted — returning to idle')
                        break
                    continue
                else:
                    log.info('[Sara] Ambiguous transcript blocked (no intent, no verb): %r',
                             transcript[:80])
                    if clarification_attempts >= MAX_CLARIFICATION_ATTEMPTS:
                        log.info('[Sara] Clarification budget exhausted — returning to idle')
                        break
                    self.respond("I didn't catch a clear command. Could you please repeat?")
                    tts_engine.wait_until_done()
                    self._record_tts_latency()
                    clarification_attempts += 1
                    continue

            log.info('[Sara] Transcript received: %r', transcript[:160])
            self._set_ui_transcript(transcript)

            # --- Check for stop / farewell phrase ------------------------
            if self._is_stop_phrase(transcript):
                log.info('[Sara] Stop phrase detected: %r', transcript[:80])
                farewell = 'Goodbye!'
                self.respond(farewell)
                tts_engine.wait_until_done()
                self._record_tts_latency()
                last_message = farewell
                break

            # --- Process the command -------------------------------------
            _hist_len_before = len(self._conv_buffer.get_history())
            last_message = self.process_command(transcript)
            self._call_ui_hook('_add_history_item', transcript, last_message)

            # Wait for TTS to finish before opening the mic again
            tts_engine.wait_until_done()
            self._record_tts_latency()
            self._set_ui_status('Done', state='idle')

            # Detect whether process_command executed a real action or only
            # asked for clarification.  process_command adds to conv_buffer
            # only on a successful dispatch; if the length is unchanged it
            # responded with "I didn't quite understand" and returned early.
            if len(self._conv_buffer.get_history()) > _hist_len_before:
                clarification_attempts = 0   # real command handled — reset
                consecutive_errors = 0
                retry_count = 0
            else:
                if clarification_attempts >= MAX_CLARIFICATION_ATTEMPTS:
                    log.info('[Sara] Clarification budget exhausted — returning to idle')
                    break
                clarification_attempts += 1

            # --- Voice confirmation for destructive actions --------------
            if self._pending_confirmation_action is not None:
                last_message = self._handle_voice_confirmation()

            # Loop continues — Sara stays in conversation mode

        return last_message

    def _record_tts_latency(self) -> None:
        """Flush the pending TTS start-time into metrics (call after wait_until_done)."""
        if self._tts_start:
            _tts_ms = (time.perf_counter() - self._tts_start) * 1000
            _metrics.record_pipeline_stage(
                'tts', latency_ms=_tts_ms, success=True,
                session_id=self._current_session_id,
            )
            self._tts_start = 0.0

    # ------------------------------------------------------------------ #
    # Voice confirmation layer
    # ------------------------------------------------------------------ #

    _YES_PHRASES = frozenset({
        'yes', 'yeah', 'yep', 'yup', 'sure', 'ok', 'okay',
        'confirm', 'do it', 'go ahead', 'proceed', 'continue',
        'affirmative', 'correct', 'right',
    })
    _NO_PHRASES = frozenset({
        'no', 'nope', 'nah', 'cancel', 'stop', 'abort',
        'never mind', 'nevermind', 'skip', 'negative', 'dont',
        "don't", 'back', 'undo',
    })

    def _handle_voice_confirmation(self) -> str:
        """Listen for a yes/no answer and execute or cancel the pending action.

        Called immediately after Sara has spoken the confirmation question.
        The pending action is stored in ``_pending_confirmation_action`` by
        ``process_command()`` when the agent returns ``requires_confirmation=True``.

        Returns the final spoken message.
        """
        action = self._pending_confirmation_action
        self._pending_confirmation_action = None  # clear regardless of outcome

        if action is None:
            return ''

        log.info('[Sara] Confirmation layer — listening for yes/no')
        self._set_ui_status('Yes or No?', state='listening')
        self._set_ui_transcript('Listening for confirmation...')

        _t_stt = time.perf_counter()
        response = listen_once(language=self.language)
        _stt_ms = (time.perf_counter() - _t_stt) * 1000
        _stt_ok = bool(response and not response.startswith('[transcription_error]'))
        _metrics.record_pipeline_stage(
            'stt', latency_ms=_stt_ms, success=_stt_ok,
            session_id=self._current_session_id,
        )

        if not response or response.startswith('[transcription_error]'):
            message = "I couldn't hear your answer. Action cancelled."
            log.info('[Sara] Confirmation: no audio — cancelling')
            self.respond(message)
            tts_engine.wait_until_done()
            self._record_tts_latency()
            return message

        normalized = response.strip().lower().rstrip('.!?,')
        log.info('[Sara] Confirmation response: %r → normalised=%r', response[:80], normalized)
        self._set_ui_transcript(response)

        # Check for yes
        is_yes = (
            normalized in self._YES_PHRASES
            or any(normalized.startswith(y) for y in self._YES_PHRASES)
        )
        # Check for no
        is_no = (
            normalized in self._NO_PHRASES
            or any(normalized.startswith(n) for n in self._NO_PHRASES)
        )

        if is_yes:
            log.info('[Sara] Confirmation: YES — executing action')
            self._set_ui_status('Executing...', state='processing')
            _t_exec = time.perf_counter()
            result = self.agent.handle_action(action, confirmed=True, auto_correct=True)
            _exec_ms = (time.perf_counter() - _t_exec) * 1000
            _metrics.record_pipeline_stage(
                'execution', latency_ms=_exec_ms, success=result.executed,
                session_id=self._current_session_id,
            )
            message = self._message_from_result(result)
            log.info('[Sara] Confirmed execution complete: %r', message[:80])
        elif is_no:
            message = 'Okay, action cancelled.'
            log.info('[Sara] Confirmation: NO — action cancelled')
        else:
            message = "I didn't catch that — action cancelled to be safe."
            log.info('[Sara] Confirmation: unclear %r — cancelling', normalized)

        self.respond(message)
        tts_engine.wait_until_done()
        self._record_tts_latency()
        return message

    @staticmethod
    def _is_stop_phrase(text: str) -> bool:
        """Return True if *text* is a conversation-ending phrase."""
        normalized = text.strip().lower().rstrip('.!?,')
        return normalized in EXIT_PHRASES

    @staticmethod
    def _enrich_with_context(text: str) -> str:
        """Prepend active memory state so the LLM has column/sheet context.

        Adds a compact prefix like::

            [file:sales.xlsx sheet:Sheet1 columns:Name,Score] add another row

        The prefix uses bracket notation so the heuristic parser's regex
        patterns are not confused by it (no bare keywords in the prefix).

        All values from memory are sanitised before injection to prevent
        prompt injection via crafted filenames or column header content.
        For example, a column named ``] ignore instructions [`` would be
        escaped to ``  ignore instructions  `` before embedding.
        """
        from memory import memory as _mem
        snap = _mem.snapshot()
        parts: list[str] = []
        if snap.get('active_file'):
            # sanitize_memory_value escapes [ ] and strips newlines
            parts.append(f"file:{sanitize_memory_value(snap['active_file'])}")
        if snap.get('active_sheet'):
            parts.append(f"sheet:{sanitize_memory_value(snap['active_sheet'])}")
        headers = snap.get('active_headers') or []
        if headers:
            safe_headers = ','.join(sanitize_memory_value(h) for h in headers)
            parts.append(f"columns:{safe_headers}")
        if parts:
            return f"[{' '.join(parts)}] {text}"
        return text

    def process_command(self, text: str) -> str:
        """Route text through the LLM router and execute the parsed action.

        Improvement 2 — conversation history:
            The current session's conversation buffer is passed to the router so
            the LLM can resolve follow-up commands (e.g. "now calculate average"
            after "add row Tom 90").  The buffer is updated with the final
            user/assistant turn after the command completes.

        Improvement 4 — confidence scoring:
            If the router returns a confidence < 0.5 the command is ambiguous
            and Sara asks for clarification instead of executing a wrong action.
        """
        self._call_ui_hook('set_processing')
        self._set_ui_status('Thinking...', state='processing')

        # Detect intent before routing so we have a fallback reference if the
        # router returns 'unknown'.
        pre_intent = detect_command_intent(text)
        log.info('[Intent] Pre-route intent detected: %s', pre_intent or 'none')

        # --- Slot completeness gate ---------------------------------------
        # If a command intent is detected but required arguments are clearly
        # missing (e.g. "add row" with no data, "write cell C1" with no value),
        # ask for the missing slots NOW instead of sending the incomplete text
        # to Ollama and getting a hallucinated or wrong action back.
        _xl_sheets = (_xl_ctrl._session.workbook.sheetnames
                      if _xl_ctrl._session is not None else None)
        _slot_msg = check_command_slots(pre_intent, text, available_sheets=_xl_sheets)
        if _slot_msg is not None:
            log.info('[Slots] Incomplete command (intent=%s) — clarifying: %r',
                     pre_intent, _slot_msg[:80])
            self.respond(_slot_msg)
            return _slot_msg

        enriched = self._enrich_with_context(text)
        log.info('[Router] Routing transcript through llm_router (enriched=%s)',
                 enriched != text)

        # Improvement 2: pass current conversation history to the router
        history = self._conv_buffer.get_history()

        # ── Deterministic fast-path — bypass Ollama/Groq entirely ─────────────
        # Commands like 'undo' and 'save' need no LLM interpretation.
        # 'undo' is especially fragile: the heuristic parser checks tokens[0],
        # but _enrich_with_context() prepends "[file:…]" which shifts 'undo' off
        # position 0.  This explicit bypass is the robust fix.
        if pre_intent in _DETERMINISTIC_ACTIONS:
            action: dict = {**_DETERMINISTIC_ACTIONS[pre_intent],
                            '_router_source': 'heuristic', 'confidence': 0.95}
            # freeze_panes cell depends on the user's phrase, not a static default.
            # "freeze header"/"freeze top row"→A2; "freeze panes"/"freeze first column"/
            # "freeze column A"→B1.  Run the heuristic on the corrected transcript so
            # all STT variants resolve correctly without involving the LLM.
            if pre_intent == 'freeze_panes':
                from llm_parser import _heuristic_natural_language_parse
                _h = _heuristic_natural_language_parse(correct_transcript(text)) or {}
                action['cell'] = _h.get('cell') or action.get('cell', 'A2')
            log.info('[Router] Deterministic fast-path: intent=%s → type=%s (LLM skipped)',
                     pre_intent, action['type'])
            _llm_ms, _llm_ok = 0.0, True
        else:
            # Thinking hint — fire a short audio cue if the LLM call is slow.
            # A threading.Timer fires after THINKING_HINT_DELAY seconds; if the
            # router returns before the delay, the timer is cancelled and no hint
            # is spoken.  The hint only queues when TTS is currently idle so it
            # never interrupts an ongoing response.
            _hint_timer: threading.Timer | None = None
            if THINKING_HINT_DELAY > 0:
                def _speak_hint():
                    if not tts_engine.is_busy():
                        hint = next(_THINKING_HINTS)
                        log.debug('[TTS] Thinking hint: %r', hint)
                        tts_engine.safe_speak(hint)
                _hint_timer = threading.Timer(THINKING_HINT_DELAY, _speak_hint)
                _hint_timer.daemon = True
                _hint_timer.start()

            _t_llm = time.perf_counter()
            try:
                action = route_llm(enriched, history=history if history else None)
            finally:
                if _hint_timer is not None:
                    _hint_timer.cancel()
            _llm_ms = (time.perf_counter() - _t_llm) * 1000
            _llm_ok = self._is_actionable(action)

            # ── Deterministic intent protection ───────────────────────────
            # Reject LLM actions that contradict a high-stakes pre-route intent.
            # Prevents STT noise from turning "rename file" into "write_cell".
            if _llm_ok and isinstance(action, dict) and pre_intent in _PROTECTED_INTENTS:
                _action_type = action.get('type', '')
                if _action_type in _PROTECTED_INTENTS[pre_intent]:
                    log.warning(
                        '[Sara] Intent protection: pre_intent=%s blocked LLM action type=%s',
                        pre_intent, _action_type,
                    )
                    action = {
                        'type': 'unknown',
                        'confidence': 0.0,
                        '_router_source': 'intent_guard',
                    }
                    _llm_ok = False

        _metrics.record_pipeline_stage(
            'llm', latency_ms=_llm_ms, success=_llm_ok,
            detail=action.get('_router_source') if isinstance(action, dict) else None,
            session_id=self._current_session_id,
        )

        # ── LLM destructive-action safety gate ───────────────────────────────
        # If the LLM (Ollama/Groq) returned a destructive or file-system action
        # but detect_command_intent() found NO deterministic intent before routing,
        # the transcript was too noisy to be trusted — block and ask for retry.
        # Examples caught:
        #   "DELETE HEAT HEAT"              → LLM said delete_row → BLOCKED
        #   "Two is cool Delete"            → LLM said delete_row → BLOCKED
        #   "Other countries often go on"   → LLM said open_file  → BLOCKED
        # Heuristic-parsed actions (_router_source='heuristic') pass through even
        # when pre_intent is None — e.g. "merge header" is deterministic regex.
        if (_llm_ok
                and isinstance(action, dict)
                and pre_intent is None
                and action.get('_router_source') not in ('heuristic',)
                and action.get('type') in _LLM_SAFETY_BLOCKED):
            log.warning(
                '[Sara] LLM safety gate: blocked type=%r source=%r (pre_intent=None)',
                action.get('type'), action.get('_router_source'),
            )
            msg = (
                "I'm not sure what you meant. "
                "Please say the command more clearly — for example: "
                "delete row 3, or sort by column A ascending."
            )
            self.respond(msg)
            return msg

        # ── Voice-mode add_row safety gate ────────────────────────────────────
        # Reject add_row actions whose values look garbled before they reach
        # the executor.  Catches LLM-hallucinated actions from unclear speech:
        #   "Insert row one"       → {'type':'add_row','values':['one.']}
        #   "Insert row tool"      → {'type':'add_row','values':['tool.']}
        #   "at row eight nine 10" → values=['eight','nine',10] (3 items)
        # Valid voice commands always have exactly one name-like string and one
        # numeric value; anything else is treated as a garbled transcript.
        if _llm_ok and isinstance(action, dict) and action.get('type') == 'add_row':
            if not _is_safe_add_row_values(action.get('values') or []):
                log.info('[Sara] add_row blocked — unsafe values: %r',
                         action.get('values'))
                msg = (
                    "I'm not sure what to add to the row. "
                    "Please say the name and score clearly, for example: add row John 90."
                )
                self.respond(msg)
                return msg

        # Improvement 4: check confidence before deciding whether to execute.
        # The check runs before the _is_actionable gate so it can distinguish
        # between "the parser gave up" (unknown, 0.10) and "the parser guessed
        # but is not very sure" (groq/ollama on an ambiguous phrase).
        confidence = action.get('confidence', 1.0)
        log.info('[Router] confidence=%.2f  source=%s  actionable=%s',
                 confidence, action.get('_router_source', '?'), _llm_ok)

        if not _llm_ok or confidence < 0.5:
            # Either the router gave up entirely (unknown) or confidence is
            # below the clarity threshold — ask the user to rephrase.
            # If we detected a pre-route intent, use it to give a targeted hint
            # rather than a generic "try again" message.  This also means no
            # Ollama fallback is attempted when the STT was just noisy.
            if confidence < 0.5 and _llm_ok:
                log.warning('[Router] Low confidence (%.2f) — requesting clarification',
                            confidence)
                if pre_intent:
                    readable = pre_intent.replace('_', ' ')
                    message = (
                        f"Almost there — I think you meant {readable}. "
                        "Could you say the full command again?"
                    )
                else:
                    message = (
                        "Hmm, I'm not sure what you meant. "
                        "Did you want to add a row, calculate something, or do something else?"
                    )
            else:
                log.warning('[Router] Unknown command — raw text: %r  pre_intent=%s',
                            text[:200], pre_intent or 'none')
                if pre_intent:
                    readable = pre_intent.replace('_', ' ')
                    message = (
                        f"Hmm, I think you meant {readable} — can you say the full command again?"
                    )
                else:
                    message = (
                        "I didn't quite catch that — try something like: "
                        "add row, open Excel, or calculate average."
                    )
            self.respond(message)
            # Do NOT add failed/unclear turns to the buffer — it adds noise to
            # the LLM context and wastes prompt tokens (see architecture review).
            return message

        log.info('[Agent] Passing routed action to agent')
        log.info('[Sara] Executing action')
        self._set_ui_status('Executing...', state='processing')
        _t_exec = time.perf_counter()
        result = self.agent.handle_action(action, auto_correct=True)
        _exec_ms = (time.perf_counter() - _t_exec) * 1000

        # Pipeline-stage metrics (per-stage latency for the thesis dashboard)
        _metrics.record_pipeline_stage(
            'execution', latency_ms=_exec_ms, success=result.executed,
            session_id=self._current_session_id,
        )

        # Structured execution record — populates the `executions` table so
        # metrics.summary() shows per-action-type success/failure rates.
        _action_type = (
            action.get('type', 'steps') if isinstance(action, dict) else 'unknown'
        )
        _metrics.record_execution(
            action_type=_action_type,
            success=result.success,
            latency_ms=_exec_ms,
            workflow_id=result.workflow_id,
            error=result.error,
            filename=result.download_url.lstrip('/files/') if result.download_url else None,
            request_id=self._current_session_id,
        )

        if result.requires_confirmation:
            # Store the parsed action so _conversation_loop() can ask yes/no
            # after Sara finishes speaking the confirmation question.
            self._pending_confirmation_action = action
            log.info('[Sara] Destructive action flagged — storing for voice confirmation')
        else:
            self._pending_confirmation_action = None

        if result.executed or result.requires_confirmation:
            log.info('[Excel] Excel command pipeline completed (executed=%s confirmation=%s)',
                     result.executed, result.requires_confirmation)
        else:
            log.info('[Excel] No Excel command executed')

        message = self._message_from_result(result)
        self.respond(message)

        # Improvement 2: record this turn so follow-up commands have context.
        # sanitize_history_turn escapes bracket characters that could inject
        # content outside the [Previous conversation:] block in the LLM prompt.
        from pipeline_sanitizer import sanitize_history_turn as _sht
        self._conv_buffer.add(user=_sht(text), assistant=_sht(message))

        # Structured pipeline summary — one log entry per executed command.
        # Fields are chosen for direct thesis analysis: action type, source
        # tier, confidence, and all stage latencies in one place.
        log.info(
            '[Pipeline] SUMMARY  transcript=%r  action=%s  source=%s  '
            'confidence=%.2f  executed=%s  llm_ms=%.0f  exec_ms=%.0f',
            text[:80],
            action.get('type', 'steps') if isinstance(action, dict) else 'unknown',
            action.get('_router_source', '?') if isinstance(action, dict) else '?',
            action.get('confidence', 0.0) if isinstance(action, dict) else 0.0,
            result.executed,
            _llm_ms,
            _exec_ms,
        )

        return message

    def respond(self, message: str) -> None:
        """Speak the message and update UI state.

        Applies ``sanitize_speech`` before passing to TTS so that LLM
        internal reasoning (<think> blocks), raw JSON fragments, or internal
        metadata keys never reach the user's ears.

        Shows "Speaking..." while TTS is active.  The caller must call
        ``tts_engine.wait_until_done()`` after this.
        """
        # Sanitize before any output — strips LLM internals, JSON blobs etc.
        clean_message = sanitize_speech(message)
        if clean_message != message:
            log.debug('[TTS] sanitize_speech stripped content: original_len=%d  clean_len=%d',
                      len(message), len(clean_message))

        if not clean_message:
            log.warning('[TTS] respond(): sanitize_speech reduced message to empty: %r', message[:80])
            return

        # Display response text immediately so the user can read it
        self._set_ui_response(clean_message)
        # Switch UI to Speaking state BEFORE starting TTS
        self._call_ui_hook('set_speaking', clean_message)
        self._set_ui_status('Speaking...', state='speaking')
        log.info('[TTS] Speaking response: %r', clean_message[:80])
        self._tts_start = time.perf_counter()
        tts_engine.safe_speak(clean_message)

    @staticmethod
    def _is_actionable(action: dict | None) -> bool:
        if not isinstance(action, dict):
            return False
        if isinstance(action.get('steps'), list) and action['steps']:
            return True
        return action.get('type') not in ('unknown', None, '')

    @staticmethod
    def _message_from_result(result: AgentResult) -> str:
        # Destructive-action confirmation prompt — speak the question
        if result.requires_confirmation:
            return result.confirmation_message or 'Can you confirm you want to do that?'

        # Auto-corrected: agent created a file automatically before retrying.
        # Prepend a brief note so the user knows what happened.
        auto_prefix = ''
        if result.auto_corrected:
            auto_prefix = "I created a new spreadsheet for you. "

        # speech_text is the primary signal — set by ErrorHandler.safe_speech()
        # for every tool call (success or error).
        if result.speech_text:
            return auto_prefix + result.speech_text

        # Explicit failure with no speech_text
        if not result.success:
            err = result.error or ''
            # Avoid exposing raw error codes or exception text to the user
            if err and not any(c in err for c in ('{', '(', '\n', ':')):
                return auto_prefix + f"Sorry, that didn't work. {err.replace('_', ' ').capitalize()}."
            return auto_prefix + "Sorry, I couldn't complete that command."

        # Multi-step result: concatenate individual step speeches
        if isinstance(result.result, dict):
            steps = result.result.get('steps')
            if isinstance(steps, list) and steps:
                parts = []
                for s in steps:
                    r = s.get('result') if isinstance(s, dict) else None
                    if isinstance(r, dict):
                        st = r.get('speech_text') or r.get('message')
                        if st:
                            parts.append(st.rstrip('.'))
                if parts:
                    return auto_prefix + '. '.join(parts) + '.'
            msg = result.result.get('speech_text') or result.result.get('message')
            if msg:
                return auto_prefix + msg

        if isinstance(result.result, str) and result.result.strip():
            return auto_prefix + result.result

        return auto_prefix + ('Done.' if result.executed else 'Got it.')

    @staticmethod
    def _transcription_error_message(transcript: str | None) -> str:
        if transcript and transcript.startswith('[transcription_error]'):
            return transcript.replace('[transcription_error]', '').strip() or 'I could not transcribe your command.'
        return 'I could not hear a command clearly.'

    def _set_idle_state(self) -> None:
        self._call_ui_hook('set_idle')   # stops animations, re-enables mic
        self._set_ui_status('Idle', state='idle')
        if self.hide_ui_when_idle:
            self._hide_ui()

    def _show_ui(self) -> None:
        if self.ui is None:
            return
        if self._call_ui_hook('show_assistant'):
            return
        def _fallback():
            for method_name in ('show', 'raise_', 'activateWindow'):
                method = getattr(self.ui, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
        self._run_on_ui_thread(_fallback)

    def _hide_ui(self) -> None:
        if self.ui is None:
            return
        if self._call_ui_hook('hide_assistant'):
            return
        def _fallback():
            method = getattr(self.ui, 'hide', None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
        self._run_on_ui_thread(_fallback)

    def _call_ui_hook(self, method_name: str, *args) -> bool:
        """Call a named method on the UI object, thread-safely."""
        if self.ui is None:
            return False
        method = getattr(self.ui, method_name, None)
        if not callable(method):
            return False
        try:
            if args:
                self._run_on_ui_thread(lambda m=method, a=args: m(*a))
            else:
                self._run_on_ui_thread(method)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Thread-safe Qt helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_on_ui_thread(func) -> None:
        """Execute *func* on the Qt main thread if a QApplication is running.

        Uses ``QTimer.singleShot(0, func)`` which is thread-safe in Qt 6 and
        schedules the call on the main event-loop thread.  Falls back to a
        direct call when PyQt is not available or there is no QApplication.
        """
        try:
            from PyQt6.QtCore import QTimer          # noqa: F811
            from PyQt6.QtWidgets import QApplication  # noqa: F811
            if QApplication.instance() is not None and threading.current_thread() is not threading.main_thread():
                QTimer.singleShot(0, func)
                return
        except ImportError:
            pass
        func()

    def _set_ui_status(self, text: str, *, state: str | None = None) -> None:
        if self.ui is None:
            return
        ui = self.ui

        def _update():
            label = getattr(ui, 'status_label', None)
            if label is not None:
                self._set_text_widget(label, text)

            dot = getattr(ui, 'status_dot', None)
            if dot is not None and state and hasattr(dot, 'set_state'):
                try:
                    dot.set_state(state)
                except Exception:
                    pass

            mic_btn = getattr(ui, 'mic_btn', None)
            if mic_btn is not None and hasattr(mic_btn, 'setText'):
                try:
                    if state == 'listening':
                        mic_btn.setText('Listening')
                    elif state == 'processing':
                        mic_btn.setText('Processing...')
                    elif state == 'speaking':
                        mic_btn.setText('Speaking...')
                    else:
                        mic_btn.setText('Start Listening')
                except Exception:
                    pass

        self._run_on_ui_thread(_update)

    def _set_ui_transcript(self, text: str) -> None:
        if self.ui is None:
            return
        ui = self.ui

        def _update():
            widget = getattr(ui, 'transcript_view', None) or getattr(ui, 'transcript', None)
            if widget is not None:
                self._set_text_widget(widget, text)

        self._run_on_ui_thread(_update)

    def _set_ui_response(self, text: str) -> None:
        if self.ui is None:
            return
        ui = self.ui

        def _update():
            widget = getattr(ui, 'response_view', None)
            if widget is not None:
                self._set_text_widget(widget, text)

        self._run_on_ui_thread(_update)

    @staticmethod
    def _set_text_widget(widget: object, text: str) -> None:
        for method_name in ('setPlainText', 'setText', 'append'):
            method = getattr(widget, method_name, None)
            if callable(method):
                try:
                    method(text)
                    return
                except Exception:
                    continue


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    runtime = SaraRuntime()
    runtime.start()


if __name__ == '__main__':
    main()