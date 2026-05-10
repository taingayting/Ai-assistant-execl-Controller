"""pipeline_sanitizer.py — Central safety layer for Sara's voice pipeline.

Centralises every sanitisation concern in one auditable module:

    1. Speech text sanitisation  — strips LLM internals before TTS
    2. Memory context injection  — escapes values injected into LLM prompts
    3. Conversation history      — escapes user speech before history storage
    4. Cell value safety         — prevents Excel formula injection
    5. Formula validation        — whitelists allowed spreadsheet functions

Import pattern::

    from pipeline_sanitizer import (
        sanitize_speech,
        sanitize_memory_value,
        sanitize_history_turn,
        sanitize_cell_value,
        sanitize_row_values,
        is_formula_safe,
    )
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger('pipeline_sanitizer')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters that make Excel treat a cell as a formula.
_FORMULA_TRIGGER = ('=', '+', '-', '@')

# Functions the assistant is authorised to write.  Any other function name
# in a write_formula action is rejected before execution.
ALLOWED_FORMULA_FUNCTIONS: frozenset[str] = frozenset({
    'SUM', 'AVERAGE', 'COUNT', 'COUNTA', 'MAX', 'MIN',
    'IF', 'IFS', 'COUNTIF', 'COUNTIFS',
    'SUMIF', 'SUMIFS', 'AVERAGEIF', 'AVERAGEIFS',
    'ROUND', 'ROUNDUP', 'ROUNDDOWN',
    'INT', 'ABS', 'MOD',
    'LEN', 'LEFT', 'RIGHT', 'MID', 'TRIM', 'UPPER', 'LOWER',
    'CONCATENATE', 'CONCAT', 'TEXTJOIN',
    'VLOOKUP', 'HLOOKUP', 'INDEX', 'MATCH', 'XLOOKUP',
    'TODAY', 'NOW', 'DATE', 'YEAR', 'MONTH', 'DAY',
    'IFERROR', 'ISBLANK', 'ISNUMBER', 'ISTEXT',
})

# Compiled regex for <think>…</think> blocks emitted by reasoning models.
_THINK_TAG_RE = re.compile(r'<think>[\s\S]*?</think>', re.IGNORECASE)

# Loose pattern that matches a JSON object or array starting a string.
_JSON_START_RE = re.compile(r'^\s*[\[{]')

# Trailing JSON blob appended after prose, e.g. "Done. {"type": "add_row"}"
_TRAILING_JSON_RE = re.compile(r'\s*\{[^}]{0,300}\}\s*$')

# Internal router metadata keys that must never reach the user.
_INTERNAL_KEY_RE = re.compile(
    r'\b(_router_source|_skipped|confidence|thinking)\s*[:=][^\n]{0,100}',
    re.IGNORECASE,
)

# Maximum characters to speak aloud — longer responses are truncated at a
# sentence boundary to keep voice output natural.
_MAX_SPEECH_LEN = 400


# ---------------------------------------------------------------------------
# 1. Speech text sanitisation
# ---------------------------------------------------------------------------

def sanitize_speech(text: str) -> str:
    """Remove LLM internals from *text* before passing it to TTS.

    Strips:
    * ``<think>…</think>`` reasoning blocks
    * Responses that are entirely raw JSON (replaced with "Done.")
    * Trailing JSON blobs appended after prose
    * Internal metadata key/value pairs
    * Excessive whitespace

    Truncates at ``_MAX_SPEECH_LEN`` characters, breaking at the last
    sentence boundary so speech ends naturally.

    Always returns a non-empty string safe for TTS.
    """
    if not isinstance(text, str) or not text.strip():
        return "Done."

    # 1. Remove <think>…</think> blocks
    text = _THINK_TAG_RE.sub('', text).strip()

    # 2. If the entire response is JSON, substitute a safe fallback
    if _JSON_START_RE.match(text):
        log.warning('sanitize_speech: raw JSON response blocked — using fallback')
        return "Done."

    # 3. Strip trailing JSON fragments after prose
    text = _TRAILING_JSON_RE.sub('', text).strip()

    # 4. Remove internal metadata key=value pairs
    text = _INTERNAL_KEY_RE.sub('', text).strip()

    # 5. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Truncate at a sentence boundary
    if len(text) > _MAX_SPEECH_LEN:
        candidate = text[:_MAX_SPEECH_LEN]
        last_end = max(
            candidate.rfind('.'),
            candidate.rfind('!'),
            candidate.rfind('?'),
        )
        if last_end > _MAX_SPEECH_LEN // 2:
            text = candidate[:last_end + 1]
        else:
            text = candidate.rstrip()

    return text or "Done."


# ---------------------------------------------------------------------------
# 2. Memory context injection safety
# ---------------------------------------------------------------------------

def sanitize_memory_value(value: str) -> str:
    """Escape a memory value before embedding it in an LLM prompt prefix.

    The context prefix format is ``[file:X sheet:Y columns:A,B]``.  If a
    stored value contains ``[`` or ``]`` it can break the bracket notation
    and inject arbitrary text outside the intended context block.  Newlines
    are also stripped because they could end the prefix line prematurely.

    Example::

        sanitize_memory_value('] ignore instructions [')
        # -> ') ignore instructions ('
    """
    if not isinstance(value, str):
        return str(value) if value is not None else ''
    return (
        value
        .replace('[', '(')
        .replace(']', ')')
        .replace('\n', ' ')
        .replace('\r', ' ')
        .strip()
    )


# ---------------------------------------------------------------------------
# 3. Conversation history sanitisation
# ---------------------------------------------------------------------------

def sanitize_history_turn(text: str) -> str:
    """Sanitise user speech before it is stored in the conversation buffer.

    The buffer is later formatted as::

        [Previous conversation:
        User: <text>
        Sara: <response>
        ]

    Bracket characters in user speech can escape the block and inject
    arbitrary content into the LLM prompt.  This function prevents that.
    Length is also capped because very long turns are unlikely to be genuine
    voice commands and waste LLM context.
    """
    if not isinstance(text, str):
        return str(text)[:300]
    cleaned = (
        text
        .replace('[', '(')
        .replace(']', ')')
        .strip()
    )
    return cleaned[:300]


# ---------------------------------------------------------------------------
# 4. Cell value safety (formula injection prevention)
# ---------------------------------------------------------------------------

def sanitize_cell_value(value: object) -> object:
    """Prevent Excel formula injection in a single cell value.

    Excel evaluates any string beginning with ``=``, ``+``, ``-``, or ``@``
    as a formula.  Prepending a tab character forces Excel to treat the
    value as plain text, which is the standard CSV / OOXML mitigation.

    Non-string values are returned unchanged.
    """
    if isinstance(value, str) and value.startswith(_FORMULA_TRIGGER):
        log.warning('sanitize_cell_value: formula injection escaped: %r', value[:40])
        return '\t' + value
    return value


def sanitize_row_values(values: list) -> list:
    """Apply :func:`sanitize_cell_value` to every element of *values*.

    Numeric values and ``None`` pass through unchanged.
    """
    return [sanitize_cell_value(v) for v in values]


# ---------------------------------------------------------------------------
# 5. Formula validation
# ---------------------------------------------------------------------------

_FUNC_RE = re.compile(r'\b([A-Z]{2,})\s*\(')

# Matches a range argument where the second cell reference is incomplete,
# e.g. "A1:A" (letter only, no row number) or "A1:1" (row only, no letter).
_INCOMPLETE_RANGE_RE = re.compile(
    r'\b[A-Z]+\d+:(?:[A-Z]+(?!\d)|(?<![A-Z])\d+)\b', re.IGNORECASE
)


def get_formula_error(formula: str) -> str | None:
    """Return a human-friendly error string if *formula* has structural problems.

    Checks (in order):
    1. Not a string / empty body — "The formula is empty."
    2. Balanced parentheses  — reports how many are missing.
    3. No-argument calls — ``SUM()`` with empty parens.
    4. Incomplete range arguments — ``A1:A`` or ``A1:10``.
    5. Unknown / disallowed function names.

    Returns ``None`` when the formula passes all checks.

    Used by :func:`is_formula_safe` and by ``write_formula`` in
    ``excel_controller`` so each check has a distinct user-facing message
    instead of a generic "formula blocked" response.
    """
    if not isinstance(formula, str):
        return "The formula must be a text string."

    stripped = formula.strip()
    body = stripped.lstrip('=').strip()
    if not body:
        return "The formula is empty — please add a function or cell reference, like =SUM(A1:A10)."

    upper = stripped.upper()

    # ---- 1. Balanced parentheses ----------------------------------------
    depth = 0
    in_string = False
    for ch in stripped:
        if ch == '"':
            in_string = not in_string
        if in_string:
            continue
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth < 0:
            return ("The formula has an extra closing parenthesis. "
                    "Check that every ')' has a matching '('.")
    if depth > 0:
        missing = 'parenthesis' if depth == 1 else 'parentheses'
        return (f"The formula is missing {depth} closing {missing}. "
                f"Make sure every '(' is closed with a matching ')'.")

    # ---- 2. Function called with no arguments ---------------------------
    # e.g. =SUM() — catches accidentally empty parens before the whitelist check.
    empty_call = re.search(r'\b([A-Z]{2,})\(\s*\)', upper)
    if empty_call:
        fname = empty_call.group(1)
        return (f"The {fname} function has no arguments. "
                f"Please add a cell range like A1:A10, e.g. ={fname}(A1:A10).")

    # ---- 3. Incomplete range arguments ----------------------------------
    # Catches =AVERAGE(A1:A) where the second cell ref has no row number,
    # or =SUM(A:10) where it has no column letter.
    bad_range = _INCOMPLETE_RANGE_RE.search(upper)
    if bad_range:
        return (f"The range '{bad_range.group()}' looks incomplete. "
                "Both ends of a range need a column letter and row number, "
                "like A1:A10.")

    # ---- 4. Unknown / disallowed function names -------------------------
    found = set(_FUNC_RE.findall(upper))
    unknown = found - ALLOWED_FORMULA_FUNCTIONS
    if unknown:
        names = ', '.join(sorted(unknown))
        log.warning('get_formula_error: rejected unknown functions: %s', unknown)
        return (f"The formula uses a function that isn't supported: {names}. "
                f"Supported functions include SUM, AVERAGE, IF, VLOOKUP, and others.")

    return None  # all checks passed


# ---------------------------------------------------------------------------
# 5b. Phonetic pre-normalization
# ---------------------------------------------------------------------------

PHONETIC_FIXES: list[tuple[str, str]] = [
    ("rise up", "write cell"),   # phrase match before single-word "rise"
    ("rice", "write"),
    ("rise", "write"),
    ("raiser", "write"),         # before "reiser"
    ("reiser", "write"),
    ("riser", "write"),
    ("right", "write"),

    ("at row", "add row"),
    ("add raw", "add row"),
    ("at raw", "add row"),       # "at raw" variant
    ("a draw", "add row"),
    ("at draw", "add row"),
    ("i draw", "add row"),       # "i draw" variant

    ("see", "c"),
    ("sea", "c"),
    ("bee", "b"),
    ("dee", "d"),
]


def apply_phonetic_fixes(text: str) -> str:
    """Apply phonetic string replacements on transcript.

    Run BEFORE regex correction passes so that rare STT corruptions like
    "reiser c1" → "write c1" are caught early, and cell-letter phonetics
    like "write see one" → "write c one" → "write c1" are resolved.
    """
    # Protect direction word "right" when STT drops the word "align" but the
    # surrounding context (cell/row/column ref + direction) is clearly alignment.
    _has_align = bool(
        re.search(r'\balign\b', text, re.IGNORECASE) or
        re.search(
            r'\b(?:row\s+\d+|column\s+[A-Za-z]+|[A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\s+'
            r'(?:right|left|center|centre)\b',
            text, re.IGNORECASE,
        )
    )
    for wrong, correct in PHONETIC_FIXES:
        # "right" → "write" must not fire in alignment context:
        # "align column B right" → direction word, not a write command.
        if wrong == 'right' and _has_align:
            continue
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# 6. STT transcript correction
# ---------------------------------------------------------------------------

# ── Pre-pass: strip leading wake-word artifacts ────────────────────────────
# When the user says "hey jarvis open excel" without pausing, STT captures
# the whole utterance including the wake word.  This pattern removes common
# wake-word prefixes from the start of the transcript so the command parser
# receives "open excel" rather than "hey jarvis open excel".
# Only strip at the very start (^) and only well-known wake phrases — never
# strip mid-transcript to avoid deleting meaningful command words.
_WAKE_PREFIX_RE = re.compile(
    r'^(?:hey\s+(?:jarvis|sara|siri|google)|ok(?:ay)?\s+(?:google|sara|jarvis))'
    r'\s*[,.]?\s*',
    re.IGNORECASE,
)

# ── Pass 1: word/phrase-level regex replacements ──────────────────────────
# Each entry is (compiled_regex, replacement_string).
# Applied in order on the transcript (case-insensitive, original case
# preserved through re.sub).  Rules are chosen where the misrecognition is
# highly specific and the correct form is unambiguous in Sara's context.

# Spoken digit words used by STT for single-digit row numbers in delete/remove
# context.  Defined here so the lambda in _TRANSCRIPT_CORRECTIONS can reference it.
_ROW_NUM_WORDS: dict[str, str] = {
    'one': '1', 'won': '1', 'two': '2', 'to': '2', 'too': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6', 'seven': '7',
    'eight': '8', 'nine': '9',
}

_TRANSCRIPT_CORRECTIONS: list[tuple[re.Pattern, str]] = [

    # ══ Punctuation normalisation ═════════════════════════════════════════════
    # Whisper often inserts commas to mark pauses in voice input
    # ("Filter, Column B, 90").  Sara commands are never comma-delimited, so
    # replace every comma with a space before any other rule runs.
    (re.compile(r','), ' '),

    # ══ Natural-language polite prefix strip ═══════════════════════════════
    # "Can I add row John" / "Could I write cell B3 100" → strip the prefix.
    # Lookahead restricts to known command verbs so conversational sentences
    # like "Can I help you?" are untouched.
    (re.compile(
        r'^(?:can|could)\s+i\s+(?=(?:add|write|create|delete|open|close|save|switch|make|'
        r'calculate|sort|format|merge|freeze|rename|list|find|replace|filter|insert|read|clear|'
        r'go|undo|set)\b)',
        re.IGNORECASE), ''),
    # "Have me rename/switch/create..." — filler prefix before command verb
    (re.compile(
        r'^have\s+me\s+(?=(?:rename|switch|create|delete|list|add|write|open|close|save|'
        r'calculate|sort|format|merge|freeze|find|replace|filter|insert|read|clear|undo|set)\b)',
        re.IGNORECASE), ''),

    # ══ Rename-verb STT misrecognitions ═══════════════════════════════════════
    # "renave" / "renee" → "rename"  (phonetic corruptions at start of utterance)
    (re.compile(r'^renave\b', re.IGNORECASE), 'rename'),
    (re.compile(r'^renee\b',  re.IGNORECASE), 'rename'),
    # "rename fire" → "rename file"  ("fire" is a common STT mishearing of "file")
    (re.compile(r'\brename\s+fire\b', re.IGNORECASE), 'rename file'),
    # "rename [file] sever/servo to X" → "rename [file] server to X"
    # STT mishears "server" (the filename) as "sever" or "servo".
    # Must run AFTER the fire→file rule so "rename fire sever" → "rename file sever"
    # → "rename file server" in two passes.
    (re.compile(r'\b(rename\s+(?:file\s+)?)sever\b', re.IGNORECASE),
     lambda m: f'{m.group(1)}server'),
    (re.compile(r'\b(rename\s+(?:file\s+)?)servo\b', re.IGNORECASE),
     lambda m: f'{m.group(1)}server'),
    # "rename, sheet" → "rename sheet"  (STT inserts a comma after the verb pause)
    (re.compile(r'\brename\s*,\s*', re.IGNORECASE), 'rename '),

    # ══ "add row" misrecognitions ══════════════════════════════════════════
    # "at Rojo 90"  → "add row 90"   ("rojo" = "row [name]" merged by STT)
    # "a rojo"      → "add row"
    # "add rojo"    → "add row"
    (re.compile(r'\bat rojo\b', re.IGNORECASE), 'add row'),
    (re.compile(r'\ba rojo\b',  re.IGNORECASE), 'add row'),
    (re.compile(r'\badd rojo\b', re.IGNORECASE), 'add row'),
    # "a role" / "ad role" / "add role" / "add roll"
    (re.compile(r'\ba role\b',   re.IGNORECASE), 'add row'),
    (re.compile(r'\bat role\b',  re.IGNORECASE), 'add row'),
    (re.compile(r'\badd role\b', re.IGNORECASE), 'add row'),
    (re.compile(r'\badd roll\b', re.IGNORECASE), 'add row'),
    # "ad row" / "and row" / "had row" / "and grow" (log-confirmed: "and grow John")
    (re.compile(r'\bad row\b',  re.IGNORECASE), 'add row'),
    (re.compile(r'\band row\b', re.IGNORECASE), 'add row'),
    (re.compile(r'\bhat row\b', re.IGNORECASE), 'add row'),
    (re.compile(r'\band grow\b', re.IGNORECASE), 'add row'),
    (re.compile(r'\badroll\b', re.IGNORECASE), 'add row'),
    # "at row …" at start of utterance
    (re.compile(r'^at row\b',   re.IGNORECASE), 'add row'),
    # plain "a row …" as first token
    (re.compile(r'^a row\b',    re.IGNORECASE), 'add row'),

    # ══ "write" + "cell" misrecognitions ══════════════════════════════════
    # "rise at C1" → "write cell C1"   (both words wrong simultaneously)
    # "rice at C1" → "write cell C1"
    # "right at C1"→ "write cell C1"
    (re.compile(r'\brise at\b',  re.IGNORECASE), 'write cell'),
    (re.compile(r'\brice at\b',  re.IGNORECASE), 'write cell'),
    (re.compile(r'\bright at\b', re.IGNORECASE), 'write cell'),
    # "rise cell" / "rice cell" / "right cell" / "rise so" / "rice so"
    (re.compile(r'\brise cell\b',  re.IGNORECASE), 'write cell'),
    (re.compile(r'\brice cell\b',  re.IGNORECASE), 'write cell'),
    (re.compile(r'\brice cell\b',  re.IGNORECASE), 'write cell'),
    (re.compile(r'\bright cell\b', re.IGNORECASE), 'write cell'),
    (re.compile(r'\brice so\b',    re.IGNORECASE), 'write cell'),
    (re.compile(r'\brise so\b',    re.IGNORECASE), 'write cell'),
    (re.compile(r'\b(rise|rice|right|wright)\s+(?:up|of|serve|says)\b', re.IGNORECASE), 'write cell'),
    # "write up" — residual when phonetic "rise up"→"write cell" was partially applied
    (re.compile(r'\bwrite\s+up\b', re.IGNORECASE), 'write cell'),
    # "write serve" — residual when phonetic "rice serve"→"write serve" was partially applied
    (re.compile(r'\bwrite serve\b', re.IGNORECASE), 'write cell'),
    # "Rice's C 100" → "write cell C 100"
    # Whisper apostrophe-s artifact: bare ^rice\b would leave "'s" dangling,
    # producing "write's C 100". This rule must come BEFORE ^rice\b.
    (re.compile(r"^rice's\b", re.IGNORECASE), 'write cell'),
    # standalone leading "rice" / "rise" / "raisa" → "write"
    # "raisa" — log-confirmed STT corruption of "write" on Realtek laptop mic
    (re.compile(r'^rice\b',  re.IGNORECASE), 'write'),
    (re.compile(r'^rise\b',  re.IGNORECASE), 'write'),
    (re.compile(r'^raisa\b', re.IGNORECASE), 'write'),
    (re.compile(r'^raiser\b', re.IGNORECASE), 'write'),
    # "raise" → "write" ONLY when the next token is an Excel-specific marker:
    #   "raise cell B3", "raise column A", "raise row 5", "raise B3 value"
    # This prevents "raise the bar" or "raise your hand" from corrupting.
    (re.compile(r'^raise(?=\s+(?:cell\b|column\b|row\b|[A-Za-z]\d+\b))',
                re.IGNORECASE), 'write'),

    # ══ "save" misrecognitions ════════════════════════════════════════════
    (re.compile(r'\bsafe and close\b',   re.IGNORECASE), 'save and close'),
    (re.compile(r'\bsake and close\b',   re.IGNORECASE), 'save and close'),
    (re.compile(r'\bself and close\b',   re.IGNORECASE), 'save and close'),  # real log
    (re.compile(r'\bsave this and close\b', re.IGNORECASE), 'save and close'),
    (re.compile(r'\bsoft self\b',        re.IGNORECASE), 'save'),
    # Bare "safe"/"sake" → "save" (Whisper phonetic confusion on Realtek laptop mic).
    # Negative lookahead guards compound forms: "safe and close" is handled above
    # and must not be rewritten here into "save and close" (double-rewrite risk).
    (re.compile(r'^safe\b(?!\s+and\b)',  re.IGNORECASE), 'save'),
    (re.compile(r'^sake\b(?!\s+and\b)',  re.IGNORECASE), 'save'),

    # ══ "open" / "close" misrecognitions ══════════════════════════════════
    # "open it Sarah" → "open excel"  (Whisper hallucination: "it Sara/Sarah"
    # for "excel"; confirmed in logs; only safe at utterance start because
    # "open it" mid-sentence is a valid English phrase in other contexts)
    (re.compile(r'\bopen\s+it\s+sir\b', re.IGNORECASE), 'open excel'),
    (re.compile(r'^open\s+it\b', re.IGNORECASE), 'open excel'),
    (re.compile(r'\bopen\s+each\s+cell\b', re.IGNORECASE), 'open excel'),
    # "open each there" / "open each sir" — Whisper phonetic corruption of "open excel"
    # where "excel" is misheard as "each [word]".  Confirmed in real logs.
    # More specific "open each cell" rule above takes precedence because it runs first.
    (re.compile(r'\bopen\s+each\b', re.IGNORECASE), 'open excel'),
    (re.compile(r'\bopened excel\b', re.IGNORECASE), 'open excel'),
    (re.compile(r'\bclosed excel\b', re.IGNORECASE), 'close excel'),
    (re.compile(r'\bclothes excel\b', re.IGNORECASE), 'close excel'),
    # "open sound" — phonetic mishearing of "open excel"
    (re.compile(r'\bopen\s+sound\b', re.IGNORECASE), 'open excel'),
    # "open the chair" / "open exhale" — Whisper phonetic corruptions of "excel"
    # (log-confirmed on Realtek laptop mic: "chair" ≈ "excel", "exhale" ≈ "excel")
    (re.compile(r'\bopen\s+(?:the\s+)?(?:chair|exhale)\b', re.IGNORECASE), 'open excel'),
    # "Brightside" — phonetic corruption of "write cell" (real harness failure)
    (re.compile(r'^Brightside\b', re.IGNORECASE), 'write cell'),
    # Natural phrasing: "open my spreadsheet/workbook/file"
    (re.compile(r'\bopen\s+my\s+(?:spreadsheet|workbook|file)\b', re.IGNORECASE), 'open excel'),
    # "close exam" — phonetic; "close the workbook/file" — natural
    (re.compile(r'\bclose\s+exam\b', re.IGNORECASE), 'close excel'),
    (re.compile(r'\bclose\s+(?:the\s+)?(?:workbook|document)\b', re.IGNORECASE), 'close excel'),

    # ══ "add row" — phonetic / natural phrasing expansions ════════════════
    # "Adele John 90" → likely "add row John 90" (STT merge artifact)
    (re.compile(r'^adele\b', re.IGNORECASE), 'add row'),
    # "at Rowena", "at Rowland" → STT merged "row <name>" into one word
    (re.compile(r'^(at|ad|hat)\s+row\w+\b', re.IGNORECASE), 'add row'),
    # Natural phrasing: "add [name] with score [n]" → "add row [name] [n]"
    # Negative lookahead avoids matching "add column/row/sheet with score"
    (re.compile(r'\badd\s+(?!(?:column|row|sheet|formula)\b)(\w+)\s+with\s+score\s+(\d+)',
                re.IGNORECASE), r'add row \1 \2'),
    (re.compile(r'\badd\s+(?!(?:column|row|sheet|formula)\b)(\w+)\s+with\s+(\d+)',
                re.IGNORECASE), r'add row \1 \2'),

    # ══ "write cell" — additional phonetic variants ════════════════════════
    # "write side C1" → "write cell C1"
    (re.compile(r'\bwrite\s+side\b', re.IGNORECASE), 'write cell'),
    # "write healthy 212" → "write cell 212" (real STT: "healthy" for "cell")
    (re.compile(r'\b(write|rice|rise|right)\s+healthy\b', re.IGNORECASE), r'\1 cell'),
    # "write sell B3 90" → "write cell B3 90" (sell/cell phonetic swap)
    (re.compile(r'\b(write|rice|rise|right)\s+sell\b', re.IGNORECASE), r'\1 cell'),
    # "write set C1" → "write cell C1"  (STT: "set" for "cell" — log-confirmed)
    (re.compile(r'\bwrite\s+set\b', re.IGNORECASE), 'write cell'),
    # "write said C1" → "write cell C1"  (STT: "said" for "cell")
    (re.compile(r'\bwrite\s+said\b', re.IGNORECASE), 'write cell'),

    # ══ "calculate sum" — phonetic variants ═══════════════════════════════
    # "calculate sam column A" → "calculate sum column A"
    # Restricted to "calculate sam" to avoid clobbering names in other contexts.
    (re.compile(r'\bcalculate\s+sam\b', re.IGNORECASE), 'calculate sum'),

    # ══ Natural language trailing filler removal ═══════════════════════════
    # "save and close excel for me" → "save and close excel"
    (re.compile(r'\s+for\s+(?:me|you)\s*$', re.IGNORECASE), ''),

    # ══ "delete" misrecognitions ══════════════════════════════════════════
    (re.compile(r'\bdelete throw\b',   re.IGNORECASE), 'delete row'),
    (re.compile(r'\bdelete the row\b', re.IGNORECASE), 'delete row'),
    (re.compile(r'\bdell row\b',       re.IGNORECASE), 'delete row'),
    # "dalit roto 3" — real mic log: STT corruption of "delete row" on Realtek laptop mic
    (re.compile(r'^dalit\s+roto\b', re.IGNORECASE), 'delete row'),
    # "Delhi Row 2" — "delhi" ≈ "delete" (real STT log: Realtek laptop mic)
    (re.compile(r'^delhi\s+row\b', re.IGNORECASE), 'delete row'),
    # "Deli Roto" — "deli" ≈ "delete", "roto" ≈ "row" (real STT log)
    (re.compile(r'^deli\s+roto\b', re.IGNORECASE), 'delete row'),
    # "D-lit roll" — "D-lit" ≈ "delete", "roll" ≈ "row" (real STT log)
    (re.compile(r'^d-lit\s+roll\b', re.IGNORECASE), 'delete row'),
    # "Do you eat rotu" — phonetic corruption of "delete row" (real STT log)
    (re.compile(r'^do\s+you\s+eat\s+rotu?\b', re.IGNORECASE), 'delete row'),
    # "Do you like Colombia/Kalumbi" — STT corruption of "delete column B" (real mic log)
    (re.compile(r'^do\s+you\s+like\s+(?:colombia|kalumbi)\b', re.IGNORECASE), 'delete column b'),
    # "D late cut love be" — STT corruption of "delete column b" (real mic log)
    (re.compile(r'^d\s+late\s+cut\s+love\s+be\b', re.IGNORECASE), 'delete column b'),
    # "Lily call up me" — STT corruption of "delete column" (real mic log)
    (re.compile(r'^lily\s+call\s+up\s+me\b', re.IGNORECASE), 'delete column'),
    # "delete row two/three/..." → "delete row 2/3/..."
    # STT returns spoken digit words for single-digit row numbers.
    # Restricted to "delete/remove row" context to avoid clobbering names in add-row.
    (re.compile(
        r'\b(delete|remove)\s+row\s+(one|won|two|to|too|three|four|five|six|seven|eight|nine)\b',
        re.IGNORECASE,
    ), lambda m: f'{m.group(1).lower()} row {_ROW_NUM_WORDS[m.group(2).lower()]}'),

    # ══ "calculate" misrecognitions ═══════════════════════════════════════
    (re.compile(r'\bcalculate some\b', re.IGNORECASE), 'calculate sum'),

    # ══ Number words → digits ════════════════════════════════════════════
    # Google STT occasionally returns number words instead of digits.
    # Map round tens first (higher priority) then teens.
    (re.compile(r'\bninety\b',    re.IGNORECASE), '90'),
    (re.compile(r'\beighty\b',    re.IGNORECASE), '80'),
    (re.compile(r'\bseventy\b',   re.IGNORECASE), '70'),
    (re.compile(r'\bsixty\b',     re.IGNORECASE), '60'),
    (re.compile(r'\bfifty\b',     re.IGNORECASE), '50'),
    (re.compile(r'\bforty\b',     re.IGNORECASE), '40'),
    (re.compile(r'\bthirty\b',    re.IGNORECASE), '30'),
    (re.compile(r'\btwenty\b',    re.IGNORECASE), '20'),
    (re.compile(r'\bnineteen\b',  re.IGNORECASE), '19'),
    (re.compile(r'\beighteen\b',  re.IGNORECASE), '18'),
    (re.compile(r'\bseventeen\b', re.IGNORECASE), '17'),
    (re.compile(r'\bsixteen\b',   re.IGNORECASE), '16'),
    (re.compile(r'\bfifteen\b',   re.IGNORECASE), '15'),
    (re.compile(r'\bfourteen\b',  re.IGNORECASE), '14'),
    (re.compile(r'\bthirteen\b',  re.IGNORECASE), '13'),
    (re.compile(r'\btwelve\b',    re.IGNORECASE), '12'),
    (re.compile(r'\beleven\b',    re.IGNORECASE), '11'),
    (re.compile(r'\bten\b',       re.IGNORECASE), '10'),

    # ══ "column width" misrecognitions ════════════════════════════════════
    # "set column with A 20" — STT mishears "width" as "with".
    # Pattern requires a column letter + number after "with" (tight anchor)
    # so "create a column with score" is NOT rewritten.
    # Must run BEFORE the cell-ref normalizer so "A 20" spacing is intact.
    (re.compile(r'\bcolumn\s+with\s+([A-Za-z])\s*(\d+(?:\.\d+)?)\b', re.IGNORECASE),
     lambda m: f'column width {m.group(1)} {m.group(2)}'),
    # "Auto-fit column A" — STT sometimes inserts a hyphen; normalise so the
    # heuristic regex (\bauto\s*fit\b) matches correctly.
    (re.compile(r'\bauto\s*[-–]\s*fit\b', re.IGNORECASE), 'auto fit'),
    # "Columbee" → "column b" — phonetic: STT merges "column B" into one token.
    # Must run BEFORE the broad auto-fit rule so "Go Fit Columbee" → "go fit column b"
    # → "auto fit column B".
    (re.compile(r'\bcolumbee\b', re.IGNORECASE), 'column b'),
    # "Although fit column B" / "All the fit color B" / "Go Fit column b"
    # → "auto fit column <letter>"
    # Verb alternatives cover real STT garbles of "auto" confirmed in logs.
    # Column-word alternatives cover "color", "colour", "col" as STT garbles of "column".
    # Anchored to a trailing single column letter to prevent false positives.
    (re.compile(
        r'\b(?:auto|although|all(?:\s+the)?|go)\s*,?\s*fit\s+'
        r'(?:column|col|color|colour|colon)\s+([A-Za-z])\b',
        re.IGNORECASE),
     lambda m: f'auto fit column {m.group(1).upper()}'),

    # ══ "front size" / "front side" → "font size" ════════════════════════════
    # "set front size A2" / "increase front side I1" → "font size"
    # STT mishears "font" as "front" and "size" as "side".
    # Lookahead ensures a cell/range/row/column target follows — prevents
    # global front→font replacement in other contexts.
    (re.compile(
        r'\bfront\s+(?:size|side)\b(?=.*(?:row\s+\d+|column\s+[A-Za-z]+|[A-Za-z]+\d+))',
        re.IGNORECASE,
    ), 'font size'),

    # ══ Filter command: insert "=" before the value ═══════════════════════
    # The cell-ref concatenation rule below turns "B 90" → "B90", making
    # "filter column B 90" unparseable.  Insert "=" first so the letter and
    # the value are separated by an operator, which the existing heuristic
    # already handles and which the concatenation rule won't touch.
    # "filter column B 90"  → "filter column B = 90"
    # "filter B 90"         → "filter B = 90"
    # Must run BEFORE the letter+digit rule immediately below.
    (re.compile(r'(\bfilter\b.{0,60}?\b([A-Za-z]))\s+(\d+(?:\.\d+)?)\b',
                re.IGNORECASE), r'\1 = \3'),
    # Already-merged case: "filter column B90" → "filter column B = 90"
    # Handles Whisper output that skips the space between column letter and value.
    (re.compile(r'(\bfilter\b.{0,60}?\bcolumn\s+)([A-Za-z])(\d+(?:\.\d+)?)\b',
                re.IGNORECASE), r'\1\2 = \3'),

    # ══ Cell reference normalization ════════════════════════════════════════
    # Must run AFTER number-word rules so "c ten" → "c 10" → "c10".
    # "write cell c 2 score" → "write cell c2 score"  (letter + space + digits)
    (re.compile(r'\b([A-Za-z])\s+(\d+)\b'), r'\1\2'),
    # Unambiguous single-digit number words in cell-ref position:
    # "c one/won" → "c1",  "c two/too" → "c2",  etc.
    (re.compile(r'\b([A-Za-z])\s+(?:one|won)\b',  re.IGNORECASE), r'\g<1>1'),
    (re.compile(r'\b([A-Za-z])\s+(?:two|too)\b',  re.IGNORECASE), r'\g<1>2'),
    (re.compile(r'\b([A-Za-z])\s+three\b',         re.IGNORECASE), r'\g<1>3'),
    (re.compile(r'\b([A-Za-z])\s+five\b',          re.IGNORECASE), r'\g<1>5'),
    (re.compile(r'\b([A-Za-z])\s+six\b',           re.IGNORECASE), r'\g<1>6'),
    (re.compile(r'\b([A-Za-z])\s+seven\b',         re.IGNORECASE), r'\g<1>7'),
    (re.compile(r'\b([A-Za-z])\s+eight\b',         re.IGNORECASE), r'\g<1>8'),
    (re.compile(r'\b([A-Za-z])\s+nine\b',          re.IGNORECASE), r'\g<1>9'),
    # "to" and "for" are too ambiguous globally — restrict to explicit cell context.
    # "write cell c to 5" → "write cell c2 5"
    (re.compile(r'\bcell\s+([A-Za-z])\s+to\b',    re.IGNORECASE), r'cell \g<1>2'),
    (re.compile(r'\bcell\s+([A-Za-z])\s+for\b',   re.IGNORECASE), r'cell \g<1>4'),
    # "go to set Me 3" — STT appends lowercase noise to the column letter
    # (e.g. "B" → "Me", "M" → "Me").  Take only the first character as the column.
    # Must run BEFORE the clean-ref rule below so "Me 3" is matched before "B3".
    (re.compile(r'\bgo\s+to\s+set\s+([A-Za-z])[a-z]+\s+(\d+)\b', re.IGNORECASE),
     lambda m: f'go to cell {m.group(1).upper()}{m.group(2)}'),
    # "go to set <cell>" → "go to cell <cell>" (real STT: "go to set B 3")
    # Runs after cell-ref spacing so "B 3" is already "B3" when this fires.
    (re.compile(r'\bgo\s+to\s+set\s+([A-Za-z]\d+)\b', re.IGNORECASE), r'go to cell \1'),
    # "go to free" → "go to F3" (real STT: "free" as phonetic for cell ref "F3")
    # Restricted to exact phrase to avoid corrupting "go to freedom" or similar.
    (re.compile(r'\bgo\s+to\s+free\b', re.IGNORECASE), 'go to F3'),
    # "go to sad" → "go to cell"  (real STT: "go to sad B 3" for "go to cell B3")
    # "sad" is a phonetic corruption of "cell"; restricted to "go to sad" context.
    (re.compile(r'\bgo\s+to\s+sad\b', re.IGNORECASE), 'go to cell'),
    # "some B2 to B5" → "sum B2 to B5" — "some" as phonetic for "sum".
    # Lookahead requires a cell ref immediately after so "add some values"
    # and "buy some groceries" are untouched.
    (re.compile(r'\bsome\b(?=\s+[A-Za-z]\d+\b)', re.IGNORECASE), 'sum'),
    # "some b2b5 into c5" — "some" before a merged-range token (two cell refs concatenated
    # with no space, e.g. "b2b5").  STT drops the colon and merges refs into one token.
    # Must run BEFORE the range normalization rule so "sum b2b5 into c5" → "sum B2:B5 into C5".
    (re.compile(r'\bsome\b(?=\s+[A-Za-z]+\d+[A-Za-z]+\d+)', re.IGNORECASE), 'sum'),

    # ══ "cell 810" → "cell A10" ═══════════════════════════════════════════
    # STT mishears the letter "A" as the digit "8" in cell references, producing
    # "810" for "A10".  Restricted to explicit "cell" keyword context to avoid
    # corrupting scores or other numeric values that happen to contain "810".
    (re.compile(
        r'\b((?:delete|write|read|clear|go\s+to)\s+cell\s+)810\b',
        re.IGNORECASE,
    ), lambda m: f'{m.group(1)}A10'),

    # ══ "make [cell]-[color]" de-hyphenation ══════════════════════════════════
    # "Make a3-orange" (STT hyphenates the cell ref and color word after number-word
    # + cell-ref normalization joins "a three" → "a3").  Must run AFTER cell-ref
    # normalization so the cell ref is already compact.
    (re.compile(
        r'\bmake\s+([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)-([A-Za-z]+)\b',
        re.IGNORECASE,
    ), lambda m: f'make {m.group(1)} {m.group(2)}'),

    # ══ "set font [cell] to [n]" / "set font [cell] [n]" → "set font size [cell] [n]"
    # STT drops the "size" keyword: "set font A1 to 20" / "set font A1 20".
    # Guard: (?!size\b) prevents double-application when "size" is already present.
    # Must run AFTER cell-ref normalization so "A 1" is already "A1".
    (re.compile(
        r'\bset\s+font\s+(?!size\b)([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\s+(?:to\s+)?(\d+)\b',
        re.IGNORECASE,
    ), lambda m: f'set font size {m.group(1)} {m.group(2)}'),
    # "set font [cell]" with no trailing number — slot-gate will ask for the missing size.
    (re.compile(
        r'\bset\s+font\s+(?!size\b)([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\b',
        re.IGNORECASE,
    ), lambda m: f'set font size {m.group(1)}'),

    # ══ "add row" — additional phonetic corruptions ═══════════════════════
    # "adroit John 90" — different STT path from "adele" (also maps to add row)
    (re.compile(r'^adroit\b', re.IGNORECASE), 'add row'),
    # "atro John 90" — real mic log: STT returned "atro John 90 at row John"
    # "atro" is not a real English word; safe to correct unconditionally.
    (re.compile(r'^atro\b', re.IGNORECASE), 'add row'),
    # "adro John 90" — real mic log: STT corruption of "add row" on Realtek laptop mic
    (re.compile(r'^adro\b', re.IGNORECASE), 'add row'),
    # "add road John 90" / "add roads on 90" — STT corruption of "add row".
    # "road"/"roads" never appear as valid Excel commands.
    (re.compile(r'\badd roads?\b', re.IGNORECASE), 'add row'),
    # "at Rolex on 90" / "at Roald John 90" — STT merges "add row" into "at Ro..."
    # where the mangled word starts with "ro" but NOT "row" (Rowena/Rowland are
    # handled by the existing ^(at|ad|hat)\s+row\w+\b rule above).
    # After this correction: "at Rolex on 90" → "add row on 90" (blocked by slot
    # check — bare number); "at Roald John 90" → "add row John 90" (correct!).
    (re.compile(r'^(at|ad|hat)\s+ro(?!w)[a-z]+\b', re.IGNORECASE), 'add row'),

    # ══ "bold" phonetic variants ═══════════════════════════════════════════
    # "make A1 bowl/ball/bolt/board" → "make A1 bold"
    # STT mishears "bold" as phonetically-similar common English words.
    # Anchored to make/format + cell-ref so "make a bowl of soup" is NOT matched
    # (no cell ref follows "bowl").  Runs AFTER the cell-ref normalizer so
    # "a one" has already become "a1".
    (re.compile(r'\b(make|format)\s+([A-Za-z]\d+(?::[A-Za-z]\d+)?)\s+(bowl|ball|bolt|board)\b',
                re.IGNORECASE),
     lambda m: f'{m.group(1)} {m.group(2)} bold'),
    # "make header bold" phonetic variants — "head" ≈ "header"; bolt/board/ball/bowl ≈ "bold"
    # "Make head the bolt" / "Make head up board" → "make header bold"
    (re.compile(r'\bmake\s+(?:a\s+)?head(?:er)?\s+(?:(?:the|up|of|a|an)\s+)?(?:bold|bolt|board|ball|bowl)\b',
                re.IGNORECASE),
     'make header bold'),

    # ══ Sheet command phonetic corrections ══════════════════════════════════
    # "speech sheet" → "switch sheet"  (real STT: "speech sheet sheet 2" for "switch sheet Sheet2")
    # Restricted to "speech sheet" so bare "speech" elsewhere is unaffected.
    (re.compile(r'\bspeech\s+sheet\b', re.IGNORECASE), 'switch sheet'),
    # "switch sheet sheet N" → "switch sheet SheetN"  (double "sheet" + bare digit from STT)
    # Fires after the speech→switch rule above so both "speech sheet sheet 2"
    # and "switch sheet sheet 2" are normalised to "switch sheet Sheet2".
    (re.compile(r'\bswitch\s+sheet\s+sheet\s+(\d+)\b', re.IGNORECASE),
     lambda m: f'switch sheet Sheet{m.group(1)}'),
    # "switch cheat sheet" — real STT: "switch cheat sheet 2" for "switch sheet Sheet2"
    # "cheat sheet" alone is unaffected (pattern requires "switch" before it).
    (re.compile(r'\bswitch\s+cheat\s+sheet\b',  re.IGNORECASE), 'switch sheet'),
    # "switch shit shit" — real STT: "shit" as phonetic for "sheet"
    # Only fires when "switch" is the command verb to avoid false positives.
    (re.compile(r'\bswitch\s+sh[i!]t\b', re.IGNORECASE), 'switch sheet'),
    # "create/delete/rename/list shit/shet" → "[verb] sheet"
    # Contextual: only fires when a sheet command verb immediately precedes the
    # garbled word, so "add shit cooking" (add_row garble) is NOT rewritten.
    (re.compile(r'\b(create|delete|rename|list)\s+sh[i!]t\b', re.IGNORECASE), r'\1 sheet'),
    (re.compile(r'\b(create|delete|rename|list)\s+shet\b',    re.IGNORECASE), r'\1 sheet'),
    # Normalise "go to sheet" / "navigate to sheet" → "switch sheet" so the
    # heuristic parser sees a consistent prefix.
    (re.compile(r'\bgo\s+to\s+sheet\b',        re.IGNORECASE), 'switch sheet'),
    (re.compile(r'\bnavigate\s+to\s+sheet\b',  re.IGNORECASE), 'switch sheet'),
    (re.compile(r'\bswitch\s+to\s+sheet\b',    re.IGNORECASE), 'switch sheet'),
    # "switch sheet to <name>" → "switch sheet <name>"
    # Real STT: "switch sheet to student" — the "to" is filler, not a slot separator.
    # Must come AFTER "switch to sheet" rule above.
    (re.compile(r'\bswitch\s+sheet\s+to\b',    re.IGNORECASE), 'switch sheet'),
    # "new sheet" → "create sheet"  (natural phrasing)
    (re.compile(r'\bnew\s+sheet\b',             re.IGNORECASE), 'create sheet'),
    # "show sheets" / "list sheets" → canonical form
    (re.compile(r'\bshow\s+(?:all\s+)?sheets?\b', re.IGNORECASE), 'list sheets'),
    # "combine cells" → canonical "merge cells"
    (re.compile(r'\bcombine\s+(?:the\s+)?cells?\b', re.IGNORECASE), 'merge cells'),

    # ══ "merge cells" STT misrecognitions ════════════════════════════════════
    # "much [cell/sir/sad/sell]" — "much" ≈ "merge", noun ≈ "cells" phonetically.
    (re.compile(r'\bmuch\s+(?:cell|cells|sir|sad|sell)\b', re.IGNORECASE), 'merge cells'),
    # "mod/mode cell" — "mod"/"mode" ≈ "merge"
    (re.compile(r'\bmod\s+(?:cell|cells)\b',  re.IGNORECASE), 'merge cells'),
    (re.compile(r'\bmode\s+(?:cell|cells)\b', re.IGNORECASE), 'merge cells'),
    # "merge sir/sad/sell" — second word ≈ "cells" (STT phonetic corruption)
    (re.compile(r'\bmerge\s+(?:the\s+)?(?:sir|sad|sell)\b', re.IGNORECASE), 'merge cells'),
    # "merge cells A1C1" → "merge cells A1:C1"
    # STT omits the colon and merges adjacent cell refs into one token.
    # Runs AFTER the verb-fix rules above so "much cell A1C1" → "merge cells A1C1"
    # (verb fix) → "merge cells A1:C1" (this rule) in two sequential passes.
    (re.compile(r'\b(merge\s+cells?\s+)([A-Za-z]+\d+)([A-Za-z]+\d+)\b', re.IGNORECASE),
     lambda m: f'{m.group(1)}{m.group(2).upper()}:{m.group(3).upper()}'),

    # ══ "unmerge cells" STT misrecognitions ══════════════════════════════════
    # STT corrupts "unmerge" as "and merge" or "and much", and doubles it
    # when the wake-word loop re-fires ("and merge and merge").
    # CRITICAL: without this rule, "and merge and merge" reaches Groq with
    # "merge" in _COMMAND_VERBS bypassing the ambiguous-transcript guard,
    # and Groq guesses open_file/write_cell from the garbled text.
    #
    # Pattern: consume ALL leading "and merge" / "and much" repetitions at once.
    # Examples:
    #   "and merge A1:C1"           → "unmerge A1:C1"
    #   "and much A1C1"             → "unmerge A1C1"  (cell-ref split rule below fixes colon)
    #   "and merge and merge"       → "unmerge "      → slot check clarifies
    #   "and merge and merge A1:C1" → "unmerge A1:C1" → heuristic produces unmerge_cells
    #   "please hand me and merge cell A1B1" → "unmerge cell A1B1"
    #   "hand me and merge cell A1B1"        → "unmerge cell A1B1"
    # Filler-prefix variants — strip "please [hand me]" / "hand me" before "and merge/much".
    # Must precede the bare ^(?:and\s+...) rule so the longer prefix is consumed first.
    (re.compile(r'^please\s+(?:hand\s+me\s+)?and\s+(?:merge|much)\b', re.IGNORECASE), 'unmerge'),
    (re.compile(r'^hand\s+me\s+and\s+(?:merge|much)\b', re.IGNORECASE), 'unmerge'),
    # Bare leading repetitions: consume ALL "and merge/much" groups at once.
    (re.compile(r'^(?:and\s+(?:merge|much)\b\s*)+', re.IGNORECASE), 'unmerge '),
    # "unmerge select C1 D1" → "unmerge cells C1 D1"
    # "select" is a phonetic corruption of "cells" in unmerge context (STT artifact).
    # Must run BEFORE the colon-insertion rule so the resulting "cells" prefix is canonical.
    (re.compile(r'\bunmerge\s+select\b', re.IGNORECASE), 'unmerge cells'),
    # "unmerge A1C1" / "unmerge cells A1C1" → insert missing colon between cell refs.
    # Mirrors the same fix for merge cells (P36).
    (re.compile(r'\b(unmerge\s+(?:cells?\s+)?)([A-Za-z]+\d+)([A-Za-z]+\d+)\b', re.IGNORECASE),
     lambda m: f'{m.group(1)}{m.group(2).upper()}:{m.group(3).upper()}'),

    # "free hada" — real mic log: STT corruption of "freeze header" on Realtek laptop mic
    (re.compile(r'^free\s+hada\b', re.IGNORECASE), 'freeze header'),
    # "freeze top row" → "freeze header" (freeze row 1, cell=A2)
    # "freeze row(s)" without "top" → "freeze panes" (cell=B1)
    (re.compile(r'\bfreeze\s+(?:the\s+)?top\s+rows?\b', re.IGNORECASE), 'freeze header'),
    (re.compile(r'\bfreeze\s+(?:the\s+)?rows?\b', re.IGNORECASE), 'freeze panes'),
    (re.compile(r'\bunfreeze\s+(?:the\s+)?rows?\b',           re.IGNORECASE), 'unfreeze panes'),
    # ══ "freeze panes" / "unfreeze panes" filler-prefix and phonetic variants ══════
    # "and freeze pen/pens/paints" → "unfreeze panes"
    # STT splits "un-" as "and/in" and hears "panes" as "pen/pens/paints/pants".
    # Must run BEFORE the plain "and freeze → freeze" strip so the unfreeze
    # interpretation wins when the panes phonetic is also garbled.
    (re.compile(
        r'^(?:and|in)\s+freeze\s+(?:pan(?:ts?|s)?|pen(?:s|ts?)?|paint(?:s)?)\b',
        re.IGNORECASE,
    ), 'unfreeze panes'),
    # "and freeze pan/pen/pants" — leading filler before "freeze" (real mic log)
    (re.compile(r'^(?:and|i\'?m|i\s+am)\s+(?=freeze\b)', re.IGNORECASE), ''),
    # "freeze pan/pants/pens" → "freeze panes" — STT phonetic corruption
    (re.compile(r'\bfreeze\s+pan(?:ts?|s)?\b', re.IGNORECASE), 'freeze panes'),
    (re.compile(r'\bfreeze\s+pen\b',            re.IGNORECASE), 'freeze panes'),

    # ══ Sort direction STT misrecognitions ════════════════════════════════════
    # Whisper corrupts "descending" inside sort commands.  Context guard:
    # both "sort" and "by" must appear earlier so corrections don't fire on
    # unrelated sentences.
    #
    # "Sort by column A decently."  → "Sort by column A descending."
    # "sort by B descently"         → "sort by B descending"
    (re.compile(r'(\bsort\b.{0,60}\bby\b.{0,20})\s+(?:decently|descently)\b',
                re.IGNORECASE), r'\1 descending'),
    # "the sending" / "be sending" / "this ending" → "descending"
    # (phonetic: de-scend-ing ≈ the-send-ing / be-send-ing / this-end-ing)
    (re.compile(r'(\bsort\b.{0,60}\bby\b.{0,20})\s+'
                r'(?:the\s+sending|be\s+sending|this\s+ending)\b',
                re.IGNORECASE), r'\1 descending'),

    # ══ Average/sum range normalization (STT colon drop) ════════════════════
    # STT often drops the colon between cell refs or hears two separate tokens
    # where the user said a range like "B2:B5".  These rules normalize the
    # transcript so the heuristic parser always sees the canonical colon form.
    # Must run AFTER cell-ref normalization so refs are already compact.
    # Lookahead enforces a "into/to [cell ref]" destination to avoid firing on
    # unrelated text.
    #
    # Merged form: "average B2B7 into C7" → "average B2:B7 into C7"
    # Also handles sum: "sum B2B5 into C6" → "sum B2:B5 into C6"
    (re.compile(
        r'\b(average|sum)\s+([A-Za-z]+\d+)([A-Za-z]+\d+)\b'
        r'(?=\s+(?:in(?:to)?|to)\s+[A-Za-z]+\d+)',
        re.IGNORECASE,
    ), lambda m: f'{m.group(1)} {m.group(2)}:{m.group(3)}'),
    # Space-separated form: "average B2 B5 into C7" → "average B2:B5 into C7"
    # The comma form "average B2, B5 into C7" collapses to this after the
    # comma→space rule runs earlier in the list.
    (re.compile(
        r'\b(average|sum)\s+([A-Za-z]+\d+)\s+([A-Za-z]+\d+)\b'
        r'(?=\s+(?:in(?:to)?|to)\s+[A-Za-z]+\d+)',
        re.IGNORECASE,
    ), lambda m: f'{m.group(1)} {m.group(2)}:{m.group(3)}'),

    # "format B2 to B10 as currency" → "format B2:B10 as currency"
    # "format C2 through C8 as percentage" → "format C2:C8 as percentage"
    # Voice speakers say "to" or "through" instead of ":" for ranges.
    (re.compile(
        r'\bformat\s+([A-Za-z]+\d+)\s+(?:to|through)\s+([A-Za-z]+\d+)\b',
        re.IGNORECASE,
    ), r'format \1:\2'),

    # ══ "sort" verb STT mishears ═══════════════════════════════════════════════
    # "swap/sword/short by B" → "sort by B" at utterance start.
    # Both anchored to ^ and require trailing "by\b" so "short by the way" is safe.
    (re.compile(r'^(?:swap|sword)\s+by\b', re.IGNORECASE), 'sort by'),
    (re.compile(r'^short\s+by\b', re.IGNORECASE), 'sort by'),
    # "start by B ascending/descending" — tight lookahead prevents matching common
    # English "start by doing X" constructions.
    (re.compile(
        r'^start\s+by\b(?=\s+\S+\s+(?:ascending|descending|asc|desc)\b)',
        re.IGNORECASE,
    ), 'sort by'),

    # ══ "color" → "column" disambiguation in column-operation context ═══════════
    # STT mishears "column" as "color" in sort/filter/width commands.
    # "sort by color B ascending" → "sort by column B ascending"
    # "filter color B = 40"       → "filter column B = 40"
    # "set color width B 20"      → "set column width B 20"
    # Tight prefix anchors: only fires when "color" immediately follows a column-operation
    # keyword, NOT a formatting verb — "color B2 red" (fill command) is UNCHANGED.
    (re.compile(r'\b(sort\s+by)\s+color\b', re.IGNORECASE), r'\1 column'),
    (re.compile(r'\b(filter)\s+color\b', re.IGNORECASE), r'\1 column'),
    (re.compile(r'\b(set)\s+color\s+width\b', re.IGNORECASE), r'\1 column width'),

    # ══ Column letter "B" STT mishears ════════════════════════════════════════
    # Whisper mishears the spoken letter "B" as "be" or "bee" (phonetic names).
    # Lambda checks the last character of the captured word: lowercase 'e' means
    # voice STT output; uppercase 'E' means the user typed "BE" or "BEE" → no change.
    # "column be" → "column B"  (voice); "column BE" → unchanged (typed).
    (re.compile(r'\b(column|col)\s+(be|bee)\b', re.IGNORECASE),
     lambda m: 'column B' if m.group(2)[-1].islower() else m.group(0)),
    # "by be ascending/descending" → "by B ascending/descending" (sort context).
    # Lookahead enforces an explicit direction word to limit false positives.
    (re.compile(r'\bby\s+(be|bee)\b(?=\s+(?:ascending|descending|asc|desc)\b)', re.IGNORECASE),
     lambda m: 'by B' if m.group(1)[-1].islower() else m.group(0)),
    # "color/fill be to [color-word]" → "color/fill column B [color-word]"
    # "be" (voice, lowercase 'e') before a spurious "to" + color → expand to full column form.
    # Prevents LLM from inventing broad ranges like A1:D10 from the garbled input.
    (re.compile(r'\b(color|colour|fill)\s+(be)\s+to\s+(?=[a-z])', re.IGNORECASE),
     lambda m: f'{m.group(1)} column B ' if m.group(2)[-1].islower() else m.group(0)),

    # ══ "bold" command: additional STT mishears ═══════════════════════════════
    # "board/ball/bolt A1" → "bold A1" at utterance start (no make/format prefix).
    # Positive lookahead ensures a cell ref follows so standalone "board"/"ball" is safe.
    # Runs AFTER cell-ref normalization so "a one" is already "a1".
    (re.compile(r'^(?:board|ball|bolt)\b(?=\s+[A-Za-z]\d+)', re.IGNORECASE), 'bold'),
    # "format/make row N board/ball/bolt" → "format/make row N bold"
    # Extends the existing (make|format) [cell_ref] bold-word rule to row/column targets.
    (re.compile(
        r'\b(format|make)\s+((?:row|column)\s+\S+)\s+(?:ball|board|bolt|bowl)\b',
        re.IGNORECASE,
    ), lambda m: f'{m.group(1)} {m.group(2)} bold'),

    # ══ Font-size increase/decrease STT mishears ═════════════════════════════════
    # "in grease font size B2" / "in greece font size B2" → "increase font size B2"
    # STT splits "in-crease" as "in grease/greece" at the word boundary.
    # Positive lookahead for "font size" or bare "size" restricts to font commands.
    (re.compile(r'\bin\s+gre(?:ase|ece)\b(?=\s+(?:font[\s-])?size\b)', re.IGNORECASE), 'increase'),
    # "degrease font size B2" → "decrease font size B2"
    (re.compile(r'\bdegrease\b(?=\s+(?:font[\s-])?size\b)', re.IGNORECASE), 'decrease'),

    # ══ "sum" command: additional STT mishears ════════════════════════════════
    # "sam B2:B5" / "sam column B" → "sum ..." — "sam" is a phonetic variant of "sum".
    # Restricted to cell-ref or "column" context to avoid clobbering the name "Sam".
    (re.compile(r'\bsam\b(?=\s+(?:[A-Za-z]\d+|column)\b)', re.IGNORECASE), 'sum'),
    # "sun column B" → "sum column B" — "sun" is a phonetic variant of "sum".
    # Restricted to cell-ref or "column" context to avoid clobbering the noun "sun".
    (re.compile(r'\bsun\b(?=\s+(?:[A-Za-z]\d+|column)\b)', re.IGNORECASE), 'sum'),
    # "some column B" → "sum column B" (the existing "some" rule covers cell refs only).
    (re.compile(r'\bsome\b(?=\s+column\b)', re.IGNORECASE), 'sum'),

    # ══ "fill" STT mishears at utterance start ════════════════════════════════
    # "fear/feel/field row 2 with red" → "fill row 2 with red"
    # STT mishears "fill" as "fear"/"feel"/"field"/"pure"/"pier"/"fur".
    # Row target: lookahead requires "row" + digit to avoid rewriting conversational phrases.
    (re.compile(r'^(?:fear|feel|field|pure|pier|fur)\s+row\b(?=\s+\d+\b)', re.IGNORECASE), 'fill row'),
    # Column target: "feel column B blue" → "fill column B blue"
    (re.compile(r'^(?:fear|feel|field)\s+column\b', re.IGNORECASE), 'fill column'),
    # Cell-ref target: "feel A1 green" → "fill A1 green"
    (re.compile(r'^(?:fear|feel|field)\s+([A-Za-z]\d+)\b', re.IGNORECASE),
     lambda m: f'fill {m.group(1)}'),

    # ══ "row" STT mishears in fill / color / format context ══════════════════
    # "fill/color/format/highlight roll/role/roto [N]" → "... row [N]"
    # Enables "fill roll 2 with red" → "fill row 2 with red".
    # Also converts "fill a roto with red" → "fill a row with red" so the
    # slot-gate can ask for the missing row number.
    (re.compile(
        r'\b(fill|color|colour|format|highlight)\s+(?:a\s+)?(?:roll|role|roto)\b',
        re.IGNORECASE,
    ), lambda m: f'{m.group(1)} row'),
    # "fill/color/format/highlight row two/to/too/..." → "... row 2/3/..."
    # Extends the existing delete/remove digit-word rule to other row commands.
    (re.compile(
        r'\b(fill|color|colour|format|highlight)\s+row\s+'
        r'(one|won|two|to|too|three|four|five|six|seven|eight|nine)\b',
        re.IGNORECASE,
    ), lambda m: f'{m.group(1)} row {_ROW_NUM_WORDS[m.group(2).lower()]}'),

    # ══ "add column" plural normalisation ════════════════════════════════════
    # "add columns Name Score Age" → "add column Name Score Age"
    # STT/user sometimes says "columns" (plural).  Normalise so intent detection
    # and slot-checking see the canonical singular form.
    (re.compile(r'\badd\s+columns\b', re.IGNORECASE), 'add column'),
]

# ── Pass 2: command-prefix normalization ──────────────────────────────────
# Matches the first 1-3 words of the transcript and replaces them with the
# canonical command prefix.  Applied AFTER the word-level rules so that
# pass-1 corrections (e.g. "rojo" → "row") simplify these patterns.
# Format: (compiled_regex_matching_prefix, canonical_replacement_prefix)
_COMMAND_PREFIX_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "add row" family — catches residual variants after pass 1
    # "at all" → "add row"  (real STT: "at all adult 19" for "add row John 90")
    (re.compile(r'^(add|at|ad|hat)\s+(row|role|roll|ro|all|raw)\b', re.IGNORECASE), 'add row'),
    # "insert row" without positional keywords → "add row" ONLY when followed
    # by alphabetic content (a person's name / data).
    # "insert row 3" and "insert row before 3" keep their insert_row_at intent;
    # bare "insert row" goes to the slot-check gate for clarification.
    (re.compile(r'^insert\s+row\b(?=\s+[A-Za-z])(?!\s+(?:at|before|after)\b)', re.IGNORECASE), 'add row'),
    # "write cell" family  (includes "price of" — real STT: "price of C2 fan")
    (re.compile(r'^(write|rice|rise|right|wright|price)\s+(cell|at|so|sel|to|the cell|of|side)\b', re.IGNORECASE), 'write cell'),
    # "Brightside" → "write cell"  (real STT: "Brightside C 212" for "write cell C2 12")
    (re.compile(r'^brightside\b', re.IGNORECASE), 'write cell'),
    # "write [CellRef]" — "rice C1" → "write C1" (cell word already implied)
    (re.compile(r'^(rice|rise|right|wright)\s+([A-Za-z]\d+)\b', re.IGNORECASE),
     lambda m: f'write {m.group(2)}'),  # type: ignore[arg-type]
    # "delete row" family
    (re.compile(r'^(delete|delet|dell|remove)\s+(row|role|roll)\b', re.IGNORECASE), 'delete row'),
    # "open excel" family — includes phonetic ("sound") and natural ("workbook").
    # NOTE: "file" is intentionally excluded: "open file [name]" routes to open_file,
    # not open_excel, so the filename is preserved and passed to the file-open action.
    (re.compile(r'^(open|opal|pen)\s+(sound|excel|spreadsheet|workbook)\b', re.IGNORECASE), 'open excel'),
    # bare "excel" alone → "open excel"
    (re.compile(r'^excel\s*$', re.IGNORECASE), 'open excel'),
    # "close excel" family — includes phonetic ("exam") and natural ("workbook")
    (re.compile(r'^(close|clothes)\s+(exam|excel|workbook|file|spreadsheet)\b', re.IGNORECASE), 'close excel'),
    # "create excel" family
    (re.compile(r'^(create|creat|great)\s+(excel|file|spreadsheet)\b', re.IGNORECASE), 'create excel'),
    # "calculate sum / average" family
    (re.compile(r'^(calculate|calc|count)\s+(sum|total|some)\b', re.IGNORECASE), 'calculate sum'),
    (re.compile(r'^(calculate|calc)\s+(average|avg|avr)\b', re.IGNORECASE), 'calculate average'),
    # "switch sheet" / "go to sheet" family
    (re.compile(r'^(switch|go)\s+(?:to\s+)?sheet\b', re.IGNORECASE), 'switch sheet'),
    # "sweet" → "switch sheet" (phonetic: "sweet city" → "switch sheet City")
    (re.compile(r'^sweet\b', re.IGNORECASE), 'switch sheet'),
    # "speech sheet" → "switch sheet"  (prefix guard catches any Pass-1 misses)
    (re.compile(r'^speech\s+sheet\b', re.IGNORECASE), 'switch sheet'),
    # "create sheet" family
    (re.compile(r'^(create|add|make)\s+(?:a\s+)?sheet\b', re.IGNORECASE), 'create sheet'),
    # "delete sheet" / "remove sheet" family
    (re.compile(r'^(delete|remove)\s+(?:the\s+)?sheet\b', re.IGNORECASE), 'delete sheet'),
    # "rename sheet" family
    (re.compile(r'^rename\s+(?:the\s+)?sheet\b', re.IGNORECASE), 'rename sheet'),
    # "list sheets" / "show sheets" family
    (re.compile(r'^(list|show)\s+(?:all\s+)?sheets?\b', re.IGNORECASE), 'list sheets'),
    # "undo" family
    (re.compile(r'^undo\b', re.IGNORECASE), 'undo'),
    # "I do" / "unto" / "and do" → "undo" — STT phonetic corruptions of "undo".
    # Restricted to standalone utterances so "I do the work" is never touched.
    (re.compile(r'^i\s+do\s*$', re.IGNORECASE), 'undo'),
    (re.compile(r'^unto\s*$', re.IGNORECASE), 'undo'),
    (re.compile(r'^and\s+do\s*$', re.IGNORECASE), 'undo'),
    # "merge cells" family
    (re.compile(r'^(merge|combine)\s+(?:the\s+)?cells?\b', re.IGNORECASE), 'merge cells'),
    # "freeze / unfreeze panes"
    (re.compile(r'^freeze\s+(?:panes?|rows?)\b', re.IGNORECASE), 'freeze panes'),
    (re.compile(r'^unfreeze\s+(?:panes?|rows?)\b', re.IGNORECASE), 'unfreeze panes'),
    # "rename file"
    (re.compile(r'^rename\s+(?:the\s+)?(?:file|excel|spreadsheet|workbook)\b', re.IGNORECASE), 'rename file'),
]


# ---------------------------------------------------------------------------
# 5c. Voice recovery helpers
# ---------------------------------------------------------------------------

# Leading filler words that STT inserts between "add row" and the actual values.
# Example: "At role of Jenny 18" → corrected to "add row of jenny 18" → strip "of"
_ADD_ROW_PAYLOAD_FILLER: frozenset[str] = frozenset({
    'of', 'a', 'an', 'the', 'at', 'least', 'please', 'and', 'with',
    'just', 'some', 'is', 'use', 'i', "i'll", 'ill',
})

# Leading filler words that appear between a cell ref and the actual value.
# Example: "write cell C1 of 18" → strip "of" → "write cell C1 18"
_WRITE_CELL_VALUE_FILLER: frozenset[str] = frozenset({
    'of', 'a', 'an', 'the', 'with', 'to', 'is', 'equals', 'equal',
})


def recover_add_row_voice(text: str) -> str | None:
    """Strip leading filler words from an add-row payload.

    Returns the cleaned transcript if fillers were removed and at least one
    name-like token remains, otherwise ``None`` (no change needed).

    Example::

        recover_add_row_voice('add row of jenny 18')  → 'add row jenny 18'
        recover_add_row_voice('add row jenny 18')     → None  (no change)
        recover_add_row_voice('add row of 18')        → None  (no name token)
    """
    m = re.match(r'^add\s+row\s+', text, re.IGNORECASE)
    if not m:
        return None
    payload = text[m.end():].lstrip(' ,.')
    words = payload.split()
    original_count = len(words)
    while words and words[0].lower() in _ADD_ROW_PAYLOAD_FILLER:
        words.pop(0)
    if not words or len(words) == original_count:
        return None
    # For voice recovery, require exactly 2 words: <name> <value> to avoid unsafe execution
    if len(words) != 2:
        return None
    # Require at least one name-like alpha token so we don't strip too aggressively.
    has_name = any(
        re.match(r'^[a-z]{2,}$', w.lower()) and w.lower() not in _ADD_ROW_PAYLOAD_FILLER
        for w in words
    )
    if not has_name:
        return None
    return 'add row ' + ' '.join(words)


def recover_write_cell_voice(text: str) -> str | None:
    """Strip leading filler words from the value portion of a write-cell command.

    Returns the cleaned transcript if fillers were removed, otherwise ``None``.

    Example::

        recover_write_cell_voice('write cell c1 of 18')  → 'write cell c1 18'
        recover_write_cell_voice('write cell c1 18')     → None  (no change)
    """
    m = re.match(r'^(write\s+cell\s+[A-Za-z]+\d+)\s+', text, re.IGNORECASE)
    if not m:
        return None
    prefix = m.group(1)
    rest = text[m.end():].strip()
    value_words = rest.split()
    if not value_words:
        return None
    original_count = len(value_words)
    while value_words and value_words[0].lower() in _WRITE_CELL_VALUE_FILLER:
        value_words.pop(0)
    if not value_words or len(value_words) == original_count:
        return None
    return prefix + ' ' + ' '.join(value_words)


def correct_transcript(text: str) -> str:
    """Apply two-pass rule-based correction for common STT misrecognitions.

    Pass 1 — word/phrase regex substitutions across the whole transcript.
    Pass 2 — command-prefix normalization on the first 1–3 words.

    Does nothing to error sentinels (strings starting with ``[``).
    Returns the corrected transcript; logs when a change is made.

    Examples::

        correct_transcript("at Rojo 90")        → "add row 90"
        correct_transcript("rise at C1 score")  → "write cell C1 score"
        correct_transcript("Rice C1 test")      → "write C1 test"
        correct_transcript("safe and close excel") → "save and close excel"
        correct_transcript("add row jane ninety")  → "add row jane 90"
    """
    if not text or text.startswith('['):
        return text

    # ── Pre-pass 0: phonetic string replacements ─────────────
    # Converts rare STT corruptions before regex passes run, e.g.:
    #   "reiser c1 100" → "write c1 100"
    #   "write see one" → "write c one" (cell-letter phonetic)
    corrected = apply_phonetic_fixes(text)

    # ── Pre-pass: strip leading wake-word artifact ────────────────────────
    # "hey jarvis open excel" → "open excel"
    corrected = _WAKE_PREFIX_RE.sub('', corrected).strip()

    # ── Pass 1: word/phrase substitutions ────────────────────────────────
    for pattern, replacement in _TRANSCRIPT_CORRECTIONS:
        corrected = pattern.sub(replacement, corrected)

    # ── Apply phonetic fixes after corrections to avoid interference ───
    corrected = apply_phonetic_fixes(corrected)

    # ── Pass 2: command-prefix normalization ─────────────────────────────
    for pattern, replacement in _COMMAND_PREFIX_PATTERNS:
        m = pattern.match(corrected)
        if m:
            if callable(replacement):
                prefix = replacement(m)
            else:
                prefix = replacement
            corrected = prefix + corrected[m.end():]
            break  # only one prefix pattern applies

    # ── Pass 3: structural rewrites ───────────────────────────────────────
    # "write/put/set [value] in [cell] [ref]" → "write cell [ref] [value]"
    # Handles: "write score in cell C1" → "write cell C1 score"
    #          "put test in cell A1"    → "write cell A1 test"
    # Guard: skip when the transcript is a write-formula command — "write formula
    #        SUM A1:A10 in C1" must stay as-is so write_formula intent is preserved.
    # Guard: skip when the captured value is just a cell ref (avoid double-rewrite).
    _m3 = re.search(
        r'\b(?:write|put|set)\s+(.+?)\s+(?:in(?:to)?|to)\s+(?:cell\s+)?([A-Za-z]+\d+)\s*$',
        corrected, re.IGNORECASE,
    )
    if _m3:
        _val = _m3.group(1).strip()
        _cell = _m3.group(2).upper()
        # Skip if value starts with "formula" — this is a write_formula command.
        if not re.match(r'formula\b', _val, re.IGNORECASE):
            corrected = f'write cell {_cell} {_val}'

    # ── Pass 3b: split merged cell+value ─────────────────────────────────────
    # STT sometimes merges a cell ref and value: "write cell C212" where the
    # user said "write cell C2 12". Pattern: single letter + 1 digit + 2+ digits
    # at the end of a write-cell command → split into ref and value.
    _m_cv = re.match(
        r'^(write\s+cell\s+)([A-Za-z]\d)(\d{2,})\s*$',
        corrected, re.IGNORECASE,
    )
    if _m_cv:
        corrected = f'{_m_cv.group(1)}{_m_cv.group(2).upper()} {_m_cv.group(3)}'

    # ── Pass 3c: strip "value"/"with" filler between cell ref and actual value ─
    # "write cell C1 value 12" → "write cell C1 12"
    # "write cell C1 with 12"  → "write cell C1 12"
    # Guard: requires something after the filler word so "write cell B3 value"
    # (no actual value yet) is left intact for the slot checker to handle.
    _m3c = re.match(
        r'^(write\s+cell\s+[A-Za-z]+\d+)\s+(?:value|with)\s+(.+)$',
        corrected, re.IGNORECASE,
    )
    if _m3c:
        corrected = f'{_m3c.group(1)} {_m3c.group(2)}'

    # ── Pass 3d: "write VALUE CELLREF" → "write cell CELLREF VALUE" ───────────
    # Handles STT word-order inversion: "write Sarah C112" → "write cell C112 Sarah"
    # Only fires for a single word before a bare cell ref (no "cell" keyword, no
    # preposition).  Skips connectors / command keywords that are not real values.
    _m3d = re.match(
        r'^write\s+(\w+)\s+([A-Za-z]\d+)\s*$',
        corrected, re.IGNORECASE,
    )
    if _m3d:
        _val3d = _m3d.group(1)
        _cell3d = _m3d.group(2).upper()
        _3d_skip = frozenset({'cell', 'set', 'said', 'the', 'to', 'in', 'into',
                              'at', 'and', 'or', 'a', 'an', 'of', 'for'})
        if _val3d.lower() not in _3d_skip:
            corrected = f'write cell {_cell3d} {_val3d}'

    # ── Pass 4: voice recovery — strip leading filler from command payloads ──
    _recovered = recover_add_row_voice(corrected)
    if _recovered is not None:
        corrected = _recovered
    _recovered = recover_write_cell_voice(corrected)
    if _recovered is not None:
        corrected = _recovered

    if corrected != text:
        log.info('correct_transcript: %r → %r', text[:120], corrected[:120])

    return corrected


# ---------------------------------------------------------------------------
# 7. Command intent detection
# ---------------------------------------------------------------------------

# Maps canonical intent name → list of compiled patterns.
# Patterns match noisy STT output; only the FIRST matching intent is returned.
# Order matters: more specific commands appear before broader ones.
_INTENT_PATTERNS: dict[str, list[re.Pattern]] = {
    'write_cell': [
        re.compile(r'\b(write|rice|rise|right|wright|raiser)\s+(cell|at|so|sel)\b', re.IGNORECASE),
        re.compile(r'\b(write|rice|rise|right|raiser)\s+[A-Za-z]\d+\b', re.IGNORECASE),
        # Broad fallback: "write" (or near-miss) anywhere AND a cell ref anywhere,
        # but NOT when "formula" is also present (those belong to write_formula).
        # Catches near-misses with noise between "write" and the cell ref, e.g.
        # "write and C12", "write Sarah C112" (after Pass 3d), "rice and C12".
        re.compile(r'\b(write|rice|rise|right|raiser)\b(?!.*\bformula\b).*\b[A-Za-z]\d+\b', re.IGNORECASE),
    ],
    'add_row': [
        re.compile(r'\b(add|at|ad|hat)\s+(row|role|roll|rojo|ro|raw)\b', re.IGNORECASE),
        re.compile(r'^(add|at|ad)\s+r[oa]', re.IGNORECASE),
        # "Adele …" — STT merges "add row" into the name "Adele"
        re.compile(r'^adele\b', re.IGNORECASE),
        # "adroit …" — different phonetic corruption of "add row"
        re.compile(r'^adroit\b', re.IGNORECASE),
        # "at Rowena/Rowland" — STT compounds "row" + name into one token
        re.compile(r'^(at|ad|hat)\s+row\w', re.IGNORECASE),
    ],
    'delete_row': [
        re.compile(r'\b(delete|delet|dell|remove)\s+(row|role|roll)\b', re.IGNORECASE),
    ],
    'delete_column': [
        re.compile(r'\b(delete|delet|dell|remove)\s+(column|col)\b', re.IGNORECASE),
    ],
    'add_column': [
        re.compile(r'\badd\s+(column|col)\b', re.IGNORECASE),
    ],
    # "open file filterr.xlsx" / "open file client" / "open the file" →
    # open a specific file by name (distinct from launching the Excel app).
    # Must appear before open_excel so intent detection prefers the more specific match.
    'open_file': [
        re.compile(r'\bopen\s+(?:the\s+)?file\b', re.IGNORECASE),
    ],
    'open_excel': [
        # "file" removed: "open file [name]" now routes to open_file above.
        re.compile(r'\b(open|opal)\s+(excel|spreadsheet|sound|workbook)\b', re.IGNORECASE),
        re.compile(r'\bopen\s+my\s+(?:spreadsheet|workbook|file)\b', re.IGNORECASE),
        # Bare "Excel" — user said only the app name; treat as open request.
        re.compile(r'^excel\s*$', re.IGNORECASE),
    ],
    'close_excel': [
        re.compile(r'\b(close|clothes|quit|exit)\s+(excel|file|workbook|exam)\b', re.IGNORECASE),
        re.compile(r'\b(save\s+and\s+close|safe\s+and\s+close|self\s+and\s+close)\b', re.IGNORECASE),
    ],
    'save': [
        re.compile(r'^(save|safe|sake)\b', re.IGNORECASE),
        re.compile(r'\bsave\s+(?:and\s+close|this|excel|the\s+file)?\b', re.IGNORECASE),
    ],
    'create_excel': [
        re.compile(r'\b(create|creat|great|new)\s+(excel|file|spreadsheet)\b', re.IGNORECASE),
    ],
    'calculate_sum': [
        re.compile(r'\b(calculate|calc)\s+(sum|total|some)\b', re.IGNORECASE),
        re.compile(r'\bsum\s+(of\s+)?(column|the)\b', re.IGNORECASE),
        # "sum B2 to B5 into C6" / "sum B2 to B5 in tubes" — bare "sum" + cell ref
        re.compile(r'\bsum\s+[A-Za-z]\d+\b', re.IGNORECASE),
    ],
    'calculate_average': [
        re.compile(r'\b(calculate|calc)\s+(average|avg|avr)\b', re.IGNORECASE),
        re.compile(r'\baverage\s+(of\s+)?(column|the)\b', re.IGNORECASE),
        # "average B2:B5 into C7" / "average B2 B5 into C7" — bare "average" + cell ref
        re.compile(r'\baverage\s+[A-Za-z]\d+\b', re.IGNORECASE),
    ],
    'write_formula': [
        re.compile(r'\b(write|add|insert)\s+formula\b', re.IGNORECASE),
        re.compile(r'\bformula\b', re.IGNORECASE),
    ],
    'create_chart': [
        re.compile(r'\b(create|make|add)\s+(chart|graph|plot)\b', re.IGNORECASE),
        # "make a bar chart" / "create a line graph" — type word sits between verb and chart noun
        re.compile(r'\b(create|make|add)\s+(?:a\s+)?\w+\s+(chart|graph|plot)\b', re.IGNORECASE),
    ],
    'sort_range': [
        re.compile(r'\b(sort)\s+(column|range|by)\b', re.IGNORECASE),
    ],
    'format_cells': [
        re.compile(r'\b(format|bold|italic|colour|color)\s+(cell|column|row|range)\b', re.IGNORECASE),
        # "make A1 bold" / "make B2:C3 italic" — cell-ref formatting via "make"
        re.compile(r'\b(?:make|mark)\s+[A-Za-z]\d+(?::[A-Za-z]\d+)?\s+(?:bold|italic)\b', re.IGNORECASE),
        # "make header bold" / "make the header italic"
        re.compile(r'\b(?:make|mark)\s+(?:the\s+)?header\s+(?:bold|italic)\b', re.IGNORECASE),
        # "bold A1" / "italic C6" / "bold A5:C5" / "italic A12:B12"
        re.compile(r'\b(?:bold|italic)\s+[A-Za-z]+\d+(?::[A-Za-z]+\d+)?\b', re.IGNORECASE),
        # "format A13:C13 bold" / "format A12:B12 italic"
        re.compile(r'\bformat\s+[A-Za-z]+\d+(?::[A-Za-z]+\d+)?\s+(?:bold|italic)\b', re.IGNORECASE),
        # "center B2" / "center A1:C1" / "center row 2" / "center column B"
        re.compile(r'\bcenter\s+(?:row\s+\d+|column\s+[A-Za-z]+|[A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\b', re.IGNORECASE),
        # "align A2 left" / "align row 1 center" / "align column B right"
        re.compile(r'\balign\s+(?:row\s+\d+|column\s+[A-Za-z]+|[A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\b', re.IGNORECASE),
        # "align column 3 right" / "align row to left" — malformed; catch for slot-gate clarification
        re.compile(r'\balign\s+(?:row|column)\b', re.IGNORECASE),
        # "color B2 red" / "color A1:C1 blue" — cell ref directly after color
        re.compile(r'\bcolou?r\s+[A-Za-z]+\d+(?::[A-Za-z]+\d+)?\b', re.IGNORECASE),
        # "set font size A2 30" / "increase font size A1" / "decrease font size row 1"
        re.compile(r'\b(?:set|increase|decrease)\s+(?:font\s+)?size\b', re.IGNORECASE),
        # "font size A2 20" / "font size row 3 16" — no set/increase/decrease prefix
        re.compile(r'\bfont\s+size\b', re.IGNORECASE),
        # "make header size 18" / "make the header font size 20"
        re.compile(r'\bmake\s+(?:the\s+)?header\s+(?:font\s+)?size\b', re.IGNORECASE),
        # "make hat size 2" / "make [any-word] size N" — STT corruption of "make header size"
        re.compile(r'\bmake\s+\w+\s+(?:font\s+)?size\b', re.IGNORECASE),
        # "set font-sign D130" / "set font-[anything]" — garbled hyphenated form
        re.compile(r'\bset\s+font-\w+\b', re.IGNORECASE),
        # "format B2:B10 as currency" / "format column A as number" / "format row 3 as currency"
        re.compile(
            r'\bformat\s+(?:row\s+\d+|column\s+[A-Za-z]+|[A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\s+as\b',
            re.IGNORECASE,
        ),
        # "format JFI as number" — garbled range; broad catch so slot-gate can clarify
        re.compile(
            r'\bformat\s+\S+\s+as\s+(?:number|integer|currency|percentage|percent|decimal|date|text)\b',
            re.IGNORECASE,
        ),
        # "format B2:B10" / "format A1" — range without format type (catch for slot-gate clarification)
        re.compile(r'\bformat\s+[A-Za-z]+\d+(?::[A-Za-z]+\d+)?\b', re.IGNORECASE),
        # "format as currency" — format type with no target (catch for slot-gate clarification)
        re.compile(
            r'\bformat\s+as\s+(?:number|integer|currency|percentage|percent|decimal|date|text)\b',
            re.IGNORECASE,
        ),
        # "make A3 orange" / "make B2:C3 red" — cell-ref + color word
        re.compile(
            r'\bmake\s+[A-Za-z]+\d+(?::[A-Za-z]+\d+)?\s+'
            r'(?:red|green|blue|yellow|orange|purple|pink|black|white|gray|grey|'
            r'cyan|magenta|brown|lime|teal|navy|coral|salmon)\b',
            re.IGNORECASE,
        ),
        # "make row 3 blue" / "make column C orange" — row/col + color word → font_color
        re.compile(
            r'\bmake\s+(?:row\s+\d+|column\s+[A-Za-z]+)\s+'
            r'(?:red|green|blue|yellow|orange|purple|pink|black|white|gray|grey|'
            r'cyan|magenta|brown|lime|teal|navy|coral|salmon)\b',
            re.IGNORECASE,
        ),
        # "fill row 2 with red" / "fill A4 pink" / "fill A5:C5 with green" / "fill column B"
        re.compile(
            r'\bfill\s+(?:row\s+\d+|column\s+[A-Za-z]+|[A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\b',
            re.IGNORECASE,
        ),
        # "fill red" / "fill blue" — color word with no target (catch for slot-gate clarification)
        re.compile(
            r'\bfill\s+(?:red|green|blue|yellow|orange|purple|pink|black|white|gray|grey|'
            r'cyan|magenta|brown|lime|teal|navy|coral|salmon)\b',
            re.IGNORECASE,
        ),
        # "highlight row 4" / "highlight A1:C1" / bare "highlight" — all caught for slot check
        re.compile(r'\bhighlight\b', re.IGNORECASE),
        # "color red" / "colour blue" — color word with no target (catch for slot-gate clarification)
        re.compile(
            r'\bcolou?r\s+(?:red|green|blue|yellow|orange|purple|pink|black|white|gray|grey|'
            r'cyan|magenta|brown|lime|teal|navy|coral|salmon)\b',
            re.IGNORECASE,
        ),
        # "fill a row with red" / "fill row with red" — row without digit (slot-gate will clarify)
        re.compile(r'\bfill\s+(?:a\s+)?row\b', re.IGNORECASE),
        # "color a row" / "color row" — row without digit (slot-gate will clarify)
        re.compile(r'\bcolou?r\s+(?:a\s+)?row\b', re.IGNORECASE),
    ],

    # ── Sheet management ──────────────────────────────────────────────────
    'switch_sheet': [
        # All three patterns require the "sheet" keyword — avoids treating
        # "go to B3" (cell ref) or bare "go to" as a switch-sheet command.
        re.compile(r'\b(switch|go|navigate)\s+(?:to\s+)?sheet\b', re.IGNORECASE),
        re.compile(r'\bswitch\s+sheet\b', re.IGNORECASE),
        re.compile(r'\bgo\s+to\s+sheet\b', re.IGNORECASE),
    ],
    'create_sheet': [
        re.compile(r'\b(create|add|make|new)\s+(?:a\s+)?sheet\b', re.IGNORECASE),
    ],
    'delete_sheet': [
        re.compile(r'\b(delete|remove)\s+(?:the\s+)?sheet\b', re.IGNORECASE),
    ],
    'rename_sheet': [
        re.compile(r'\brename\s+(?:the\s+)?sheet\b', re.IGNORECASE),
        # "rename [name] to [new-name]" structural form — no "sheet" keyword needed.
        # Negative lookahead excludes file-related keywords so "rename file to X"
        # still routes to rename_file (checked later in the intent dict).
        re.compile(r'\brename\s+(?!(?:file|excel|spreadsheet|workbook)\b)\S+\s+to\b',
                   re.IGNORECASE),
    ],
    'list_sheets': [
        re.compile(r'\b(list|show)\s+(?:all\s+)?sheets?\b', re.IGNORECASE),
        re.compile(r'\bwhat\s+sheets?\b', re.IGNORECASE),
    ],

    # ── Cell read / clear ─────────────────────────────────────────────────
    'read_cell': [
        re.compile(r'\b(read|show)\s+(?:the\s+)?cell\b', re.IGNORECASE),
        re.compile(r"\bwhat(?:'s|\s+is)\s+in\s+(?:cell\s+)?[A-Za-z]\d+\b", re.IGNORECASE),
        # "go to B3" / "go to cell B3" — cell navigation, not sheet switch.
        # Requires a cell ref (letter + digits) so "go to sheet X" is NOT caught here.
        re.compile(r'\bgo\s+to\s+(?:cell\s+)?[A-Za-z]\d+\b', re.IGNORECASE),
    ],
    'clear_cell': [
        re.compile(r'\b(clear|empty|erase)\s+(?:the\s+)?cell\b', re.IGNORECASE),
        re.compile(r'\bdelete\s+(?:the\s+)?cell\s+content\b', re.IGNORECASE),
    ],

    # ── Range operations ──────────────────────────────────────────────────
    'read_range': [
        re.compile(r'\b(read|show)\s+(?:the\s+)?range\b', re.IGNORECASE),
    ],

    # ── Formatting helpers ────────────────────────────────────────────────
    'freeze_panes': [
        re.compile(r'\bfreeze\s+(?:panes?|rows?|top)\b', re.IGNORECASE),
        # "freeze header" / "freeze the header" / "freeze top row" — canonical voice forms
        re.compile(r'\bfreeze\s+(?:the\s+)?(?:header|top\s+row)\b', re.IGNORECASE),
        # "freeze first column" / "freeze column A" — freeze leftmost column (cell=B1)
        re.compile(r'\bfreeze\s+(?:first\s+col(?:umn)?|col(?:umn)?\s+[A-Za-z])\b', re.IGNORECASE),
    ],
    'unfreeze_panes': [
        re.compile(r'\bunfreeze\s+(?:panes?|rows?)\b', re.IGNORECASE),
        # bare "unfreeze" with no qualifier — the only sensible Excel action
        re.compile(r'\bunfreeze\b', re.IGNORECASE),
    ],
    'merge_cells': [
        re.compile(r'\b(merge|combine)\s+(?:the\s+)?cells?\b', re.IGNORECASE),
    ],

    # ── File operations ───────────────────────────────────────────────────
    'undo': [
        re.compile(r'^undo\b', re.IGNORECASE),
        re.compile(r'\bundo\s+(?:last|that|it)?\b', re.IGNORECASE),
    ],
    'rename_file': [
        re.compile(r'\brename\s+(?:the\s+)?(?:file|excel|spreadsheet|workbook)\b', re.IGNORECASE),
    ],

    # ── Previously missing: row/column insertion ──────────────────────────
    # These were falling through to the LLM every time — now pre-routed.
    'insert_row_at': [
        re.compile(r'\binsert\s+(?:a\s+)?row\s+(?:at|before|after)\b', re.IGNORECASE),
        re.compile(r'\badd\s+row\s+(?:at|before|after)\s+\d+\b', re.IGNORECASE),
        re.compile(r'\binsert\s+(?:a\s+)?row\s+\d+\b', re.IGNORECASE),   # "insert row 3"
        re.compile(r'\binsert\s+before\s+row\b', re.IGNORECASE),          # "insert before row 3"
        re.compile(r'\binsert\s+(?:a\s+)?row\s*$', re.IGNORECASE),        # bare → slot check
    ],
    'insert_column_at': [
        re.compile(r'\binsert\s+(?:a\s+)?column\s+(?:at|before|after)\b', re.IGNORECASE),
        re.compile(r'\badd\s+column\s+(?:at|before|after)\b', re.IGNORECASE),
        re.compile(r'\binsert\s+(?:a\s+)?column\s+[A-Za-z]{1,3}\b', re.IGNORECASE),  # "insert column B"
        re.compile(r'\binsert\s+before\s+column\b', re.IGNORECASE),                   # "insert before column B"
        re.compile(r'\binsert\s+(?:a\s+)?column\s*$', re.IGNORECASE),                 # bare → slot check
    ],

    # ── Previously missing: cell/range deletion ───────────────────────────
    # "delete cell B3" is distinct from "clear cell B3" (shifts cells vs clears).
    'delete_cell': [
        re.compile(r'\bdelete\s+(?:the\s+)?cell\s+[A-Za-z]\d+\b', re.IGNORECASE),
        re.compile(r'\bdelete\s+(?:the\s+)?cell\b', re.IGNORECASE),
    ],
    'delete_range': [
        re.compile(r'\b(delete|clear|erase|remove)\s+(?:the\s+)?range\b', re.IGNORECASE),
    ],

    # ── Previously missing: column width ─────────────────────────────────
    'set_column_width': [
        re.compile(r'\b(?:set|change|resize|adjust)\s+(?:the\s+)?column\s+\w+\s+width\b', re.IGNORECASE),
        re.compile(r'\bauto\s*[-\s]?fit\s+column\b', re.IGNORECASE),
        re.compile(r'\bauto\s*fit\s+all\b', re.IGNORECASE),
        re.compile(r'\bresize\s+column\b', re.IGNORECASE),
        re.compile(r'\bcolumn\s+\w+\s+(?:wider|narrower|width)\b', re.IGNORECASE),
        re.compile(r'\bmake\s+column\s+\w+\s+(?:wider|narrower|smaller|bigger|larger)\b', re.IGNORECASE),
    ],

    # ── Previously missing: find/replace ─────────────────────────────────
    'find_replace': [
        re.compile(r'\b(?:find|search)\s+(?:and\s+)?replace\b', re.IGNORECASE),
        re.compile(r'\bsearch\s+and\s+replace\b', re.IGNORECASE),
        # "find [term] replace with [replacement]" — term sits between find and replace
        re.compile(r'\bfind\s+\S+\s+replace\b', re.IGNORECASE),
        # "replace [old] with [new]" — direct imperative form
        re.compile(r'\breplace\s+\S+\s+with\b', re.IGNORECASE),
    ],

    # ── Previously missing: filter / clear-filter ────────────────────────
    'filter_table': [
        re.compile(r'\bfilter\s+(?:by|column|where|the)\b', re.IGNORECASE),
        re.compile(r'\bshow\s+only\s+(?:rows?\s+)?where\b', re.IGNORECASE),
        # Bare voice forms: "filter B 90" / "filter Score 90" (no keyword)
        re.compile(r'\bfilter\s+[A-Za-z]\w*\s+\S', re.IGNORECASE),
        # "filter B" alone — intent detected so slot check can ask for value
        re.compile(r'\bfilter\s+[A-Za-z]\w*\s*$', re.IGNORECASE),
    ],
    'clear_filter': [
        re.compile(r'\b(?:clear|remove|reset)\s+(?:the\s+)?filters?\b', re.IGNORECASE),
        re.compile(r'\bshow\s+all\s+(?:rows?|data)\b', re.IGNORECASE),
    ],

    # ── Previously missing: unmerge ───────────────────────────────────────
    'unmerge_cells': [
        re.compile(r'\bunmerge\s+(?:the\s+)?(?:merged\s+)?cells?\b', re.IGNORECASE),
        re.compile(r'\bsplit\s+(?:the\s+)?(?:merged\s+)?cells?\b', re.IGNORECASE),
        re.compile(r'\bunmerge\b', re.IGNORECASE),
    ],
}


def detect_command_intent(text: str) -> str | None:
    """Return the canonical intent name for *text*, or ``None`` if unrecognised.

    Works on noisy STT output — patterns are broad enough to catch common
    phonetic corruptions of each command.

    Returns one of: ``'add_row'``, ``'write_cell'``, ``'delete_row'``,
    ``'delete_column'``, ``'add_column'``, ``'open_excel'``, ``'close_excel'``,
    ``'save'``, ``'create_excel'``, ``'calculate_sum'``, ``'calculate_average'``,
    ``'write_formula'``, ``'create_chart'``, ``'sort_range'``, ``'format_cells'``,
    ``'switch_sheet'``, ``'create_sheet'``, ``'delete_sheet'``, ``'rename_sheet'``,
    ``'list_sheets'``, ``'read_cell'``, ``'clear_cell'``, ``'read_range'``,
    ``'freeze_panes'``, ``'unfreeze_panes'``, ``'merge_cells'``,
    ``'undo'``, ``'rename_file'``, or ``None``.

    Examples::

        detect_command_intent("at Rojo 90")        → "add_row"
        detect_command_intent("rise at C1 score")  → "write_cell"
        detect_command_intent("safe and close")    → "close_excel"
        detect_command_intent("taste")             → None
    """
    if not text or text.startswith('['):
        return None
    # Fast-path for deterministic commands
    if text.strip().lower() == "undo":
        return "undo"
    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            if pat.search(text):
                return intent
    return None


# ---------------------------------------------------------------------------
# 8. Low-confidence transcript filter
# ---------------------------------------------------------------------------

# Profanity tokens — almost never valid slot values (e.g. sheet names).
# Used by check_command_slots() to reject noisy sheet-name candidates and by
# is_low_confidence_transcript() to block short all-profanity transcripts.
_PROFANITY_TOKENS: frozenset[str] = frozenset({
    'shit', 'fuck', 'fucking', 'fucked', 'damn', 'damned',
    'ass', 'arse', 'crap', 'piss', 'bitch', 'bastard',
})

# Single words / short phrases that are definitely noise, not commands.
_NOISE_WORDS: frozenset[str] = frozenset({
    'taste', 'dying', 'oh', 'um', 'uh', 'hmm', 'huh',
    'what', 'hello', 'hey', 'hi', 'okay', 'ok', 'okay',
    'test', 'testing', 'check', 'nothing', 'never mind',
    'wait', 'hold on', 'sorry', 'pardon', 'come on',
    # reported in problem statement
    "i'm dying",
})

# Single-word utterances that ARE valid commands and must not be filtered.
_VALID_SOLO_COMMANDS: frozenset[str] = frozenset({
    'save', 'undo', 'redo', 'stop', 'exit', 'quit',
    'done', 'cancel', 'confirm', 'yes', 'no',
})


def is_low_confidence_transcript(text: str) -> bool:
    """Return True when *text* is too noisy to route to the LLM.

    Criteria (any one is sufficient):
    * The whole transcript is a single known noise word.
    * The whole transcript is a single word that is not a valid solo command
      AND is not recognised by :func:`detect_command_intent`.
    * The transcript has exactly two words and no intent is recognised AND
      neither word is a known command keyword.

    Does NOT filter error sentinels (``[…]`` strings) — those are handled
    separately by the conversation loop.

    Examples::

        is_low_confidence_transcript("taste")         → True
        is_low_confidence_transcript("rice")          → False  # intent: write_cell
        is_low_confidence_transcript("oh")            → True
        is_low_confidence_transcript("add row john")  → False
        is_low_confidence_transcript("save")          → False
    """
    if not text or text.startswith('['):
        return False

    normalized = text.strip().lower().rstrip('.!?,')
    words = normalized.split()

    # Single word
    if len(words) == 1:
        word = words[0]
        if word in _NOISE_WORDS:
            return True
        if word in _VALID_SOLO_COMMANDS:
            return False
        # Single unrecognised word — reject unless intent detected
        if detect_command_intent(text) is None:
            return True

    # Two words with no detectable intent
    if len(words) == 2 and detect_command_intent(text) is None:
        return True

    # Three words with no detectable intent AND no recognisable Excel keyword.
    # Real failure: "John 19 at" (garbled "add row John 90") passed through to
    # the LLM router because the 1–2 word check didn't catch it.
    # Guard: preserve partial-but-meaningful commands that DO contain a command
    # keyword — e.g. "add a row" (has 'add'), "go to cell" (has 'go', 'cell').
    if len(words) == 3 and detect_command_intent(text) is None:
        _CMD_KW: frozenset[str] = frozenset({
            'add', 'write', 'save', 'open', 'close', 'delete', 'create',
            'switch', 'go', 'read', 'calculate', 'sort', 'format', 'merge',
            'freeze', 'unfreeze', 'undo', 'rename', 'insert', 'find',
            'replace', 'filter', 'set', 'sum', 'average',
            'column', 'row', 'cell', 'sheet',
            'bold', 'make', 'italic', 'header', 'width', 'font',
            'center', 'align', 'color', 'colour', 'increase', 'decrease',
        })
        if not any(w in _CMD_KW for w in words):
            return True

    # Short transcript composed entirely of profanity — reject as noise
    if len(words) <= 3 and all(w in _PROFANITY_TOKENS for w in words):
        return True

    return False


# ---------------------------------------------------------------------------
# 8b. Suspicious-transcript detection
# ---------------------------------------------------------------------------

# Known garbled word fragments that indicate the STT produced a corrupted
# command rather than noise.  Transcripts containing any of these fragments
# after correction are "suspicious" — the conversation loop attempts a silent
# retry before surfacing a clarification message.
_SUSPICIOUS_FRAGMENTS: frozenset[str] = frozenset({
    'adro',   # "add row" garbled by Realtek laptop mic (real log)
    'dalit',  # "delete" garbled by Realtek laptop mic (real log)
    'roto',   # "row" garbled by Realtek laptop mic (real log: "dalit roto 3")
    'hada',   # "header" garbled by Realtek laptop mic (real log: "free hada")
})


def is_suspicious_transcript(text: str) -> bool:
    """Return True when *text* contains known STT garble fragments after correction.

    The conversation loop uses this to trigger one silent retry before giving
    feedback — avoids surfacing a confusing "I didn't catch that" when a
    simple re-listen would produce a clean result.

    Returns False for error sentinels (``[…]`` strings).

    Examples::

        is_suspicious_transcript('adro John 90')  → True
        is_suspicious_transcript('dalit roto 3')  → True
        is_suspicious_transcript('free hada')     → True
        is_suspicious_transcript('add row John')  → False
        is_suspicious_transcript('freeze header') → False
    """
    if not text or text.startswith('['):
        return False
    words = text.strip().lower().split()
    return any(w in _SUSPICIOUS_FRAGMENTS for w in words)


# ---------------------------------------------------------------------------
# 9. Slot completeness check (pre-routing gate)
# ---------------------------------------------------------------------------

# Connectors and filler tokens that are never valid slot values.
_SLOT_GARBAGE: frozenset[str] = frozenset({
    'and', 'the', 'a', 'an', 'to', 'in', 'of', 'for', 'is', 'it',
    'that', 'this', 'my', 'please', 'just', 'with', 'on', 'at', 'or',
    'into', 'from', 'by', 'as', 'be', 'do', 'did', 'are', 'was',
    'least', 'at least',
})

# STT noise values that are almost never real data — block before filter executes.
_FILTER_VALUE_JUNK: frozenset[str] = frozenset({
    'table', 'the arts', 'theater', 'theatre', 'the art',
    'the answer', 'the office', 'the data', 'the table',
    'arts', 'the range', 'the filter',
})

# Recognised command verb tokens — must be stripped before counting data tokens.
_COMMAND_VERBS: frozenset[str] = frozenset({
    'add', 'row', 'insert', 'write', 'cell', 'set', 'update',
    'open', 'close', 'save', 'delete', 'create', 'remove',
    'calculate', 'average', 'sum', 'formula',
    # phonetic variants treated as command verbs
    'at', 'ad', 'hat', 'adele', 'rice', 'rise', 'right',
})

_SLOT_CELL_REF_RE = re.compile(r'\b[A-Za-z]+\d+\b')
_SLOT_BARE_NUM_RE = re.compile(r'^\d+$')


def check_command_slots(intent: str | None, text: str,
                        available_sheets: list[str] | None = None) -> str | None:
    """Return a clarification question when required slots are missing for *intent*.

    Called in ``process_command()`` **before** ``route_llm()`` so that
    incomplete commands trigger a targeted clarification instead of an
    expensive and often wrong Ollama fallback.

    Returns ``None`` when the command looks complete and should be routed
    normally.

    Supported intents: ``'add_row'``, ``'write_cell'``, ``'delete_row'``,
    ``'switch_sheet'``, ``'create_sheet'``, ``'rename_sheet'``.
    All others pass through immediately (they either have no required slots or
    the heuristic handles them reliably without extra help).

    Examples::

        check_command_slots('add_row', 'add row')          → "What values…?"
        check_command_slots('add_row', 'add row and')      → "What values…?"
        check_command_slots('add_row', 'add row John 90')  → None
        check_command_slots('write_cell', 'write cell C1') → "What value for C1?"
        check_command_slots('write_cell', 'write cell C1 score') → None
        check_command_slots('delete_row', 'delete row')          → "Which row number…?"
        check_command_slots('delete_row', 'delete row 3')        → None
        check_command_slots('switch_sheet', 'switch sheet')      → "Which sheet…?"
        check_command_slots('switch_sheet', 'switch sheet data') → None
        check_command_slots('create_sheet', 'create sheet')      → "What should I call…?"
        check_command_slots('rename_sheet', 'rename sheet foo')  → "…what should it be renamed to?"
        check_command_slots('save', 'save')                      → None
    """
    if not text or text.startswith('['):
        return None

    lower = text.strip().lower()
    words = lower.split()

    # Guard: known STT garbage phrases that should never execute as commands.
    _BAD_PATTERNS = ('and not', 'i ate', 'experience', 'something',)
    if any(p in lower for p in _BAD_PATTERNS):
        return "I didn't catch that clearly. Please repeat."

    # Guard: profanity when intent is unknown → clarify, never send to LLM.
    # When intent IS known, each intent branch does its own profanity check so
    # it can return a context-specific message (e.g. "which sheet would you like?").
    _clean_words = [w.rstrip('.,!?…') for w in words]
    _non_verb_clean = [w for w in _clean_words if w not in _COMMAND_VERBS and w not in _SLOT_GARBAGE]
    if intent is None and any(w in _PROFANITY_TOKENS for w in _non_verb_clean):
        return "I didn't catch that clearly. Please repeat your command."

    # Guard: "create/make/new" with no detectable intent and no Excel-context word.
    # Prevents garbage transcripts like "create a sad anymore" from reaching Ollama
    # and coming back as create_excel with a nonsense filename.
    if not intent:
        if re.search(r'\b(?:create|make|new)\b', lower):
            _EXCEL_ANCHORS = frozenset({
                'excel', 'file', 'spreadsheet', 'workbook',
            })
            # Formatting commands use "make" as a verb (e.g. "make A1 bold",
            # "make header size 18"). If any formatting keyword is present, this
            # is NOT a create-intent command and must not be blocked here.
            _FORMAT_ANCHORS = frozenset({
                'bold', 'italic', 'center', 'centre', 'align', 'font', 'size',
                'fill', 'color', 'colour', 'format', 'highlight', 'merge',
                'freeze', 'width', 'wider', 'narrower',
            })
            if (not any(a in words for a in _EXCEL_ANCHORS)
                    and not any(a in words for a in _FORMAT_ANCHORS)):
                return (
                    "I'm not sure what to create. "
                    "Try: create excel teacher, or create sheet report."
                )
        # Guard: "write Saturday to 3" or "write Yvonne Studin" — 'write' is
        # present but no cell ref (e.g. B3) exists anywhere in the transcript.
        # Removing the digit requirement catches name-only garbles like
        # "write Yvonne Studin" that have no digit at all; they would otherwise
        # reach the LLM and come back as a wrong or hallucinated action.
        if re.search(r'\bwrite\b', lower) and not re.search(r'[A-Za-z]\d+', lower):
            return (
                "Which cell should I write to? "
                "For example: write 100 to B3."
            )
        # Guard: "go to" with no valid destination (no cell ref, no sheet keyword).
        # Catches garbled phrases like "do let go to" — sending these to the LLM
        # burns 10-30 s on Ollama/Groq and always returns the wrong action.
        if re.search(r'\bgo\s+to\b', lower):
            if not re.search(r'[A-Za-z]\d+', lower) and 'sheet' not in words and 'sheets' not in words:
                return (
                    "Did you mean to switch to a sheet or navigate to a cell? "
                    "Say 'switch sheet <name>' or 'go to cell B3'."
                )
        # Guard: "switch to [X]" without the "sheet" keyword — ambiguous, no "sheet"
        # in the transcript means either wrong intent or garbled command.
        # "Switch to 8" / "switch to shit" must not go to Ollama and waste 14 s.
        if re.search(r'\bswitch\s+to\b', lower) and 'sheet' not in words:
            return (
                "Did you mean switch sheet <name> or go to cell <ref>? "
                "Say 'switch sheet <name>' or 'go to cell B3'."
            )
        # Guard: bare "switch [no sheet keyword]" — catches "switch it to them",
        # "switch that" etc. that slip through corrections because "sheet" is absent.
        if re.search(r'\bswitch\b', lower) and 'sheet' not in words and 'sheets' not in words:
            return (
                "Which sheet should I switch to? "
                "Please say: switch sheet <name>."
            )
        # Guard: "N row M" with no delete verb — destructive action must never
        # be inferred from a bare number sequence.  "3 row 2" could be STT
        # corruption of "delete row 3" but without an explicit verb it is safer
        # to ask for clarification than to guess and mutate the sheet.
        if re.search(r'^\d+\s+row\s+\d+', lower):
            return (
                "Did you mean to delete a row? "
                "Please say 'delete row 3' clearly."
            )

        # Guard: "sheet" present but no sheet command intent detected — e.g.
        # "Do it, sheet, style" / "Do let the sheet start".  In production the
        # ambiguity guard blocks these, but an explicit slot message here ensures
        # pipeline tests can verify they never trigger LLM.
        # Use _clean_words (punctuation stripped) so "sheet," is still matched.
        if 'sheet' in _clean_words or 'sheets' in _clean_words:
            return (
                "I didn't catch that clearly. "
                "Did you mean to switch, delete, create, or rename a sheet?"
            )
        # Guard: "filter [digit/value only]" — column missing.
        # "filter 90" has no recognisable column → intent=None here.
        if re.search(r'\bfilter\b', lower):
            if not re.search(r'[A-Za-z]', re.sub(r'\bfilter\b', '', lower)):
                return (
                    "Which column and value should I filter by? "
                    "For example: filter column B 90."
                )
        # Guard: "average" present but no valid colon-form range found.
        # Catches "Average B to B5 into C7" (start row missing) and bare "average".
        # Runs on the corrected transcript, so well-formed variants like
        # "average B2 B5 into C7" already have a colon inserted by the
        # transcript correction pass and will NOT trigger this guard.
        if re.search(r'\baverage\b', lower):
            if not re.search(r'\b[A-Za-z]+\d+\s*:\s*[A-Za-z]+\d+\b', lower):
                return (
                    "Please specify the average range clearly. "
                    "For example: average B2:B5 into C7."
                )
        return None

    if intent == 'add_row':
        # Strip command verbs and garbage connectors; what remains are data tokens.
        data = [w.rstrip('.,!?…') for w in words if w not in _COMMAND_VERBS and w not in _SLOT_GARBAGE]
        data = [w for w in data if w]  # drop empty strings after stripping
        # Profanity in data slot → never write it to the sheet
        if any(w in _PROFANITY_TOKENS for w in data):
            return (
                "I didn't catch that clearly. "
                "Please say the name and score, for example: add row John 90."
            )
        if not data:
            return (
                "I need values for the row. "
                "Please say the name and score, for example: add row John 90."
            )
        # Too many data tokens — likely garbled STT (e.g. "add row eight nine 10").
        # Valid voice commands have exactly one name and one score, so more than
        # two residual tokens almost always means the transcript is corrupted.
        if len(data) > 2:
            return (
                "That looks like too many values for one row. "
                "Please say just the name and score, for example: add row John 90."
            )
        # Single very-short residual (e.g. "wena" from "Rowena") = corrupted artifact
        if len(data) == 1 and len(data[0]) <= 4 and not _SLOT_BARE_NUM_RE.match(data[0]):
            return (
                "I think you want to add a row, but I still need the name and score. "
                "Please try again."
            )
        # Bare number with no string token (e.g. "add row 290") — almost certainly
        # a STT garbling of "add row <name> <number>" where the name was dropped or
        # merged into the digit.  Adding a row with only a numeric value would write
        # the number into the Name column, which is wrong.  Ask for a retry.
        if len(data) == 1 and _SLOT_BARE_NUM_RE.match(data[0]):
            return (
                "I heard a number but no name. "
                "Please say the name and score together, for example: add row John 90."
            )
        # Require at least one alpha token (name) AND one numeric token (score).
        # Catches garbled phrases like "add row not 8" where no real name is present.
        _has_alpha = any(re.match(r'^[A-Za-z]{2,}$', d) and d not in _SLOT_GARBAGE
                         for d in data)
        _has_num   = any(_SLOT_BARE_NUM_RE.match(d) or re.match(r'^\d+\.?\d*$', d)
                         for d in data)
        if not _has_alpha or not _has_num:
            return (
                "Please say name and number clearly. "
                "Example: add row John 90."
            )

    elif intent == 'write_cell':
        cell_ref = _SLOT_CELL_REF_RE.search(text)
        if not cell_ref:
            return (
                "Which cell should I write to? "
                "Please say something like: write cell C1 score."
            )
        # Check that a meaningful value exists after the cell reference.
        # Strip generic filler AND command keywords that land in the value
        # position when STT garbles the utterance — e.g. "write cell E1 cell then"
        # from "Rise to then to E1" must not write "cell then" into the sheet.
        after_ref = text[cell_ref.end():].strip()
        _WRITE_VALUE_NOISE = _SLOT_GARBAGE | {'cell', 'then', 'now'}
        value_words = [w for w in after_ref.lower().split() if w not in _WRITE_VALUE_NOISE]
        if not value_words:
            cell = cell_ref.group(0).upper()
            return f"What value should I write into cell {cell}?"

    elif intent == 'delete_row':
        # Need an explicit row number
        if not re.search(r'\b\d+\b', lower):
            return (
                "Which row number should I delete? "
                "Please say the number, for example: delete row 3."
            )

    elif intent == 'delete_column':
        # Need an explicit column letter
        if not re.search(r'\bcolumn\s+[A-Za-z]\b', lower):
            return (
                "Which column should I delete? "
                "Please say the letter, for example: delete column B."
            )

    elif intent == 'insert_row_at':
        if not re.search(r'\b\d+\b', lower):
            return (
                "Which row number should I insert before? "
                "Please say the number, for example: insert row 3."
            )

    elif intent == 'insert_column_at':
        if not re.search(r'\b[A-Za-z]{1,3}\b', lower.split('column')[-1] if 'column' in lower else ''):
            return (
                "Which column should I insert before? "
                "Please say the letter, for example: insert column B."
            )

    elif intent == 'delete_cell':
        # Require a recognisable cell reference (letter + digit) before executing.
        # Garbled transcripts like "delete cell XYZ" have no digit → action=None
        # without this guard, which sends ambiguous text to the LLM.
        if not re.search(r'\b[A-Za-z]+\d+\b', lower):
            return (
                "Which cell should I delete? "
                "Please say something like: delete cell C3."
            )

    elif intent == 'merge_cells':
        # A range needs two cell refs; a single cell cannot be merged.
        _has_range = bool(re.search(r'\b[A-Za-z]+\d+\s*:\s*[A-Za-z]+\d+\b', lower))
        _has_two   = len(re.findall(r'\b[A-Za-z]+\d+\b', lower)) >= 2
        if not _has_range and not _has_two:
            return (
                "Which range should I merge? "
                "Please say something like: merge cells A1:C1."
            )

    elif intent == 'unmerge_cells':
        # Same guard as merge_cells: need a range, not just a single cell or bare verb.
        # This is the primary blocker that prevents "and merge and merge" (corrected
        # to "unmerge") from reaching Groq — slot check returns clarification first.
        _has_range = bool(re.search(r'\b[A-Za-z]+\d+\s*:\s*[A-Za-z]+\d+\b', lower))
        _has_two   = len(re.findall(r'\b[A-Za-z]+\d+\b', lower)) >= 2
        if not _has_range and not _has_two:
            return (
                "Which range should I unmerge? "
                "Please say something like: unmerge cells A1:C1."
            )

    elif intent == 'delete_sheet':
        _DEL_VERBS = {'delete', 'remove', 'the', 'sheet'}
        name_words = [
            w.rstrip('.,!?…') for w in words
            if w.rstrip('.,!?…') not in _COMMAND_VERBS
            and w.rstrip('.,!?…') not in _SLOT_GARBAGE
            and w.rstrip('.,!?…') not in _DEL_VERBS
            and w.rstrip('.,!?…')
        ]
        if not name_words:
            # The word "sheet" itself may be the sheet name (e.g. "delete sheet Sheet").
            # _DEL_VERBS strips it — recover it via a direct regex before clarifying.
            _m = re.search(r'^(?:delete|remove)\s+(?:the\s+)?sheet\s+(\S+)', lower)
            if _m:
                _candidate = _m.group(1).rstrip('.,!?…')
                if _candidate and _candidate not in _SLOT_GARBAGE:
                    name_words = [_candidate]
        if not name_words:
            return "Which sheet should I delete? Please say the sheet name."
        if any(w in _PROFANITY_TOKENS for w in name_words):
            return (
                "I didn't catch the sheet name clearly. "
                "Which sheet should I delete?"
            )
        # Multiple residual tokens = garbled STT (same logic as switch_sheet).
        if len(name_words) > 1:
            if available_sheets:
                return (
                    "Which sheet should I delete? "
                    f"Available sheets are: {', '.join(available_sheets)}."
                )
            return (
                "I'm not sure which sheet you mean. "
                "Please say: delete sheet <name>."
            )
        # Single candidate that doesn't match any known sheet.
        if available_sheets is not None:
            if not any(s.lower() == name_words[0] for s in available_sheets):
                return (
                    "Which sheet should I delete? "
                    f"Available sheets are: {', '.join(available_sheets)}."
                )

    elif intent == 'switch_sheet':
        # Extract candidate sheet-name words (strip command verbs / filler)
        _SWITCH_VERBS = {'switch', 'go', 'navigate', 'to', 'sheet', 'the'}
        name_words = [
            w.rstrip('.,!?…') for w in words
            if w.rstrip('.,!?…') not in _COMMAND_VERBS
            and w.rstrip('.,!?…') not in _SLOT_GARBAGE
            and w.rstrip('.,!?…') not in _SWITCH_VERBS
            and w.rstrip('.,!?…')
        ]
        if not name_words:
            # "switch sheet Sheet" — the target IS the word "sheet" (a real tab name).
            # words is already lowercased: count 'sheet' occurrences.
            # 1 = only the syntactic marker ("switch sheet" with nothing after) → clarify.
            # 2 = marker + target name "Sheet"                                  → allow.
            # 3+ = "switch sheet sheet sheet" triple-repetition garbage          → clarify.
            if words.count('sheet') == 2:
                name_words = ['sheet']
            else:
                return "Which sheet should I switch to? Please say the sheet name."
        # Profanity in the name slot → always block (e.g. "switch sheet shit").
        # Bare numbers are NO LONGER blocked here: a sheet named "18" is valid and
        # the Excel layer resolves case-insensitively, returning sheet_not_found when
        # the number genuinely has no matching tab.
        if any(w in _PROFANITY_TOKENS for w in name_words):
            return (
                "I didn't catch the sheet name clearly. "
                "Which sheet would you like to go to?"
            )
        # Multiple residual tokens almost always mean garbled STT — valid sheet
        # names in Sara's context are single words.  Sending "sweet egg" or similar
        # multi-token noise to Ollama/Groq wastes 20-30 s and always fails.
        if len(name_words) > 1:
            if available_sheets:
                return (
                    "Which sheet should I switch to? "
                    f"Available sheets are: {', '.join(available_sheets)}."
                )
            return (
                "I'm not sure which sheet you mean. "
                "Please say: switch sheet <name>."
            )
        # When the caller provides the open workbook's sheet list, verify the
        # single candidate actually exists — catches STT that produces one
        # plausible-looking but wrong token (e.g. "sweet" for "Style").
        if available_sheets is not None:
            if not any(s.lower() == name_words[0] for s in available_sheets):
                return (
                    "Which sheet should I switch to? "
                    f"Available sheets are: {', '.join(available_sheets)}."
                )

    elif intent == 'create_sheet':
        _CREATE_VERBS = {'create', 'add', 'make', 'new', 'a', 'sheet'}
        # Strip trailing punctuation so "by..." is normalised to "by" before the
        # garbage check — Whisper appends "..." when the user trails off mid-phrase.
        clean = [w.rstrip('.,!?…') for w in words]
        name_words = [
            w for w in clean
            if w and w not in _COMMAND_VERBS
            and w not in _SLOT_GARBAGE
            and w not in _CREATE_VERBS
        ]
        if not name_words:
            return "What should I call the new sheet? Please say a name."
        if (any(w in _PROFANITY_TOKENS for w in name_words)
                or any(re.match(r'^\d+$', w) for w in name_words)):
            return "That doesn't sound like a valid sheet name. What should I call the new sheet?"

    elif intent == 'create_excel':
        _CREATE_FILE_VERBS = {
            'create', 'creat', 'great', 'new', 'make', 'a', 'an',
            'excel', 'file', 'spreadsheet', 'workbook',
        }
        clean = [w.rstrip('.,!?…') for w in words]
        name_words = [
            w for w in clean
            if w and w not in _COMMAND_VERBS
            and w not in _SLOT_GARBAGE
            and w not in _CREATE_FILE_VERBS
        ]
        if not name_words:
            return (
                "What should I call the new Excel file? "
                "Please say a filename, for example: create excel Teacher."
            )

    elif intent == 'add_column':
        _COL_VERBS = {'add', 'insert', 'create', 'column', 'col', 'a', 'an', 'new'}
        name_words = [
            w for w in words
            if w not in _COMMAND_VERBS
            and w not in _SLOT_GARBAGE
            and w not in _COL_VERBS
        ]
        if not name_words:
            return (
                "What should I call the new column? "
                "Please say a column name, for example: add column Score."
            )

    elif intent == 'rename_sheet':
        _RENAME_VERBS = {'rename', 'sheet', 'the', 'as', 'call', 'called'}
        _clean_r = [w.rstrip('.,!?…') for w in words]
        if 'to' in _clean_r:
            # "rename sheet [old] to [new]" — split on "to" to find each name
            to_idx = _clean_r.index('to')
            _pre_to = [w for w in _clean_r[:to_idx]
                       if w and w not in _COMMAND_VERBS
                       and w not in _SLOT_GARBAGE and w not in _RENAME_VERBS]
            _post_to = [w for w in _clean_r[to_idx + 1:]
                        if w and w not in _COMMAND_VERBS and w not in _SLOT_GARBAGE]
            if not _pre_to:
                if _post_to:
                    return (
                        f"Which sheet should I rename to '{_post_to[0]}'? "
                        "Say: rename sheet [name] to [new name]."
                    )
                return (
                    "Which sheet should I rename? "
                    "Say: rename sheet [name] to [new name]."
                )
            # Old name is profanity → it's a misrecognition.
            # Bare numbers are allowed: a sheet named "18" is a valid target.
            if any(w in _PROFANITY_TOKENS for w in _pre_to):
                return (
                    "Which sheet should I rename? "
                    "Say: rename sheet [name] to [new name]."
                )
            # Old name not in known sheets → clarify with the list.
            if available_sheets is not None:
                if not any(s.lower() == _pre_to[0].lower() for s in available_sheets):
                    return (
                        "Which sheet should I rename? "
                        f"Available sheets are: {', '.join(available_sheets)}."
                    )
            if not _post_to:
                return f"What should I rename '{_pre_to[-1]}' to?"
        else:
            name_words = [
                w for w in _clean_r
                if w and w not in _COMMAND_VERBS
                and w not in _SLOT_GARBAGE and w not in _RENAME_VERBS
            ]
            if not name_words:
                return (
                    "Which sheet should I rename, and what should the new name be? "
                    "For example: rename sheet Sheet1 to Summary."
                )
            if any(w in _PROFANITY_TOKENS for w in name_words):
                return (
                    "Which sheet should I rename, and what should the new name be? "
                    "For example: rename sheet Sheet1 to Summary."
                )
            if len(name_words) == 1:
                return (
                    f"I heard '{name_words[0]}' as the sheet name — "
                    "what should it be renamed to?"
                )

    elif intent == 'format_cells':
        # "fill/color/highlight row" without a row number → ask for number.
        # Catches "fill a row with red" produced by roll/roto → row correction.
        # Excludes "color row A1 blue" — the cell ref IS the valid target.
        if (re.search(r'\b(?:fill|color|colour|highlight)\b', lower)
                and re.search(r'\brow\b', lower)
                and not re.search(r'\brow\s+\d+\b', lower)
                and not re.search(r'\brow\s+[A-Za-z]\d+\b', lower)):
            return "Which row number? For example: fill row 3 with red."
        # "align column 3 right" — column must be a letter, not a number
        if re.search(r'\balign\s+column\s+\d+\b', lower):
            return "Column references must use a letter, like 'align column B right'."
        # "align row to left" / "align row left" — row number is missing
        if re.search(r'\balign\s+row\b', lower) and not re.search(r'\balign\s+row\s+\d+\b', lower):
            return "Which row number? Please say something like 'align row 3 left'."
        # "align A2" / "align column B" with no direction word → ask clarification
        if re.search(r'\balign\b', lower) and not re.search(r'\b(?:left|right|center|centre)\b', lower):
            _has_align_target = bool(re.search(
                r'(?:row\s+\d+|column\s+[A-Za-z]+|[A-Za-z]+\d+)', lower
            ))
            if _has_align_target:
                return "Which direction should I align? Please say left, right, or center."
        # "set font-sign D130" / "set font-[anything]" — garbled hyphenated font attribute.
        # Valid hyphenated forms: font-size, font-color/colour, font-bold, font-italic,
        # font-style, font-name, font-face.  Anything else → clarify immediately.
        _font_hyphen_m = re.search(r'\bset\s+font-(\w+)\b', lower)
        if _font_hyphen_m:
            _valid_font_attrs = {'size', 'color', 'colour', 'bold', 'italic', 'style', 'name', 'face'}
            if _font_hyphen_m.group(1) not in _valid_font_attrs:
                return (
                    "I didn't catch that font command. "
                    "Did you mean set font size? For example: set font size A2 20."
                )
        # Font-size slot checks — covers set/increase/decrease, bare "font size",
        # and "make [word] size N" (catches STT corruptions like "make hat size 2").
        _font_size_cmd = bool(
            re.search(r'\bset\s+(?:font\s+)?size\b', lower) or
            re.search(r'\b(?:increase|decrease)\s+(?:font\s+)?size\b', lower) or
            re.search(r'\bfont\s+size\b', lower) or
            re.search(r'\bmake\s+\w+\s+(?:font\s+)?size\b', lower)
        )
        if _font_size_cmd:
            # "header" counts as a valid target for "make header size 18" forms.
            _fs_target = bool(re.search(
                r'[A-Za-z]+\d+|row\s+\d+|column\s+[A-Za-z]+|\bheader\b', lower
            ))
            # Also treat digits embedded in "column B14" as the size number:
            # cell-ref normalization merges "B 14" → "B14" so \b\d+\b misses them.
            _fs_num = bool(
                re.search(r'\b\d+\b', lower) or
                re.search(r'\bcolumn\s+[A-Za-z]\d+\b', lower)
            )
            _fs_delta = bool(re.search(r'\b(?:increase|decrease)\b', lower))
            if _fs_target and not _fs_num and not _fs_delta:
                return "What font size should I use?"
            if _fs_num and not _fs_target:
                return "Which cell or range should I set the font size for? For example: set font size A2 30."
            if not _fs_target and not _fs_num and not _fs_delta:
                return "Which cell and what font size? For example: set font size A2 30."
            if _fs_delta and not _fs_target:
                return "Which cell or range should I change the font size for? For example: increase font size A2."
        # "format JFI as number" — single-token range not a valid cell/range/column → clarify
        _fmt_as = re.search(
            r'\bformat\s+(\S+)\s+as\s+(?:number|integer|currency|percentage|percent|decimal|date|text)\b',
            lower,
        )
        if _fmt_as:
            _rng = _fmt_as.group(1)
            _valid_rng = bool(
                re.fullmatch(r'[a-z]+\d+(?::[a-z]+\d+)?', _rng) or
                re.match(r'^column\s+[a-z]+$', _rng)
            )
            if not _valid_rng:
                return (
                    "Which cell range should I format? "
                    "For example: format B2:B10 as currency."
                )

        # ── Number-format slot checks ──────────────────────────────────────────
        if re.search(r'\bformat\b', lower):
            _VALID_NUM_FMT = frozenset({
                'number', 'integer', 'currency', 'percentage', 'percent', 'decimal', 'date', 'text',
            })
            _nf_tokens = set(lower.split())
            _nf_has_as = bool(re.search(r'\bas\b', lower))
            _nf_has_target = bool(re.search(
                r'(?:row\s+\d+|column\s+[a-z]+|[a-z]+\d+(?::[a-z]+\d+)?)', lower
            ))
            _nf_has_valid_fmt = bool(_nf_tokens & _VALID_NUM_FMT)
            # "format as currency" — format type present but no target
            if _nf_has_as and _nf_has_valid_fmt and not _nf_has_target:
                return (
                    "Which cell or range should I format? "
                    "For example: format B2:B10 as currency."
                )
            # "format B2:B10" or "format column B" — target present but no "as <format>"
            # Guard: skip when bold/italic/align qualifiers are already the format type
            _nf_has_other_qual = bool(
                re.search(r'\b(?:bold|italic|center|centre|align|left|right)\b', lower)
            )
            if _nf_has_target and not _nf_has_as and not _nf_has_other_qual:
                return (
                    "Which format should I apply? "
                    "For example: format B2:B10 as currency."
                )
            # "format B2 as xyz" — "as" present + target present but unrecognised format name
            if _nf_has_as and _nf_has_target and not _nf_has_valid_fmt:
                return (
                    "I don't recognise that format. "
                    "Supported: currency, percentage, number, decimal."
                )

        # ── Color / fill / highlight slot checks ──────────────────────────────
        _COLOR_SLOT_WORDS = frozenset({
            'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink',
            'black', 'white', 'gray', 'grey', 'cyan', 'magenta', 'brown',
            'lime', 'teal', 'navy', 'maroon', 'olive', 'coral', 'salmon',
        })
        _lower_tokens = set(lower.split())

        # "highlight" with no cell/row/col target → clarify
        if re.search(r'\bhighlight\b', lower):
            _hl_target = bool(re.search(
                r'(?:row\s+\d+|column\s+[a-z]+|[a-z]+\d+(?::[a-z]+\d+)?)', lower
            ))
            if not _hl_target:
                return (
                    "Which cell or row should I highlight? "
                    "For example: highlight row 4, or highlight A1:C1."
                )

        # "color"/"colour" or "fill" as action verb — check target + color word
        _is_color_verb = bool(
            re.search(r'\bcolou?r\b', lower) and
            not re.search(r'\bfont[\s-]colou?r\b', lower)
        )
        _is_fill_verb = bool(re.search(r'\bfill\b', lower))
        if _is_color_verb or _is_fill_verb:
            _fc_target = bool(re.search(
                r'(?:row\s+\d+|column\s+[a-z]+|[a-z]+\d+(?::[a-z]+\d+)?)', lower
            ))
            _fc_color = bool(_lower_tokens & _COLOR_SLOT_WORDS)
            if not _fc_target and _fc_color:
                return (
                    "Which cell or range should I color? "
                    "For example: color B2 red, or fill A4 pink."
                )
            if _fc_target and not _fc_color:
                _verb = 'fill' if _is_fill_verb else 'color'
                return f"Which color? For example: {_verb} A3 orange."

    elif intent == 'calculate_sum':
        # Need a cell ref (e.g. B2) OR a column reference (e.g. column A) to identify the range.
        cell_refs = re.findall(r'\b[A-Za-z]\d+\b', text)
        has_column_ref = bool(re.search(r'\bcolumn\s+[A-Za-z]\b', lower))
        if not cell_refs and not has_column_ref:
            return (
                "Which cells should I sum? "
                "For example: sum B2 to B5 into C6."
            )
        # "in/into [word]" is present but word is not a valid cell ref → destination garbled
        dest_m = re.search(r'\bin(?:to)?\s+(\w+)\b', lower)
        if dest_m:
            dest = dest_m.group(1).rstrip('.,!?…')
            if not re.match(r'^[A-Za-z]\d+$', dest):
                return (
                    "Which cell should I put the sum in? "
                    "For example: sum B2 to B5 into C6."
                )

    elif intent == 'calculate_average':
        # Range+target form: "average B2:B5 into C7".
        # If "into/to" is present but no valid cell ref follows, the destination is
        # missing — ask before attempting heuristic parse or LLM routing.
        if re.search(r'\b(?:in(?:to)?|to)\b', lower):
            if not re.search(r'\b(?:in(?:to)?|to)\s+[A-Za-z]+\d+\b', lower):
                return (
                    "Which cell should I write the average into? "
                    "For example: average B2:B5 into C7."
                )

    elif intent == 'filter_table':
        # Need at least a column identifier AND a value.
        # Strip filler words, then count remaining tokens.
        _FILTER_FILLER = {'filter', 'column', 'by', 'where', 'the',
                          'equals', 'equal', 'is', 'are'}
        _ftokens = [
            w.rstrip('.,!?…') for w in lower.split()
            if w.rstrip('.,!?…') and w.rstrip('.,!?…') not in _FILTER_FILLER
        ]
        if len(_ftokens) == 0:
            return (
                "Which column and value should I filter by? "
                "For example: filter column B 90."
            )
        if len(_ftokens) == 1:
            tok = _ftokens[0]
            if re.match(r'^[a-z]$', tok):        # single letter → column without value
                return f"What value should I filter column {tok.upper()} by?"
            else:                                  # value without column (or ambiguous)
                return (
                    "Which column should I filter, and by what value? "
                    "For example: filter column B 90."
                )
        if len(_ftokens) >= 2:
            # Identify column token (first single-letter) and remaining value tokens.
            _col_found = False
            _val_toks: list[str] = []
            for _tok in _ftokens:
                if not _col_found and re.match(r'^[a-z]$', _tok):
                    _col_found = True
                else:
                    _val_toks.append(_tok)
            _suspected = ' '.join(_val_toks)
            if _suspected in _FILTER_VALUE_JUNK:
                return (
                    f"I heard ‘{_suspected}’ as the filter value — "
                    "that doesn’t look like data. "
                    "Please say: filter column B 90."
                )

    elif intent == 'sort_range':
        # Require an explicit direction word.  Silently defaulting to ascending
        # when the direction is absent or STT-corrupted would execute the wrong
        # sort order — a silent data mutation that is hard to notice.
        if not re.search(r'\b(ascending|descending)\b', lower):
            return (
                "Should I sort ascending or descending? "
                "Please say: sort by column A ascending."
            )
        # "sort by V descending" — 'V' is a common STT mishear of 'B' in some
        # accents and microphone setups.  Ask before executing rather than silently
        # using the wrong column.
        if re.search(r'\bby\s+v\s+(?:ascending|descending|asc|desc)\b', lower):
            _v_dir_m = re.search(r'\b(ascending|descending)\b', lower)
            _v_dir   = _v_dir_m.group(1) if _v_dir_m else 'ascending'
            return (
                f"Did you mean column B {_v_dir}? "
                "If you meant column V, please say it again clearly."
            )
        # "sort by pre-ascending" — STT garbles a column+direction into a single
        # hyphenated token (e.g. "pre-ascending" for "B ascending").
        # No valid sort command uses a hyphenated column-direction form, so any
        # match here is always a garble — block unconditionally and ask.
        _sort_hyph = re.search(r'\bby\s+(\w+)-(ascending|descending)\b', lower)
        if _sort_hyph:
            _dir_part = _sort_hyph.group(2)
            return (
                f"I heard '{_sort_hyph.group(1)}' as the column but that doesn't look right. "
                f"Did you mean column B {_dir_part}?"
            )
        # Lowercase 2-letter column starting with "b" after "by" likely means
        # STT heard "B" and output a phonetically-similar 2-letter sequence.
        # Uses the original (non-lowercased) text so typed uppercase "BD"/"BE"
        # columns pass through; only voice-style lowercase "bd"/"be"... (rare
        # post-correction residual) triggers the clarification.
        _sort_col_m = re.search(
            r'\bby\s+([a-z]{2})\s+(?:ascending|descending|asc|desc)\b', text
        )
        if _sort_col_m and _sort_col_m.group(1)[0] == 'b':
            _col = _sort_col_m.group(1).upper()
            return (
                f"Did you mean column B? "
                f"If you meant column {_col}, please say it again clearly."
            )

    return None


# ---------------------------------------------------------------------------
# 10. Low-confidence message helper
# ---------------------------------------------------------------------------

def describe_low_confidence(text: str) -> str | None:
    """Return a targeted retry message if *text* is low-confidence, else None.

    Combines the :func:`is_low_confidence_transcript` filter with a reason-
    specific message so the conversation loop can give better feedback than
    a generic "please repeat" prompt.

    Examples::

        describe_low_confidence("18")    → "I heard a number only…"
        describe_low_confidence("taste") → "I didn't catch that clearly…"
        describe_low_confidence("add row john 90") → None  (not filtered)
    """
    if not is_low_confidence_transcript(text):
        return None

    normalized = text.strip().lower().rstrip('.!?,')

    # Bare number — specific message
    if _SLOT_BARE_NUM_RE.match(normalized):
        return (
            "I heard a number only. "
            "Please say the full command, for example: add row John 90."
        )

    # Single word that matches a known command action → not really noise, but
    # the low-confidence filter still flagged it — give a gentle hint.
    if ' ' not in normalized and detect_command_intent(text) is not None:
        return f"I heard '{normalized}' — could you say the full command?"

    # Bare navigation stub — two-word partial command that could mean two things.
    # "go to" alone is ambiguous: sheet switch vs cell navigation.
    if normalized == 'go to':
        return (
            "Did you mean to switch to a sheet or navigate to a cell? "
            "Say 'switch sheet <name>' or 'go to cell B3'."
        )

    return "I didn't catch that clearly. Could you please repeat?"


def is_formula_safe(formula: str) -> bool:
    """Return True if *formula* passes all structural and whitelist checks.

    Delegates to :func:`get_formula_error` so both functions share exactly
    the same logic.  Callers that only need a pass/fail result (e.g.
    ``tool_registry._excel_write_formula``) can keep calling this function;
    callers that need the specific reason should call ``get_formula_error``
    directly.

    Usage::

        if not is_formula_safe(formula):
            return None  # reject in _validate_single_action
    """
    return get_formula_error(formula) is None
