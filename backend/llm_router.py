"""LLM router for Sara.

Hybrid router with a three-tier fallback chain and comprehensive failure
classification.

Chain::

    route_llm(text)
    │
    ├─ Step 1 — Heuristic regex parser (instant, always first)
    │     • ~80 % of common Excel commands matched here in < 1 ms
    │     • returns immediately — no LLM called, no network cost
    │
    ├─ Step 2 — Ollama (local, no internet required)
    │     • skipped when USE_OLLAMA=false
    │     • longer timeout (OLLAMA_TIMEOUT, default 120 s) for cold-start
    │     • up to LLM_MAX_RETRIES retries on transient errors
    │     • preferred over Groq because it is local-first and always available
    │
    └─ Step 3 — Groq (cloud, final fallback)
          • only reached when Ollama unavailable or disabled
          • requires GROQ_API_KEY and internet connectivity
          • short timeout (GROQ_TIMEOUT, default 8 s) for fast voice UX
          • 1 retry on 429/5xx; timeouts are NOT retried (fail fast)

Every attempt — success or failure — is recorded in the metrics DB so
per-backend failure rates are measurable for thesis evaluation.

Configuration (environment variables / .env)::

    USE_OLLAMA=true       – enable Ollama (default: true)
    OLLAMA_MODEL=qwen3:8b – model pulled in Ollama
    OLLAMA_BASE_URL       – default http://localhost:11434
    OLLAMA_TIMEOUT        – seconds, default 120
    LLM_MAX_RETRIES       – max retries for Ollama transient errors (default: 3)
    LLM_BASE_BACKOFF      – initial backoff in seconds (default: 0.5)
    GROQ_API_KEY          – required for Groq cloud fallback
    GROQ_TIMEOUT          – Groq request timeout in seconds (default: 8)
    GROQ_MAX_RETRIES      – max retries for Groq 429/5xx (default: 1)
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
import requests

# ── Persistent HTTP session for Ollama ───────────────────────────────────────
# Reuses the underlying TCP connection across requests so each Ollama call
# does not pay the TCP handshake cost (~10–50 ms on localhost).
# Thread-safe: requests.Session connection pool handles concurrent callers.
_OLLAMA_SESSION = requests.Session()

from metrics import metrics as _metrics
from llm_parser import (
    _parse_groq_direct,
    RobustJsonExtractor,
    _validate_llm_result,
    _plan_and_validate,
    _heuristic_natural_language_parse,
)

try:
    from pipeline_sanitizer import sanitize_history_turn as _sanitize_history_turn
except ImportError:
    def _sanitize_history_turn(text: str) -> str:  # type: ignore[misc]
        return text

log = logging.getLogger('llm_router')

# ── Configuration ───────────────────────────────────────────────────────────

USE_OLLAMA         = os.environ.get('USE_OLLAMA', 'true').lower() in ('true', '1', 'yes')
OLLAMA_BASE_URL    = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL       = os.environ.get('OLLAMA_MODEL', 'qwen2.5:3b')
OLLAMA_TIMEOUT     = int(os.environ.get('OLLAMA_TIMEOUT', '120'))

# GROQ_FIRST=true  → try Groq BEFORE Ollama when internet is available.
# This gives faster responses (~1 s vs 5–30 s) when the user is online.
# Default: false (Ollama first — preserves the original offline-first behaviour).
# Set GROQ_FIRST=true in backend/.env for online-priority mode.
GROQ_FIRST         = os.environ.get('GROQ_FIRST', 'false').lower() in ('true', '1', 'yes')

# Groq-specific timeout and retry cap.  Keep GROQ_TIMEOUT short (≤ 10 s) so
# that slow/unavailable Groq fails fast and hands off to Ollama quickly.
GROQ_TIMEOUT       = int(os.environ.get('GROQ_TIMEOUT', '8'))
GROQ_MAX_RETRIES   = int(os.environ.get('GROQ_MAX_RETRIES', '1'))
GROQ_MODEL_NAME    = os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant')

# Ollama retry settings (longer, model may be loading)
MAX_RETRIES        = int(os.environ.get('LLM_MAX_RETRIES', '3'))
BASE_BACKOFF       = float(os.environ.get('LLM_BASE_BACKOFF', '0.5'))

# HTTP status codes considered transient (worth retrying)
_TRANSIENT_CODES   = {429, 500, 502, 503, 504}

# ── Failure reason constants ─────────────────────────────────────────────────
# Stored in metrics so per-backend failure modes are queryable for the thesis.

FR_OK           = 'ok'
FR_TIMEOUT      = 'timeout'
FR_AUTH         = 'auth_failure'       # 401 / 403
FR_RATE_LIMITED = 'rate_limited'       # 429 after all retries
FR_SERVICE_DOWN = 'service_down'       # 5xx after all retries
FR_PARSE_ERROR  = 'parse_error'        # valid HTTP 200 but unreadable JSON
FR_VALIDATION   = 'validation_error'   # JSON parsed but action type invalid
FR_UNAVAILABLE  = 'unavailable'        # connection refused / DNS failure
FR_UNEXPECTED   = 'unexpected_error'   # anything else
FR_ALL_FAILED   = 'all_failed'         # sentinel used in final catch-all

# ── Ollama keep-alive ─────────────────────────────────────────────────────────
# Controls how long Ollama keeps the model loaded in GPU/RAM after the last
# request.  Default "30m" prevents the 5-30s cold-start between sessions.
# Set OLLAMA_KEEP_ALIVE=-1 to never unload.  Set to "0" to unload immediately.
_keep_alive_raw = os.environ.get('OLLAMA_KEEP_ALIVE', '30m')
try:
    OLLAMA_KEEP_ALIVE = int(_keep_alive_raw)   # bare int ("-1", "0") → JSON number (seconds)
except ValueError:
    OLLAMA_KEEP_ALIVE = _keep_alive_raw        # duration string ("30m", "5m") → JSON string

# ── Connectivity cache ────────────────────────────────────────────────────────
# Avoids a 2-second HEAD request on every command.

_CONN_CACHE_TTL = 30.0
_conn_cache: dict = {"result": None, "ts": 0.0}

# ── Groq circuit-breaker state ────────────────────────────────────────────────
# After GROQ_CB_THRESHOLD consecutive Groq failures (timeout or 5xx), Sara
# stops trying Groq for GROQ_CB_COOLDOWN seconds.  This avoids paying
# GROQ_TIMEOUT (8 s) on every voice command when Groq is degraded.
# The counter resets on the first successful Groq response.

GROQ_CB_THRESHOLD = 3       # failures before tripping
GROQ_CB_COOLDOWN  = 60.0    # seconds to stay tripped

_groq_cb: dict = {
    "failures":   0,       # consecutive failure count
    "tripped_at": 0.0,     # monotonic timestamp when circuit opened
}


def _groq_cb_tripped() -> bool:
    """Return True if the Groq circuit breaker is currently open."""
    state = _groq_cb
    if state["failures"] < GROQ_CB_THRESHOLD:
        return False
    elapsed = time.monotonic() - state["tripped_at"]
    if elapsed >= GROQ_CB_COOLDOWN:
        # Cooldown expired — reset and allow one probe attempt
        state["failures"] = 0
        state["tripped_at"] = 0.0
        log.info('[Groq-CB] Cooldown expired — circuit closed (half-open probe)')
        return False
    return True


def _groq_cb_record_success() -> None:
    """Reset circuit breaker on a successful Groq call."""
    _groq_cb["failures"] = 0
    _groq_cb["tripped_at"] = 0.0


def _groq_cb_record_failure() -> None:
    """Increment failure counter; trip circuit if threshold reached."""
    _groq_cb["failures"] += 1
    if _groq_cb["failures"] >= GROQ_CB_THRESHOLD:
        if _groq_cb["tripped_at"] == 0.0:
            _groq_cb["tripped_at"] = time.monotonic()
            log.warning(
                '[Groq-CB] Circuit TRIPPED after %d consecutive failures — '
                'Groq will be skipped for %.0f s',
                _groq_cb["failures"], GROQ_CB_COOLDOWN,
            )

# ── Conversation Memory (Improvement 2) ──────────────────────────────────────
# Keeps the last N user/assistant turns so follow-up commands are understood.
# Buffer is cleared at the start of each new wake-word session.

class ConversationBuffer:
    """Lightweight ring buffer for multi-turn conversation context.

    Stores up to *max_turns* ``{"user": str, "assistant": str}`` pairs.
    Uses ``collections.deque(maxlen=N)`` so oldest entries are evicted in O(1)
    without shifting the underlying list.  Only successfully executed turns
    should be added — clarification and error responses add noise.
    """

    def __init__(self, max_turns: int = 5):
        self._history: deque[dict] = deque(maxlen=max_turns)

    def add(self, *, user: str, assistant: str) -> None:
        """Append a completed turn.  Oldest entry auto-evicts when full."""
        self._history.append({"user": user, "assistant": assistant})

    def get_history(self) -> list[dict]:
        """Return a plain list copy of the current history."""
        return list(self._history)

    def clear(self) -> None:
        """Discard all history — call when a new wake-word session begins."""
        self._history.clear()

    def format_for_prompt(self) -> str:
        """Return a compact history block to prepend to an LLM prompt.

        Example output::

            [Previous conversation:
            User: add row Tom 90
            Sara: Row added.
            User: now calculate average
            Sara: The average score is 90.
            ]

        Used by ``_build_ollama_prompt``; kept as a public method so callers
        outside the router can format history without reimplementing the layout.
        """
        if not self._history:
            return ''
        lines = ['[Previous conversation:']
        for turn in self._history:
            lines.append(f'User: {turn["user"]}')
            lines.append(f'Sara: {turn["assistant"]}')
        lines.append(']')
        return '\n'.join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────
# Shared by both LLM backends so output is schema-compatible.

SYSTEM_PROMPT = """CRITICAL RULE: Return ONLY valid JSON. No explanations. No markdown. No code blocks. No text before or after the JSON object.
If you cannot parse the command, return exactly: {"type":"unknown"}

You are an expert office assistant. Convert spoken Excel commands into structured JSON actions.
Infer intent intelligently — do NOT be a strict keyword parser.

========================
OUTPUT CONTRACT (STRICT)
========================
Return ONLY valid JSON. No markdown. No explanations.
You MUST choose from the allowed actions only. Do NOT invent or use tool names that are not listed below.

WRONG (never do this):
  Here is the JSON: {"type":"add_row",...}
  {"row": [{"name": "john", "score": 90}]}
  ```json {"type":"add_row"} ```

CORRECT (do exactly this):
  {"type":"add_row","values":["John",90]}

Preferred shapes:
1) Single action:
    {"type":"add_row","values":["Tom",90]}

2) Multi-step plan when command has multiple operations:
    {"steps":[
      {"type":"create_excel","headers":["Name","Score"]},
      {"type":"add_row","values":["Tom",90]},
      {"type":"add_row","values":["Alex",85]}
    ]}

Use "steps" whenever there are 2+ operations in one sentence.

========================
ALLOWED ACTION TYPES
========================
- create_excel:  fields -> headers (required list e.g. ["Name","Score"]), filename (optional)
- open_excel_app: no extra fields (launches the Excel application)
- open_file:      no extra fields (opens the active workbook in Excel)
- create_sheet:   fields -> sheet_name (required)
- switch_sheet:   fields -> sheet_name (required)
- add_row:        fields -> values (required list, e.g. ["Tom", 90])
- add_column:     fields -> name (required string)
- read_cell:      fields -> cell (required e.g. "B3")
- write_cell:     fields -> cell (required e.g. "A1"), value (required)
- write_formula:  fields -> cell (required e.g. "B10"), formula (required e.g. "=SUM(B2:B9)")
- create_chart:   fields -> chart_type (optional: bar/line/pie/scatter), data_range (optional e.g. "A1:C10"), title (optional)
- format_cells:   fields -> range (required e.g. "A1:C10"), bold/italic/font_size/font_color/fill_color (all optional)
- undo_last:      no extra fields (reverts the last change)

DO NOT use any other type name. If none of the above fits, return {"type":"unknown"}.

========================
DOMAIN HEADER INFERENCE TABLE
========================
When the user mentions one of the following topics WITHOUT giving explicit column names,
automatically use the corresponding headers:

  Topic keywords                       | Inferred headers
  -------------------------------------|-------------------------------------------
  score, grades, marks, result, exam   | ["Name", "Score"]
  student score, grade sheet           | ["Name", "Score"]
  attendance, present, absent, roll    | ["Name", "Date", "Status"]
  inventory, stock, items, products    | ["Item", "Quantity"]
  expense, expenses, cost, costs,      | ["Item", "Cost"]
    payment                            |
  budget, finance, spending            | ["Category", "Budget", "Actual"]
  sales, revenue, orders               | ["Product", "Quantity", "Revenue"]
  employee, staff, hr, payroll         | ["Name", "Department", "Salary"]
  project, task, todo, milestone       | ["Task", "Owner", "Status", "Due Date"]
  meeting, discussion, agenda          | ["Topic", "Date", "Participants"]

If the user provides explicit column names, always use those instead.

========================
INTENT INFERENCE RULES
========================
1) Identify the domain from keywords and apply the header table above when no explicit headers are given.
2) If user says create/make/build + add/insert in one request, always produce a "steps" plan.
3) If multiple data records are mentioned in one command, output one add_row per record.
4) Convert word numbers to integers when obvious ("ninety" -> 90, "ten" -> 10, "twenty-five" -> 25).
5) Keep person/item names as strings with natural capitalization.
6) For attendance: map "present"/"yes"/"here" -> "Present", "absent"/"no" -> "Absent" in the Status field.
7) For inventory/expense amounts: treat the numeric value as Quantity or Cost respectively.
8) Prefer a concrete guess over returning unknown - if intent is at all inferable, produce a valid action.
9) Use {"type":"unknown","raw":"..."} ONLY when there is truly zero actionable intent.

========================
FEW-SHOT EXAMPLES
========================
Input: "add row Tom 90"
Output: {"type":"add_row","values":["Tom",90]}

Input: "create file with headers name, age"
Output: {"type":"create_excel","headers":["Name","Age"]}

Input: "delete the 2nd row"
Output: {"type":"delete_row","index":2}

Input: "write 100 to B3"
Output: {"type":"write_cell","cell":"B3","value":100}

Input: "write formula SUM(A1:A10) in cell A11"
Output: {"type":"write_formula","cell":"A11","formula":"=SUM(A1:A10)"}

Input: "calculate total using sum formula in B10 for B2 to B9"
Output: {"type":"write_formula","cell":"B10","formula":"=SUM(B2:B9)"}

Input: "Create an excel sheet for student score and add Tom with 90 and Alex with 85"
Output: {"steps":[{"type":"create_excel","headers":["Name","Score"]},{"type":"add_row","values":["Tom",90]},{"type":"add_row","values":["Alex",85]}]}

Input: "what is in cell B3"
Output: {"type":"read_cell","cell":"B3"}

Input: "switch to Sheet2"
Output: {"type":"switch_sheet","sheet_name":"Sheet2"}

Input: "create a new sheet called Sales"
Output: {"type":"create_sheet","sheet_name":"Sales"}

Input: "undo"
Output: {"type":"undo"}
"""

# ── Exception helpers ─────────────────────────────────────────────────────────

def _classify_exc(exc: Exception) -> str:
    """Map an exception to a failure-reason constant."""
    if isinstance(exc, requests.Timeout):
        return FR_TIMEOUT
    if isinstance(exc, requests.ConnectionError):
        return FR_UNAVAILABLE
    if isinstance(exc, requests.HTTPError):
        code = exc.response.status_code if exc.response is not None else 0
        if code in (401, 403):
            return FR_AUTH
        if code == 429:
            return FR_RATE_LIMITED
        if code in _TRANSIENT_CODES:
            return FR_SERVICE_DOWN
    return FR_UNEXPECTED


def _is_transient_error(exc: Exception, *, retry_timeouts: bool = True) -> bool:
    """Return True if *exc* is worth retrying.

    ``retry_timeouts=False`` causes ``requests.Timeout`` to be treated as
    non-transient so it is not retried.  Use this for Groq to ensure a single
    slow response does not block the fallback chain for up to
    ``MAX_RETRIES * GROQ_TIMEOUT`` seconds.
    """
    if isinstance(exc, requests.Timeout):
        return retry_timeouts
    if isinstance(exc, requests.ConnectionError):
        return True
    if isinstance(exc, requests.HTTPError):
        resp = exc.response
        if resp is not None and resp.status_code in _TRANSIENT_CODES:
            return True
    return False


# ── Retry helper ─────────────────────────────────────────────────────────────

def _retry_with_backoff(fn, *, label: str,
                        max_retries: int | None = None,
                        retry_timeouts: bool = True):
    """Call *fn()* with exponential backoff on transient failures.

    Parameters:
        max_retries:    override the global ``MAX_RETRIES`` cap.
        retry_timeouts: when False, ``requests.Timeout`` causes immediate
                        re-raise (no retry).  Set False for Groq so a slow
                        cloud response does not block for multiple timeouts.
    """
    cap = max_retries if max_retries is not None else MAX_RETRIES
    last_exc: Exception | None = None
    for attempt in range(1, cap + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not _is_transient_error(exc, retry_timeouts=retry_timeouts):
                log.warning('[Retry:%s] non-transient on attempt %d/%d: %s',
                            label, attempt, cap, exc)
                raise
            delay = BASE_BACKOFF * (2 ** (attempt - 1))
            log.warning('[Retry:%s] transient on attempt %d/%d: %s — retry in %.1fs',
                        label, attempt, cap, exc, delay)
            time.sleep(delay)
    log.error('[Retry:%s] all %d attempts failed', label, cap)
    raise last_exc  # type: ignore[misc]


# ── Connectivity check ────────────────────────────────────────────────────────

def check_internet_connection(timeout: float = 2.0) -> bool:
    """Return True if the machine can reach the public internet.

    Result is cached for ``_CONN_CACHE_TTL`` seconds (default 30 s) to avoid
    a network round-trip on every command.
    """
    now = time.monotonic()
    if _conn_cache["result"] is not None and (now - _conn_cache["ts"]) < _CONN_CACHE_TTL:
        return _conn_cache["result"]
    try:
        requests.head('https://www.google.com', timeout=timeout)
        result = True
    except Exception:
        result = False
    _conn_cache["result"] = result
    _conn_cache["ts"] = now
    return result


def check_groq_available() -> bool:
    """Return True if the Groq API key is configured and the service is reachable."""
    if not os.environ.get('GROQ_API_KEY'):
        return False
    return check_internet_connection()


# ── Groq wrapper ──────────────────────────────────────────────────────────────

def try_groq(text: str) -> dict:
    """Parse *text* using the cloud Groq backend (public API, backward-compat).

    Uses a short timeout (``GROQ_TIMEOUT``) and at most ``GROQ_MAX_RETRIES``
    retries on 429/5xx.  Timeouts are **not** retried so a slow Groq response
    fails fast and lets callers fall through to Ollama.

    Respects the module-level circuit breaker: if Groq has failed
    GROQ_CB_THRESHOLD times in a row, calls are skipped until the cooldown
    expires so Sara does not pay 8 s of timeout on every voice command.

    Returns a structured action dict or ``{"type": "unknown", ...}`` on any
    failure.
    """
    if _groq_cb_tripped():
        log.info('[Groq] Circuit breaker OPEN — skipping Groq call')
        return {"type": "unknown", "raw": text}
    try:
        result = _retry_with_backoff(
            lambda: _parse_groq_direct(text),
            label='Groq',
            max_retries=GROQ_MAX_RETRIES,
            retry_timeouts=False,
        )
        _groq_cb_record_success()
        return result
    except Exception as exc:
        log.warning('[Groq] Failed: %s', exc)
        _groq_cb_record_failure()
        return {"type": "unknown", "raw": text}


# Backward-compatible alias
parse_with_groq = try_groq


# ── Ollama wrapper ────────────────────────────────────────────────────────────

_THINK_TAG_RE = re.compile(r'<think>[\s\S]*?</think>', re.IGNORECASE)

# ── Pre-compiled regexes for route_llm() normalisation ───────────────────────
# These were previously compiled INSIDE route_llm() on every call.
# Moving them here means re.compile() runs once at import, not per command.
_FILLER_RE     = re.compile(
    r'\b(?:the|please|just|kindly|can\s+you|could\s+you)\b\s*', re.IGNORECASE
)
_WHITESPACE_RE  = re.compile(r'\s+')
_AND_ROW_RE     = re.compile(r'\band\s+row\b')
_ROW_START_RE   = re.compile(r'^row\s+')
_OPEN_APP_RE    = re.compile(r'\bopen\s+(?:microsoft\s+)?excel\b')
_OPEN_FILE_RE   = re.compile(r'\bopen\s+(?:spreadsheet|workbook|file|sheet)\b')
_FILENAME_RE    = re.compile(r'\b(\S+\.(?:xlsx|xls|csv|xlsm|xlsb))\b', re.IGNORECASE)


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. qwen3).

    These blocks appear before the JSON answer and waste context window space
    in the debug log.  Stripping them lets RobustJsonExtractor find the JSON
    immediately without scanning through the reasoning chain.
    """
    return _THINK_TAG_RE.sub('', text).strip()


def _build_ollama_prompt(text: str, history: list[dict] | None) -> str:
    """Combine optional conversation history with the current command text.

    Delegates formatting to ``ConversationBuffer.format_for_prompt`` so the
    layout is defined in exactly one place.  Heuristic regex patterns receive
    only the raw *text* — history is injected at the LLM layer only.

    Each history turn is re-sanitized here as a defence-in-depth measure: if
    a turn was stored without sanitization (e.g., a direct call that bypassed
    the runtime), bracket injection is still neutralised before the text reaches
    the model.
    """
    if not history:
        return text
    safe_history = [
        {"user": _sanitize_history_turn(t.get("user", "")),
         "assistant": _sanitize_history_turn(t.get("assistant", ""))}
        for t in history
    ]

    buf = ConversationBuffer()
    buf._history.extend(safe_history)
    prefix = buf.format_for_prompt()
    return f"{prefix}\n\nNew command: {text}"


def _ollama_call(text: str, history: list[dict] | None = None) -> dict:
    """Low-level Ollama API call — non-streaming, no retries.

    Raises on transient errors so the retry wrapper can back off and retry.

    Improvement 1 — internal reasoning is hidden from the user:
        The Ollama response JSON may contain a ``"thinking"`` field (used by
        reasoning models like qwen3).  We deliberately read ONLY the
        ``"response"`` field.  The full body is written to DEBUG logs for
        diagnostics but never passed to TTS or any user-facing code path.

    Improvement 2 — conversation history:
        When *history* is provided the prompt is prefixed with recent turns so
        the LLM can resolve pronouns and follow-up references.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    prompt_text = _build_ollama_prompt(text, history)
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt_text,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,       # keeps model loaded in RAM/VRAM
        "options": {"temperature": 0.0, "num_predict": 150},  # Excel JSON ≤ 100 tokens
    }

    log.info('[Ollama] POST %s  model=%s  prompt_len=%d  timeout=%ds',
             url, OLLAMA_MODEL, len(prompt_text), OLLAMA_TIMEOUT)
    _t0 = time.perf_counter()
    resp = _OLLAMA_SESSION.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    log.info('[Ollama] response received in %.2fs', time.perf_counter() - _t0)
    body = resp.json()

    # ── Improvement 1: isolate the "response" field ───────────────────────
    # The full body (including any "thinking" chain-of-thought) is logged at
    # DEBUG level only — it is NEVER forwarded to TTS or user-facing code.
    if isinstance(body, dict):
        thinking_snippet = str(body.get('thinking', ''))[:120]
        log.debug('[Ollama] full body keys=%s  thinking_snippet=%r',
                  list(body.keys()), thinking_snippet)

    # Only the clean "response" text is used downstream.
    raw_generated = body.get('response', '') if isinstance(body, dict) else ''
    if not raw_generated:
        log.warning('[Ollama] empty response body')
        return {"type": "unknown", "raw": text}

    generated = _strip_thinking_tags(raw_generated)
    log.debug('[Ollama] generated (stripped) = %r', generated[:200])

    try:
        result = RobustJsonExtractor.extract(generated)
    except Exception as _extract_exc:
        log.warning('[Ollama] JSON extraction failed — raw model output was: %r  error: %s',
                    generated[:500], _extract_exc)
        return {"type": "unknown", "raw": text}
    result = _repair_llm_output(result)
    validated = _validate_llm_result(result, text)
    planned = _plan_and_validate(validated, text)
    log.info('[Ollama] action = %s', planned.get('type', 'steps'))
    return planned


# ── Improvement 3: streaming Ollama call ──────────────────────────────────────
# Ollama's streaming API delivers NDJSON — one JSON object per line.  Each
# object has a ``"response"`` token fragment and a ``"done"`` boolean.
# We accumulate all tokens, then parse the complete text as normal.
# This allows Sara to show a "Thinking…" indicator in the UI immediately and
# start speaking as soon as the full JSON is assembled, rather than waiting
# for the whole HTTP body to close.

def _ollama_call_streaming(text: str, history: list[dict] | None = None) -> dict:
    """Ollama API call using ``stream=True``.

    Tokens are accumulated progressively.  When ``done=true`` is received the
    full response string is parsed exactly like ``_ollama_call``.

    Falls back to the non-streaming path on any protocol error so the system
    degrades gracefully on older Ollama versions.

    Improvement 1 applies here too — only the accumulated ``"response"``
    tokens are used; ``"thinking"`` or other fields in the done-event are
    logged at DEBUG level only.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    prompt_text = _build_ollama_prompt(text, history)
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt_text,
        "stream": True,   # request streaming NDJSON
        "keep_alive": OLLAMA_KEEP_ALIVE,       # keeps model loaded in RAM/VRAM
        "options": {"temperature": 0.0, "num_predict": 150},  # Excel JSON ≤ 100 tokens
    }

    log.info('[Ollama-stream] POST %s  model=%s  prompt_len=%d  timeout=%ds',
             url, OLLAMA_MODEL, len(prompt_text), OLLAMA_TIMEOUT)
    _t0 = time.perf_counter()

    accumulated = []
    with _OLLAMA_SESSION.post(url, json=payload, timeout=OLLAMA_TIMEOUT, stream=True) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except Exception:
                log.debug('[Ollama-stream] non-JSON line skipped: %r', raw_line[:80])
                continue

            token = chunk.get('response', '')
            if token:
                accumulated.append(token)

            if chunk.get('done'):
                # Log internal fields (thinking etc.) at DEBUG only — never forwarded
                log.debug('[Ollama-stream] done event keys=%s', list(chunk.keys()))
                break

    elapsed = time.perf_counter() - _t0
    log.info('[Ollama-stream] all tokens received in %.2fs  tokens=%d',
             elapsed, len(accumulated))

    raw_generated = ''.join(accumulated)
    if not raw_generated:
        log.warning('[Ollama-stream] empty accumulated response')
        return {"type": "unknown", "raw": text}

    generated = _strip_thinking_tags(raw_generated)
    log.debug('[Ollama-stream] generated (stripped) = %r', generated[:200])

    try:
        result = RobustJsonExtractor.extract(generated)
    except Exception as _extract_exc:
        log.warning('[Ollama-stream] JSON extraction failed — raw model output was: %r  error: %s',
                    generated[:500], _extract_exc)
        return {"type": "unknown", "raw": text}
    result = _repair_llm_output(result)
    validated = _validate_llm_result(result, text)
    planned = _plan_and_validate(validated, text)
    log.info('[Ollama-stream] action = %s', planned.get('type', 'steps'))
    return planned


def try_ollama(text: str, history: list[dict] | None = None) -> dict:
    """Parse *text* via a local Ollama instance (public API, backward-compat).

    Attempts the streaming path once (no retries — it is an optimisation, not
    the correctness path).  If streaming raises for any reason the full retry
    budget is given to the non-streaming path, which is the stable fallback.

    This avoids doubling the retry budget (streaming MAX_RETRIES + non-streaming
    MAX_RETRIES), which would multiply worst-case latency by 2× for a voice app.

    Returns a structured action dict or ``{"type": "unknown", ...}`` on any
    failure.
    """
    # Attempt streaming once — no retries; streaming is best-effort.
    try:
        return _ollama_call_streaming(text, history)
    except Exception as stream_exc:
        if isinstance(stream_exc, requests.HTTPError) and stream_exc.response is not None:
            log.warning('[Ollama] Streaming failed HTTP %s — body: %s — falling back to non-streaming',
                        stream_exc.response.status_code, stream_exc.response.text[:200])
        else:
            log.warning('[Ollama] Streaming failed (%s) — falling back to non-streaming', stream_exc)

    # Non-streaming path carries the full retry budget (handles cold model start).
    try:
        return _retry_with_backoff(
            lambda: _ollama_call(text, history),
            label='Ollama',
            max_retries=MAX_RETRIES,
            retry_timeouts=True,
        )
    except Exception as exc:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            log.warning('[Ollama] Failed HTTP %s — body: %s',
                        exc.response.status_code, exc.response.text[:200])
        else:
            log.warning('[Ollama] Failed: %s', exc)
        return {"type": "unknown", "raw": text}


# Backward-compatible alias
parse_with_ollama = try_ollama


# ── Ollama health check ───────────────────────────────────────────────────────

def check_ollama_model_installed(model_name: str | None = None) -> bool:
    """Return True if *model_name* is pulled in Ollama."""
    target = model_name or OLLAMA_MODEL
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        return any(
            m.get('name', '').startswith(target)
            for m in resp.json().get('models', [])
        )
    except Exception:
        return False


def is_ollama_available() -> bool:
    """Return True if Ollama is reachable and has the configured model."""
    return check_ollama_model_installed()


# ── Internal routing helpers ──────────────────────────────────────────────────

def _action_label(result: dict | None) -> str:
    """Short human-readable label for the parsed action (used in log lines)."""
    if not isinstance(result, dict):
        return 'none'
    if isinstance(result.get('steps'), list) and result['steps']:
        n = len(result['steps'])
        first = result['steps'][0].get('type', '?') if result['steps'] else '?'
        return f'steps[{n}]:{first}…'
    return result.get('type', 'unknown')


# ── Improvement 4: Intent confidence scoring ──────────────────────────────────
# Every parsed action receives a ``confidence`` score (0.0–1.0).
# The score reflects how certain the system is that the parsed action matches
# the user's intent.  Scores below 0.5 trigger a clarification prompt in the
# runtime rather than executing a potentially wrong command.
#
# Scoring rules:
#   heuristic match  → 0.95  (deterministic regex, very high certainty)
#   ollama / stream  → 0.82  (local LLM, well-tuned prompt)
#   groq             → 0.75  (cloud LLM, short timeout, less context)
#   unknown action   → 0.10  (parser gave up — user should rephrase)

_CONFIDENCE = {
    'heuristic': 0.95,
    'ollama':    0.82,
    'groq':      0.75,
    'unknown':   0.10,
}


def _add_confidence(result: dict, source: str) -> dict:
    """Annotate *result* with a ``confidence`` float (0.0–1.0) and return it.

    Base score is the source-tier trust level.  Two structural penalties are
    then applied so the score is not purely static:

    * ``-0.15`` when the parsed action type is ``'unknown'`` (parser gave up).
    * ``-0.10`` when a multi-step plan has only one step (LLM may have
      over-planned a simple command, suggesting partial understanding).

    These penalties are heuristic but make the score somewhat responsive to
    parse quality rather than being a pure tier label.

    NOTE: this is still not true probabilistic confidence.  A proper
    implementation would require the LLM to emit a self-assessed score or use
    a secondary classifier.  Documented as a known limitation.
    """
    if not isinstance(result, dict):
        return result

    if result.get('type') == 'unknown':
        result['confidence'] = _CONFIDENCE['unknown']
        return result

    base = _CONFIDENCE.get(source, 0.75)

    # Penalty: multi-step plan with only one step is suspicious
    steps = result.get('steps')
    if isinstance(steps, list) and len(steps) == 1:
        base -= 0.10

    result['confidence'] = round(max(0.0, min(1.0, base)), 4)
    return result


def _record(
    text: str,
    provider: str,
    model: str | None,
    result: dict | None,
    latency_ms: float,
    failure_reason: str,
    *,
    fallback: bool,
) -> None:
    """Write one metrics row for a backend attempt — success **or** failure.

    Recording every attempt (not just successes) makes per-backend failure
    rates and fallback frequencies measurable in the thesis data.
    """
    success = failure_reason == FR_OK and _is_valid(result)
    atype = _action_label(result) if result else 'none'
    try:
        _metrics.record_llm_call(
            text,
            provider=provider,
            model=model,
            action_type=atype,
            success=success,
            latency_ms=latency_ms,
            fallback=fallback,
            raw_result={"failure_reason": failure_reason} if not success else None,
        )
    except Exception:
        log.debug('[Metrics] record_llm_call failed', exc_info=True)


# ── Public router ─────────────────────────────────────────────────────────────

def route_llm(text: str, history: list[dict] | None = None) -> dict:
    """Route one parse request through the three-tier fallback chain.

    Routing::

        Heuristic (instant, always first)
        → Ollama (local, 120 s timeout)  — no internet needed, preferred LLM
        → Groq   (cloud,   8 s timeout)  — only when Ollama unavailable/disabled
        → unknown                         — all tiers exhausted

    The heuristic runs FIRST because it handles ~80 % of common Excel commands
    (add row, create sheet, calculate average …) in under 1 ms with zero
    network cost.  Only genuinely ambiguous or complex commands reach the LLM.

    Ollama is tried before Groq because it is local (no internet, no API cost,
    no latency variance) and always available once the service is running.
    Groq is used as a final cloud fallback when Ollama is disabled or down.

    Every backend attempt — success or failure — is recorded in the metrics DB
    so that per-backend failure rates and fallback frequencies are measurable.

    The returned dict always has at least ``{"type": "..."}`` and carries an
    ``_router_source`` key (``'heuristic'``, ``'ollama'``, ``'groq'``, or
    ``'unknown'``) and a ``confidence`` float (Improvement 4) that callers can
    inspect to know which tier answered and how certain it was.

    Parameters:
        text:    The raw (or context-enriched) transcript to parse.
        history: Optional list of recent ``{"user": str, "assistant": str}``
                 turns from the conversation buffer (Improvement 2).  Passed to
                 Ollama so it can resolve pronouns and follow-up references.
                 Groq receives history embedded in the text prefix instead.

    Never raises.
    """
    # ── Pre-normalization (filler-word removal) ───────────────────────────
    # Build a normalised copy of *text* for pattern matching only.
    # Strips leading/trailing whitespace, lowercases, and removes filler words
    # ("the", "please", etc.) so that "please open the excel" matches the same
    # pattern as "open excel".  The original *text* is preserved and forwarded
    # unchanged to every LLM backend so natural phrasing is not lost.
    # NOTE: regexes are module-level constants — no re.compile() cost per call.
    _norm = _WHITESPACE_RE.sub(' ', _FILLER_RE.sub(' ', text.lower().strip())).strip()
    # STT artifact normalisation: common mis-transcriptions of "add row".
    # Applied to _norm only — the original *text* is still forwarded to LLMs.
    _norm = _AND_ROW_RE.sub('add row', _norm)    # "and row X" → "add row X"
    _norm = _ROW_START_RE.sub('add row ', _norm) # "row X" at start → "add row X"

    # ── Fast-path: open excel / open workbook ─────────────────────────────
    # Common voice phrases that fail the heuristic regex in llm_parser when
    # the word "the" appears between "open" and "excel"/"spreadsheet".
    # Resolved here before Tier 1 so they never reach the LLM.
    if _OPEN_APP_RE.search(_norm):
        _r = {'type': 'open_excel_app', '_router_source': 'heuristic'}
        _add_confidence(_r, 'heuristic')
        _record(text, 'heuristic', None, _r, 0.0, FR_OK, fallback=False)
        log.info('[Router] ✓ Fast-path open_excel_app  0 ms  (normalised=%r)', _norm)
        return _r
    if _OPEN_FILE_RE.search(_norm):
        _r = {'type': 'open_file', '_router_source': 'heuristic'}
        # Extract the filename from the command tail after "open [the] file/..."
        # MUST NOT use _FILENAME_RE.search(text) here: when `text` is context-enriched
        # (e.g. "[file:server.xlsx sheet:sheet1] open file server"), the regex finds
        # "file:server.xlsx" from the memory prefix instead of the actual argument.
        _tail_m = re.search(
            r'\bopen\s+(?:the\s+)?(?:file|spreadsheet|workbook|sheet)\s+(.+)',
            text, re.IGNORECASE,
        )
        if _tail_m:
            _tail = re.sub(r'[.,!?;]+$', '', _tail_m.group(1).strip()).strip()
            # Safety guard: if the tail still contains context-prefix artifacts
            # (file:, sheet:, or bracket chars) it was captured from the prefix,
            # not the user command — discard it and open the active workbook.
            if _tail and not re.search(r'file:|sheet:|\[|\]', _tail):
                if not re.search(r'\.(?:xlsx|xls|csv|xlsm|xlsb)$', _tail, re.IGNORECASE):
                    _tail += '.xlsx'
                _r['filename'] = _tail
        _add_confidence(_r, 'heuristic')
        _record(text, 'heuristic', None, _r, 0.0, FR_OK, fallback=False)
        log.info('[Router] ✓ Fast-path open_file  0 ms  (normalised=%r)', _norm)
        return _r

    # ── Tier 1: Heuristic regex parser (instant, always offline) ─────────
    # Run this FIRST so that common, well-understood Excel commands never
    # pay LLM latency.  The heuristic operates on raw text only (no history).
    heuristic_raw = _heuristic_natural_language_parse(text)
    if heuristic_raw:
        planned = _plan_and_validate(heuristic_raw, text)
        if _is_valid(planned):
            log.info('[Router] ✓ Heuristic fast-path  action=%s  0 ms',
                     _action_label(planned))
            _record(text, 'heuristic', None, planned, 0.0, FR_OK, fallback=False)
            planned['_router_source'] = 'heuristic'
            # Improvement 4: annotate with confidence
            _add_confidence(planned, 'heuristic')
            return planned
        log.debug('[Router] Heuristic matched but plan invalid — falling through to LLM')

    # ── GROQ_FIRST fast-path (optional, online mode) ──────────────────────
    # When GROQ_FIRST=true and internet is available, try Groq before Ollama.
    # Groq responses arrive in ~1 s vs 5–30 s for a cold Ollama model.
    # Falls through to the normal Ollama→Groq chain below on any failure.
    if GROQ_FIRST and os.environ.get('GROQ_API_KEY') and check_internet_connection():
        log.info('[Router] GROQ_FIRST mode — trying Groq before Ollama')
        groq_prompt = _build_ollama_prompt(text, history) if history else text
        t0 = time.perf_counter()
        gf_result: dict = {"type": "unknown", "raw": text}
        gf_reason = FR_UNEXPECTED
        try:
            gf_result = try_groq(groq_prompt)
            gf_reason = FR_OK
        except Exception as exc:
            gf_reason = _classify_exc(exc)
            log.warning('[Router] GROQ_FIRST: Groq raised %s: %s', gf_reason, exc)
        groq_first_ms = (time.perf_counter() - t0) * 1000

        if _is_valid(gf_result):
            log.info('[Router] ✓ Groq (GROQ_FIRST)  action=%-22s  %.0f ms',
                     _action_label(gf_result), groq_first_ms)
            gf_result['_router_source'] = 'groq'
            _add_confidence(gf_result, 'groq')
            _record(text, 'groq', GROQ_MODEL_NAME, gf_result, groq_first_ms,
                    FR_OK, fallback=False)
            return gf_result

        log.info('[Router] GROQ_FIRST: Groq miss (%.0f ms) → Ollama fallback', groq_first_ms)

    # ── Tier 2: Ollama (local, preferred LLM) ─────────────────────────────
    # try_ollama now passes *history* to the model so follow-up commands are
    # resolved in context (Improvement 2).
    if not USE_OLLAMA:
        log.info('[Router] Ollama disabled (USE_OLLAMA=false) → Groq')
    else:
        log.info('[Router] Heuristic miss → Ollama (local)')
        t0 = time.perf_counter()
        o_result: dict = {"type": "unknown", "raw": text}
        o_reason = FR_UNEXPECTED
        try:
            o_result = try_ollama(text, history)
            o_reason = FR_OK
        except Exception as exc:
            o_reason = _classify_exc(exc)
            log.warning('[Router] Ollama raised %s: %s', o_reason, exc)
        ollama_ms = (time.perf_counter() - t0) * 1000

        _record(text, 'ollama', OLLAMA_MODEL, o_result, ollama_ms,
                FR_OK if _is_valid(o_result) else o_reason, fallback=False)

        if _is_valid(o_result):
            # Reject hallucinated tool types that the agent cannot execute.
            # Multi-step plans have no root "type"; validate each step instead.
            _o_type = o_result.get('type', '')
            try:
                from llm_parser import _ALLOWED_TYPES as _AP
                if 'steps' in o_result:
                    _type_ok = bool(o_result['steps']) and all(
                        isinstance(s, dict) and s.get('type') in _AP
                        for s in o_result['steps']
                    )
                else:
                    _type_ok = _o_type in _AP
            except Exception:
                _type_ok = True  # Don't block if import fails
            if _type_ok:
                log.info('[Router] ✓ Ollama  action=%-28s  %.0f ms',
                         _action_label(o_result), ollama_ms)
                o_result['_router_source'] = 'ollama'
                _add_confidence(o_result, 'ollama')  # Improvement 4
                return o_result
            log.warning('[Router] ✗ Ollama  hallucinated type=%r → Groq fallback', _o_type)
            o_reason = 'hallucinated_type'

        log.warning('[Router] ✗ Ollama  reason=%-18s  %.0f ms  → Groq',
                    o_reason, ollama_ms)

    # ── Tier 3: Groq (cloud fallback) ────────────────────────────────────
    # History is injected as a text prefix so Groq's existing interface is
    # unchanged (Improvement 2 compatibility).
    online = check_internet_connection()
    if not online:
        log.info('[Router] Groq skipped — no internet connection')
    else:
        # Build history-enriched text for Groq (it has no native history param)
        groq_text = _build_ollama_prompt(text, history) if history else text
        t1 = time.perf_counter()
        g_result: dict = {"type": "unknown", "raw": text}
        g_reason = FR_UNEXPECTED
        try:
            g_result = try_groq(groq_text)
            g_reason = FR_OK
        except Exception as exc:
            g_reason = _classify_exc(exc)
            log.warning('[Router] Groq raised %s: %s', g_reason, exc)
        groq_ms = (time.perf_counter() - t1) * 1000

        _record(text, 'groq', GROQ_MODEL_NAME, g_result, groq_ms,
                FR_OK if _is_valid(g_result) else g_reason, fallback=True)

        if _is_valid(g_result):
            log.info('[Router] ✓ Groq  action=%-30s  %.0f ms',
                     _action_label(g_result), groq_ms)
            g_result['_router_source'] = 'groq'
            _add_confidence(g_result, 'groq')  # Improvement 4
            return g_result

        log.warning('[Router] ✗ Groq  reason=%-20s  %.0f ms',
                    g_reason, groq_ms)

    # ── All tiers exhausted ───────────────────────────────────────────────
    log.error('[Router] ✗ All tiers failed  text=%r', text[:80])
    _record(text, 'none', None, None, 0.0, FR_ALL_FAILED, fallback=True)

    # Last-resort: if the heuristic parsed something but _plan_and_validate
    # rejected it (e.g. strict field check failed), return the raw heuristic
    # result rather than dropping the command entirely.  The agent will surface
    # a clearer error than a silent "unknown" drop.
    if isinstance(heuristic_raw, dict) and heuristic_raw.get('type'):
        try:
            from llm_parser import _ALLOWED_TYPES as _AT
            _type_ok = heuristic_raw['type'] in _AT
        except Exception:
            _type_ok = True
        if _type_ok:
            log.warning('[Router] All LLMs failed — using raw heuristic result (type=%s)',
                        heuristic_raw.get('type'))
            heuristic_raw['_router_source'] = 'heuristic'
            _add_confidence(heuristic_raw, 'heuristic')
            return heuristic_raw

    fallback_result = {"type": "unknown", "raw": text, "_router_source": "unknown"}
    _add_confidence(fallback_result, 'unknown')  # Improvement 4: score=0.10
    return fallback_result


# ── LLM output repair ────────────────────────────────────────────────────────

_TYPE_SYNONYMS = {
    'insert_row': 'add_row',
    'write':      'write_cell',
}


def _repair_llm_output(obj: dict) -> dict:
    """Attempt lightweight structural repairs on raw LLM output before validation.

    Small models (e.g. qwen2.5:3b) occasionally return correct *intent* but
    wrong *schema* — wrong key names, missing "type", synonym type names, etc.
    This function fixes the most common patterns deterministically and logs
    every repair so issues are visible in the test logs.

    Returns the (possibly mutated) dict.  Never raises.
    """
    if not isinstance(obj, dict):
        return obj

    original = dict(obj)  # shallow copy for logging
    repaired = False

    # ── Repair 1: row shorthand ──────────────────────────────────────────────
    # {"row": [{"name": "john", "score": 90}]}  →  {"type":"add_row","values":["john",90]}
    if 'row' in obj and 'type' not in obj:
        row_val = obj['row']
        if isinstance(row_val, list) and row_val and isinstance(row_val[0], dict):
            values = list(row_val[0].values())
            obj = {'type': 'add_row', 'values': values}
            repaired = True

    # ── Repair 2: bare values list ───────────────────────────────────────────
    # {"values": ["Tom", 90]}  →  {"type":"add_row","values":["Tom",90]}
    if 'values' in obj and 'type' not in obj:
        obj['type'] = 'add_row'
        repaired = True

    # ── Repair 3: bare cell write ────────────────────────────────────────────
    # {"cell": "A1", "value": 42}  →  {"type":"write_cell","cell":"A1","value":42}
    if 'cell' in obj and 'value' in obj and 'type' not in obj:
        obj['type'] = 'write_cell'
        repaired = True

    # ── Repair 4: type synonyms ──────────────────────────────────────────────
    # "insert_row" → "add_row",  "write" → "write_cell"
    raw_type = obj.get('type')
    if raw_type in _TYPE_SYNONYMS:
        obj['type'] = _TYPE_SYNONYMS[raw_type]
        repaired = True

    if repaired:
        log.info('[Router] repaired LLM output: %s → %s', original, obj)

    return obj


# ── Validation helper ─────────────────────────────────────────────────────────

def _is_valid(action: dict | None) -> bool:
    """Return True if *action* is a real parsed result (not unknown/empty)."""
    if not isinstance(action, dict):
        return False
    if isinstance(action.get('steps'), list) and action['steps']:
        return True
    return action.get('type') not in ('unknown', None, '')
