"""Production-ready LLM router for Sara.

Provides :class:`LLMRouter` — an intelligent dispatcher that routes
natural-language commands through a prioritised chain of LLM backends,
with automatic fallback, retry logic, and **streaming support**.

Routing chain
-------------
* **Online**:  Groq (cloud) → Ollama (local) → Heuristic regex
* **Offline**: Ollama (local) → Heuristic regex  (Groq skipped)

Streaming
---------
* ``route_stream()`` yields partial text chunks as they arrive from the
  model, suitable for real-time UI feedback.
* The final chunk is always the complete, validated action dict.
* Only Groq and Ollama support true token-level streaming; the heuristic
  fallback emits the full result in one chunk.

Configuration (env vars)::

    GROQ_API_KEY            – required for Groq
    GROQ_API_URL            – default https://api.groq.com/openai/v1/chat/completions
    GROQ_MODEL              – default llama-3.3-70b-versatile
    USE_OLLAMA=true         – enable Ollama fallback
    OLLAMA_MODEL=qwen3:8b   – Ollama model name
    OLLAMA_BASE_URL         – default http://localhost:11434
    OLLAMA_TIMEOUT=60       – seconds
    LLM_MAX_RETRIES=3       – retries per backend
    LLM_BASE_BACKOFF=0.5    – initial backoff seconds
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Generator

import requests

# ── Resolve parent package so we can import sibling backend modules ──
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from metrics import metrics as _metrics
from llm_parser import (
    _parse_groq_direct,
    RobustJsonExtractor,
    _validate_llm_result,
    _plan_and_validate,
    _heuristic_natural_language_parse,
)

# Re-use the canonical system prompt from the existing router so both
# code-paths produce structurally identical output.
from llm_router import SYSTEM_PROMPT

log = logging.getLogger("llm.route_llm")

# ───────────────────── Configuration ─────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = os.environ.get(
    "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"
)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TIMEOUT = int(os.environ.get("GROQ_TIMEOUT", "30"))

USE_OLLAMA = os.environ.get("USE_OLLAMA", "true").lower() in ("true", "1", "yes")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))

MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
BASE_BACKOFF = float(os.environ.get("LLM_BASE_BACKOFF", "0.5"))

log.info("[Config] GROQ_MODEL=%s  GROQ_API_URL=%s", GROQ_MODEL, GROQ_API_URL)

_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})


# ───────────────────── Helpers ───────────────────────────────────

def _is_transient(exc: Exception) -> bool:
    """Return True for network errors worth retrying."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError):
        resp = getattr(exc, "response", None)
        if resp is not None and resp.status_code in _TRANSIENT_STATUS:
            return True
    return False


def _retry(fn, *, label: str, max_retries: int = MAX_RETRIES):
    """Call *fn()* with exponential backoff on transient failures."""
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not _is_transient(exc):
                raise
            delay = BASE_BACKOFF * (2 ** (attempt - 1))
            log.warning(
                "[%s] transient error attempt %d/%d: %s — retry in %.1fs",
                label, attempt, max_retries, exc, delay,
            )
            time.sleep(delay)
    raise last  # type: ignore[misc]


def _is_valid(action: dict | None) -> bool:
    """True if *action* is a real parsed result (not unknown)."""
    if not isinstance(action, dict):
        return False
    if isinstance(action.get("steps"), list) and action["steps"]:
        return True
    return action.get("type") not in ("unknown", None, "")


def check_internet(timeout: float = 2.0) -> bool:
    """Return True if the machine has internet connectivity."""
    try:
        requests.head("https://www.google.com", timeout=timeout)
        return True
    except Exception:
        return False


def is_ollama_available() -> bool:
    """Return True if Ollama is reachable and has the configured model."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = resp.json().get("models", [])
        return any(m.get("name", "").startswith(OLLAMA_MODEL) for m in models)
    except Exception:
        return False


# ───────────────────── Groq backend ──────────────────────────────

def _groq_call(text: str) -> dict:
    """Blocking Groq API call (no retries). Returns parsed action dict."""
    return _parse_groq_direct(text)


def _groq_stream(text: str) -> Generator[str, None, None]:
    """Streaming Groq call. Yields text chunks, not parsed JSON."""
    if not GROQ_API_KEY:
        return

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "stream": True,
    }

    resp = requests.post(
        GROQ_API_URL, headers=headers, json=payload,
        timeout=GROQ_TIMEOUT, stream=True,
    )
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content
        except (json.JSONDecodeError, IndexError, KeyError):
            continue


# ───────────────────── Ollama backend ────────────────────────────

def _ollama_call(text: str) -> dict:
    """Blocking Ollama API call (no retries). Returns parsed action dict."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": text,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    log.info("[Ollama] calling %s model=%s", url, OLLAMA_MODEL)
    resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()

    body = resp.json()
    generated = body.get("response", "") if isinstance(body, dict) else ""
    if not generated:
        return {"type": "unknown", "raw": text}

    result = RobustJsonExtractor.extract(generated)
    validated = _validate_llm_result(result, text)
    return _plan_and_validate(validated, text)


def _ollama_stream(text: str) -> Generator[str, None, None]:
    """Streaming Ollama call. Yields text chunks."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": text,
        "stream": True,
        "options": {"temperature": 0.0},
    }

    resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT, stream=True)
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break
        except json.JSONDecodeError:
            continue


# ───────────────────── Router class ──────────────────────────────

class LLMRouter:
    """Intelligent LLM router with automatic fallback and streaming.

    Parameters
    ----------
    use_ollama:
        Override env-var to enable/disable Ollama fallback.
    max_retries:
        Override env-var for retry count per backend.

    Usage::

        router = LLMRouter()
        action = router.route("add row Tom 90")

        for chunk in router.route_stream("create excel with name score"):
            print(chunk, end="")
    """

    def __init__(
        self,
        *,
        use_ollama: bool | None = None,
        max_retries: int | None = None,
    ):
        self._use_ollama = USE_OLLAMA if use_ollama is None else use_ollama
        self._max_retries = MAX_RETRIES if max_retries is None else max_retries

    # ── Blocking route ──────────────────────────────────────────

    def route(self, text: str) -> dict:
        """Route *text* through Groq → Ollama → Heuristic and return a
        structured action dict.

        Equivalent to the existing ``llm_router.route_llm()`` but accessed
        via the new modular API.
        """
        online = check_internet()
        groq_ms = 0.0
        ollama_ms = 0.0
        result: dict | None = None

        if online:
            log.info("[Router] Online — Groq → Ollama → Heuristic")
        else:
            log.info("[Router] Offline — Ollama → Heuristic")

        # 1. Groq (cloud)
        if online:
            t0 = time.perf_counter()
            try:
                result = _retry(
                    lambda: _groq_call(text),
                    label="Groq",
                    max_retries=self._max_retries,
                )
            except Exception as exc:
                log.warning("[Router] Groq failed: %s", exc)
                result = {"type": "unknown", "raw": text}
            groq_ms = (time.perf_counter() - t0) * 1000

            if _is_valid(result):
                log.info("[Router] ✓ Groq — %s (%.0f ms)", result.get("type", "steps"), groq_ms)
                _metrics.record_llm_call(
                    text, provider="groq", model=GROQ_MODEL,
                    action_type=result.get("type", "steps"),
                    success=True, latency_ms=groq_ms, fallback=False,
                )
                return result
            log.warning("[Router] Groq invalid (%.0f ms) — fallback", groq_ms)

        # 2. Ollama (local)
        if self._use_ollama:
            t1 = time.perf_counter()
            try:
                result = _retry(
                    lambda: _ollama_call(text),
                    label="Ollama",
                    max_retries=self._max_retries,
                )
            except Exception as exc:
                log.warning("[Router] Ollama failed: %s", exc)
                result = {"type": "unknown", "raw": text}
            ollama_ms = (time.perf_counter() - t1) * 1000

            if _is_valid(result):
                log.info("[Router] ✓ Ollama — %s (%.0f ms)", result.get("type", "steps"), ollama_ms)
                _metrics.record_llm_call(
                    text, provider="ollama", model=OLLAMA_MODEL,
                    action_type=result.get("type", "steps"),
                    success=True, latency_ms=ollama_ms, fallback=not online,
                )
                return result
            log.warning("[Router] Ollama invalid (%.0f ms) — fallback", ollama_ms)
        else:
            log.info("[Router] Ollama disabled — skipping")

        # 3. Heuristic regex
        return self._heuristic_fallback(text, groq_ms + ollama_ms)

    # ── Streaming route ─────────────────────────────────────────

    def route_stream(self, text: str) -> Generator[str | dict, None, None]:
        """Route *text* with streaming — yields str chunks during generation,
        then yields the final parsed action dict as the last item.

        Callers should check ``isinstance(chunk, dict)`` to detect the
        final parsed result.

        Usage::

            full_text = ""
            for chunk in router.route_stream("add row Tom 90"):
                if isinstance(chunk, dict):
                    action = chunk   # final parsed action
                else:
                    full_text += chunk
                    print(chunk, end="", flush=True)
        """
        online = check_internet()

        # 1. Try Groq streaming
        if online and GROQ_API_KEY:
            log.info("[Stream] trying Groq streaming")
            collected = ""
            try:
                for token in _groq_stream(text):
                    collected += token
                    yield token

                if collected:
                    action = self._parse_collected(collected, text)
                    if _is_valid(action):
                        log.info("[Stream] ✓ Groq stream — %s", action.get("type", "steps"))
                        yield action
                        return
                log.warning("[Stream] Groq stream empty/invalid — fallback")
            except Exception as exc:
                log.warning("[Stream] Groq stream error: %s — fallback", exc)

        # 2. Try Ollama streaming
        if self._use_ollama:
            log.info("[Stream] trying Ollama streaming")
            collected = ""
            try:
                for token in _ollama_stream(text):
                    collected += token
                    yield token

                if collected:
                    action = self._parse_collected(collected, text)
                    if _is_valid(action):
                        log.info("[Stream] ✓ Ollama stream — %s", action.get("type", "steps"))
                        yield action
                        return
                log.warning("[Stream] Ollama stream empty/invalid — fallback")
            except Exception as exc:
                log.warning("[Stream] Ollama stream error: %s — fallback", exc)

        # 3. Heuristic (no streaming — emit full result at once)
        action = self._heuristic_fallback(text, 0)
        yield json.dumps(action)
        yield action

    # ── Internals ───────────────────────────────────────────────

    @staticmethod
    def _parse_collected(raw: str, original_text: str) -> dict:
        """Parse accumulated streamed text into a validated action dict."""
        try:
            result = RobustJsonExtractor.extract(raw)
            validated = _validate_llm_result(result, original_text)
            return _plan_and_validate(validated, original_text)
        except Exception as exc:
            log.warning("[Stream] parse failed: %s", exc)
            return {"type": "unknown", "raw": original_text}

    @staticmethod
    def _heuristic_fallback(text: str, elapsed_ms: float) -> dict:
        """Run the regex heuristic parser as last resort."""
        log.info("[Router] trying heuristic for %r", text[:80])
        heuristic = _heuristic_natural_language_parse(text)
        if heuristic:
            planned = _plan_and_validate(heuristic, text)
            if _is_valid(planned):
                log.info("[Router] ✓ Heuristic — %s", planned.get("type", "steps"))
                _metrics.record_llm_call(
                    text, provider="heuristic",
                    action_type=planned.get("type", "steps"),
                    success=True, latency_ms=0, fallback=True,
                )
                return planned

        _metrics.record_llm_call(
            text, provider="none", success=False,
            latency_ms=elapsed_ms, fallback=True,
        )
        log.error("[Router] ✗ all backends failed for %r", text[:80])
        return {"type": "unknown", "raw": text}


# ───────────────────── Module-level convenience ──────────────────

_default_router = LLMRouter()


def route(text: str) -> dict:
    """Module-level shortcut: ``route(text)`` ≡ ``LLMRouter().route(text)``."""
    return _default_router.route(text)


def route_stream(text: str) -> Generator[str | dict, None, None]:
    """Module-level shortcut for streaming."""
    return _default_router.route_stream(text)
