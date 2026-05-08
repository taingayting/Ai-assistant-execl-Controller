"""Dedicated Ollama client for Sara.

Consolidates all Ollama interactions into a single client class with:

* **Server detection** — check if Ollama is reachable.
* **Auto-start** — launch ``ollama serve`` when the server isn't running.
* **Model management** — list, pull, switch, and inspect models.
* **Generation** — blocking and streaming text generation with timeout
  protection, retry logic, and structured JSON response parsing.

Recommended models::

    qwen3:8b   — fast, strong instruction-following  (default)
    llama3:8b   — great general reasoning
    mistral     — balanced speed / quality

Configuration (env vars / .env)::

    OLLAMA_BASE_URL   — default http://localhost:11434
    OLLAMA_MODEL      — default qwen3:8b
    OLLAMA_TIMEOUT    — seconds, default 60
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Generator

import requests

# ── Resolve sibling backend modules ──────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from llm_parser import (
    RobustJsonExtractor,
    _validate_llm_result,
    _plan_and_validate,
)
from llm_router import SYSTEM_PROMPT

log = logging.getLogger("llm.ollama_client")

# ───────────────────── Constants ─────────────────────────────────

RECOMMENDED_MODELS: list[str] = ["qwen3:8b", "llama3:8b", "mistral"]

_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})

# How long to wait for Ollama to become healthy after auto-start.
_STARTUP_POLL_INTERVAL = 0.5  # seconds between health-check polls
_STARTUP_MAX_WAIT = 15.0       # total seconds before giving up


# ───────────────────── Exceptions ────────────────────────────────

class OllamaError(Exception):
    """Base exception for Ollama client errors."""


class OllamaServerNotRunning(OllamaError):
    """Raised when Ollama cannot be reached and auto-start is disabled or failed."""


class OllamaModelNotFound(OllamaError):
    """Raised when the requested model is not available locally."""


class OllamaStartupFailed(OllamaError):
    """Raised when the client tried but failed to start Ollama."""


class OllamaGenerationError(OllamaError):
    """Raised when a generation request fails after retries."""


# ───────────────────── Client ────────────────────────────────────

class OllamaClient:
    """Self-contained client for the Ollama local LLM server.

    Parameters
    ----------
    base_url:
        Ollama REST API base (default from env or ``http://localhost:11434``).
    model:
        Default model name (default from env or ``qwen3:8b``).
    timeout:
        Request timeout in seconds (default from env or ``60``).
    auto_start:
        If True, try to launch ``ollama serve`` automatically when the
        server is not running.  Default ``True``.
    max_retries:
        Number of retries on transient failures.  Default ``3``.
    base_backoff:
        Initial backoff in seconds (doubled each retry).  Default ``0.5``.
    system_prompt:
        System prompt sent with every generation.  Defaults to the
        shared Sara system prompt.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        auto_start: bool = True,
        max_retries: int = 3,
        base_backoff: float = 0.5,
        system_prompt: str | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self.model = (
            model or os.environ.get("OLLAMA_MODEL", "qwen3:8b")
        )
        self.timeout = (
            timeout
            if timeout is not None
            else int(os.environ.get("OLLAMA_TIMEOUT", "60"))
        )
        self.auto_start = auto_start
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self._server_process: subprocess.Popen | None = None

    # ── Server detection ─────────────────────────────────────────

    def is_server_running(self, timeout: float = 3.0) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags", timeout=timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ── Auto-start ───────────────────────────────────────────────

    @staticmethod
    def find_ollama_executable() -> str | None:
        """Return the path to the ``ollama`` binary, or None."""
        return shutil.which("ollama")

    def start_server(self) -> None:
        """Attempt to start ``ollama serve`` as a background process.

        Raises :class:`OllamaStartupFailed` if the binary is missing or
        the server doesn't become healthy within a reasonable time.
        """
        exe = self.find_ollama_executable()
        if exe is None:
            raise OllamaStartupFailed(
                "Ollama binary not found on PATH. "
                "Install from https://ollama.com"
            )

        log.info("Starting Ollama server: %s serve", exe)
        try:
            self._server_process = subprocess.Popen(
                [exe, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise OllamaStartupFailed(f"Failed to start Ollama: {exc}") from exc

        # Wait for the server to become healthy.
        elapsed = 0.0
        while elapsed < _STARTUP_MAX_WAIT:
            time.sleep(_STARTUP_POLL_INTERVAL)
            elapsed += _STARTUP_POLL_INTERVAL
            if self.is_server_running(timeout=2.0):
                log.info("Ollama server healthy after %.1fs", elapsed)
                return

        raise OllamaStartupFailed(
            f"Ollama process started but server not healthy after "
            f"{_STARTUP_MAX_WAIT}s"
        )

    def ensure_server(self) -> None:
        """Make sure the Ollama server is available.

        If ``auto_start`` is enabled and the server is not running, this
        method tries to start it.  Raises on failure.
        """
        if self.is_server_running():
            return
        if not self.auto_start:
            raise OllamaServerNotRunning(
                f"Ollama server at {self.base_url} is not reachable "
                "and auto_start is disabled."
            )
        self.start_server()

    # ── Model management ─────────────────────────────────────────

    def list_models(self) -> list[dict[str, Any]]:
        """Return the list of locally available models.

        Each entry is a dict with at least ``name`` and ``size`` keys.
        """
        self.ensure_server()
        resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("models", [])

    def is_model_available(self, model: str | None = None) -> bool:
        """Return True if *model* is pulled locally."""
        target = model or self.model
        try:
            models = self.list_models()
        except Exception:
            return False
        return any(m.get("name", "").startswith(target) for m in models)

    def pull_model(self, model: str | None = None) -> None:
        """Pull (download) a model from the Ollama registry.

        Blocks until the download completes.

        Raises :class:`OllamaError` on failure.
        """
        target = model or self.model
        self.ensure_server()
        log.info("Pulling model %s …", target)
        resp = requests.post(
            f"{self.base_url}/api/pull",
            json={"name": target, "stream": False},
            timeout=600,  # models can be large
        )
        if resp.status_code != 200:
            raise OllamaError(
                f"Failed to pull model {target}: "
                f"HTTP {resp.status_code} — {resp.text[:200]}"
            )
        log.info("Model %s pulled successfully", target)

    def get_model_info(self, model: str | None = None) -> dict[str, Any]:
        """Return metadata for a locally available model.

        Calls ``POST /api/show``.
        """
        target = model or self.model
        self.ensure_server()
        resp = requests.post(
            f"{self.base_url}/api/show",
            json={"name": target},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def switch_model(self, model: str) -> None:
        """Switch the active model.

        If the model is not already available locally it will be pulled
        automatically before switching.
        """
        if not self.is_model_available(model):
            log.info("Model %s not found locally — pulling …", model)
            self.pull_model(model)
        self.model = model
        log.info("Active model switched to %s", self.model)

    # ── Retry helper ─────────────────────────────────────────────

    def _retry(self, fn, *, label: str):
        """Call *fn()* with exponential backoff on transient errors."""
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                last = exc
                if not self._is_transient(exc):
                    raise
                delay = self.base_backoff * (2 ** (attempt - 1))
                log.warning(
                    "[%s] transient error attempt %d/%d: %s — retry in %.1fs",
                    label, attempt, self.max_retries, exc, delay,
                )
                time.sleep(delay)
        raise last  # type: ignore[misc]

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
            return True
        if isinstance(exc, requests.HTTPError):
            resp = getattr(exc, "response", None)
            if resp is not None and resp.status_code in _TRANSIENT_STATUS:
                return True
        return False

    # ── Generation (blocking) ────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        raw: bool = False,
    ) -> dict[str, Any]:
        """Send a blocking generation request and return a parsed action dict.

        Parameters
        ----------
        prompt:
            The user's natural-language command.
        model:
            Override the active model for this request only.
        system:
            Override the system prompt for this request only.
        temperature:
            Sampling temperature (default 0 for deterministic output).
        raw:
            If True, return ``{"response": "<text>"}`` instead of parsing
            the output as a structured action.
        """
        self.ensure_server()
        target_model = model or self.model

        def _call() -> dict[str, Any]:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": target_model,
                "system": system or self.system_prompt,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            }
            log.info("[generate] %s model=%s", url, target_model)
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            body = resp.json()
            generated = body.get("response", "") if isinstance(body, dict) else ""

            if raw:
                return {"response": generated}

            if not generated:
                log.warning("[generate] empty response")
                return {"type": "unknown", "raw": prompt}

            result = RobustJsonExtractor.extract(generated)
            validated = _validate_llm_result(result, prompt)
            return _plan_and_validate(validated, prompt)

        try:
            return self._retry(_call, label="Ollama-generate")
        except Exception as exc:
            raise OllamaGenerationError(
                f"Generation failed after {self.max_retries} attempts: {exc}"
            ) from exc

    # ── Generation (streaming) ───────────────────────────────────

    def generate_stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> Generator[str, None, None]:
        """Stream tokens from Ollama and yield them as they arrive.

        Each yielded value is a text fragment (string).  The caller can
        concatenate them to reconstruct the full response.

        Parameters
        ----------
        prompt:
            The user's natural-language command.
        model:
            Override the active model for this request.
        system:
            Override the system prompt for this request.
        temperature:
            Sampling temperature.
        """
        self.ensure_server()
        target_model = model or self.model

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": target_model,
            "system": system or self.system_prompt,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }

        log.info("[stream] %s model=%s", url, target_model)
        resp = requests.post(
            url, json=payload, timeout=self.timeout, stream=True,
        )
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

    # ── Convenience ──────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Return a summary of the server & model status.

        Useful for status endpoints / UI health bars.
        """
        server_ok = self.is_server_running()
        model_ok = self.is_model_available() if server_ok else False
        models = []
        if server_ok:
            try:
                models = [m.get("name", "?") for m in self.list_models()]
            except Exception:
                pass
        return {
            "server_running": server_ok,
            "model_available": model_ok,
            "active_model": self.model,
            "available_models": models,
            "base_url": self.base_url,
        }

    def __repr__(self) -> str:
        return (
            f"OllamaClient(base_url={self.base_url!r}, "
            f"model={self.model!r}, auto_start={self.auto_start})"
        )
