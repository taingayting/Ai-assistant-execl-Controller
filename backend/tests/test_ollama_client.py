"""Tests for backend.llm.ollama_client — OllamaClient.

Every test is fully offline; HTTP calls and subprocess launches are mocked.
"""
from __future__ import annotations

import json
import os
import sys
import types
from unittest.mock import MagicMock, patch, PropertyMock

import requests
import pytest

# ── path fixup ──────────────────────────────────────────────────
_BACKEND_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.insert(0, os.path.abspath(_BACKEND_DIR))

from llm.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaGenerationError,
    OllamaModelNotFound,
    OllamaServerNotRunning,
    OllamaStartupFailed,
    RECOMMENDED_MODELS,
)


# ── Helpers ─────────────────────────────────────────────────────

def _fake_tags_response(models: list[str]):
    """Return a mock requests.Response for GET /api/tags."""
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "models": [{"name": m, "size": 1_000_000} for m in models],
    }
    return r


def _fake_generate_response(text: str):
    """Return a mock requests.Response for POST /api/generate."""
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"response": text}
    return r


def _fake_stream_lines(tokens: list[str]):
    """Return lines as if coming from streaming /api/generate."""
    lines = []
    for tok in tokens:
        lines.append(json.dumps({"response": tok, "done": False}))
    lines.append(json.dumps({"response": "", "done": True}))
    return lines


# ───────────────────────────────────────────────────────────────
# Test: constants & recommended models
# ───────────────────────────────────────────────────────────────

class TestConstants:
    def test_recommended_models_present(self):
        assert "qwen3:8b" in RECOMMENDED_MODELS
        assert "llama3:8b" in RECOMMENDED_MODELS
        assert "mistral" in RECOMMENDED_MODELS

    def test_recommended_not_empty(self):
        assert len(RECOMMENDED_MODELS) >= 3


# ───────────────────────────────────────────────────────────────
# Test: server detection
# ───────────────────────────────────────────────────────────────

class TestServerDetection:
    @patch("llm.ollama_client.requests.get")
    def test_server_running(self, mock_get):
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        c = OllamaClient(auto_start=False)
        assert c.is_server_running() is True

    @patch("llm.ollama_client.requests.get", side_effect=ConnectionError)
    def test_server_not_running(self, mock_get):
        c = OllamaClient(auto_start=False)
        assert c.is_server_running() is False

    @patch("llm.ollama_client.requests.get")
    def test_server_bad_status(self, mock_get):
        r = MagicMock()
        r.status_code = 500
        mock_get.return_value = r
        c = OllamaClient(auto_start=False)
        assert c.is_server_running() is False


# ───────────────────────────────────────────────────────────────
# Test: auto-start
# ───────────────────────────────────────────────────────────────

class TestAutoStart:
    @patch("llm.ollama_client.shutil.which", return_value=None)
    def test_start_server_no_binary(self, _):
        c = OllamaClient(auto_start=True)
        with pytest.raises(OllamaStartupFailed, match="not found"):
            c.start_server()

    @patch("llm.ollama_client.time.sleep")
    @patch("llm.ollama_client.subprocess.Popen")
    @patch("llm.ollama_client.shutil.which", return_value="/usr/bin/ollama")
    @patch("llm.ollama_client.requests.get")
    def test_start_server_success(self, mock_get, _which, mock_popen, _sleep):
        # Server becomes healthy on first poll.
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        c = OllamaClient(auto_start=True)
        c.start_server()
        mock_popen.assert_called_once()
        assert c._server_process is not None

    @patch("llm.ollama_client._STARTUP_MAX_WAIT", 0.5)
    @patch("llm.ollama_client._STARTUP_POLL_INTERVAL", 0.1)
    @patch("llm.ollama_client.subprocess.Popen")
    @patch("llm.ollama_client.shutil.which", return_value="/usr/bin/ollama")
    @patch("llm.ollama_client.requests.get", side_effect=ConnectionError)
    def test_start_server_timeout(self, _get, _which, _popen):
        c = OllamaClient(auto_start=True)
        with pytest.raises(OllamaStartupFailed, match="not healthy"):
            c.start_server()

    @patch("llm.ollama_client.requests.get", side_effect=ConnectionError)
    def test_ensure_server_raises_when_no_auto_start(self, _get):
        c = OllamaClient(auto_start=False)
        with pytest.raises(OllamaServerNotRunning, match="auto_start"):
            c.ensure_server()

    @patch("llm.ollama_client.time.sleep")
    @patch("llm.ollama_client.subprocess.Popen")
    @patch("llm.ollama_client.shutil.which", return_value="/usr/bin/ollama")
    @patch("llm.ollama_client.requests.get")
    def test_ensure_server_auto_starts(self, mock_get, _which, mock_popen, _sleep):
        # First call (is_server_running) -> fail, second call (poll) -> ok.
        fail = MagicMock()
        fail.status_code = 500
        mock_get.side_effect = [fail, _fake_tags_response(["qwen3:8b"])]
        c = OllamaClient(auto_start=True)
        c.ensure_server()  # should not raise
        mock_popen.assert_called_once()


# ───────────────────────────────────────────────────────────────
# Test: model management
# ───────────────────────────────────────────────────────────────

class TestModelManagement:
    @patch("llm.ollama_client.requests.get")
    def test_list_models(self, mock_get):
        mock_get.return_value = _fake_tags_response(["qwen3:8b", "mistral:latest"])
        c = OllamaClient(auto_start=False)
        models = c.list_models()
        assert len(models) == 2
        assert models[0]["name"] == "qwen3:8b"

    @patch("llm.ollama_client.requests.get")
    def test_is_model_available_true(self, mock_get):
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        c = OllamaClient(auto_start=False, model="qwen3:8b")
        assert c.is_model_available() is True

    @patch("llm.ollama_client.requests.get")
    def test_is_model_available_false(self, mock_get):
        mock_get.return_value = _fake_tags_response(["mistral:latest"])
        c = OllamaClient(auto_start=False, model="qwen3:8b")
        assert c.is_model_available() is False

    @patch("llm.ollama_client.requests.post")
    @patch("llm.ollama_client.requests.get")
    def test_get_model_info(self, mock_get, mock_post):
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        info_resp = MagicMock()
        info_resp.status_code = 200
        info_resp.json.return_value = {"modelfile": "FROM qwen3:8b", "parameters": "8B"}
        mock_post.return_value = info_resp
        c = OllamaClient(auto_start=False)
        info = c.get_model_info()
        assert "modelfile" in info

    @patch("llm.ollama_client.requests.post")
    @patch("llm.ollama_client.requests.get")
    def test_pull_model(self, mock_get, mock_post):
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        pull_resp = MagicMock()
        pull_resp.status_code = 200
        mock_post.return_value = pull_resp
        c = OllamaClient(auto_start=False)
        c.pull_model("llama3:8b")  # should not raise

    @patch("llm.ollama_client.requests.post")
    @patch("llm.ollama_client.requests.get")
    def test_pull_model_fail(self, mock_get, mock_post):
        mock_get.return_value = _fake_tags_response([])
        pull_resp = MagicMock()
        pull_resp.status_code = 404
        pull_resp.text = "not found"
        mock_post.return_value = pull_resp
        c = OllamaClient(auto_start=False)
        with pytest.raises(OllamaError, match="Failed to pull"):
            c.pull_model("nonexistent:9b")

    @patch("llm.ollama_client.requests.post")
    @patch("llm.ollama_client.requests.get")
    def test_switch_model_already_available(self, mock_get, mock_post):
        mock_get.return_value = _fake_tags_response(["qwen3:8b", "mistral:latest"])
        c = OllamaClient(auto_start=False, model="qwen3:8b")
        c.switch_model("mistral")
        assert c.model == "mistral"

    @patch("llm.ollama_client.requests.post")
    @patch("llm.ollama_client.requests.get")
    def test_switch_model_pulls_missing(self, mock_get, mock_post):
        # First list_models calls: model not found.
        tags_no = _fake_tags_response(["qwen3:8b"])
        tags_yes = _fake_tags_response(["qwen3:8b", "llama3:8b"])
        mock_get.side_effect = [tags_no, tags_no, tags_yes]
        pull_resp = MagicMock()
        pull_resp.status_code = 200
        mock_post.return_value = pull_resp
        c = OllamaClient(auto_start=False, model="qwen3:8b")
        c.switch_model("llama3:8b")
        assert c.model == "llama3:8b"


# ───────────────────────────────────────────────────────────────
# Test: blocking generation
# ───────────────────────────────────────────────────────────────

class TestGenerate:
    @patch("llm.ollama_client.requests.post")
    @patch("llm.ollama_client.requests.get")
    def test_generate_returns_parsed(self, mock_get, mock_post):
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        mock_post.return_value = _fake_generate_response(
            '{"type": "add_row", "values": ["Tom", 90]}'
        )
        c = OllamaClient(auto_start=False, max_retries=1)
        result = c.generate("add row Tom 90")
        assert isinstance(result, dict)
        # _plan_and_validate may wrap in steps or return flat
        if "steps" in result:
            assert any(s.get("type") == "add_row" for s in result["steps"])
        else:
            assert result.get("type") == "add_row"

    @patch("llm.ollama_client.requests.post")
    @patch("llm.ollama_client.requests.get")
    def test_generate_empty_returns_unknown(self, mock_get, mock_post):
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        mock_post.return_value = _fake_generate_response("")
        c = OllamaClient(auto_start=False, max_retries=1)
        result = c.generate("gibberish")
        assert result.get("type") == "unknown"

    @patch("llm.ollama_client.requests.post")
    @patch("llm.ollama_client.requests.get")
    def test_generate_raw_mode(self, mock_get, mock_post):
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        mock_post.return_value = _fake_generate_response("Hello world")
        c = OllamaClient(auto_start=False, max_retries=1)
        result = c.generate("hi", raw=True)
        assert result == {"response": "Hello world"}

    @patch("llm.ollama_client.requests.post", side_effect=requests.Timeout)
    @patch("llm.ollama_client.requests.get")
    def test_generate_timeout_retries(self, mock_get, mock_post):
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        mock_post.side_effect = requests.Timeout("connection timeout")
        c = OllamaClient(auto_start=False, max_retries=2, base_backoff=0.01)
        with pytest.raises(OllamaGenerationError, match="failed"):
            c.generate("test")
        # Should have been called max_retries times.
        assert mock_post.call_count == 2

    @patch("llm.ollama_client.requests.post")
    @patch("llm.ollama_client.requests.get")
    def test_generate_with_model_override(self, mock_get, mock_post):
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        mock_post.return_value = _fake_generate_response(
            '{"type": "save"}'
        )
        c = OllamaClient(auto_start=False, max_retries=1)
        c.generate("save", model="mistral")
        # Verify the payload uses the override model.
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "mistral"


# ───────────────────────────────────────────────────────────────
# Test: streaming generation
# ───────────────────────────────────────────────────────────────

class TestGenerateStream:
    @patch("llm.ollama_client.requests.post")
    @patch("llm.ollama_client.requests.get")
    def test_stream_yields_tokens(self, mock_get, mock_post):
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = _fake_stream_lines(["Hel", "lo ", "world"])
        mock_post.return_value = resp

        c = OllamaClient(auto_start=False)
        tokens = list(c.generate_stream("hi"))
        assert tokens == ["Hel", "lo ", "world"]

    @patch("llm.ollama_client.requests.post")
    @patch("llm.ollama_client.requests.get")
    def test_stream_empty(self, mock_get, mock_post):
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = _fake_stream_lines([])
        mock_post.return_value = resp

        c = OllamaClient(auto_start=False)
        tokens = list(c.generate_stream("empty"))
        assert tokens == []


# ───────────────────────────────────────────────────────────────
# Test: health summary
# ───────────────────────────────────────────────────────────────

class TestHealth:
    @patch("llm.ollama_client.requests.get")
    def test_health_when_running(self, mock_get):
        mock_get.return_value = _fake_tags_response(["qwen3:8b"])
        c = OllamaClient(auto_start=False, model="qwen3:8b")
        h = c.health()
        assert h["server_running"] is True
        assert h["model_available"] is True
        assert "qwen3:8b" in h["available_models"]

    @patch("llm.ollama_client.requests.get", side_effect=ConnectionError)
    def test_health_when_down(self, _):
        c = OllamaClient(auto_start=False)
        h = c.health()
        assert h["server_running"] is False
        assert h["model_available"] is False
        assert h["available_models"] == []


# ───────────────────────────────────────────────────────────────
# Test: repr
# ───────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_format(self):
        c = OllamaClient(auto_start=False, model="test:1b", base_url="http://x:1234")
        r = repr(c)
        assert "OllamaClient" in r
        assert "test:1b" in r
        assert "http://x:1234" in r


# ───────────────────────────────────────────────────────────────
# Test: retry logic
# ───────────────────────────────────────────────────────────────

class TestRetry:
    def test_retry_succeeds_after_transient(self):
        c = OllamaClient(auto_start=False, max_retries=3, base_backoff=0.01)
        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise requests.ConnectionError("temp")
            return {"ok": True}

        result = c._retry(flaky, label="test")
        assert result == {"ok": True}
        assert call_count["n"] == 3

    def test_retry_raises_non_transient(self):
        c = OllamaClient(auto_start=False, max_retries=3, base_backoff=0.01)
        def bad():
            raise ValueError("permanent")
        with pytest.raises(ValueError):
            c._retry(bad, label="test")
