"""Tests for llm_router module.

Covers:
  * try_groq — delegates to the Groq backend
  * try_ollama — Ollama /api/generate parsing, connection errors, timeouts
  * is_ollama_available — health-check logic
  * route_llm — Groq-first routing with Ollama fallback
  * USE_OLLAMA toggle
  * _is_valid helper
"""
import pytest
from unittest.mock import patch, MagicMock
import os
import requests

import llm_router


# ── try_groq ───────────────────────────────────────────────────

class TestTryGroq:
    def test_delegates_to_groq_parser(self):
        """try_groq should call _parse_groq_direct from llm_parser."""
        with patch('llm_router._parse_groq_direct', return_value={"type": "add_row", "values": ["A"]}) as mock:
            result = llm_router.try_groq("add row A")
            mock.assert_called_once_with("add row A")
            assert result == {"type": "add_row", "values": ["A"]}

    def test_returns_unknown_on_failure(self):
        with patch('llm_router._parse_groq_direct', return_value={"type": "unknown", "raw": "xyz"}):
            result = llm_router.try_groq("xyz")
            assert result["type"] == "unknown"

    def test_backward_compat_alias(self):
        """parse_with_groq alias should point to try_groq."""
        assert llm_router.parse_with_groq is llm_router.try_groq


# ── try_ollama ─────────────────────────────────────────────────

class TestTryOllama:
    def test_successful_parse(self):
        """Ollama /api/generate returns {response: '...'}."""
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "response": '{"type":"add_row","values":["Tom",90]}'
        }
        fake_resp.raise_for_status = MagicMock()

        with patch('llm_router.requests.post', return_value=fake_resp):
            result = llm_router.try_ollama("add row Tom 90")
            assert result.get("type") == "add_row" or "steps" in result

    def test_connection_error_returns_heuristic_or_unknown(self):
        with patch('llm_router.requests.post', side_effect=requests.ConnectionError):
            result = llm_router.try_ollama("add row Tom 90")
            assert isinstance(result, dict)
            assert "type" in result or "steps" in result

    def test_timeout_returns_unknown(self):
        with patch('llm_router.requests.post', side_effect=requests.Timeout):
            result = llm_router.try_ollama("xyzzy gibberish")
            assert isinstance(result, dict)

    def test_bad_json_in_response(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "response": "I can't help with that"
        }
        fake_resp.raise_for_status = MagicMock()

        with patch('llm_router.requests.post', return_value=fake_resp):
            result = llm_router.try_ollama("do something")
            assert isinstance(result, dict)

    def test_empty_response(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"response": ""}
        fake_resp.raise_for_status = MagicMock()

        with patch('llm_router.requests.post', return_value=fake_resp):
            result = llm_router.try_ollama("test")
            assert result["type"] == "unknown"

    def test_multi_step_response(self):
        multi = '{"steps":[{"type":"create_excel","headers":["Name"]},{"type":"add_row","values":["A"]}]}'
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"response": multi}
        fake_resp.raise_for_status = MagicMock()

        with patch('llm_router.requests.post', return_value=fake_resp):
            result = llm_router.try_ollama("create and add")
            assert "steps" in result
            assert len(result["steps"]) == 2

    def test_backward_compat_alias(self):
        """parse_with_ollama alias should point to try_ollama."""
        assert llm_router.parse_with_ollama is llm_router.try_ollama


# ── check_ollama_model_installed ────────────────────────────────

class TestCheckOllamaModelInstalled:
    def test_model_present(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "models": [{"name": "qwen3:8b"}, {"name": "codellama:latest"}]
        }
        with patch('llm_router.requests.get', return_value=fake_resp):
            assert llm_router.check_ollama_model_installed('qwen3') is True

    def test_model_absent(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "models": [{"name": "codellama:latest"}]
        }
        with patch('llm_router.requests.get', return_value=fake_resp):
            assert llm_router.check_ollama_model_installed('qwen3') is False

    def test_defaults_to_ollama_model(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "models": [{"name": "qwen3:8b"}]
        }
        with patch('llm_router.requests.get', return_value=fake_resp), \
             patch('llm_router.OLLAMA_MODEL', 'qwen3:8b'):
            assert llm_router.check_ollama_model_installed() is True

    def test_connection_error(self):
        with patch('llm_router.requests.get', side_effect=requests.ConnectionError):
            assert llm_router.check_ollama_model_installed('qwen3') is False

    def test_bad_status(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 500
        with patch('llm_router.requests.get', return_value=fake_resp):
            assert llm_router.check_ollama_model_installed('qwen3') is False


# ── is_ollama_available (delegates to check_ollama_model_installed) ─

class TestOllamaAvailable:
    def test_available_with_model(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "models": [{"name": "llama3:latest"}, {"name": "codellama:latest"}]
        }
        with patch('llm_router.requests.get', return_value=fake_resp), \
             patch('llm_router.OLLAMA_MODEL', 'llama3'):
            assert llm_router.is_ollama_available() is True

    def test_not_available_wrong_model(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "models": [{"name": "codellama:latest"}]
        }
        with patch('llm_router.requests.get', return_value=fake_resp), \
             patch('llm_router.OLLAMA_MODEL', 'llama3'):
            assert llm_router.is_ollama_available() is False

    def test_not_available_connection_error(self):
        with patch('llm_router.requests.get', side_effect=requests.ConnectionError):
            assert llm_router.is_ollama_available() is False

    def test_not_available_bad_status(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 500
        with patch('llm_router.requests.get', return_value=fake_resp):
            assert llm_router.is_ollama_available() is False


# ── route_llm ──────────────────────────────────────────────────

# All route_llm tests mock check_internet_connection to control online/offline.

class TestRouteLlm:
    def test_groq_success_no_ollama_call(self):
        """When Groq returns a valid action, Ollama should NOT be called."""
        with patch.object(llm_router, 'check_internet_connection', return_value=True), \
             patch.object(llm_router, 'try_groq',
                          return_value={"type": "add_row", "values": ["X"]}) as g, \
             patch.object(llm_router, 'try_ollama') as o:
            result = llm_router.route_llm("add row X")
            g.assert_called_once()
            o.assert_not_called()
            assert result["type"] == "add_row"

    def test_groq_unknown_falls_back_to_ollama(self):
        """Groq returning unknown triggers automatic Ollama fallback."""
        with patch.object(llm_router, 'check_internet_connection', return_value=True), \
             patch.object(llm_router, 'try_groq',
                          return_value={"type": "unknown", "raw": "blah"}) as g, \
             patch.object(llm_router, 'try_ollama',
                          return_value={"type": "add_row", "values": ["Y"]}) as o:
            result = llm_router.route_llm("blah")
            g.assert_called_once()
            o.assert_called_once()
            assert result["type"] == "add_row"

    def test_groq_exception_falls_back_to_ollama(self):
        """Groq raising an exception triggers automatic Ollama fallback."""
        with patch.object(llm_router, 'check_internet_connection', return_value=True), \
             patch.object(llm_router, 'try_groq',
                          side_effect=Exception("network down")) as g, \
             patch.object(llm_router, 'try_ollama',
                          return_value={"type": "save"}) as o:
            result = llm_router.route_llm("save file")
            g.assert_called_once()
            o.assert_called_once()
            assert result["type"] == "save"

    def test_both_fail_returns_unknown(self):
        with patch.object(llm_router, 'check_internet_connection', return_value=True), \
             patch.object(llm_router, 'try_groq',
                          return_value={"type": "unknown", "raw": "???"}) as g, \
             patch.object(llm_router, 'try_ollama',
                          return_value={"type": "unknown", "raw": "???"}) as o:
            result = llm_router.route_llm("???")
            assert result["type"] == "unknown"

    def test_groq_steps_result_accepted(self):
        steps = {"steps": [{"type": "create_excel", "headers": ["A"]}]}
        with patch.object(llm_router, 'check_internet_connection', return_value=True), \
             patch.object(llm_router, 'try_groq', return_value=steps):
            result = llm_router.route_llm("create excel")
            assert "steps" in result

    def test_groq_empty_steps_falls_back(self):
        with patch.object(llm_router, 'check_internet_connection', return_value=True), \
             patch.object(llm_router, 'try_groq',
                          return_value={"steps": []}) as g, \
             patch.object(llm_router, 'try_ollama',
                          return_value={"type": "create_excel", "headers": ["X"]}) as o:
            result = llm_router.route_llm("create")
            o.assert_called_once()

    def test_use_ollama_disabled_skips_fallback(self):
        """When USE_OLLAMA is False, Ollama must not be called."""
        with patch.object(llm_router, 'check_internet_connection', return_value=True), \
             patch.object(llm_router, 'try_groq',
                          return_value={"type": "unknown", "raw": "??"}) as g, \
             patch.object(llm_router, 'try_ollama') as o, \
             patch.object(llm_router, 'USE_OLLAMA', False):
            result = llm_router.route_llm("??")
            g.assert_called_once()
            o.assert_not_called()
            assert result["type"] == "unknown"

    def test_offline_skips_groq_uses_ollama(self):
        """When offline, Groq should be skipped and Ollama called directly."""
        with patch.object(llm_router, 'check_internet_connection', return_value=False), \
             patch.object(llm_router, 'try_groq') as g, \
             patch.object(llm_router, 'try_ollama',
                          return_value={"type": "add_row", "values": ["Z"]}) as o:
            result = llm_router.route_llm("add row Z")
            g.assert_not_called()
            o.assert_called_once()
            assert result["type"] == "add_row"

    def test_offline_ollama_fails_falls_to_heuristic(self):
        """When offline and Ollama fails, heuristic should be tried."""
        with patch.object(llm_router, 'check_internet_connection', return_value=False), \
             patch.object(llm_router, 'try_groq') as g, \
             patch.object(llm_router, 'try_ollama',
                          return_value={"type": "unknown", "raw": "??"}) as o, \
             patch.object(llm_router, '_heuristic_natural_language_parse',
                          return_value={"type": "save"}) as h, \
             patch.object(llm_router, '_plan_and_validate',
                          return_value={"type": "save"}) as p:
            result = llm_router.route_llm("save the file")
            g.assert_not_called()
            o.assert_called_once()
            assert result["type"] == "save"


# ── check_internet_connection ──────────────────────────────────

class TestCheckInternet:
    def test_returns_true_when_reachable(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        with patch('llm_router.requests.head', return_value=fake_resp):
            assert llm_router.check_internet_connection() is True

    def test_returns_false_on_connection_error(self):
        with patch.object(llm_router, '_conn_cache', {"result": None, "ts": 0.0}), \
             patch('llm_router.requests.head', side_effect=requests.ConnectionError):
            assert llm_router.check_internet_connection() is False

    def test_returns_false_on_timeout(self):
        with patch.object(llm_router, '_conn_cache', {"result": None, "ts": 0.0}), \
             patch('llm_router.requests.head', side_effect=requests.Timeout):
            assert llm_router.check_internet_connection() is False


# ── check_groq_available ───────────────────────────────────────

class TestCheckGroqAvailable:
    def test_returns_false_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(llm_router, 'check_internet_connection', return_value=True):
            # Ensure GROQ_API_KEY is absent
            os.environ.pop('GROQ_API_KEY', None)
            assert llm_router.check_groq_available() is False

    def test_returns_true_with_key_and_internet(self):
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
             patch.object(llm_router, 'check_internet_connection', return_value=True):
            assert llm_router.check_groq_available() is True

    def test_returns_false_with_key_but_no_internet(self):
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
             patch.object(llm_router, 'check_internet_connection', return_value=False):
            assert llm_router.check_groq_available() is False


# ── _is_valid ──────────────────────────────────────────────────

class TestIsValid:
    def test_valid_single_action(self):
        assert llm_router._is_valid({"type": "add_row"}) is True

    def test_valid_steps(self):
        assert llm_router._is_valid({"steps": [{"type": "add_row"}]}) is True

    def test_unknown_invalid(self):
        assert llm_router._is_valid({"type": "unknown"}) is False

    def test_empty_type_invalid(self):
        assert llm_router._is_valid({"type": ""}) is False

    def test_none_type_invalid(self):
        assert llm_router._is_valid({"type": None}) is False

    def test_non_dict_invalid(self):
        assert llm_router._is_valid("not a dict") is False

    def test_empty_steps_invalid(self):
        assert llm_router._is_valid({"steps": []}) is False
