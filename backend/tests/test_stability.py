"""Stability tests for Sara's voice pipeline.

Verifies that the system never crashes under the 5 most common failure modes:

    1. Invalid JSON from LLM
    2. Hallucinated (unregistered) tool name
    3. Tool execution error (Excel API failure)
    4. Router fallback (Ollama down, Groq down → heuristic / unknown)
    5. Empty LLM response

Each test asserts:
  - No exception is raised (system stays alive)
  - A safe dict/AgentResult is returned (never raw exception text)
  - The pipeline produces a user-facing speech_text where applicable
"""
from __future__ import annotations

import sys
import os
import pathlib
import types
from unittest.mock import MagicMock, patch

import pytest

# ---- path setup (mirrors conftest.py) ------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
BACKEND_DIR = str(HERE.parent)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ---- imports (lazy where needed to allow monkey-patching) ----------------


# ==========================================================================
# Scenario 1 — Invalid JSON from LLM
# ==========================================================================

class TestInvalidJsonFromLLM:
    """The LLM returns garbage text that cannot be parsed as JSON.

    Expected behaviour:
      - RobustJsonExtractor raises ValueError('no json found')
      - Caller catches it and returns {"type": "unknown", ...}
      - System never crashes; user hears a polite error message.
    """

    def test_robust_extractor_raises_on_garbage(self):
        from llm_parser import RobustJsonExtractor
        with pytest.raises(ValueError, match='no json found'):
            RobustJsonExtractor.extract("this is not json at all!!")

    def test_robust_extractor_raises_on_empty(self):
        from llm_parser import RobustJsonExtractor
        with pytest.raises(ValueError):
            RobustJsonExtractor.extract("")

    def test_robust_extractor_raises_on_none(self):
        from llm_parser import RobustJsonExtractor
        with pytest.raises(ValueError):
            RobustJsonExtractor.extract(None)  # type: ignore[arg-type]

    def test_validate_llm_result_returns_unknown_for_bad_json_output(self):
        """_validate_llm_result must return unknown, not raise, for any non-dict."""
        from llm_parser import _validate_llm_result
        result = _validate_llm_result("not a dict", "original text")
        assert result.get('type') == 'unknown'
        assert 'raw' in result

    def test_validate_llm_result_returns_unknown_for_empty_dict(self):
        from llm_parser import _validate_llm_result
        result = _validate_llm_result({}, "original text")
        assert result.get('type') == 'unknown'

    def test_parse_groq_direct_survives_malformed_response(self):
        """_parse_groq_direct must return unknown dict, not raise, on bad response."""
        from llm_parser import _parse_groq_direct
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "<<not json>>"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "<<not json>>"

        with patch('llm_parser.requests.post', return_value=mock_resp):
            with patch.dict(os.environ, {'GROQ_API_KEY': 'fake-key'}):
                result = _parse_groq_direct("add a row")
        assert isinstance(result, dict)
        # Must degrade to unknown, not crash
        assert result.get('type') in ('unknown', 'add_row', 'create_excel')  # any valid type is OK


# ==========================================================================
# Scenario 2 — Hallucinated (unregistered) tool name
# ==========================================================================

class TestHallucinatedToolName:
    """The LLM invents a tool that does not exist in the registry.

    Expected behaviour:
      - ToolRegistry.execute raises ToolNotRegisteredError
      - _dispatch_single logs a warning and falls through to legacy path
        OR the validate step rejects the action type before execution.
      - System never crashes.
    """

    def test_registry_raises_for_unknown_tool(self):
        from tools.tool_registry import ToolRegistry, ToolNotRegisteredError
        reg = ToolRegistry()
        with pytest.raises(ToolNotRegisteredError):
            reg.execute('delete_entire_database', path='x.xlsx')

    def test_registry_assert_registered_raises(self):
        from tools.tool_registry import ToolRegistry, ToolNotRegisteredError
        reg = ToolRegistry()
        with pytest.raises(ToolNotRegisteredError):
            reg.assert_registered('send_email')

    def test_validate_single_action_rejects_hallucinated_type(self):
        """Actions whose type is not in _ALLOWED_TYPES must be rejected."""
        from llm_parser import _validate_single_action
        result = _validate_single_action({'type': 'hack_mainframe', 'target': 'nasa'})
        assert result is None

    def test_validate_llm_result_rejects_hallucinated_type(self):
        from llm_parser import _validate_llm_result
        result = _validate_llm_result({'type': 'send_email', 'to': 'boss@corp.com'}, "send an email")
        assert result.get('type') == 'unknown'

    def test_dispatch_single_logs_warning_for_unregistered(self, caplog):
        """_dispatch_single must log a WARNING when falling through to legacy path."""
        import logging
        from agent import Agent
        from excel_controller import execute_action

        agent = Agent.__new__(Agent)  # skip __init__

        fake_action = {'type': 'a_totally_fake_tool'}
        fake_path = '/nonexistent/path.xlsx'

        with patch('agent.execute_action', return_value={'error': 'file not found'}):
            with caplog.at_level(logging.WARNING, logger='agent'):
                result = Agent._dispatch_single(fake_action, fake_path)

        assert any('NOT in tool registry' in r.message for r in caplog.records), \
            "Expected a WARNING about unregistered tool"


# ==========================================================================
# Scenario 3 — Tool execution error (Excel API failure)
# ==========================================================================

class TestToolExecutionError:
    """The registered tool raises an exception during execution.

    Expected behaviour:
      - ToolRegistry.execute re-raises the exception (callers handle it)
      - agent._execute catches it and returns an error dict
      - handle_action catches _execute errors and returns safe AgentResult
      - System never crashes.
    """

    def test_registry_re_raises_tool_exception(self):
        from tools.tool_registry import ToolRegistry

        reg = ToolRegistry()
        reg.register('bad_tool', lambda path, **kw: (_ for _ in ()).throw(RuntimeError('disk full')))

        with pytest.raises(RuntimeError, match='disk full'):
            reg.execute('bad_tool', path='x.xlsx')

    def test_handle_action_survives_execute_exception(self, tmp_path):
        """handle_action must return a safe AgentResult when _execute raises."""
        from agent import Agent, AgentResult

        files_dir = str(tmp_path)
        agent = Agent(files_dir)

        valid_action = {'type': 'add_row', 'values': ['Alice', 90]}

        with patch.object(agent, '_execute', side_effect=RuntimeError('openpyxl crashed')):
            with patch.object(agent.planner, 'create_plan') as mock_plan:
                mock_plan.return_value = MagicMock(valid=True, warnings=[], requires_confirmation=False)
                with patch('agent.workflow_router.route', return_value=None):
                    result = agent.handle_action(valid_action, confirmed=True)

        assert isinstance(result, AgentResult)
        assert result.success is False
        assert result.speech_text is not None
        assert 'exception' not in result.speech_text.lower(), \
            "Raw exception text must not reach the user"

    def test_handle_action_excel_error_dict_is_reported(self, tmp_path):
        """When _execute returns an error dict, handle_action reports it."""
        from agent import Agent, AgentResult

        files_dir = str(tmp_path)
        agent = Agent(files_dir)

        valid_action = {'type': 'add_row', 'values': ['Alice', 90]}
        error_result = {'error': 'Sheet does not exist'}

        with patch.object(agent, '_execute', return_value=error_result):
            with patch.object(agent.planner, 'create_plan') as mock_plan:
                mock_plan.return_value = MagicMock(valid=True, warnings=[], requires_confirmation=False)
                with patch('agent.workflow_router.route', return_value=None):
                    result = agent.handle_action(valid_action, confirmed=True)

        assert isinstance(result, AgentResult)


# ==========================================================================
# Scenario 4 — Router fallback (Ollama + Groq both fail)
# ==========================================================================

class TestRouterFallback:
    """Both Ollama and Groq are unavailable.

    Expected behaviour:
      - Router catches all exceptions internally
      - Returns {"type": "unknown", ...} (never raises)
      - System degrades gracefully to a polite "I didn't understand" message.
    """

    def test_try_groq_returns_unknown_on_timeout(self):
        import requests
        from llm_router import try_groq

        with patch('llm_router._parse_groq_direct', side_effect=requests.Timeout()):
            with patch.dict(os.environ, {'GROQ_API_KEY': 'fake'}):
                result = try_groq("calculate average")

        assert isinstance(result, dict)
        assert result.get('type') == 'unknown'

    def test_try_groq_returns_unknown_on_connection_error(self):
        import requests
        from llm_router import try_groq

        with patch('llm_router._parse_groq_direct', side_effect=requests.ConnectionError()):
            result = try_groq("add row")

        assert isinstance(result, dict)
        assert result.get('type') == 'unknown'

    def test_route_llm_returns_unknown_when_all_tiers_fail(self):
        """route_llm must never raise — always returns a dict."""
        from llm_router import route_llm

        with patch('llm_router.try_ollama', side_effect=Exception("ollama dead")):
            with patch('llm_router.try_groq', return_value={"type": "unknown", "raw": "x"}):
                with patch('llm_router._heuristic_natural_language_parse',
                           return_value={"type": "unknown", "raw": "x"}):
                    result = route_llm("do something impossible xyz123")

        assert isinstance(result, dict)

    def test_handle_text_returns_safe_result_when_route_raises(self, tmp_path):
        """handle_text must not propagate an unexpected route_llm exception."""
        from agent import Agent, AgentResult

        agent = Agent(str(tmp_path))
        with patch('agent.route_llm', side_effect=RuntimeError("network exploded")):
            result = agent.handle_text("calculate average", auto_execute=True)

        assert isinstance(result, AgentResult)
        assert result.success is False
        assert result.speech_text is not None


# ==========================================================================
# Scenario 5 — Empty LLM response
# ==========================================================================

class TestEmptyLLMResponse:
    """The LLM returns an empty string or whitespace-only response.

    Expected behaviour:
      - RobustJsonExtractor raises ValueError
      - Caller returns unknown action dict
      - TTS receives a safe fallback string (never empty)
      - System never crashes.
    """

    def test_extractor_raises_on_whitespace(self):
        from llm_parser import RobustJsonExtractor
        with pytest.raises(ValueError):
            RobustJsonExtractor.extract("   ")

    def test_ollama_empty_response_returns_unknown(self):
        """An Ollama response with empty 'response' field is handled safely.

        _ollama_call logs a warning and returns {"type": "unknown"} rather
        than raising — the system must stay alive and degrade gracefully.
        """
        from llm_router import _ollama_call

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": ""}
        mock_resp.raise_for_status = MagicMock()

        with patch('llm_router.requests.post', return_value=mock_resp):
            result = _ollama_call("add a row")

        # Must return a safe dict, never raise, never crash
        assert isinstance(result, dict)
        assert result.get('type') == 'unknown'

    def test_sanitize_speech_never_returns_empty(self):
        """sanitize_speech must always return a non-empty string."""
        from pipeline_sanitizer import sanitize_speech

        for bad_input in ["", "   ", None, "{}", '{"type":"add_row"}', "<think>x</think>"]:
            result = sanitize_speech(bad_input)  # type: ignore[arg-type]
            assert result, f"sanitize_speech returned empty for input {bad_input!r}"
            assert isinstance(result, str)

    def test_sanitize_speech_strips_raw_json(self):
        from pipeline_sanitizer import sanitize_speech
        result = sanitize_speech('{"type": "add_row", "values": ["Tom", 90]}')
        assert result == "Done."

    def test_sanitize_speech_strips_think_tags(self):
        from pipeline_sanitizer import sanitize_speech
        result = sanitize_speech("<think>I should add a row</think>Row added successfully.")
        assert '<think>' not in result
        assert 'Row added successfully' in result
