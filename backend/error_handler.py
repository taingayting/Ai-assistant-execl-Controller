"""error_handler.py — Domain-aware error message system for Sara.

Three categories of errors, each mapped to a natural, helpful voice response:

    1. UNKNOWN_COMMAND  — LLM couldn't parse the intent
    2. EXECUTION_ERROR  — A tool ran but returned an error dict
    3. SYSTEM_ERROR     — Infrastructure failure (LLM timeout, STT, etc.)

Design principle:
    Never let a Python exception or internal error code reach the user's ears.
    Every error path must end with a complete, actionable English sentence.

Usage::

    from error_handler import ErrorHandler

    msg = ErrorHandler.for_unknown_command(transcript)
    msg = ErrorHandler.for_execution(error_code, context)
    msg = ErrorHandler.for_system(failure_type)
    msg = ErrorHandler.from_result(result_dict, action_type)   # convenience
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("error_handler")


class ErrorHandler:
    """
    Converts every error type Sara can encounter into a safe, helpful
    voice response.

    All methods are @staticmethod so the class can be used without
    instantiation (import and call directly).
    """

    # ------------------------------------------------------------------
    # 1. Unknown / ambiguous command
    # ------------------------------------------------------------------

    _UNKNOWN_RESPONSES = {
        "default": (
            "I didn't quite catch that. "
            "You can try: 'add a row', 'create a chart', or 'calculate the total'."
        ),
        "low_confidence": (
            "I'm not sure I understood that correctly. "
            "Could you rephrase it? For example: 'add column Revenue' "
            "or 'delete row 3'."
        ),
        "empty": (
            "I didn't hear anything. Please try again."
        ),
        "too_short": (
            "That was too short for me to understand. "
            "Try a complete command like 'create a new spreadsheet'."
        ),
    }

    @staticmethod
    def for_unknown_command(
        transcript: str = "",
        confidence: float = 0.0,
    ) -> str:
        """Return a helpful response when the LLM couldn't parse the command.

        Parameters
        ----------
        transcript : The original user speech (used to pick the right message).
        confidence : The parser confidence score (0.0–1.0).
        """
        if not transcript or not transcript.strip():
            return ErrorHandler._UNKNOWN_RESPONSES["empty"]

        words = transcript.strip().split()
        if len(words) <= 1:
            return ErrorHandler._UNKNOWN_RESPONSES["too_short"]

        if 0.0 < confidence < 0.5:
            return ErrorHandler._UNKNOWN_RESPONSES["low_confidence"]

        return ErrorHandler._UNKNOWN_RESPONSES["default"]

    # ------------------------------------------------------------------
    # 2. Execution errors — tool returned {"error": ...}
    # ------------------------------------------------------------------

    # Maps error code strings (from excel_controller return values) to
    # a natural language response.  Uses .format(**context) for variables.
    _EXECUTION_RESPONSES: dict[str, str] = {
        # File / workbook
        "file_not_found": (
            "I couldn't find a spreadsheet called '{filename}'. "
            "Say 'create a new spreadsheet' to make one."
        ),
        "no_active_file": (
            "No spreadsheet is open right now. "
            "Say 'create a new spreadsheet' or 'open {filename}' to get started."
        ),
        # Sheet errors
        "sheet_not_found": (
            "I couldn't find a sheet called '{sheet_name}'. "
            "Say 'list sheets' to see what's available."
        ),
        "sheet_already_exists": (
            "A sheet called '{sheet_name}' already exists. "
            "Please choose a different name."
        ),
        # Row / column errors
        "index_out_of_range": (
            "Row {index} doesn't exist. "
            "The spreadsheet only has {max_row} rows."
        ),
        "column_out_of_range": (
            "That column doesn't exist. "
            "The spreadsheet only has {max_col} columns."
        ),
        "invalid_column": (
            "I don't recognise '{col}' as a valid column. "
            "Try using a letter like A, B, or C."
        ),
        "header_row_protected": (
            "Row 1 is the header row and can't be deleted. "
            "Did you mean a different row number?"
        ),
        # Cell / formula errors
        "invalid_cell": (
            "'{cell}' isn't a valid cell reference. "
            "Use a format like A1 or B10."
        ),
        "formula_blocked": (
            "That formula isn't allowed for safety reasons. "
            "I support standard functions like SUM, AVERAGE, IF, and VLOOKUP."
        ),
        # Undo
        "no_undo_available": (
            "There's nothing to undo right now."
        ),
        # Fallback
        "default": (
            "Something went wrong with that Excel command. "
            "Your file has not been changed."
        ),
    }

    @staticmethod
    def for_execution(
        error_code: str,
        context: dict | None = None,
        action_type: str = "",
    ) -> str:
        """Return a helpful response for a tool execution error.

        Parameters
        ----------
        error_code  : The ``error`` key from the tool's result dict.
        context     : Extra values for template substitution (filename, index…).
        action_type : The action that was being executed (used in logs).
        """
        log.warning("[ErrorHandler] execution error: code=%r  action=%s  ctx=%s",
                    error_code, action_type, context)

        template = (
            ErrorHandler._EXECUTION_RESPONSES.get(error_code)
            or ErrorHandler._EXECUTION_RESPONSES["default"]
        )
        if context:
            try:
                return template.format(**context)
            except KeyError:
                pass

        # If template has unfilled placeholders, return a safe generic message
        try:
            return template.format(
                filename="your file",
                sheet_name="that sheet",
                index="?",
                max_row="?",
                max_col="?",
                col="?",
                cell="?",
            )
        except Exception:
            return ErrorHandler._EXECUTION_RESPONSES["default"]

    @staticmethod
    def from_result(result: dict, action_type: str = "", context: dict | None = None) -> str | None:
        """Convenience: inspect a tool result dict and return an error message
        if the result contains an ``"error"`` key, otherwise return None.

        Usage::

            msg = ErrorHandler.from_result(tool_result, action_type="delete_row")
            if msg:
                sara.speak(msg)
        """
        if not isinstance(result, dict) or "error" not in result:
            return None   # no error — caller should use result["speech_text"]

        error_code = result["error"]
        merged_ctx = dict(result)   # error dict often contains useful context fields
        if context:
            merged_ctx.update(context)

        return ErrorHandler.for_execution(error_code, merged_ctx, action_type)

    # ------------------------------------------------------------------
    # 3. System errors — LLM timeout, STT failure, etc.
    # ------------------------------------------------------------------

    _SYSTEM_RESPONSES: dict[str, str] = {
        # Speech recognition
        "stt_empty":          "I didn't catch anything. Could you please repeat that?",
        "stt_error":          "I had trouble hearing you. Please try again.",
        "stt_no_voice":       "I didn't detect any speech. I'm still here when you're ready.",
        # LLM failures
        "llm_timeout":        "That took too long. Please try a simpler command.",
        "llm_all_failed":     "I'm having trouble thinking right now. Please try again in a moment.",
        "groq_unavailable":   "The cloud assistant isn't available. I'll use the local model instead.",
        "ollama_unavailable":  "The local model isn't responding. Please check that Ollama is running.",
        # Tool failures
        "tool_not_found":     "I don't know how to do that yet. Try a different command.",
        "tool_crashed":       "Something unexpected happened. Please try again.",
        # Excel application
        "excel_locked":       "The spreadsheet seems to be open in Excel. Please close it and try again.",
        "excel_corrupt":      "I couldn't open that file — it may be damaged.",
        # File system
        "disk_full":          "There isn't enough disk space to save the file.",
        "permission_denied":  "I don't have permission to write to that file.",
        # Undo
        "undo_failed":        "I couldn't undo that action. Your file may not have changed.",
        # Default
        "default":            "I encountered an unexpected problem. Please try again.",
    }

    @staticmethod
    def for_system(failure_type: str, detail: str = "") -> str:
        """Return a safe system-error message.

        Parameters
        ----------
        failure_type : Key from _SYSTEM_RESPONSES (e.g. "llm_timeout").
        detail       : Optional internal detail — logged but NOT spoken to user.
        """
        if detail:
            log.error("[ErrorHandler] system error: type=%r  detail=%s", failure_type, detail)
        else:
            log.warning("[ErrorHandler] system error: type=%r", failure_type)

        return (
            ErrorHandler._SYSTEM_RESPONSES.get(failure_type)
            or ErrorHandler._SYSTEM_RESPONSES["default"]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def classify_exception(exc: Exception) -> str:
        """Map a Python exception to a failure_type string for for_system().

        Keeps the classification logic in one place so no raw exception text
        ever reaches voice output.
        """
        msg = str(exc).lower()
        if "timeout" in msg or "timed out" in msg:
            return "llm_timeout"
        if "permission" in msg or "access" in msg:
            return "permission_denied"
        if "disk" in msg or "space" in msg or "no space" in msg:
            return "disk_full"
        if "corrupt" in msg or "bad zip" in msg or "invalid" in msg:
            return "excel_corrupt"
        if "locked" in msg or "sharing violation" in msg:
            return "excel_locked"
        return "default"

    @staticmethod
    def safe_speech(result: dict, action_type: str = "", context: dict | None = None) -> str:
        """Return the best available speech text from a result dict.

        Priority:
            1. result["speech_text"]  — already formatted by the tool
            2. ErrorHandler.from_result() — if result has "error" key
            3. Generic "Done." fallback

        This is the single call site that assistant_runtime should use
        after every tool execution.
        """
        if isinstance(result, dict):
            if "speech_text" in result:
                return result["speech_text"]
            err_msg = ErrorHandler.from_result(result, action_type, context)
            if err_msg:
                return err_msg
        return "Done."
