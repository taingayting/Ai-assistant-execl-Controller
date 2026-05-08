"""base_tool.py — Universal tool interface for Sara.

Every tool in Sara's tool layer must follow this contract.

Why a standard interface?
  • The Executor can call any tool without knowing its internals.
  • New tools (Word, PowerPoint, database) can be added without modifying
    the Executor or Agent — just inherit BaseTool and register.
  • Consistent ``speech_text`` key in every response means the runtime
    always knows what to speak, regardless of which tool ran.
  • ``_success()`` and ``_error()`` helpers prevent inconsistent response
    shapes across tools.

Usage::

    # Defining a new tool:
    class AddColumnTool(BaseTool):
        name        = "add_column"
        description = "Add a new column with the given name to the spreadsheet."

        def execute(self, path: str, *, context: dict | None = None, **params) -> dict:
            col_name = params.get("name")
            if not col_name:
                return self._error("column name is required",
                                   "Please tell me the column name.")
            from excel_controller import add_column
            result = add_column(path, col_name)
            if "error" in result:
                return self._error(result["error"])
            return self._success(
                "column_added",
                f"Done! I've added the '{col_name}' column.",
                column_name=col_name,
            )

    # Registering the tool:
    tool_registry.register("add_column", AddColumnTool())

    # Calling through registry (Executor never instantiates tools directly):
    result = tool_registry.execute("add_column", path="sales.xlsx", name="Revenue")
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """
    Abstract base class for all Sara tools.

    Subclasses must define:
        name        – registry key (str, snake_case)
        description – one-line human description for UI/logs
        execute()   – the implementation

    The ``__call__`` method allows registered tools to be called as plain
    callables, which is what ToolRegistry.execute() does internally.
    """

    # Subclasses must override these two class attributes
    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, path: str, *, context: dict | None = None, **params: Any) -> dict:
        """
        Run the tool against the Excel file at *path*.

        Parameters
        ----------
        path    : Absolute path to the target Excel file.
        context : Runtime context snapshot from StateManager:
                  {"active_file", "active_sheet", "active_headers", ...}
                  May be None if the tool doesn't need context.
        **params: Action-specific keyword arguments parsed from the LLM output.
                  E.g. for add_column: name="Revenue"
                       for delete_row: index=3
                       for write_cell: cell="B5", value=1500

        Returns
        -------
        dict — must always contain:
            "message"     : snake_case status string
            "speech_text" : natural language response for TTS

        On failure, include instead:
            "error"       : error code string (used by ErrorHandler)
            "speech_text" : (optional) override for the error message

        The tool must NEVER raise an exception — catch all errors and return
        an appropriate error dict.
        """

    def __call__(self, path: str, **kwargs: Any) -> dict:
        """Allow instances to be used as plain callables in the registry."""
        context = kwargs.pop("context", None)
        return self.execute(path, context=context, **kwargs)

    # ------------------------------------------------------------------
    # Response builder helpers
    # ------------------------------------------------------------------

    def _success(self, message: str, speech: str, **extra: Any) -> dict:
        """Build a standard success response dict.

        Parameters
        ----------
        message : snake_case status code, e.g. "column_added"
        speech  : Natural language sentence for TTS output.
        **extra : Any additional fields to include (e.g. column_name=...).
        """
        return {"message": message, "speech_text": speech, **extra}

    def _error(self, error_code: str, speech: str | None = None, **extra: Any) -> dict:
        """Build a standard error response dict.

        Parameters
        ----------
        error_code : Snake_case error identifier, matched by ErrorHandler.
        speech     : Optional human-readable override. If None, ErrorHandler
                     will generate one from the error_code.
        **extra    : Context fields that ErrorHandler may use for formatting
                     (e.g. index=3, filename="sales.xlsx").
        """
        result: dict[str, Any] = {"error": error_code, **extra}
        if speech:
            result["speech_text"] = speech
        return result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


# ---------------------------------------------------------------------------
# Convenience base classes for common tool patterns
# ---------------------------------------------------------------------------

class ExcelReadTool(BaseTool):
    """Base for tools that read from Excel without modifying the file.

    Subclasses only need to override ``execute()`` — the class documents
    intent and allows the registry to categorise tools by type.
    """


class ExcelWriteTool(BaseTool):
    """Base for tools that write to Excel (add/modify data).

    These tools must call ``_undo_snapshot()`` in excel_controller before
    any destructive write so the undo system can restore the file.
    """


class ExcelDestructiveTool(BaseTool):
    """Base for tools that permanently delete data (rows, columns, sheets).

    These tools always require an undo snapshot and should trigger the
    confirmation flow in the Planner when ``auto_confirm=False``.
    """
