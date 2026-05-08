"""state_manager.py — Central session state for Sara.

Wraps the SQLite-backed ``Memory`` singleton with higher-level helpers that
are called by the Agent after every successful tool execution.

Responsibilities
----------------
* Track the current workbook, active sheet, and column headers.
* Build the LLM context prefix so the model always knows what file/sheet
  it is working on — enabling natural follow-up commands.
* Update state atomically after every action so context never drifts.
* Provide a clean snapshot for the Planner's precondition checks.

Usage::

    from state.state_manager import state_manager

    # After every successful tool execution:
    state_manager.update_after_action(action_type, params, result)

    # Before routing a command through the LLM:
    enriched = state_manager.enrich_text("calculate total")
    # → "[file:sales.xlsx sheet:Sheet1 columns:Name,Revenue] calculate total"

    # For the Planner:
    ctx = state_manager.get_context()
    # → {"active_file": "sales.xlsx", "active_sheet": "Sheet1",
    #    "active_headers": ["Name", "Revenue"], "recent_actions": [...]}
"""
from __future__ import annotations

import logging
import os
import sys

# Ensure backend/ is on sys.path so 'memory' is importable
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from memory import memory as _mem

log = logging.getLogger("state_manager")


class StateManager:
    """
    High-level state facade wrapping the Memory singleton.

    All writes go through ``update_after_action()`` so state transitions are
    centralised and auditable.  The Memory singleton handles thread safety
    and SQLite persistence — StateManager only adds the domain logic layer.
    """

    # Action types that modify the active file reference
    _FILE_CHANGING = frozenset({
        "create_excel", "create_sheet_and_add_row", "open_file",
    })

    # Action types that change the active sheet
    _SHEET_CHANGING = frozenset({
        "create_excel", "create_sheet_and_add_row", "create_sheet",
        "switch_sheet", "delete_sheet", "rename_sheet",
    })

    # Action types that modify the column header list
    _HEADER_CHANGING = frozenset({
        "create_excel", "create_sheet_and_add_row",
        "add_column", "delete_column",
    })

    def update_after_action(
        self,
        action_type: str,
        params: dict,
        result: dict,
    ) -> None:
        """
        Update stored context after a successful tool execution.

        Called by Agent._update_memory() (and optionally the Executor directly).
        Skips update if ``result`` contains an ``"error"`` key.

        Parameters
        ----------
        action_type : The action that was executed (e.g. "add_column").
        params      : The parameters that were passed to the tool.
        result      : The dict returned by the tool.
        """
        if not isinstance(result, dict) or "error" in result:
            return   # do not update on failure

        atype = action_type or ""

        # ---- Active file ------------------------------------------------
        filename = params.get("filename")
        if filename and atype in self._FILE_CHANGING | {"open_file", "save"}:
            _mem.set_active_file(filename)
            log.info("[State] active_file → %s", filename)
            # Keep excel_controller session in sync with memory so pathless
            # operations (add_row with no path) use the correct file.
            self._sync_controller_session(filename)
        elif filename and _mem.get_active_file() is None:
            # First-time: whatever file was just touched becomes active
            _mem.set_active_file(filename)
            log.info("[State] active_file (first touch) → %s", filename)
            self._sync_controller_session(filename)

        # ---- Active sheet -----------------------------------------------
        if atype in ("create_excel", "create_sheet_and_add_row"):
            _mem.set_active_sheet("Sheet1")
        elif atype == "create_sheet":
            sheet = params.get("sheet_name") or result.get("sheet")
            if sheet:
                _mem.set_active_sheet(sheet)
        elif atype in ("switch_sheet", "rename_sheet"):
            sheet = params.get("sheet_name") or params.get("new_name") or result.get("sheet")
            if sheet:
                _mem.set_active_sheet(sheet)
        elif atype == "delete_sheet":
            # After deletion active sheet reverts to first sheet
            active = result.get("active_sheet", "Sheet1")
            _mem.set_active_sheet(active)

        # ---- Column headers ---------------------------------------------
        if atype in ("create_excel", "create_sheet_and_add_row"):
            headers = params.get("headers") or result.get("headers", [])
            if headers:
                _mem.set_active_headers(list(headers))
            else:
                _mem.set_active_headers([])

        elif atype == "add_column":
            col_name = params.get("name") or params.get("column_name")
            if col_name:
                existing = _mem.get_active_headers() or []
                if col_name not in existing:
                    _mem.set_active_headers(existing + [col_name])

        elif atype == "delete_column":
            col_id = str(params.get("column", "")).upper()
            existing = _mem.get_active_headers() or []
            # Best-effort: remove by letter-position or name match
            # (headers list is ordered, column letter maps to 0-based index)
            if col_id.isalpha() and existing:
                from openpyxl.utils import column_index_from_string
                try:
                    idx = column_index_from_string(col_id) - 1  # 0-based
                    if 0 <= idx < len(existing):
                        updated = existing[:idx] + existing[idx + 1:]
                        _mem.set_active_headers(updated)
                except Exception:
                    pass
            elif col_id in existing:
                _mem.set_active_headers([h for h in existing if h != col_id])

        elif atype == "switch_sheet":
            # Changing sheets invalidates the known header list
            _mem.set_active_headers([])

        # ---- Recent commands --------------------------------------------
        _mem.add_command({"type": atype, **params})
        log.debug("[State] updated after %s", atype)

    def get_context(self) -> dict:
        """Return the full current context as a plain dict.

        Used by the Planner for precondition checks and by
        ``enrich_text()`` for the LLM prompt prefix.
        """
        snap = _mem.snapshot()
        return {
            "active_file":    snap.get("active_file"),
            "active_sheet":   snap.get("active_sheet") or "Sheet1",
            "active_headers": snap.get("active_headers") or [],
            "recent_actions": [
                c.get("type") for c in (snap.get("recent_commands") or [])[-5:]
                if isinstance(c, dict)
            ],
        }

    def enrich_text(self, text: str) -> str:
        """Prepend a compact context prefix so the LLM always knows the
        current file, sheet, and columns.

        Example::

            state_manager.enrich_text("calculate total")
            # → "[file:sales.xlsx sheet:Sheet1 columns:Name,Revenue] calculate total"

        All values are sanitised (brackets escaped) to prevent prompt injection.
        The prefix uses bracket notation so the heuristic parser's regex
        patterns are not confused by it.
        """
        try:
            from pipeline_sanitizer import sanitize_memory_value as _smv
        except ImportError:
            def _smv(v):   # type: ignore[misc]
                return str(v).replace('[', '(').replace(']', ')')

        ctx = self.get_context()
        parts: list[str] = []

        if ctx["active_file"]:
            parts.append(f"file:{_smv(ctx['active_file'])}")
        if ctx["active_sheet"]:
            parts.append(f"sheet:{_smv(ctx['active_sheet'])}")
        if ctx["active_headers"]:
            safe_headers = ",".join(_smv(h) for h in ctx["active_headers"])
            parts.append(f"columns:{safe_headers}")

        if parts:
            return f"[{' '.join(parts)}] {text}"
        return text

    # ------------------------------------------------------------------ #
    # Session-management helpers                                          #
    # ------------------------------------------------------------------ #

    def has_active_file(self) -> bool:
        """Return ``True`` when a workbook is currently open for editing.

        Checks the SQLite-backed memory (survives process restarts).  Use
        ``require_active_file()`` to get a human-readable error string.
        """
        return bool(_mem.get_active_file())

    def require_active_file(self) -> str | None:
        """Return an error message if no file is open, otherwise ``None``.

        Designed as a one-line guard at the start of any tool that needs an
        open workbook::

            err = state_manager.require_active_file()
            if err:
                return AgentResult(success=False, speech_text=err)
        """
        if not _mem.get_active_file():
            return (
                "No Excel file is open. "
                "Say 'create a new file' or 'open filename.xlsx' first."
            )
        return None

    def open_session(self, filename: str, headers: list | None = None) -> None:
        """Open a new file session in both memory and the excel controller.

        Replaces any previous session.  Prefer calling this over setting
        ``active_file`` directly so the excel controller's ``_active_path``
        stays in sync with the SQLite-backed memory.

        Parameters
        ----------
        filename : Basename or relative/absolute path of the Excel file.
        headers  : Optional column headers; stored for LLM context enrichment.
        """
        _mem.set_active_file(filename)
        _mem.set_active_sheet("Sheet1")
        if headers:
            _mem.set_active_headers(list(headers))
        self._sync_controller_session(filename)
        log.info("[State] open_session: %s", filename)

    def close_session(self, filename: str | None = None) -> dict:
        """Save and close the active session.

        Delegates to ``excel_controller.close_excel()`` so the controller's
        ``_active_path`` is cleared (preventing pathless commands from
        accidentally targeting the old file), then wipes the memory context
        so future LLM prompts do not reference a stale file/sheet/headers.

        Parameters
        ----------
        filename : When given, only closes if it matches the active session.
                   When omitted, closes whatever is currently active.

        Returns
        -------
        The result dict from ``close_excel()`` (contains ``"speech_text"``).
        """
        result: dict = {}
        try:
            from excel_controller import close_excel as _close_excel
            result = _close_excel(filename)
        except Exception as exc:
            log.warning("[State] close_excel raised (non-fatal): %s", exc)

        _mem.set_active_file(None)
        _mem.set_active_sheet(None)
        _mem.set_active_headers([])
        log.info("[State] close_session: memory cleared (file=%s)", filename or "active")
        return result

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sync_controller_session(filename: str) -> None:
        """Push *filename* to ``excel_controller.open_file_session``.

        This keeps the controller's in-memory ``_active_path`` consistent
        with the SQLite-backed memory so that operations which omit the
        *path* argument always target the right file.

        Failures are swallowed — the controller session is a performance
        convenience, not a correctness requirement (callers always pass an
        explicit path through the agent→tool_registry pipeline).
        """
        try:
            from excel_controller import open_file_session as _open
            _open(filename)
        except Exception as exc:
            log.debug("[State] _sync_controller_session failed (non-fatal): %s", exc)

    def clear_session(self) -> None:
        """Reset all context — called when a new wake-word session starts."""
        _mem.clear()
        log.info("[State] session context cleared")

    def snapshot(self) -> dict:
        """Alias for get_context() — keeps API consistent with Memory."""
        return self.get_context()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# Import and use this everywhere instead of creating new instances.
#
#     from state.state_manager import state_manager
#
state_manager = StateManager()
