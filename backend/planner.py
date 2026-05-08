"""Planner module for Sara.

Receives a parsed command (from ``llm_router.route_llm``) and converts it
into a validated, structured execution plan before the Executor runs it.

Responsibilities
----------------
* Validate that the target file / sheet / column / cell references exist.
* Expand high-level intents into concrete multi-step plans when needed.
* Annotate each step with metadata (destructive flag, estimated effect).
* Return warnings for anything suspicious but non-blocking.
* Flag ``requires_confirmation`` for destructive operations.

Usage::

    from planner import Planner

    planner = Planner(files_dir="backend/files")
    plan = planner.create_plan(action, filename="data.xlsx")
    # plan = {
    #     "steps": [...],
    #     "warnings": [...],
    #     "requires_confirmation": False,
    #     "valid": True,
    #     "error": None,
    # }
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

log = logging.getLogger("planner")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell_ref_valid(ref: str) -> bool:
    """Return True if *ref* looks like a valid Excel cell reference (e.g. A1, BC200)."""
    return bool(re.fullmatch(r"[A-Za-z]{1,3}\d{1,7}", ref))


def _range_ref_valid(ref: str) -> bool:
    """Return True if *ref* is a valid cell or range (e.g. A1, A1:C10)."""
    parts = ref.split(":")
    return all(_cell_ref_valid(p.strip()) for p in parts if p.strip())


def _column_letter_valid(col: str) -> bool:
    """Return True if *col* is a column letter like A, B, ... XFD."""
    return bool(re.fullmatch(r"[A-Za-z]{1,3}", col))


def _file_exists(filepath: str) -> bool:
    return filepath is not None and os.path.exists(filepath)


def _sheet_exists(filepath: str, sheet_name: str) -> bool:
    """Check if *sheet_name* exists in the workbook at *filepath*."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(filepath, read_only=True)
        names = wb.sheetnames
        wb.close()
        return sheet_name in names
    except Exception:
        return False


def _get_sheet_names(filepath: str) -> list[str]:
    """Return sheet names of an existing workbook, or empty list."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(filepath, read_only=True)
        names = list(wb.sheetnames)
        wb.close()
        return names
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Plan result container
# ---------------------------------------------------------------------------

class PlanResult:
    """Structured output from :meth:`Planner.create_plan`."""

    __slots__ = ("steps", "warnings", "requires_confirmation", "valid",
                 "error", "auto_fix_steps", "destructive_count")

    def __init__(
        self,
        steps: list[dict] | None = None,
        warnings: list[str] | None = None,
        requires_confirmation: bool = False,
        valid: bool = True,
        error: str | None = None,
        auto_fix_steps: list[dict] | None = None,
    ):
        self.steps = steps or []
        self.warnings = warnings or []
        self.requires_confirmation = requires_confirmation
        self.valid = valid
        self.error = error
        self.auto_fix_steps = auto_fix_steps or []
        # Count of destructive steps — used by Executor to decide when to
        # create a pre-execution backup for rollback.
        self.destructive_count: int = sum(
            1 for s in self.steps
            if isinstance(s, dict) and s.get("type") in _DESTRUCTIVE_TYPES
        ) if self.steps else 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "steps": self.steps,
            "warnings": self.warnings,
            "requires_confirmation": self.requires_confirmation,
            "valid": self.valid,
        }
        if self.error:
            d["error"] = self.error
        if self.auto_fix_steps:
            d["auto_fix_steps"] = self.auto_fix_steps
        return d


# ---------------------------------------------------------------------------
# Destructive action types (mirrors workflow_router.DESTRUCTIVE_TYPES)
# ---------------------------------------------------------------------------
# "delete_row" and "delete_sheet" are intentionally excluded: both are
# preceded by an undo snapshot, so the user can say "undo" to restore any
# accidentally deleted data.  Requiring confirmation here would permanently
# block voice commands (process_command never passes confirmed=True).
_DESTRUCTIVE_TYPES = frozenset({
    "clear_range",
})

# Action types that require an existing file to operate on.
_REQUIRES_FILE = frozenset({
    "add_row", "write_cell", "insert_cell", "write_formula",
    "delete_row", "insert_row_at", "insert_column_at", "delete_column", "add_column",
    "calculate_row_sum", "calculate_average",
    "format_cells", "create_chart", "save",
    "create_sheet", "switch_sheet", "delete_sheet", "list_sheets",
    "rename_sheet", "rename_file", "find_replace", "sort_range",
    "set_column_width", "freeze_panes", "merge_cells", "unmerge_cells", "read_range",
    "read_cell", "filter_table", "clear_filter",
})

# Action types that create a file (file need not pre-exist).
_CREATES_FILE = frozenset({
    "create_excel", "create_sheet_and_add_row", "open_file",
    "create_sales_report",
})


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """Build and validate a structured execution plan from a parsed action."""

    def __init__(self, files_dir: str):
        self.files_dir = files_dir

    # ---- public API -------------------------------------------------------

    def create_plan(
        self,
        action: dict,
        filename: str | None = None,
        active_sheet: str | None = None,
    ) -> PlanResult:
        """Analyse *action* and return a :class:`PlanResult`.

        Parameters
        ----------
        action:
            Parsed action dict from ``route_llm`` / ``llm_parser``.
            May contain a ``"steps"`` list for multi-step commands.
        filename:
            Target Excel file name (relative to *files_dir*).
        active_sheet:
            Currently active sheet name (from memory), used for validation.

        Returns
        -------
        PlanResult with validated steps, warnings, and confirmation flag.
        """
        if not isinstance(action, dict):
            return PlanResult(valid=False, error="Action must be a dict.")

        # ---- Special case: undo ----------------------------------------
        # "undo" / "undo_last" need no validation — they restore the previous
        # file state.  Pass straight through as a valid single-step plan.
        atype_top = action.get("type", "")
        if atype_top in ("undo", "undo_last"):
            return PlanResult(steps=[action], valid=True)

        filepath = os.path.join(self.files_dir, filename) if filename else None

        # Normalise to a list of steps.
        raw_steps = action.get("steps")
        if isinstance(raw_steps, list) and raw_steps:
            steps = [s for s in raw_steps if isinstance(s, dict)]
        elif action.get("type"):
            steps = [action]
        else:
            return PlanResult(valid=False, error="Action has no type or steps.")

        if not steps:
            return PlanResult(valid=False, error="No executable steps found.")

        # Walk every step: validate + annotate
        validated: list[dict] = []
        warnings: list[str] = []
        auto_fix: list[dict] = []
        needs_confirmation = False

        for idx, step in enumerate(steps):
            atype = step.get("type", "")
            prefix = f"Step {idx + 1}" if len(steps) > 1 else "Action"

            # -- destructive check --
            if atype in _DESTRUCTIVE_TYPES:
                needs_confirmation = True

            # -- file existence --
            if atype in _REQUIRES_FILE:
                if not filepath or not _file_exists(filepath):
                    if filename:
                        # Suggest creating the file first.
                        auto_fix.append({"type": "create_excel", "filename": filename})
                        warnings.append(
                            f"{prefix}: file '{filename}' does not exist — "
                            "will auto-create if auto_correct is enabled."
                        )
                    else:
                        return PlanResult(
                            valid=False,
                            error=f"{prefix} ({atype}): no target file specified.",
                        )

            # -- sheet validation --
            sheet = step.get("sheet_name") or step.get("sheet")
            if atype == "switch_sheet" and sheet and filepath and _file_exists(filepath):
                if not _sheet_exists(filepath, sheet):
                    available = _get_sheet_names(filepath)
                    warnings.append(
                        f"{prefix}: sheet '{sheet}' not found. "
                        f"Available: {available or ['(none)']}."
                    )

            if atype == "delete_sheet" and sheet and filepath and _file_exists(filepath):
                if not _sheet_exists(filepath, sheet):
                    return PlanResult(
                        valid=False,
                        error=f"{prefix}: cannot delete sheet '{sheet}' — it does not exist.",
                    )

            if atype == "create_sheet" and sheet and filepath and _file_exists(filepath):
                if _sheet_exists(filepath, sheet):
                    warnings.append(
                        f"{prefix}: sheet '{sheet}' already exists — "
                        "create_sheet may raise an error."
                    )

            # -- cell / range validation --
            cell = step.get("cell") or step.get("range")
            if atype in ("write_cell", "insert_cell", "write_formula") and cell:
                if not _range_ref_valid(cell):
                    return PlanResult(
                        valid=False,
                        error=f"{prefix}: invalid cell reference '{cell}'.",
                    )

            # -- column validation --
            col = step.get("column") or step.get("col")
            if atype == "delete_column" and col:
                if isinstance(col, str) and not _column_letter_valid(col):
                    # Might be an integer index — that's also fine.
                    try:
                        int(col)
                    except (ValueError, TypeError):
                        return PlanResult(
                            valid=False,
                            error=f"{prefix}: invalid column reference '{col}'.",
                        )

            # -- data range for charts --
            # create_chart requires a full range (e.g. A1:C10) not a single cell,
            # so use a stricter check than _range_ref_valid which also accepts A1.
            if atype == "create_chart":
                dr = step.get("data_range")
                if dr:
                    dr_stripped = str(dr).strip().upper()
                    # Must be START:END where both parts are valid cell refs.
                    if ":" not in dr_stripped or not _range_ref_valid(dr_stripped):
                        return PlanResult(
                            valid=False,
                            error=(
                                f"The chart range '{dr}' isn't valid. "
                                "Please use a format like A1:C10 with a start "
                                "and end cell."
                            ),
                        )

            validated.append(step)

        log.info(
            "create_plan: %d step(s), confirmation=%s, warnings=%d, auto_fix=%d",
            len(validated), needs_confirmation, len(warnings), len(auto_fix),
        )

        return PlanResult(
            steps=validated,
            warnings=warnings,
            requires_confirmation=needs_confirmation,
            valid=True,
            auto_fix_steps=auto_fix,
        )
