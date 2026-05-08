"""Workflow router for Sara.

Maps LLM-parsed action types to workflow definitions and validates
preconditions before execution.  Each workflow is registered as a dict
with its ID, category, the action types it handles, whether it is
destructive, and what preconditions must hold in memory.

The router is the bridge between the Agent (intent) and the Tool layer
(excel_controller.execute_action).
"""
import logging
import os

log = logging.getLogger('workflow_router')


# --------------- Workflow registry ---------------

# Each entry mirrors the metadata block in workflows/*.md
# Keys:
#   id            – matches the Workflow ID in the .md file
#   category      – logical grouping
#   action_types  – set of LLM action types this workflow handles
#   requires_file – True if memory must contain active_file
#   destructive   – True if the action needs user confirmation
#   description   – one-line purpose

WORKFLOWS = [
    {
        "id": "create_excel_file",
        "category": "file_management",
        "action_types": {"create_excel"},
        "requires_file": False,
        "destructive": False,
        "description": "Create a new Excel workbook with optional headers.",
    },
    {
        "id": "open_excel_file",
        "category": "file_management",
        "action_types": {"open_file"},
        "requires_file": False,
        "destructive": False,
        "description": "Open an existing Excel workbook.",
    },
    {
        "id": "launch_excel_app",
        "category": "file_management",
        "action_types": {"open_excel_app"},
        "requires_file": False,
        "destructive": False,
        "description": "Launch the Microsoft Excel application (no workbook required).",
    },
    {
        "id": "insert_data",
        "category": "data_entry",
        "action_types": {"add_row", "write_cell", "insert_cell", "create_sheet_and_add_row"},
        "requires_file": True,
        "destructive": False,
        "description": "Insert data by appending a row or writing to a cell.",
    },
    {
        "id": "add_column",
        "category": "structure",
        "action_types": {"add_column"},
        "requires_file": True,
        "destructive": False,
        "description": "Add a new column with a header and optional default.",
    },
    {
        "id": "delete_column",
        "category": "structure",
        "action_types": {"delete_column"},
        "requires_file": True,
        "destructive": False,
        "description": "Delete a column by letter or index.",
    },
    {
        "id": "calculate_sum",
        "category": "calculation",
        "action_types": {"calculate_row_sum"},
        "requires_file": True,
        "destructive": False,
        "description": "Calculate the sum of values in a row.",
    },
    {
        "id": "calculate_average",
        "category": "calculation",
        "action_types": {"calculate_average"},
        "requires_file": True,
        "destructive": False,
        "description": "Calculate the average of values in a column.",
    },
    {
        "id": "format_cells",
        "category": "formatting",
        "action_types": {"format_cells"},
        "requires_file": True,
        "destructive": False,
        "description": "Apply formatting to a range of cells.",
    },
    {
        "id": "create_chart",
        "category": "visualization",
        "action_types": {"create_chart"},
        "requires_file": True,
        "destructive": False,
        "description": "Create and embed a chart in the workbook.",
    },
    {
        "id": "create_sales_report",
        "category": "reporting",
        "action_types": {"create_sales_report"},
        "requires_file": False,
        "destructive": False,
        "description": "Create a sales report with standard columns and formatting.",
    },
    {
        "id": "save",
        "category": "file_management",
        "action_types": {"save"},
        "requires_file": True,
        "destructive": False,
        "description": "Save the current workbook.",
    },
    {
        "id": "delete_row",
        "category": "data_entry",
        "action_types": {"delete_row"},
        "requires_file": True,
        "destructive": False,
        "description": "Delete a row by 1-based index.",
    },
    {
        "id": "sheet_management",
        "category": "structure",
        "action_types": {"create_sheet", "switch_sheet", "list_sheets", "rename_sheet"},
        "requires_file": True,
        "destructive": False,
        "description": "Create, switch, rename, or list sheets in a workbook.",
    },
    {
        "id": "rename_file",
        "category": "file_management",
        "action_types": {"rename_file"},
        "requires_file": True,
        "destructive": False,
        "description": "Rename the current Excel workbook file on disk.",
    },
    {
        "id": "delete_sheet",
        "category": "structure",
        "action_types": {"delete_sheet"},
        "requires_file": True,
        "destructive": False,
        "description": "Delete a sheet from a workbook.",
    },
    {
        "id": "find_replace",
        "category": "data_entry",
        "action_types": {"find_replace"},
        "requires_file": True,
        "destructive": False,
        "description": "Find and replace values across the sheet.",
    },
    {
        "id": "insert_row_at",
        "category": "data_entry",
        "action_types": {"insert_row_at"},
        "requires_file": True,
        "destructive": False,
        "description": "Insert a blank row at a specific position.",
    },
    {
        "id": "insert_column_at",
        "category": "data_entry",
        "action_types": {"insert_column_at"},
        "requires_file": True,
        "destructive": False,
        "description": "Insert a blank column at a specific position.",
    },
    {
        "id": "sort_range",
        "category": "data_entry",
        "action_types": {"sort_range"},
        "requires_file": True,
        "destructive": False,
        "description": "Sort sheet data by a column.",
    },
    {
        "id": "filter_table",
        "category": "data_entry",
        "action_types": {"filter_table"},
        "requires_file": True,
        "destructive": False,
        "description": "Filter rows into a Filtered sheet.",
    },
    {
        "id": "clear_filter",
        "category": "data_entry",
        "action_types": {"clear_filter"},
        "requires_file": True,
        "destructive": False,
        "description": "Delete the Filtered sheet and return to source sheet.",
    },
    {
        "id": "column_width",
        "category": "formatting",
        "action_types": {"set_column_width"},
        "requires_file": True,
        "destructive": False,
        "description": "Set the width of a column.",
    },
    {
        "id": "freeze_panes",
        "category": "formatting",
        "action_types": {"freeze_panes"},
        "requires_file": True,
        "destructive": False,
        "description": "Freeze rows or columns in a sheet.",
    },
    {
        "id": "merge_cells",
        "category": "formatting",
        "action_types": {"merge_cells"},
        "requires_file": True,
        "destructive": False,
        "description": "Merge a range of cells.",
    },
    {
        "id": "unmerge_cells",
        "category": "formatting",
        "action_types": {"unmerge_cells"},
        "requires_file": True,
        "destructive": False,
        "description": "Unmerge previously merged cells.",
    },
    {
        "id": "read_range",
        "category": "data_entry",
        "action_types": {"read_range"},
        "requires_file": True,
        "destructive": False,
        "description": "Read and return values from a cell range.",
    },
    {
        "id": "write_formula",
        "category": "calculation",
        "action_types": {"write_formula"},
        "requires_file": True,
        "destructive": False,
        "description": "Write an Excel formula to a cell.",
    },
]

# Build a fast lookup: action_type -> workflow dict
_ACTION_TYPE_INDEX: dict[str, dict] = {}
for _wf in WORKFLOWS:
    for _at in _wf["action_types"]:
        _ACTION_TYPE_INDEX[_at] = _wf

# Also index by workflow ID for direct lookup
_WORKFLOW_ID_INDEX: dict[str, dict] = {wf["id"]: wf for wf in WORKFLOWS}

# Destructive action types (for confirmation checks)
DESTRUCTIVE_TYPES = frozenset(
    at for wf in WORKFLOWS if wf["destructive"] for at in wf["action_types"]
) | {"clear_range"}
# Note: "delete_row" and "delete_sheet" are NOT listed here — both have undo
# support (_undo_snapshot before deletion) so confirmation is unnecessary and
# would block voice commands.


def get_workflow_by_action_type(action_type: str) -> dict | None:
    """Return the workflow definition that handles *action_type*, or None."""
    return _ACTION_TYPE_INDEX.get(action_type)


def get_workflow_by_id(workflow_id: str) -> dict | None:
    """Return a workflow definition by its ID, or None."""
    return _WORKFLOW_ID_INDEX.get(workflow_id)


def list_workflows() -> list[dict]:
    """Return all registered workflow definitions (JSON-safe)."""
    out = []
    for wf in WORKFLOWS:
        d = dict(wf)
        d["action_types"] = sorted(d["action_types"])  # set -> list
        out.append(d)
    return out


def route(action: dict) -> dict | None:
    """Given a parsed action dict, find the matching workflow.

    For multi-step plans (action has 'steps' list) the first step's type
    is used to identify the primary workflow.
    """
    if not isinstance(action, dict):
        return None
    # multi-step plan
    steps = action.get("steps")
    if isinstance(steps, list) and steps:
        first = steps[0] if isinstance(steps[0], dict) else {}
        return get_workflow_by_action_type(first.get("type", ""))
    return get_workflow_by_action_type(action.get("type", ""))


# --------------- Precondition checking ---------------

def check_preconditions(workflow: dict, memory_snapshot: dict,
                        filepath: str | None = None,
                        all_action_types: list[str] | None = None) -> list[str]:
    """Return a list of unmet precondition messages (empty = all OK).

    Parameters:
        workflow          – primary workflow dict from the registry
        memory_snapshot   – result of memory.snapshot()
        filepath          – resolved file path on disk (for existence checks)
        all_action_types  – ALL action types in the plan (including every step
                            of a multi-step plan).  When provided, every type
                            in the list is checked for file requirements so that
                            a multi-step plan cannot sneak a requires_file step
                            past validation just because it isn't the first step.
    """
    errors: list[str] = []
    active = memory_snapshot.get("active_file")
    file_exists = filepath and os.path.exists(filepath)

    # Collect workflows for ALL action types in the plan, not just the first.
    workflows_to_check: list[dict] = [workflow] if workflow else []
    if all_action_types:
        for atype in all_action_types:
            wf = get_workflow_by_action_type(atype)
            if wf and wf not in workflows_to_check:
                workflows_to_check.append(wf)

    file_required = any(wf.get("requires_file") for wf in workflows_to_check)
    if file_required:
        if not active:
            errors.append("No active file. Please open or create a file first.")
        elif filepath and not file_exists:
            # Only an error for workflows that read an existing file.
            _create_ids = {"create_excel_file", "create_sales_report", "insert_data"}
            non_create = [
                wf for wf in workflows_to_check
                if wf.get("requires_file") and wf.get("id") not in _create_ids
            ]
            if non_create:
                errors.append(f"File '{active}' not found.")

    return errors


def is_destructive(action: dict) -> bool:
    """Return True if the action (or any step) is destructive."""
    if not isinstance(action, dict):
        return False
    steps = action.get("steps")
    if isinstance(steps, list) and steps:
        return any(
            isinstance(s, dict) and s.get("type") in DESTRUCTIVE_TYPES
            for s in steps
        )
    return action.get("type") in DESTRUCTIVE_TYPES
