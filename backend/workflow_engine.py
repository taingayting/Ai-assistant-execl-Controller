"""Workflow engine for markdown-defined Sara workflows.

Reads workflow definitions from workflows/*.md and exposes helpers to:
- load all workflows
- fetch a workflow by id
- execute a workflow's tool calls sequentially
"""

from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path

from excel_controller import execute_action

log = logging.getLogger("workflow_engine")

WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / "workflows"

# In-memory cache keyed by workflow_id
_WORKFLOW_CACHE: dict[str, dict] = {}


def load_workflows() -> dict[str, dict]:
    """Load and parse all markdown workflows from workflows/*.md.

    Returns:
        dict keyed by workflow_id with parsed metadata.
    """
    workflows: dict[str, dict] = {}
    if not WORKFLOWS_DIR.exists():
        log.warning("workflow_engine: workflows dir does not exist: %s", WORKFLOWS_DIR)
        _WORKFLOW_CACHE.clear()
        return _WORKFLOW_CACHE

    for md_file in sorted(WORKFLOWS_DIR.glob("*.md")):
        try:
            parsed = _parse_workflow_markdown(md_file.read_text(encoding="utf-8"), md_file)
            wf_id = parsed.get("workflow_id")
            if wf_id:
                workflows[wf_id] = parsed
        except Exception as exc:
            log.warning("workflow_engine: failed parsing %s: %s", md_file.name, exc)

    _WORKFLOW_CACHE.clear()
    _WORKFLOW_CACHE.update(workflows)
    return _WORKFLOW_CACHE


def get_workflow(workflow_id: str) -> dict | None:
    """Get a workflow by id from cache (auto-load if needed)."""
    if not _WORKFLOW_CACHE:
        load_workflows()
    return _WORKFLOW_CACHE.get(workflow_id)


def execute_workflow(workflow_id: str, parameters: dict | None = None) -> dict:
    """Execute workflow tool calls sequentially using excel_controller.

    Parameters:
        workflow_id: workflow metadata id from markdown
        parameters:
            Expected keys include:
            - filepath: absolute xlsx path (required for tool execution)
            - action: parsed action dict (optional)
            - any extra fields used as placeholder values in tool calls

    Returns:
        dict with workflow execution summary.
    """
    wf = get_workflow(workflow_id)
    if not wf:
        return {"error": "workflow_not_found", "workflow_id": workflow_id}

    params = dict(parameters or {})
    filepath = params.get("filepath")
    if not filepath:
        return {"error": "filepath_required", "workflow_id": workflow_id}

    tool_calls = wf.get("tool_calls") or []
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if not isinstance(tool_calls, list) or not tool_calls:
        return {
            "error": "no_tool_calls",
            "workflow_id": workflow_id,
            "message": "Workflow does not define executable tool calls.",
        }

    step_results: list[dict] = []
    for idx, raw_call in enumerate(tool_calls, start=1):
        if not isinstance(raw_call, dict) or not raw_call.get("type"):
            return {
                "error": "invalid_tool_call",
                "workflow_id": workflow_id,
                "step": idx,
                "tool_call": raw_call,
            }

        action = _build_action(raw_call, params)
        result = execute_action(action, filepath)
        step_results.append({"step": idx, "action": action, "result": result})

        if isinstance(result, dict) and result.get("error"):
            return {
                "error": "workflow_step_failed",
                "workflow_id": workflow_id,
                "failed_step": idx,
                "failed_action": action,
                "step_result": result,
                "steps": step_results,
            }

    speech_text = None
    if step_results:
        last_result = step_results[-1].get("result")
        if isinstance(last_result, dict):
            speech_text = last_result.get("speech_text")

    return {
        "message": "workflow_executed",
        "workflow_id": workflow_id,
        "steps": step_results,
        "speech_text": speech_text,
    }


def _parse_workflow_markdown(text: str, source_file: Path) -> dict:
    workflow_id = _extract_metadata_value(text, "Workflow ID") or source_file.stem
    category = _extract_metadata_value(text, "Category") or "unknown"

    preconditions_section = _extract_section(text, "Preconditions")
    preconditions = _extract_bullets(preconditions_section)

    steps_section = _extract_section(text, "Steps")
    steps = _extract_numbered_steps(steps_section)

    tool_calls = _extract_tool_calls(text)

    return {
        "workflow_id": workflow_id,
        "category": category,
        "preconditions": preconditions,
        "steps": steps,
        "tool_calls": tool_calls,
        "source_file": str(source_file),
    }


def _extract_metadata_value(text: str, label: str) -> str | None:
    # Handles lines like: > **Workflow ID:** `create_sales_report`
    pattern = rf"^>\s*\*\*{re.escape(label)}:\*\*\s*(.+?)\s*$"
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    raw = m.group(1).strip()
    # strip markdown backticks and trailing spacing artifacts
    raw = raw.strip("`").strip()
    return raw or None


def _extract_section(text: str, name: str) -> str:
    # Capture section content from "## Name" until next level-2 heading or EOF.
    pattern = rf"^##\s+{re.escape(name)}\s*$([\s\S]*?)(?=^##\s+|\Z)"
    m = re.search(pattern, text, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def _extract_bullets(section_text: str) -> list[str]:
    out = []
    for line in section_text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            out.append(line[2:].strip())
    return out


def _extract_numbered_steps(section_text: str) -> list[str]:
    out = []
    for line in section_text.splitlines():
        line = line.strip()
        if re.match(r"^\d+\.\s+", line):
            out.append(re.sub(r"^\d+\.\s+", "", line).strip())
    return out


def _extract_tool_calls(text: str):
    # Support headings like "## Tool Calls" and "## Tool Calls (Sequential)"
    m = re.search(r"^##\s+Tool Calls[^\n]*$([\s\S]*?)(?=^##\s+|\Z)", text, flags=re.MULTILINE)
    if not m:
        return []
    block = m.group(1)

    # Prefer fenced JSON blocks; use the first parseable one.
    for fm in re.finditer(r"```json\s*([\s\S]*?)\s*```", block, flags=re.IGNORECASE):
        payload = fm.group(1).strip()
        if not payload:
            continue
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue

    return []


def _build_action(tool_call: dict, parameters: dict) -> dict:
    action = copy.deepcopy(tool_call)

    # Simple {{placeholder}} substitution inside string values.
    action = _resolve_placeholders(action, parameters)

    atype = action.get("type")

    # Normalize common schema mismatches between markdown and excel_controller.
    if atype == "add_column" and "column_name" in action and "name" not in action:
        action["name"] = action.pop("column_name")

    # Parameter-driven overrides for reusable workflows.
    if atype == "create_excel" and "headers" in parameters:
        action["headers"] = parameters["headers"]
    if atype == "add_column" and "name" in parameters:
        action["name"] = parameters["name"]
    if atype == "delete_column" and "column" in parameters:
        action["column"] = parameters["column"]

    return action


def _resolve_placeholders(value, parameters: dict):
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, parameters) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(v, parameters) for v in value]
    if isinstance(value, str):
        # Replace exact placeholder token, e.g. "{{filename}}".
        mm = re.fullmatch(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}", value)
        if mm:
            key = mm.group(1)
            return parameters.get(key, value)
    return value
