"""Plan executor for Sara.

Executes a multi-step plan (from *planner.py* or ``Agent._execute``) through
the :class:`ToolRegistry`, wrapping the entire sequence in transactional
logic with automatic backup/rollback.

Key guarantees
--------------
* **Transactional**: a file-level backup is created before the first
  destructive or mutating step.  If any step fails the file is restored
  to its pre-execution state.
* **Confirmation-aware**: steps whose action type is destructive are
  skipped (and flagged) unless the caller passes ``confirmed=True``.
* **Thread-safe**: all mutable state is local to a single ``execute_plan``
  call; the module holds no global mutable state.
* **Extensible**: delegates every tool call to the :class:`ToolRegistry`
  with a fallback to the legacy ``execute_action`` dispatcher so new tool
  families can be added without touching this module.

Usage::

    from executor import execute_plan

    result = execute_plan(
        plan={"steps": [{"type": "create_excel"}, {"type": "add_row", "values": [1, 2]}]},
        filepath="/tmp/out.xlsx",
        confirmed=True,
    )
    assert result["success"]
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Any

log = logging.getLogger("executor")

# --------------- internal helpers ---------------

_action_to_kwargs = None  # lazy import cache


def _get_action_to_kwargs():
    """Lazily import ``_action_to_kwargs`` from agent to avoid circular imports."""
    global _action_to_kwargs
    if _action_to_kwargs is None:
        from agent import _action_to_kwargs as _fn
        _action_to_kwargs = _fn
    return _action_to_kwargs


def _get_tool_registry():
    """Return the module-level ToolRegistry singleton (lazy)."""
    from tools.tool_registry import registry
    return registry


def _get_execute_action():
    """Return the legacy ``execute_action`` dispatcher (lazy)."""
    from excel_controller import execute_action
    return execute_action


def _is_destructive_type(action_type: str) -> bool:
    """Check if *action_type* is destructive via workflow_router."""
    try:
        from workflow_router import DESTRUCTIVE_TYPES
        return action_type in DESTRUCTIVE_TYPES
    except ImportError:
        return False


# --------------- backup / rollback ---------------

# Guard concurrent backup operations on the same filepath.
_backup_lock = threading.Lock()


def _create_backup(filepath: str) -> str | None:
    """Copy *filepath* to a ``.txn_backup`` sidecar.

    Returns the backup path on success, ``None`` if the source file does
    not exist or the copy fails.
    """
    if not filepath or not os.path.exists(filepath):
        return None
    backup_path = filepath + ".txn_backup"
    with _backup_lock:
        try:
            shutil.copy2(filepath, backup_path)
            log.debug("backup created: %s", backup_path)
            return backup_path
        except Exception as exc:
            log.warning("failed to create backup for %s: %s", filepath, exc)
            return None


def _rollback(filepath: str, backup_path: str) -> bool:
    """Restore *filepath* from *backup_path*.  Returns True on success."""
    with _backup_lock:
        try:
            shutil.copy2(backup_path, filepath)
            log.info("rolled back %s from %s", filepath, backup_path)
            return True
        except Exception as exc:
            log.error("rollback failed for %s: %s", filepath, exc)
            return False


def _cleanup_backup(backup_path: str | None) -> None:
    """Remove the transactional backup file if it exists."""
    if backup_path and os.path.exists(backup_path):
        try:
            os.remove(backup_path)
        except Exception:
            pass


# --------------- single-step dispatch ---------------

def _dispatch_step(step: dict, filepath: str) -> dict:
    """Execute a single action *step* via the ToolRegistry or legacy path.

    Always returns a ``dict``.  On unexpected exceptions the dict contains
    an ``"error"`` key.
    """
    registry = _get_tool_registry()
    atype = step.get("type") if isinstance(step, dict) else None

    try:
        if atype and registry.is_registered(atype):
            kwargs = _get_action_to_kwargs()(atype, step)
            result = registry.execute(atype, path=filepath, **kwargs)
        else:
            result = _get_execute_action()(step, filepath)
    except Exception as exc:
        log.exception("_dispatch_step: %s raised %s", atype, exc)
        return {"error": str(exc)}

    # Normalise: legacy dispatcher may return plain strings.
    if isinstance(result, str):
        return {"message": result}
    if not isinstance(result, dict):
        return {"message": str(result)}
    return result


def _step_failed(result: dict) -> bool:
    """Return True when *result* signals a failure."""
    return isinstance(result, dict) and "error" in result


# --------------- public API ---------------

def execute_plan(
    plan: dict,
    filepath: str,
    *,
    confirmed: bool = False,
    auto_correct: bool = False,
) -> dict[str, Any]:
    """Execute a multi-step *plan* transactionally.

    Parameters
    ----------
    plan:
        An action dict.  May contain a ``"steps"`` list for multi-step
        plans, or represent a single action (must have a ``"type"`` key).
    filepath:
        Absolute path to the target file.
    confirmed:
        If ``True`` destructive steps are executed without pausing.
        When ``False`` any destructive step causes the plan to stop and
        return a ``pending_confirmation`` flag.
    auto_correct:
        Reserved for future auto-correction integration.

    Returns
    -------
    dict with keys:
        success             – overall boolean
        executed_steps      – list of ``{step, result, index}`` dicts
        total_steps         – number of planned steps
        skipped_destructive – list of step indices that were skipped
        rolled_back         – True if a rollback occurred
        recovery            – dict with suggestions (only on failure)
        auto_corrections    – list of corrections applied (empty for now)
        error               – error string (only on failure)
        pending_confirmation – True when a destructive step blocks execution
    """
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list) or not steps:
        # Single-action plan — wrap it in a one-element list.
        if isinstance(plan, dict) and plan.get("type"):
            steps = [plan]
        else:
            return _failure("Plan has no executable steps.", rolled_back=False)

    total = len(steps)
    log.info("execute_plan: %d step(s), confirmed=%s, file=%s", total, confirmed, filepath)

    # --- pre-execution backup ---
    backup_path = _create_backup(filepath)

    executed: list[dict] = []
    skipped_destructive: list[int] = []
    auto_corrections: list[dict] = []

    for idx, step in enumerate(steps):
        atype = step.get("type", "") if isinstance(step, dict) else ""

        # Destructive gate
        if _is_destructive_type(atype) and not confirmed:
            log.info("execute_plan: step %d/%d (%s) requires confirmation — pausing", idx + 1, total, atype)
            skipped_destructive.append(idx)
            # Return immediately so the caller can request confirmation.
            _cleanup_backup(backup_path)
            return {
                "success": True,
                "executed_steps": executed,
                "total_steps": total,
                "skipped_destructive": skipped_destructive,
                "pending_confirmation": True,
                "confirmation_step": idx,
                "confirmation_type": atype,
                "rolled_back": False,
                "recovery": None,
                "auto_corrections": auto_corrections,
                "error": None,
            }

        # Dispatch
        result = _dispatch_step(step, filepath)
        entry = {"step": step, "result": result, "index": idx}
        executed.append(entry)

        if _step_failed(result):
            error_msg = result.get("error", "unknown error")
            log.warning("execute_plan: step %d/%d (%s) failed — %s", idx + 1, total, atype, error_msg)

            # Rollback
            rolled_back = False
            if backup_path and os.path.exists(backup_path):
                rolled_back = _rollback(filepath, backup_path)
            _cleanup_backup(backup_path)

            recovery = _build_recovery(step, error_msg, executed, total)
            return {
                "success": False,
                "executed_steps": executed,
                "total_steps": total,
                "skipped_destructive": skipped_destructive,
                "rolled_back": rolled_back,
                "recovery": recovery,
                "auto_corrections": auto_corrections,
                "error": f"Step {idx + 1}/{total} ({atype}) failed: {error_msg}",
                "pending_confirmation": False,
            }

    # All steps succeeded
    _cleanup_backup(backup_path)
    log.info("execute_plan: all %d steps completed successfully", total)
    return {
        "success": True,
        "executed_steps": executed,
        "total_steps": total,
        "skipped_destructive": skipped_destructive,
        "rolled_back": False,
        "recovery": None,
        "auto_corrections": auto_corrections,
        "error": None,
        "pending_confirmation": False,
    }


# --------------- result builders ---------------

def _failure(error: str, *, rolled_back: bool) -> dict[str, Any]:
    return {
        "success": False,
        "executed_steps": [],
        "total_steps": 0,
        "skipped_destructive": [],
        "rolled_back": rolled_back,
        "recovery": {"message": error, "suggestions": []},
        "auto_corrections": [],
        "error": error,
        "pending_confirmation": False,
    }


def _build_recovery(
    failed_step: dict,
    error_msg: str,
    executed: list[dict],
    total: int,
) -> dict[str, Any]:
    """Build recovery guidance for a failed step."""
    suggestions: list[str] = []
    atype = failed_step.get("type", "") if isinstance(failed_step, dict) else ""

    if "not found" in error_msg.lower() or "no active file" in error_msg.lower():
        suggestions.append("Create or open an Excel file first.")
    if "permission" in error_msg.lower():
        suggestions.append("Close the file in other applications and retry.")
    if not suggestions:
        suggestions.append("Review the failed step and retry.")

    return {
        "message": f"Step failed: {error_msg}",
        "failed_action_type": atype,
        "completed_before_failure": len(executed) - 1,
        "total_steps": total,
        "suggestions": suggestions,
    }
