"""Agent layer for Sara.

The Agent is the central orchestrator that ties together:
  1. LLM parsing   (llm_parser)
  2. Workflow routing & precondition checking (workflow_router)
  3. Tool execution (excel_controller)
  4. Context memory (memory)

Call ``Agent.handle_text()`` with raw user text, or ``Agent.handle_action()``
with an already-parsed action dict. Both return a uniform ``AgentResult``.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field

# Load .env before any backend module reads env vars at import time.
# Mirrors the same pattern used by run_sara.py and app.py.
try:
    from dotenv import load_dotenv as _load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(_env_path):
        _load_dotenv(_env_path, override=True)
except ImportError:
    pass

from llm_parser import parse_with_llm, generate_suggestions
from llm_router import route_llm
from excel_controller import execute_action
from memory import memory
from pipeline_sanitizer import correct_transcript, sanitize_memory_value
from planner import Planner
from tools.tool_registry import registry as tool_registry, register_default_excel_tools
from error_handler import ErrorHandler
import workflow_router
import workflow_engine

log = logging.getLogger('agent')


def _enrich_text_with_context(text: str) -> str:
    """Prepend active memory state so the LLM has file/sheet/column context.

    Produces the same ``[file:X sheet:Y columns:A,B]`` prefix as
    ``AssistantRuntime._enrich_with_context`` so both pipelines (run_sara.py
    and desktop UI) give the LLM identical context.  The heuristic parser
    in llm_parser already strips this prefix before token-based checks, so
    deterministic commands like ``undo`` are unaffected.
    """
    snap = memory.snapshot()
    parts: list[str] = []
    if snap.get('active_file'):
        parts.append(f"file:{sanitize_memory_value(snap['active_file'])}")
    if snap.get('active_sheet'):
        parts.append(f"sheet:{sanitize_memory_value(snap['active_sheet'])}")
    headers = snap.get('active_headers') or []
    if headers:
        safe_headers = ','.join(sanitize_memory_value(h) for h in headers)
        parts.append(f"columns:{safe_headers}")
    if parts:
        return f"[{' '.join(parts)}] {text}"
    return text


# --------------- Result container ---------------

@dataclass
class AgentResult:
    """Uniform result returned by every Agent method."""
    success: bool = True
    action: dict | None = None
    workflow_id: str | None = None
    result: dict | str | None = None
    download_url: str | None = None
    speech_text: str | None = None
    suggestions: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    confirmation_id: str | None = None
    confirmation_message: str | None = None
    error: str | None = None
    memory_snapshot: dict | None = None
    executed: bool = False
    recovery: dict | None = None
    auto_corrected: bool = False

    def to_dict(self) -> dict:
        d: dict = {"success": self.success}
        if self.action is not None:
            d["action"] = self.action
        if self.workflow_id is not None:
            d["workflow_id"] = self.workflow_id
        if self.result is not None:
            d["result"] = self.result
        if self.download_url:
            d["download_url"] = self.download_url
        if self.speech_text:
            d["speech_text"] = self.speech_text
        if self.suggestions:
            d["suggestions"] = self.suggestions
        if self.requires_confirmation:
            d["requires_confirmation"] = True
            d["confirmation_id"] = self.confirmation_id
            d["message"] = self.confirmation_message
        if self.error:
            d["error"] = self.error
        if self.memory_snapshot is not None:
            d["memory"] = self.memory_snapshot
        if self.recovery is not None:
            d["recovery"] = self.recovery
        if self.auto_corrected:
            d["auto_corrected"] = True
        d["executed"] = self.executed
        return d


class Agent:
    """Sara agent — stateless orchestrator (state lives in *memory* singleton)."""

    def __init__(self, files_dir: str):
        self.files_dir = files_dir
        self.planner = Planner(files_dir)
        # Load markdown workflows once at startup; engine also lazy-loads if needed.
        workflow_engine.load_workflows()
        # Populate the tool registry with default Excel tools.
        if len(tool_registry) == 0:
            register_default_excel_tools()

    # ---- public entry points ----

    def handle_text(self, text: str, filename: str | None = None,
                    auto_execute: bool = False,
                    auto_correct: bool = False) -> AgentResult:
        """Parse *text* with the LLM, then optionally execute.

        Parameters:
            text         – raw user command (natural language)
            filename     – target Excel file; falls back to memory / default
            auto_execute – if True, execute immediately (unless destructive)
        Returns:
            AgentResult with at least .action populated; .result populated if
            executed.
        """
        log.info('handle_text: text=%r  auto_execute=%s', text[:120], auto_execute)
        corrected = correct_transcript(text)
        if corrected != text:
            log.info('handle_text: transcript corrected %r → %r', text[:80], corrected[:80])
        text = corrected
        enriched = _enrich_text_with_context(text)
        try:
            action = route_llm(enriched)
        except Exception as exc:
            log.error('[Pipeline] handle_text: route_llm raised unexpectedly: %s', exc)
            failure = ErrorHandler.classify_exception(exc)
            return AgentResult(
                success=False,
                error=str(exc),
                speech_text=ErrorHandler.for_system(failure, detail=str(exc)),
            )

        res = AgentResult(action=action)
        is_unknown = _is_unknown(action)

        if is_unknown:
            res.suggestions = _safe_suggestions(text)

        if auto_execute and not is_unknown:
            exec_res = self.handle_action(action, filename=filename, auto_correct=auto_correct)
            # merge execution result into our response
            res.result = exec_res.result
            res.workflow_id = exec_res.workflow_id
            res.download_url = exec_res.download_url
            res.speech_text = exec_res.speech_text
            res.requires_confirmation = exec_res.requires_confirmation
            res.confirmation_id = exec_res.confirmation_id
            res.confirmation_message = exec_res.confirmation_message
            res.error = exec_res.error
            res.success = exec_res.success
            res.executed = exec_res.executed
            res.memory_snapshot = exec_res.memory_snapshot
            res.recovery = exec_res.recovery
            res.auto_corrected = exec_res.auto_corrected
        else:
            res.executed = False

        return res

    def handle_action(self, action: dict, filename: str | None = None,
                       confirmed: bool = False,
                       auto_correct: bool = False) -> AgentResult:
        """Execute a pre-parsed *action* dict through the full pipeline.

        Pipeline: route → preconditions → execute → memory update.

        Set *confirmed* to True when the caller has already obtained user
        confirmation for destructive actions.
        """
        # Guard: unknown / empty action type must never reach the executor.
        # Return immediately so the caller gets a clean success=False, executed=False
        # result and can speak the right "I didn't understand" message.
        if _is_unknown(action):
            atype_raw = action.get('type', '') if isinstance(action, dict) else ''
            log.info('handle_action: action type %r is unknown — skipping execution', atype_raw)
            transcript = action.get('raw', '')
            confidence = float(action.get('confidence', 0.0)) if isinstance(action, dict) else 0.0
            return AgentResult(
                success=False,
                action=action,
                executed=False,
                speech_text=ErrorHandler.for_unknown_command(transcript, confidence),
                suggestions=_safe_suggestions(transcript),
            )

        # Honour filename embedded in the action dict (e.g. from heuristic parser)
        # when no explicit filename was passed by the caller.
        # The planner wraps single-step actions in {'steps': [...]}, so we must
        # also look inside the first step to find the embedded filename.
        action_filename = None
        if isinstance(action, dict):
            action_filename = action.get('filename')
            if action_filename is None:
                steps = action.get('steps')
                if isinstance(steps, list) and steps and isinstance(steps[0], dict):
                    action_filename = steps[0].get('filename')
        # For open_file the action's embedded filename IS the target — it must
        # win over the caller's active-file hint so "open file server" opens
        # server.xlsx regardless of which file is currently active in memory.
        if isinstance(action, dict) and action.get('type') == 'open_file' and action_filename:
            resolved = self._resolve_filename(action_filename)
        else:
            resolved = self._resolve_filename(filename or action_filename)
        filepath = os.path.join(self.files_dir, resolved)

        # 1. Route
        wf = workflow_router.route(action)
        wf_id = wf["id"] if wf else None
        log.info('handle_action: workflow=%s  file=%s', wf_id, resolved)

        # 2. Precondition check
        if wf:
            snap = memory.snapshot()
            # If caller passed an explicit filename, treat it as the active file
            # for precondition purposes (e.g. confirm endpoint stores filepath).
            if filename and filename.strip():
                snap = dict(snap, active_file=resolved)
            # Collect every action type present in the plan (all steps)
            all_types: list[str] = []
            steps = action.get('steps') if isinstance(action, dict) else None
            if isinstance(steps, list):
                all_types = [s.get('type') for s in steps if isinstance(s, dict) and s.get('type')]
            elif isinstance(action, dict) and action.get('type'):
                all_types = [action['type']]
            errors = workflow_router.check_preconditions(
                wf, snap, filepath, all_action_types=all_types
            )
            if errors:
                recovery = self._build_recovery(action, errors, resolved)
                if auto_correct:
                    corrected = self._attempt_auto_correction(action, resolved, confirmed, recovery)
                    if corrected is not None:
                        return corrected
                return AgentResult(
                    success=False,
                    action=action,
                    workflow_id=wf_id,
                    error=errors[0],
                    speech_text=recovery["message"],
                    suggestions=recovery.get("suggestions", []),
                    recovery=recovery,
                )

        # 3. Planner validation (cell refs, sheet existence, etc.)
        plan = self.planner.create_plan(
            action, filename=resolved, active_sheet=memory.get_active_sheet(),
        )
        if not plan.valid:
            return AgentResult(
                success=False,
                action=action,
                workflow_id=wf_id,
                error=plan.error,
                speech_text=plan.error,
                suggestions=[],
            )
        if plan.warnings:
            log.info('handle_action: planner warnings: %s', plan.warnings)

        # 4. Destructive check (caller / app.py handles confirmation flow)
        if not confirmed and (workflow_router.is_destructive(action) or plan.requires_confirmation):
            return AgentResult(
                success=True,
                action=action,
                workflow_id=wf_id,
                requires_confirmation=True,
                confirmation_message=self._destructive_message(action, filepath),
            )

        # 5. Execute (prefer markdown workflow engine when appropriate)
        try:
            result = self._execute(action, filepath, workflow_id=wf_id, filename=resolved)
        except Exception as exc:
            log.error('[Pipeline] handle_action: _execute raised unexpectedly: %s', exc)
            failure = ErrorHandler.classify_exception(exc)
            return AgentResult(
                success=False,
                action=action,
                workflow_id=wf_id,
                error=str(exc),
                speech_text=ErrorHandler.for_system(failure, detail=str(exc)),
                executed=False,
            )

        # 6. Memory update (non-fatal — SQLite failure must not abort the pipeline)
        try:
            self._update_memory(action, resolved, result)
        except Exception as _mem_exc:
            log.warning('handle_action: memory update failed (non-fatal): %s', _mem_exc)

        return self._build_exec_result(action, wf_id, resolved, result)

    # ---- internals ----

    def _resolve_filename(self, explicit: str | None) -> str:
        """Return best filename: explicit > memory > default."""
        if explicit and explicit.strip():
            return explicit
        return memory.get_active_file() or 'data.xlsx'

    def _execute(self, action: dict, filepath: str,
                 workflow_id: str | None = None,
                 filename: str | None = None) -> dict | str:
        """Run action through excel_controller, handling multi-step plans.

        Multi-step plans are executed transactionally: a backup is taken before
        the first step, and if any step fails, the file is rolled back to the
        pre-execution state.
        """
        if self._should_use_workflow_engine(action, workflow_id):
            params = dict(action)
            params["action"] = action
            params["filepath"] = filepath
            if filename:
                params["filename"] = filename
            return workflow_engine.execute_workflow(workflow_id, params)

        has_steps = isinstance(action.get('steps'), list) and action['steps']
        if has_steps:
            # --- transactional multi-step execution ---
            txn_backup = None
            if os.path.exists(filepath):
                txn_backup = filepath + '.txn_backup'
                try:
                    shutil.copy2(filepath, txn_backup)
                except Exception as exc:
                    log.warning('_execute: failed to create txn backup: %s', exc)
                    txn_backup = None

            step_results = []
            for idx, step in enumerate(action['steps']):
                try:
                    res = self._dispatch_single(step, filepath)
                except Exception as exc:
                    res = {"error": str(exc)}

                step_results.append({"step": step, "result": res})

                if isinstance(res, dict) and 'error' in res:
                    # "no_undo_available" is an expected informational response,
                    # not a failure — skip rollback and surface the message.
                    if res.get('error') == 'no_undo_available':
                        speech = res.get('speech_text', 'Nothing to undo.')
                        if txn_backup and os.path.exists(txn_backup):
                            try:
                                os.remove(txn_backup)
                            except Exception:
                                pass
                        return {'message': 'undo_skipped', 'speech_text': speech}

                    log.warning('_execute: step %d/%d failed (%s), rolling back',
                                idx + 1, len(action['steps']), res.get('error'))
                    if txn_backup and os.path.exists(txn_backup):
                        try:
                            shutil.copy2(txn_backup, filepath)
                            log.info('_execute: rolled back to txn backup')
                        except Exception:
                            log.error('_execute: rollback copy failed')
                    return {
                        "error": f"Step {idx + 1} failed: {res.get('error', 'unknown')}",
                        "message": "transaction_rolled_back",
                        "completed_steps": idx,
                        "total_steps": len(action['steps']),
                        "steps": step_results,
                    }

            # All steps succeeded — remove transactional backup
            if txn_backup and os.path.exists(txn_backup):
                try:
                    os.remove(txn_backup)
                except Exception:
                    pass
            return {"message": "steps_executed", "steps": step_results}
        return self._dispatch_single(action, filepath)

    @staticmethod
    def _dispatch_single(action: dict, filepath: str) -> dict | str:
        """Dispatch a single action through the tool registry or legacy path.

        Registry path is always preferred.  If the action type is not registered,
        the legacy ``execute_action`` path is used as a fallback — but this is
        logged at WARNING level so any hallucinated or unregistered types are
        clearly visible in production logs.
        """
        atype = action.get('type') if isinstance(action, dict) else None
        if atype and tool_registry.is_registered(atype):
            kwargs = _action_to_kwargs(atype, action)
            return tool_registry.execute(atype, path=filepath, **kwargs)
        # Fallback: legacy excel_controller.execute_action for unregistered tools.
        # Log at WARNING so unregistered/hallucinated types are never silent.
        log.warning(
            '[Tool] _dispatch_single: action type %r is NOT in tool registry — '
            'falling through to legacy execute_action. '
            'If this type is unexpected, it may be a hallucinated command.',
            atype,
        )
        return execute_action(action, filepath)

    def _should_use_workflow_engine(self, action: dict, workflow_id: str | None) -> bool:
        """Decide if action should execute via markdown workflow engine."""
        if not workflow_id:
            return False
        wf = workflow_engine.get_workflow(workflow_id)
        if not wf:
            return False
        tool_calls = wf.get("tool_calls")
        if not tool_calls:
            return False
        if isinstance(action.get('steps'), list) and action.get('steps'):
            return False

        # For standard direct tool actions, keep the existing path to preserve behavior.
        direct_types = {
            'create_excel',
            'create_sheet_and_add_row',
            'add_row',
            'delete_row',
            'insert_row_at',
            'insert_column_at',
            'add_column',
            'save',
            'delete_column',
            'delete_cell',
            'delete_range',
            'write_cell',
            'insert_cell',
            'write_formula',
            'write_sum_formula',
            'write_average_formula',
            'calculate_row_sum',
            'calculate_average',
            'open_file',
            'open_excel_app',
            'close_excel_app',
            'format_cells',
            'create_chart',
            'create_sheet',
            'switch_sheet',
            'delete_sheet',
            'list_sheets',
            'find_replace',
            'sort_range',
            'set_column_width',
            'freeze_panes',
            'unfreeze_panes',
            'merge_cells',
            'unmerge_cells',
            'rename_sheet',
            'rename_file',
            'read_range',
            'filter_table',
            'clear_filter',
        }
        atype = action.get('type') if isinstance(action, dict) else None
        return atype not in direct_types

    def _update_memory(self, action: dict, filename: str, result) -> None:
        """Mirror the memory-update logic from app.py, including header context."""
        if isinstance(result, dict) and 'error' in result:
            return
        atype = action.get('type', '') if isinstance(action, dict) else ''
        # Also check the first step when the planner wraps in a steps[] plan
        if not atype and isinstance(action, dict):
            steps = action.get('steps')
            if isinstance(steps, list) and steps and isinstance(steps[0], dict):
                atype = steps[0].get('type', '')

        # App-launch tools don't operate on a file — don't pollute active_file.
        if atype == 'open_excel_app':
            memory.add_command(action)
            return

        memory.set_active_file(filename)

        # ---- Sheet tracking ----
        if atype in ('create_excel', 'create_sheet_and_add_row'):
            memory.set_active_sheet('Sheet1')
            # Store headers so subsequent commands have column context
            headers = action.get('headers')
            if isinstance(headers, list) and headers:
                memory.set_active_headers(headers)
        elif atype == 'open_file' and isinstance(result, dict):
            memory.set_active_sheet(result.get('active_sheet', 'Sheet1'))
            # open_excel now includes row-1 values; use them so follow-up commands
            # have accurate column context immediately after opening a file.
            raw_hdrs = result.get('headers')
            memory.set_active_headers(raw_hdrs if isinstance(raw_hdrs, list) else [])
        elif atype == 'create_sheet' and isinstance(result, dict) and 'error' not in result:
            memory.set_active_sheet(result.get('sheet', action.get('sheet_name')))
            memory.set_active_headers([])
        elif atype == 'switch_sheet' and isinstance(result, dict) and 'error' not in result:
            memory.set_active_sheet(result.get('sheet', action.get('sheet_name')))
            # switch_sheet now includes row-1 values of the target sheet; use them
            # so follow-up commands have accurate column context on the new sheet.
            raw_hdrs = result.get('headers')
            memory.set_active_headers(raw_hdrs if isinstance(raw_hdrs, list) else [])
        elif atype == 'delete_sheet' and isinstance(result, dict) and 'error' not in result:
            memory.set_active_sheet(result.get('active_sheet', 'Sheet1'))
            memory.set_active_headers([])
        elif atype == 'rename_sheet' and isinstance(result, dict) and 'error' not in result:
            memory.set_active_sheet(result.get('new_name', action.get('new_name')))
        elif atype == 'rename_file' and isinstance(result, dict) and 'error' not in result:
            # active_file must point at the new filename so subsequent commands
            # resolve to the renamed file rather than the (now-deleted) old path.
            new_name = result.get('new_name', action.get('new_name'))
            if new_name:
                memory.set_active_file(new_name)

        # ---- Column header tracking ----
        # add_column appends the new header; delete_column removes it by position.
        # Without this, the context prefix sent to the LLM stays stale and
        # follow-up commands (e.g. "calculate bonus average") fail to find the column.
        elif atype == 'add_column':
            col_name = action.get('name') or action.get('column_name')
            if col_name:
                existing = memory.get_active_headers() or []
                if col_name not in existing:
                    memory.set_active_headers(existing + [col_name])
        elif atype == 'delete_column':
            col_id = str(action.get('column', '')).upper()
            existing = memory.get_active_headers() or []
            if col_id.isalpha() and existing:
                try:
                    from openpyxl.utils import column_index_from_string
                    idx = column_index_from_string(col_id) - 1  # 0-based
                    if 0 <= idx < len(existing):
                        memory.set_active_headers(existing[:idx] + existing[idx + 1:])
                except Exception:
                    pass

        elif atype == 'insert_column_at' and isinstance(result, dict) and 'error' not in result:
            # A blank column was inserted at position X — shift cached headers right
            # so positions stay aligned with the real worksheet.
            col_id = str(action.get('column', '')).upper()
            existing = memory.get_active_headers() or []
            if col_id.isalpha() and existing:
                try:
                    from openpyxl.utils import column_index_from_string
                    idx = column_index_from_string(col_id) - 1  # 0-based insertion point
                    if 0 <= idx <= len(existing):
                        memory.set_active_headers(existing[:idx] + [''] + existing[idx:])
                except Exception:
                    pass

        elif atype in ('undo', 'undo_last') and isinstance(result, dict) and 'error' not in result:
            # After undo, the workbook is restored to a snapshot.  Reload headers
            # from the live session so memory stays accurate (e.g., a previously
            # added column is gone again after undoing add_column).
            try:
                from excel_controller import get_sheet_headers
                filepath = os.path.join(self.files_dir, filename)
                reloaded = get_sheet_headers(filepath)
                memory.set_active_headers(reloaded)
            except Exception:
                memory.set_active_headers([])

        # ---- Multi-step: propagate headers from create_excel sub-step ----
        if isinstance(action.get('steps'), list):
            for step in action['steps']:
                if isinstance(step, dict) and step.get('type') in ('create_excel', 'create_sheet_and_add_row'):
                    step_headers = step.get('headers')
                    if isinstance(step_headers, list) and step_headers:
                        memory.set_active_headers(step_headers)
                    break

        memory.add_command(action)

    def _build_exec_result(self, action, wf_id, filename, result) -> AgentResult:
        action_type = action.get('type', '') if isinstance(action, dict) else ''
        context = {"filename": filename, "action_type": action_type}

        # Determine whether this is a tool-level error (not a pipeline error).
        # A tool error is indicated by an "error" key in the result dict that
        # is NOT accompanied by "message" == "steps_executed" / "transaction_rolled_back",
        # which are special multi-step sentinel values.
        tool_error: str | None = None
        if isinstance(result, dict) and 'error' in result:
            msg = result.get('message', '')
            if msg not in ('steps_executed', 'transaction_rolled_back'):
                tool_error = result['error']

        if isinstance(result, dict):
            # Multi-step plan fully executed: summarise each step's speech
            if result.get('message') == 'steps_executed':
                speech = _summarize_steps(result.get('steps', []))
            # Transactional rollback: report how many steps completed
            elif result.get('message') == 'transaction_rolled_back':
                done = result.get('completed_steps', 0)
                total = result.get('total_steps', 0)
                speech = (
                    f"I completed {done} of {total} steps but hit an error "
                    "and rolled back all changes to keep your file safe."
                )
            else:
                # Single step — ErrorHandler resolves speech_text or error code
                speech = ErrorHandler.safe_speech(result, action_type, context)
        else:
            speech = "Done."

        # A tool error means we executed the call but the operation failed.
        # Mark success=False so callers and tests can distinguish execution
        # errors from pre-execution failures (precondition / confirmation).
        is_success = tool_error is None
        log.debug(
            '_build_exec_result: type=%s  tool_error=%s  success=%s',
            action_type, tool_error, is_success,
        )

        return AgentResult(
            success=is_success,
            action=action,
            workflow_id=wf_id,
            result=result,
            download_url=f"/files/{filename}" if is_success else None,
            speech_text=speech,
            executed=True,          # the call was made (even if it returned an error)
            error=tool_error,       # surface the error code for callers
            memory_snapshot=memory.snapshot(),
        )

    def _destructive_message(self, action: dict, filepath: str) -> str:
        """Build a human-readable confirmation prompt."""
        atype = action.get('type', '')
        if atype == 'delete_column':
            col = str(action.get('column', '')).upper()
            return f"Are you sure you want to delete column {col}?"
        if atype == 'delete_row':
            row = action.get('row', '?')
            return f"Are you sure you want to delete row {row}?"
        return "Are you sure you want to perform this destructive action?"

    def _build_recovery(self, action: dict, errors: list[str], filename: str) -> dict:
        """Build user-facing recovery guidance for precondition failures."""
        message = errors[0] if errors else "Precondition check failed."
        suggestions: list[str] = []
        auto_correct_action: dict | None = None

        missing_file = any('No active file' in e for e in errors)
        missing_on_disk = any('not found' in e for e in errors)
        if missing_file:
            message = "No Excel file is open. Would you like me to create a new file?"
            suggestions = [
                "Say: create a new Excel file",
                f"Provide a filename (default: {filename})",
            ]
            auto_correct_action = {"type": "create_excel"}
        elif missing_on_disk:
            message = "The active Excel file was not found. Would you like me to create a new file?"
            suggestions = [
                f"Create a new workbook named {filename}",
                "Or open an existing file first",
            ]
            auto_correct_action = {"type": "create_excel"}

        return {
            "message": message,
            "errors": errors,
            "suggestions": suggestions,
            "can_auto_correct": auto_correct_action is not None,
            "auto_correct_action": auto_correct_action,
        }

    def _attempt_auto_correction(
        self,
        action: dict,
        filename: str,
        confirmed: bool,
        recovery: dict,
    ) -> AgentResult | None:
        """Apply a safe correction and retry once, or return None if not possible."""
        auto_action = recovery.get("auto_correct_action") if isinstance(recovery, dict) else None
        if not isinstance(auto_action, dict) or not auto_action.get("type"):
            return None

        filepath = os.path.join(self.files_dir, filename)
        fix_result = execute_action(auto_action, filepath)
        if isinstance(fix_result, dict) and fix_result.get("error"):
            return AgentResult(
                success=False,
                action=action,
                error=fix_result.get("error"),
                speech_text=recovery.get("message"),
                suggestions=recovery.get("suggestions", []),
                recovery=recovery,
            )

        try:
            self._update_memory(auto_action, filename, fix_result)
        except Exception as _mem_exc:
            log.warning('handle_action: memory update failed (non-fatal): %s', _mem_exc)
        retried = self.handle_action(
            action,
            filename=filename,
            confirmed=confirmed,
            auto_correct=False,
        )
        retried.auto_corrected = True
        if not retried.speech_text:
            retried.speech_text = "I created a new Excel file and retried your command."
        if retried.recovery is None:
            retried.recovery = recovery
        return retried


# --------------- module-level helpers ---------------

def _summarize_steps(steps: list) -> str:
    """Build a human-friendly TTS summary from a completed multi-step result."""
    if not steps:
        return "Done."
    parts = []
    for s in steps:
        r = s.get('result') if isinstance(s, dict) else None
        if isinstance(r, dict) and r.get('speech_text'):
            parts.append(r['speech_text'].rstrip('.'))
    if parts:
        return '. '.join(parts) + '.'
    n = len(steps)
    return f"Completed {n} step{'s' if n != 1 else ''}."


def _is_unknown(action) -> bool:
    if not action or not isinstance(action, dict):
        return True
    has_steps = isinstance(action.get('steps'), list) and action['steps']
    if has_steps:
        return False
    return action.get('type') in ('unknown', None, '')


def _safe_suggestions(text: str) -> list[str]:
    try:
        return generate_suggestions(text) or []
    except Exception:
        return []


# Maps action type -> list of (action_key, kwarg_name) pairs for the registry adapters.
# Keys listed first take precedence (e.g. 'column' beats 'col').
_ACTION_KWARG_MAP: dict[str, list[tuple[str, str]]] = {
    'create_excel':             [('headers', 'headers')],
    'create_sheet_and_add_row': [('headers', 'headers'), ('values', 'values'), ('sheet_name', 'sheet_name')],
    'add_row':                  [('values', 'values')],
    'delete_row':               [('index', 'index'), ('row', 'index')],
    'insert_row_at':            [('index', 'index'), ('row', 'index')],
    'insert_column_at':         [('column', 'column'), ('col', 'column')],
    'delete_column':            [('column', 'column'), ('col', 'column')],
    'delete_cell':              [('cell', 'cell')],
    'delete_range':             [('range', 'range'), ('cell_range', 'range')],
    'add_column':               [('name', 'name'), ('column_name', 'name'), ('default', 'default'), ('default_value', 'default')],
    'write_cell':               [('cell', 'cell'), ('range', 'cell'), ('value', 'value'), ('text', 'value')],
    'insert_cell':              [('cell', 'cell'), ('range', 'cell'), ('value', 'value'), ('text', 'value')],
    'write_formula':            [('cell', 'cell'), ('range', 'cell'), ('formula', 'formula'), ('value', 'formula')],
    'write_sum_formula':        [('range_ref', 'range_ref'), ('range', 'range_ref'), ('cell', 'cell')],
    'write_average_formula':    [('range_ref', 'range_ref'), ('range', 'range_ref'), ('cell', 'cell')],
    'calculate_row_sum':        [('index', 'index'), ('row', 'index')],
    'calculate_average':        [('column', 'column'), ('col', 'column')],
    'open_file':                [],
    'open_excel_app':           [],
    'close_excel_app':          [],
    'format_cells':             [('range', 'cell_range'), ('cell_range', 'cell_range'),
                                 ('bold', 'bold'), ('italic', 'italic'),
                                 ('font_size', 'font_size'), ('font_color', 'font_color'),
                                 ('fill_color', 'fill_color'),
                                 ('num_format', 'num_format'), ('number_format', 'num_format'),
                                 ('align', 'align'), ('font_size_change', 'font_size_change')],
    'create_chart':             [('chart_type', 'chart_type'), ('data_range', 'data_range'),
                                 ('title', 'title'), ('x_axis', 'x_axis'), ('y_axis', 'y_axis'),
                                 ('position', 'position')],
    'save':                     [],
    'create_sheet':             [('sheet_name', 'sheet_name')],
    'switch_sheet':             [('sheet_name', 'sheet_name')],
    'delete_sheet':             [('sheet_name', 'sheet_name')],
    'list_sheets':              [],
    'find_replace':             [('find_value', 'find_value'), ('find', 'find_value'),
                                 ('replace_value', 'replace_value'), ('replace', 'replace_value')],
    'sort_range':               [('column', 'column'), ('col', 'column'),
                                 ('ascending', 'ascending')],
    'set_column_width':         [('column', 'column'), ('col', 'column'), ('width', 'width'), ('auto', 'auto')],
    'freeze_panes':             [('cell', 'cell')],
    'unfreeze_panes':           [],
    'merge_cells':              [('range', 'cell_range'), ('cell_range', 'cell_range')],
    'unmerge_cells':            [('range', 'cell_range'), ('cell_range', 'cell_range')],
    'rename_sheet':             [('old_name', 'old_name'), ('new_name', 'new_name')],
    'rename_file':              [('new_name', 'new_name')],
    'read_range':               [('range', 'cell_range'), ('cell_range', 'cell_range')],
    'read_cell':                [('cell', 'cell')],
    'undo':                     [],
    'undo_last':                [],
    'filter_table':             [('column', 'column'), ('operator', 'operator'), ('value', 'value')],
    'clear_filter':             [],
    # NOTE: write_formula appears only once here.  The duplicate entry that
    # previously existed at line 565 was removed — Python dict literals silently
    # drop the first of two identical keys, so only the last one survived anyway.
}


def _action_to_kwargs(atype: str, action: dict) -> dict:
    """Extract keyword arguments from an *action* dict for the tool registry.

    Uses ``_ACTION_KWARG_MAP`` to translate action dict keys to the kwarg
    names expected by the registry adapters, handling aliases (e.g. ``col``
    → ``column``).
    """
    mapping = _ACTION_KWARG_MAP.get(atype)
    if mapping is None:
        # Unknown type — pass everything except 'type'
        return {k: v for k, v in action.items() if k != 'type'}
    kwargs: dict = {}
    for action_key, kwarg_name in mapping:
        if kwarg_name not in kwargs and action_key in action:
            kwargs[kwarg_name] = action[action_key]
    return kwargs


# --------------- REPL / quick-test helper ---------------

_default_agent: Agent | None = None


def process_command(text: str, filename: str | None = None) -> str:
    """Parse and execute a natural-language command; return the speech response.

    Convenience wrapper around ``Agent.handle_text(auto_execute=True)`` for
    quick testing from the Python REPL or scripts::

        >>> from agent import process_command
        >>> process_command("open excel")
        'Opening sales.xlsx in Excel.'
        >>> process_command("add row John 90")
        'Row added successfully.'
        >>> process_command("write cell A1 100")
        'Cell A1 updated.'

    Parameters:
        text:     Natural-language command (same as a voice transcript).
        filename: Optional Excel file name/path to target.  When omitted the
                  agent uses the active file from memory or creates a default
                  one automatically.

    Returns:
        The plain-text speech response string.
    """
    global _default_agent
    if _default_agent is None:
        import os
        files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
        os.makedirs(files_dir, exist_ok=True)
        _default_agent = Agent(files_dir)

    result = _default_agent.handle_text(
        text,
        filename=filename,
        auto_execute=True,
        auto_correct=True,
    )

    # Build a human-readable response identical to what assistant_runtime would speak
    if result.speech_text:
        return result.speech_text
    if not result.success:
        return f"Error: {result.error or 'command failed'}"
    if isinstance(result.result, dict):
        steps = result.result.get('steps')
        if isinstance(steps, list) and steps:
            return _summarize_steps(steps)
        msg = result.result.get('speech_text') or result.result.get('message')
        if msg:
            return str(msg)
    if isinstance(result.result, str) and result.result.strip():
        return result.result
    return 'Done.' if result.executed else 'Parsed but not executed.'
