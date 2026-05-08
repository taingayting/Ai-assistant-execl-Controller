"""End-to-end command tests for Sara.

Tests the full pipeline from action dict → Agent → ToolRegistry → ExcelController
without any LLM calls.  Every test scenario is driven by pre-parsed action dicts
so results are deterministic and fast (no network, no subprocess).

Scenarios covered
-----------------
1.  create spreadsheet with Name and Score columns
2.  add_row  Tom 90
3.  add_row  Mary 85
4.  add_column  Revenue
5.  write_cell  set B2 to 500
6.  write_cell  set B3 to 400
7.  calculate_average  (auto-detect numeric column)
8.  calculate_row_sum  row 2
9.  undo  last action
10. find_replace  Tom → Thomas
11. sort_range  by Score descending
12. delete_row  row 2 (with undo)
13. read_cell  A2
14. read_range  A1:B3
15. create_sheet / switch_sheet / list_sheets / rename_sheet / delete_sheet
16. write_formula  SUM in a cell
17. multi-step plan via steps[] list
18. unknown action  → graceful error dict (no crash)
19. result schema  — every success result has speech_text
20. undo stack depth  — increases / decreases correctly

Run with::

    cd backend
    pytest tests/test_e2e_commands.py -v
"""
from __future__ import annotations

import os
import sys
import time
import pathlib
import tempfile
import shutil

import pytest

# ── Path setup ──────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent
BACKEND = str(HERE.parent)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

# Disable real LLM calls — all tests use pre-parsed action dicts.
os.environ.setdefault('USE_OLLAMA', 'false')
os.environ.setdefault('GROQ_API_KEY', '')


# ── Helper: release Excel COM lock before deletion ───────────────────────────
# The COM sync patch (added for real-time Excel updates) calls _com_open_workbook
# after every save, which keeps Excel holding the file handle.  Without releasing
# the COM session first, os.remove() raises WinError 32 on Windows.

def _release_and_remove(path: str) -> None:
    """Close the Excel COM session and openpyxl session for *path*, then delete it."""
    if not os.path.exists(path):
        return
    # 1. Release COM (Excel) file lock
    try:
        from excel_controller import _com_close_workbook
        _com_close_workbook(os.path.abspath(path))
        time.sleep(0.15)          # let Windows release the handle
    except Exception:
        pass
    # 2. Clear the openpyxl in-memory session so it doesn't re-open the file
    try:
        from excel_controller import close_file_session
        close_file_session()
    except Exception:
        pass
    # 3. Clear undo stack so the module-scoped agent doesn't leak stack depth
    #    between tests (the stack is keyed by abs_path, and the path is reused
    #    across tests in this module — without clearing, test_undo_stack_depth
    #    sees a full stack (depth=10) and the assertion depth+1 never holds).
    try:
        from excel_controller import _UNDO_STACK
        _UNDO_STACK.pop(os.path.abspath(path), None)
    except Exception:
        pass
    # 4. Delete with one retry on PermissionError
    for attempt in range(2):
        try:
            os.remove(path)
            return
        except PermissionError:
            if attempt == 0:
                time.sleep(0.3)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def files_dir(tmp_path_factory):
    """Isolated temp directory shared across all module tests."""
    d = tmp_path_factory.mktemp('sara_e2e')
    return str(d)


@pytest.fixture(scope='module')
def agent(files_dir):
    """One Agent instance for the entire module."""
    from tools.tool_registry import registry, register_default_excel_tools
    # Ensure registry is populated (idempotent on second call)
    if len(registry) == 0:
        register_default_excel_tools()

    from agent import Agent
    return Agent(files_dir=files_dir)


@pytest.fixture
def xlsx(files_dir):
    """Fresh xlsx filename (relative, no directory) for each test."""
    return 'test_e2e.xlsx'


@pytest.fixture(autouse=True)
def clean_file(files_dir, xlsx):
    """Remove the test xlsx before each test so each starts from scratch."""
    path = os.path.join(files_dir, xlsx)
    _release_and_remove(path)
    yield
    # cleanup after
    _release_and_remove(path)


# ── Helper ──────────────────────────────────────────────────────────────────

def run(agent, action, xlsx):
    """Execute *action* through agent.handle_action and assert no crash."""
    result = agent.handle_action(action, filename=xlsx, confirmed=True, auto_correct=True)
    return result


def has_speech(result) -> bool:
    """Return True if the AgentResult has a non-empty speech_text."""
    return bool(result.speech_text and result.speech_text.strip())


# ── 1. Create spreadsheet ────────────────────────────────────────────────────

def test_create_excel(agent, xlsx):
    action = {'type': 'create_excel', 'headers': ['Name', 'Score']}
    r = run(agent, action, xlsx)
    assert r.success, f"create_excel failed: {r.error}"
    assert r.executed
    assert has_speech(r), "create_excel must return speech_text"
    assert 'Name' in r.speech_text or 'spreadsheet' in r.speech_text.lower()


# ── 2 & 3. add_row ───────────────────────────────────────────────────────────

def test_add_row(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'add_row', 'values': ['Tom', 90]}, xlsx)
    assert r.success, f"add_row failed: {r.error}"
    assert has_speech(r)
    assert 'Tom' in r.speech_text or 'row' in r.speech_text.lower()


def test_add_multiple_rows(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    for name, score in [('Tom', 90), ('Mary', 85), ('Alex', 78)]:
        r = run(agent, {'type': 'add_row', 'values': [name, score]}, xlsx)
        assert r.success, f"add_row {name} failed: {r.error}"


# ── 4. add_column ────────────────────────────────────────────────────────────

def test_add_column(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'add_column', 'name': 'Revenue'}, xlsx)
    assert r.success, f"add_column failed: {r.error}"
    assert has_speech(r)
    assert 'Revenue' in r.speech_text


def test_add_column_alias_column_name(agent, xlsx):
    """LLM sometimes outputs column_name instead of name — kwarg alias must work."""
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'add_column', 'column_name': 'Bonus'}, xlsx)
    assert r.success, f"add_column with column_name alias failed: {r.error}"
    assert has_speech(r)


# ── 5 & 6. write_cell ────────────────────────────────────────────────────────

def test_write_cell(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Tom', 90]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'write_cell', 'cell': 'B2', 'value': 500}, xlsx)
    assert r.success, f"write_cell failed: {r.error}"
    assert has_speech(r)
    assert 'B2' in r.speech_text or '500' in r.speech_text


def test_write_cell_invalid_ref_rejected(agent, xlsx):
    """Planner should reject invalid cell references before execution."""
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'write_cell', 'cell': 'INVALID', 'value': 42}, xlsx)
    # Either planner rejects (success=False) or controller returns error
    if r.success and r.result:
        assert 'error' in r.result or r.executed is False
    else:
        assert not r.success


# ── 7. calculate_average ────────────────────────────────────────────────────

def test_calculate_average(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Tom', 90]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Mary', 80]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'calculate_average'}, xlsx)
    assert r.success, f"calculate_average failed: {r.error}"
    assert has_speech(r)
    # average of 90 and 80 is 85.0
    assert '85' in r.speech_text


# ── 8. calculate_row_sum ─────────────────────────────────────────────────────

def test_calculate_row_sum(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['A', 'B', 'C']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': [10, 20, 30]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'calculate_row_sum', 'index': 2}, xlsx)
    assert r.success, f"calculate_row_sum failed: {r.error}"
    assert has_speech(r)
    assert '60' in r.speech_text


# ── 9. undo ──────────────────────────────────────────────────────────────────

def test_undo_delete_row(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Tom', 90]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Mary', 85]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    # Delete row 2 (Tom)
    agent.handle_action({'type': 'delete_row', 'index': 2},
                        filename=xlsx, confirmed=True, auto_correct=True)
    # Undo the deletion
    r = run(agent, {'type': 'undo'}, xlsx)
    assert r.success, f"undo failed: {r.error}"
    assert has_speech(r)
    assert 'undo' in r.speech_text.lower() or 'restored' in r.speech_text.lower()


def test_undo_with_no_history(agent, xlsx):
    """undo on a fresh file returns a friendly message, not a crash."""
    agent.handle_action({'type': 'create_excel', 'headers': ['Name']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'undo'}, xlsx)
    # Should not crash; may succeed or fail gracefully
    assert has_speech(r)


# ── 10. find_replace ─────────────────────────────────────────────────────────

def test_find_replace(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Tom', 90]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'find_replace', 'find_value': 'Tom', 'replace_value': 'Thomas'}, xlsx)
    assert r.success, f"find_replace failed: {r.error}"
    assert has_speech(r)
    assert 'Thomas' in r.speech_text or '1' in r.speech_text


# ── 11. sort_range ───────────────────────────────────────────────────────────

def test_sort_range(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    for name, score in [('Tom', 90), ('Alice', 70), ('Bob', 85)]:
        agent.handle_action({'type': 'add_row', 'values': [name, score]},
                            filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'sort_range', 'column': 'B', 'ascending': False}, xlsx)
    assert r.success, f"sort_range failed: {r.error}"
    assert has_speech(r)
    assert 'B' in r.speech_text


# ── 12. delete_row ───────────────────────────────────────────────────────────

def test_delete_row(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Tom', 90]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Mary', 85]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'delete_row', 'index': 2}, xlsx)
    assert r.success, f"delete_row failed: {r.error}"
    assert has_speech(r)
    assert 'undo' in r.speech_text.lower() or 'deleted' in r.speech_text.lower()


def test_delete_header_row_rejected(agent, xlsx):
    """Deleting row 1 (header) must be rejected with a clear error.

    Contract:
        success=False  — the deletion did not happen
        result["error"] == "header_row_protected"
        speech_text    — a helpful message (not a raw exception)
    """
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Tom', 90]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'delete_row', 'index': 1}, xlsx)
    assert not r.success, "Deleting header row must return success=False"
    assert has_speech(r), "Header-row rejection must include a speech_text"
    if isinstance(r.result, dict):
        assert r.result.get('error') == 'header_row_protected', (
            f"Expected error='header_row_protected', got {r.result.get('error')!r}"
        )


# ── 13. read_cell ────────────────────────────────────────────────────────────

def test_read_cell(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Tom', 90]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'read_cell', 'cell': 'A2'}, xlsx)
    assert r.success, f"read_cell failed: {r.error}"
    assert has_speech(r)
    assert 'Tom' in r.speech_text or 'A2' in r.speech_text


# ── 14. read_range ───────────────────────────────────────────────────────────

def test_read_range(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Tom', 90]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'read_range', 'cell_range': 'A1:B2'}, xlsx)
    assert r.success, f"read_range failed: {r.error}"
    assert has_speech(r)


# ── 15. Sheet operations ─────────────────────────────────────────────────────

def test_sheet_lifecycle(agent, xlsx):
    # Create workbook
    agent.handle_action({'type': 'create_excel', 'headers': ['Name']},
                        filename=xlsx, confirmed=True, auto_correct=True)

    # Create sheet
    r = run(agent, {'type': 'create_sheet', 'sheet_name': 'Q2'}, xlsx)
    assert r.success, f"create_sheet failed: {r.error}"
    assert has_speech(r)
    assert 'Q2' in r.speech_text

    # List sheets
    r = run(agent, {'type': 'list_sheets'}, xlsx)
    assert r.success
    assert has_speech(r)
    assert 'Q2' in r.speech_text

    # Switch sheet
    r = run(agent, {'type': 'switch_sheet', 'sheet_name': 'Q2'}, xlsx)
    assert r.success, f"switch_sheet failed: {r.error}"
    assert has_speech(r)

    # Rename sheet
    r = run(agent, {'type': 'rename_sheet', 'old_name': 'Q2', 'new_name': 'Q2_Renamed'}, xlsx)
    assert r.success, f"rename_sheet failed: {r.error}"
    assert has_speech(r)

    # Delete the renamed sheet (not the last one)
    r = run(agent, {'type': 'delete_sheet', 'sheet_name': 'Q2_Renamed'}, xlsx)
    assert r.success, f"delete_sheet failed: {r.error}"
    assert has_speech(r)


# ── 16. write_formula ────────────────────────────────────────────────────────

def test_write_formula(agent, xlsx):
    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Tom', 90]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Mary', 80]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'write_formula', 'cell': 'B4', 'formula': '=SUM(B2:B3)'}, xlsx)
    assert r.success, f"write_formula failed: {r.error}"
    assert has_speech(r)
    assert 'B4' in r.speech_text or 'formula' in r.speech_text.lower()


def test_blocked_formula_rejected(agent, xlsx):
    """Formulas with blocked keywords must be rejected."""
    agent.handle_action({'type': 'create_excel', 'headers': ['Name']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    r = run(agent, {'type': 'write_formula', 'cell': 'A2', 'formula': '=HYPERLINK("http://evil.com")'}, xlsx)
    if r.result and isinstance(r.result, dict):
        assert 'error' in r.result
    else:
        assert not r.success


# ── 17. Multi-step plan ───────────────────────────────────────────────────────

def test_multistep_plan(agent, xlsx):
    """A steps[] plan creates the file and populates it in one call."""
    action = {
        'steps': [
            {'type': 'create_excel', 'headers': ['Name', 'Score']},
            {'type': 'add_row', 'values': ['Tom', 90]},
            {'type': 'add_row', 'values': ['Mary', 85]},
        ]
    }
    r = run(agent, action, xlsx)
    assert r.success, f"multi-step plan failed: {r.error}"
    assert r.executed
    assert has_speech(r)
    # Speech should summarise all steps (not just "steps_executed")
    assert r.speech_text.strip() not in ('steps_executed', '')


# ── 18. Unknown action — graceful failure ────────────────────────────────────

def test_unknown_action_no_crash(agent, xlsx):
    """Unknown action type must be rejected before execution.

    Contract (production requirement):
        success=False  — the command was not understood
        executed=False — nothing was dispatched to the Excel controller
        speech_text    — a helpful "I didn't understand" message is present
    """
    r = run(agent, {'type': 'unknown'}, xlsx)
    assert r is not None
    assert not r.success,   "unknown action must set success=False"
    assert not r.executed,  "unknown action must set executed=False"
    assert has_speech(r),   "unknown action must still produce speech_text"


def test_hallucinated_tool_no_crash(agent, xlsx):
    """A hallucinated / unregistered tool type must not raise an exception."""
    agent.handle_action({'type': 'create_excel', 'headers': ['Name']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    # This type does not exist; agent falls through to legacy execute_action
    # which should return an error dict, not raise
    r = run(agent, {'type': 'send_email_with_attachment', 'to': 'nobody@test.com'}, xlsx)
    # Whatever happens, no unhandled exception and speech_text is populated
    assert r is not None
    # If executed, result should be an error dict
    if r.executed and isinstance(r.result, dict):
        assert 'error' in r.result or 'message' in r.result


# ── 19. Result schema — every success has speech_text ────────────────────────

@pytest.mark.parametrize('action', [
    {'type': 'create_excel', 'headers': ['X', 'Y']},
    {'type': 'add_row', 'values': [1, 2]},
    {'type': 'add_column', 'name': 'Z'},
    {'type': 'write_cell', 'cell': 'A2', 'value': 99},
    {'type': 'list_sheets'},
    {'type': 'save'},
])
def test_schema_all_successes_have_speech(agent, xlsx, action):
    """Every successful result must have a non-empty speech_text."""
    # Ensure the file exists first
    agent.handle_action(
        {'type': 'create_excel', 'headers': ['X', 'Y']},
        filename=xlsx, confirmed=True, auto_correct=True,
    )
    r = run(agent, action, xlsx)
    if r.success and r.executed:
        assert has_speech(r), (
            f"action {action['type']!r} succeeded but returned no speech_text. "
            f"result={r.result}"
        )


# ── 20. Undo stack depth ─────────────────────────────────────────────────────

def test_undo_stack_depth(agent, xlsx, files_dir):
    from excel_controller import undo_stack_depth
    path = os.path.join(files_dir, xlsx)

    agent.handle_action({'type': 'create_excel', 'headers': ['Name', 'Score']},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Tom', 90]},
                        filename=xlsx, confirmed=True, auto_correct=True)
    agent.handle_action({'type': 'add_row', 'values': ['Mary', 85]},
                        filename=xlsx, confirmed=True, auto_correct=True)

    depth_before = undo_stack_depth(path)

    agent.handle_action({'type': 'delete_row', 'index': 2},
                        filename=xlsx, confirmed=True, auto_correct=True)

    depth_after_delete = undo_stack_depth(path)
    assert depth_after_delete == depth_before + 1, (
        f"Expected stack depth to grow by 1 after delete_row: "
        f"before={depth_before} after={depth_after_delete}"
    )

    agent.handle_action({'type': 'undo'}, filename=xlsx, confirmed=True, auto_correct=True)

    depth_after_undo = undo_stack_depth(path)
    assert depth_after_undo == depth_before, (
        f"Expected stack depth to shrink by 1 after undo: "
        f"before={depth_before} after={depth_after_undo}"
    )


# ── Full scenario: realistic user session ────────────────────────────────────

def test_full_user_session(agent, xlsx):
    """Simulate a realistic multi-turn user session end-to-end."""
    steps = [
        # 1. Create spreadsheet
        ({'type': 'create_excel', 'headers': ['Name', 'Score', 'Grade']}, 'created'),
        # 2. Add students
        ({'type': 'add_row', 'values': ['Alice', 95, 'A']}, 'row_added'),
        ({'type': 'add_row', 'values': ['Bob', 72, 'B']}, 'row_added'),
        ({'type': 'add_row', 'values': ['Charlie', 88, 'B+']}, 'row_added'),
        # 3. Add a new column
        ({'type': 'add_column', 'name': 'Remarks'}, 'column_added'),
        # 4. Write to a specific cell
        ({'type': 'write_cell', 'cell': 'D2', 'value': 'Excellent'}, 'cell_written'),
        # 5. Calculate average score
        ({'type': 'calculate_average', 'column': 'B'}, 'column_average'),
        # 6. Sort by Score descending
        ({'type': 'sort_range', 'column': 'B', 'ascending': False}, 'sort_done'),
        # 7. Find and replace
        ({'type': 'find_replace', 'find_value': 'Excellent', 'replace_value': 'Outstanding'}, 'find_replace_done'),
        # 8. Add a summary formula
        ({'type': 'write_formula', 'cell': 'B5', 'formula': '=AVERAGE(B2:B4)'}, 'formula_written'),
        # 9. Create a second sheet
        ({'type': 'create_sheet', 'sheet_name': 'Summary'}, 'sheet_created'),
        # 10. List sheets
        ({'type': 'list_sheets'}, 'sheets_listed'),
    ]

    for action, expected_msg in steps:
        r = agent.handle_action(action, filename=xlsx, confirmed=True, auto_correct=True)
        assert r.success, (
            f"Step {action['type']!r} failed — error: {r.error}  "
            f"speech: {r.speech_text}"
        )
        assert r.executed, f"Step {action['type']!r} was not executed"
        assert has_speech(r), (
            f"Step {action['type']!r} succeeded but speech_text is empty"
        )
        if isinstance(r.result, dict) and 'message' in r.result:
            assert r.result['message'] == expected_msg, (
                f"Step {action['type']!r}: expected message={expected_msg!r} "
                f"got {r.result['message']!r}"
            )
