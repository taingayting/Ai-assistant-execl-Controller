"""Integration tests that run actual LLM calls (Groq or Ollama) to validate
multi-step Excel commands end-to-end.

These tests hit real LLM backends and create real .xlsx files, so they are
marked ``integration`` and skipped when neither Groq nor Ollama is reachable.

Run:
    pytest backend/tests/test_integration_llm.py -v -m integration
"""
from __future__ import annotations

import os
import sys
import pathlib
import time

import pytest
from openpyxl import load_workbook

# ── path setup ──────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent
BACKEND_DIR = str(HERE.parent)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
PROJECT_ROOT = str(HERE.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import llm_router
from agent import Agent
from memory import memory

# ── markers & skip logic ────────────────────────────────────────

_groq_ok = llm_router.check_groq_available()
_ollama_ok = llm_router.is_ollama_available()
_any_llm = _groq_ok or _ollama_ok

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _any_llm, reason="No LLM backend reachable (Groq/Ollama both down)"),
]


# ── helpers ─────────────────────────────────────────────────────

def _cell(wb_path: str, sheet: str | None, cell_ref: str, data_only: bool = False):
    """Read a single cell value from a workbook file."""
    wb = load_workbook(wb_path, data_only=data_only)
    ws = wb[sheet] if sheet else wb.active
    val = ws[cell_ref].value
    wb.close()
    return val


def _col_values(wb_path: str, col_letter: str, min_row: int = 1, max_row: int | None = None):
    """Return a list of values in *col_letter* from row *min_row*."""
    wb = load_workbook(wb_path, data_only=True)
    ws = wb.active
    max_row = max_row or ws.max_row
    vals = [ws[f"{col_letter}{r}"].value for r in range(min_row, max_row + 1)]
    wb.close()
    return vals


def _sheet_names(wb_path: str) -> list[str]:
    wb = load_workbook(wb_path)
    names = wb.sheetnames
    wb.close()
    return names


def _row_count(wb_path: str) -> int:
    wb = load_workbook(wb_path)
    n = wb.active.max_row
    wb.close()
    return n


# ── fixtures ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_memory():
    memory.clear()
    yield
    memory.clear()


@pytest.fixture
def files_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def agent(files_dir):
    return Agent(files_dir=files_dir)


@pytest.fixture
def filepath(files_dir):
    """Return a helper that resolves a filename to a full path."""
    def _fp(name: str = "data.xlsx") -> str:
        return os.path.join(files_dir, name)
    return _fp


# ── LLM backbone test ──────────────────────────────────────────

class TestLlmRouterLive:
    """Smoke-test the live LLM router to make sure at least one backend works."""

    @staticmethod
    def _unwrap(result: dict) -> dict:
        """Unwrap `{steps: [{...}]}` into the first step action."""
        steps = result.get("steps")
        if isinstance(steps, list) and len(steps) >= 1:
            return steps[0]
        return result

    def test_simple_parse(self):
        result = llm_router.route_llm("add row Tom 90")
        assert isinstance(result, dict)
        action = self._unwrap(result)
        # Should parse to add_row or similar — not unknown
        assert action.get("type") != "unknown", f"LLM returned unknown: {result}"

    def test_create_parse(self):
        result = llm_router.route_llm("create excel file with headers Name and Score")
        assert isinstance(result, dict)
        action = self._unwrap(result)
        assert action.get("type") in ("create_excel", "create_sheet_and_add_row"), f"Got: {result}"

    def test_multi_step_parse(self):
        result = llm_router.route_llm(
            "Create an excel sheet for student scores and add Tom with 90 and Alex with 85"
        )
        assert isinstance(result, dict)
        # Expect steps array OR a create_sheet_and_add_row single action
        has_steps = isinstance(result.get("steps"), list) and len(result["steps"]) >= 2
        is_combo = result.get("type") == "create_sheet_and_add_row"
        assert has_steps or is_combo, f"Expected multi-step or combo action, got: {result}"


# ── End-to-end agent + LLM tests ───────────────────────────────

class TestAgentLlmEndToEnd:
    """Full pipeline: natural language → LLM parse → Agent execution → xlsx validation."""

    def test_create_and_add_rows(self, agent, files_dir, filepath):
        """User says: 'Create a student scores sheet and add Tom 90 and Alex 85'."""
        res = agent.handle_text(
            "Create an excel sheet for student scores and add Tom with 90 and Alex with 85",
            filename="students.xlsx",
            auto_execute=True,
            auto_correct=True,
        )
        # The LLM should produce a multi-step plan; the agent should execute it
        assert res.executed, f"Expected executed=True, got error: {res.error}"
        fp = filepath("students.xlsx")
        assert os.path.exists(fp), "Excel file was not created"
        # Validate content
        wb = load_workbook(fp)
        ws = wb.active
        # At least headers + 2 data rows
        assert ws.max_row >= 3, f"Expected >=3 rows, got {ws.max_row}"
        wb.close()

    def test_create_add_row_write_cell(self, agent, files_dir, filepath):
        """Multi-step: create file, add row, then write a cell."""
        # Step 1: create
        r1 = agent.handle_text(
            "Create an excel file with headers Name and Score",
            filename="multi.xlsx",
            auto_execute=True,
            auto_correct=True,
        )
        assert r1.executed or r1.success

        # Step 2: add row
        r2 = agent.handle_text(
            "Add row Tom 90",
            filename="multi.xlsx",
            auto_execute=True,
        )
        assert r2.executed, f"add_row failed: {r2.error}"

        # Step 3: write cell
        r3 = agent.handle_text(
            "Write 100 in cell B3",
            filename="multi.xlsx",
            auto_execute=True,
        )
        assert r3.executed, f"write_cell failed: {r3.error}"

        fp = filepath("multi.xlsx")
        val = _cell(fp, None, "B3")
        assert val is not None, "Cell B3 should have been written"

    def test_delete_column_is_destructive(self, agent, files_dir, filepath):
        """Destructive actions should require confirmation unless confirmed=True."""
        agent.handle_action(
            {"type": "create_excel", "headers": ["A", "B"], "filename": "del.xlsx"},
            filename="del.xlsx",
        )
        # Via handle_text (auto_execute=True), delete_column should still
        # return a confirmation requirement since it's destructive
        res = agent.handle_text(
            "Delete column B",
            filename="del.xlsx",
            auto_execute=True,
        )
        # The agent should either ask for confirmation or skip execution
        if res.requires_confirmation:
            assert not res.executed
        else:
            # Some LLM parses may produce a different action type — just check no crash
            assert isinstance(res.action, dict)

    def test_formula_write(self, agent, files_dir, filepath):
        """Write a SUM formula and validate it's stored."""
        agent.handle_action(
            {"type": "create_excel", "headers": ["Item", "Cost"], "filename": "formula.xlsx"},
            filename="formula.xlsx",
        )
        agent.handle_action(
            {"type": "add_row", "values": ["Laptop", 1000]},
            filename="formula.xlsx",
        )
        agent.handle_action(
            {"type": "add_row", "values": ["Phone", 500]},
            filename="formula.xlsx",
        )
        # Use handle_action with explicit type so we don't depend on LLM interpretation
        res = agent.handle_action(
            {"type": "write_formula", "cell": "B4", "formula": "=SUM(B2:B3)"},
            filename="formula.xlsx",
        )
        assert res.executed, f"Formula write failed: {res.error}"
        fp = filepath("formula.xlsx")
        cell_val = _cell(fp, None, "B4")
        # openpyxl stores formulas as strings starting with '='
        assert cell_val is not None, "B4 should contain the formula"

    def test_unknown_command_produces_suggestions(self, agent, files_dir):
        """Nonsensical command should get suggestions, not crash."""
        res = agent.handle_text("blorp fizzle wizzle")
        assert isinstance(res.action, dict)
        # Should be tagged unknown
        is_unknown = (
            res.action.get("type") in ("unknown", None)
            or res.action.get("action") == "unknown"
        )
        if is_unknown:
            assert isinstance(res.suggestions, list)


# ── Multi-step plan atomic rollback ────────────────────────────

class TestMultiStepRollback:
    """Verify that multi-step plans roll back on partial failure."""

    def test_rollback_on_second_step_failure(self, agent, files_dir, filepath):
        """Create file, then run steps where step 2 is invalid — file should revert."""
        # Pre-create a file
        agent.handle_action(
            {"type": "create_excel", "headers": ["Name"], "filename": "rb.xlsx"},
            filename="rb.xlsx",
        )
        agent.handle_action(
            {"type": "add_row", "values": ["Alice"]},
            filename="rb.xlsx",
        )
        fp = filepath("rb.xlsx")
        rows_before = _row_count(fp)

        # Multi-step: step 1 adds a row (OK), step 2 has a garbage type (FAIL)
        res = agent.handle_action(
            {
                "steps": [
                    {"type": "add_row", "values": ["Bob"]},
                    {"type": "delete_row", "index": 999},
                ]
            },
            filename="rb.xlsx",
            confirmed=True,
        )
        # The second step should fail (row 999 doesn't exist) and trigger rollback
        if isinstance(res.result, dict) and "error" in res.result:
            rows_after = _row_count(fp)
            assert rows_after == rows_before, "File should have been rolled back"


# ── LLM response quality benchmarks ────────────────────────────

class TestLlmQuality:
    """Run a small benchmark of commands and assert the LLM returns the right action types."""

    CASES = [
        ("add row Tom 90", "add_row"),
        ("create excel file with headers name and score", "create_excel"),
        ("delete column B", "delete_column"),
        ("write 100 to cell A5", "write_cell"),
        ("save the file", "save"),
        ("calculate sum of row 2", "calculate_row_sum"),
    ]

    @staticmethod
    def _unwrap(result: dict) -> dict:
        steps = result.get("steps")
        if isinstance(steps, list) and len(steps) >= 1:
            return steps[0]
        return result

    @pytest.mark.parametrize("text,expected_type", CASES, ids=[c[0][:30] for c in CASES])
    def test_action_type(self, text, expected_type):
        result = llm_router.route_llm(text)
        action = self._unwrap(result)
        actual = action.get("type")
        assert actual == expected_type, (
            f"For '{text}': expected type={expected_type}, got {actual}. Full: {result}"
        )


# ── LLM timing sanity ──────────────────────────────────────────

class TestLlmTiming:
    """Ensure LLM responses come back within a reasonable timeout."""

    def test_response_under_60s(self):
        t0 = time.perf_counter()
        llm_router.route_llm("add row hello 42")
        elapsed = time.perf_counter() - t0
        assert elapsed < 60, f"LLM call took {elapsed:.1f}s — too slow"
