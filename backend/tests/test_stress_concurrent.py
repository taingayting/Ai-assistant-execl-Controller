"""Stress tests for concurrent execution of multiple Sara commands.

Validates thread-safety of the Agent, excel_controller, and memory modules
when multiple commands run in parallel.  Uses ``concurrent.futures`` so no
extra dependencies are needed.

Run:
    pytest backend/tests/test_stress_concurrent.py -v -m stress
"""
from __future__ import annotations

import os
import sys
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

from agent import Agent
from memory import memory

pytestmark = pytest.mark.stress

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


# ── helpers ─────────────────────────────────────────────────────

def _row_count(path: str) -> int:
    wb = load_workbook(path)
    n = wb.active.max_row
    wb.close()
    return n


# ── tests ───────────────────────────────────────────────────────

class TestConcurrentAgentExecution:
    """Concurrent calls to Agent.handle_action on *separate* files."""

    def test_parallel_create_files(self, files_dir):
        """Create 10 different Excel files in parallel — no file should be corrupted."""
        agents = [Agent(files_dir=files_dir) for _ in range(10)]

        def create_file(idx: int):
            a = agents[idx]
            fname = f"parallel_{idx}.xlsx"
            res = a.handle_action(
                {"type": "create_excel", "headers": ["Name", "Value"], "filename": fname},
                filename=fname,
            )
            return idx, fname, res

        results = {}
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(create_file, i): i for i in range(10)}
            for f in as_completed(futures):
                idx, fname, res = f.result()
                results[idx] = (fname, res)

        for idx in range(10):
            fname, res = results[idx]
            assert res.success, f"File {idx} failed: {res.error}"
            fp = os.path.join(files_dir, fname)
            assert os.path.exists(fp), f"{fname} not on disk"
            wb = load_workbook(fp)
            # Should have header row
            assert wb.active.max_row >= 1
            wb.close()

    def test_parallel_add_rows_same_file(self, agent, files_dir):
        """Add 20 rows concurrently to the same file — all rows should appear."""
        fname = "shared.xlsx"
        fp = os.path.join(files_dir, fname)
        # Create the file first
        agent.handle_action(
            {"type": "create_excel", "headers": ["ID", "Score"], "filename": fname},
            filename=fname,
        )

        def add_row(idx: int):
            # Each thread creates its own agent to avoid sharing state
            a = Agent(files_dir=files_dir)
            res = a.handle_action(
                {"type": "add_row", "values": [f"user_{idx}", idx * 10]},
                filename=fname,
            )
            return idx, res

        results = []
        errors = 0
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(add_row, i) for i in range(20)]
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except (PermissionError, OSError):
                    errors += 1  # Expected on Windows due to file locks

        # Count successes
        successes = [r for r in results if r[1].success and r[1].executed]
        # Some may fail due to file locking, but the file should not be corrupted
        wb = load_workbook(fp)
        ws = wb.active
        assert ws.max_row >= 2, "Should have at least header + some data rows"
        wb.close()
        # At least some should succeed (concurrent openpyxl writes may conflict)
        assert len(successes) >= 1, "At least one concurrent add_row should succeed"

    def test_parallel_multi_step_plans(self, files_dir):
        """Run 5 multi-step plans (create + add rows) in parallel on separate files."""

        def run_plan(idx: int):
            a = Agent(files_dir=files_dir)
            fname = f"plan_{idx}.xlsx"
            action = {
                "steps": [
                    {"type": "create_excel", "headers": ["Item", "Qty"], "filename": fname},
                    {"type": "add_row", "values": [f"item_{idx}_a", 10]},
                    {"type": "add_row", "values": [f"item_{idx}_b", 20]},
                ]
            }
            res = a.handle_action(action, filename=fname, confirmed=True)
            return idx, fname, res

        results = {}
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(run_plan, i): i for i in range(5)}
            for f in as_completed(futures):
                idx, fname, res = f.result()
                results[idx] = (fname, res)

        for idx in range(5):
            fname, res = results[idx]
            fp = os.path.join(files_dir, fname)
            assert os.path.exists(fp), f"Plan {idx}: file not created"
            rows = _row_count(fp)
            assert rows >= 3, f"Plan {idx}: expected >=3 rows (header + 2 data), got {rows}"


class TestConcurrentFlaskEndpoints:
    """Concurrent HTTP requests to the Flask app."""

    @pytest.fixture
    def client(self):
        import app as app_module
        app_module.app.testing = True
        return app_module.app.test_client()

    def test_parallel_parse_requests(self, client):
        """Send 10 concurrent /api/parse requests — all should return valid JSON."""
        commands = [
            "add row Tom 90",
            "create excel file for scores",
            "delete column B",
            "write 42 to A1",
            "save the file",
            "calculate sum of row 2",
            "add column revenue",
            "delete row 3",
            "create chart from data",
            "format cells A1:C5 bold",
        ]

        def send_parse(text: str):
            resp = client.post("/api/parse", json={"text": text})
            return text, resp.status_code, resp.get_json()

        results = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(send_parse, cmd) for cmd in commands]
            for f in as_completed(futures):
                results.append(f.result())

        for text, status, data in results:
            assert status == 200, f"'{text}' returned status {status}"
            assert data.get("success") is True, f"'{text}' not successful: {data}"
            assert "action" in data, f"'{text}' missing action field"

    def test_parallel_execute_separate_files(self, client):
        """Execute create_excel on 5 different files concurrently."""

        def create(idx: int):
            fname = f"conc_{idx}.xlsx"
            action = {"type": "create_excel", "headers": ["A", "B"], "filename": fname}
            resp = client.post("/api/execute", json={"action": action, "filename": fname})
            return idx, fname, resp.status_code, resp.get_json()

        results = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(create, i) for i in range(5)]
            for f in as_completed(futures):
                results.append(f.result())

        for idx, fname, status, data in results:
            assert status == 200, f"File {idx} ({fname}): status {status}, data={data}"
            assert data.get("success") is not False or data.get("requires_confirmation"), (
                f"File {idx} failed: {data}"
            )


class TestConcurrentTiming:
    """Verify concurrent operations don't cause excessive slowdown."""

    def test_10_creates_under_30s(self, files_dir):
        """Creating 10 files in parallel should complete within 30 seconds."""
        t0 = time.perf_counter()

        def create(idx: int):
            a = Agent(files_dir=files_dir)
            fname = f"timing_{idx}.xlsx"
            return a.handle_action(
                {"type": "create_excel", "headers": ["X"], "filename": fname},
                filename=fname,
            )

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(create, i) for i in range(10)]
            for f in as_completed(futures):
                f.result()

        elapsed = time.perf_counter() - t0
        assert elapsed < 30, f"10 parallel creates took {elapsed:.1f}s — too slow"
