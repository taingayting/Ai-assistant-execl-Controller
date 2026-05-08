"""Tests for the Agent layer (agent.py)."""
import os
import json
import pytest
from memory import memory


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_memory():
    """Reset memory before every test."""
    memory.clear()
    yield
    memory.clear()


@pytest.fixture
def files_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def make_agent(files_dir):
    """Create a fresh Agent pointing at a temp directory."""
    from agent import Agent
    return Agent(files_dir=files_dir)


# ── Unit tests: Agent.handle_action ─────────────────────────────

class TestHandleAction:
    def test_create_excel(self, make_agent, files_dir):
        action = {"type": "create_excel", "filename": "test.xlsx"}
        res = make_agent.handle_action(action, filename="test.xlsx")
        assert res.success
        assert res.executed
        assert res.workflow_id == "create_excel_file"
        assert os.path.exists(os.path.join(files_dir, "test.xlsx"))
        # memory updated
        assert memory.get_active_file() == "test.xlsx"

    def test_add_row_requires_active_file(self, make_agent):
        """add_row should fail precondition when no file is active."""
        action = {"type": "add_row", "values": ["a", "b"]}
        res = make_agent.handle_action(action)
        assert not res.success
        assert "No active file" in (res.error or "")
        assert res.speech_text == "No Excel file is open. Would you like me to create a new file?"
        assert isinstance(res.recovery, dict)
        assert res.recovery.get("can_auto_correct") is True

    def test_add_column_auto_correct_creates_file_and_retries(self, make_agent, files_dir):
        action = {"type": "add_column", "name": "price"}
        res = make_agent.handle_action(action, auto_correct=True)
        assert res.success
        assert res.executed
        assert res.auto_corrected is True
        assert os.path.exists(os.path.join(files_dir, "data.xlsx"))
        assert memory.get_active_file() == "data.xlsx"

    def test_add_row_after_create(self, make_agent, files_dir):
        """Full flow: create, then add row."""
        # Step 1: create
        make_agent.handle_action(
            {"type": "create_excel", "filename": "t.xlsx"},
            filename="t.xlsx",
        )
        # Step 2: add row (memory now has active_file)
        res = make_agent.handle_action(
            {"type": "add_row", "values": ["x", "y"]},
        )
        assert res.success
        assert res.executed

    def test_destructive_returns_confirmation(self, make_agent, files_dir):
        # Create a file first
        make_agent.handle_action(
            {"type": "create_excel", "filename": "d.xlsx"},
            filename="d.xlsx",
        )
        res = make_agent.handle_action({"type": "delete_column", "column": "A"})
        assert res.requires_confirmation
        assert not res.executed

    def test_write_cell(self, make_agent, files_dir):
        make_agent.handle_action(
            {"type": "create_excel", "filename": "w.xlsx"},
            filename="w.xlsx",
        )
        res = make_agent.handle_action(
            {"type": "write_cell", "cell": "B2", "value": "hello"},
        )
        assert res.success
        assert res.executed

    def test_multi_step(self, make_agent, files_dir):
        action = {
            "steps": [
                {"type": "create_excel", "filename": "ms.xlsx"},
                {"type": "add_row", "values": ["1", "2"]},
            ]
        }
        res = make_agent.handle_action(action, filename="ms.xlsx")
        assert res.success
        assert res.executed
        assert isinstance(res.result, dict)
        assert res.result.get("message") == "steps_executed"

    def test_unknown_type_no_workflow(self, make_agent):
        """Unknown types should still attempt execution (no precondition block)."""
        action = {"type": "unknown"}
        res = make_agent.handle_action(action)
        # Workflow is None so no precondition check — execution proceeds
        # The result depends on excel_controller.execute_action handling 'unknown'
        assert res.executed or res.result is not None

    def test_create_sales_report_via_workflow_engine(self, make_agent, files_dir):
        action = {"type": "create_sales_report"}
        res = make_agent.handle_action(action, filename="sales_report.xlsx")
        assert res.success
        assert res.executed
        assert res.workflow_id == "create_sales_report"
        assert os.path.exists(os.path.join(files_dir, "sales_report.xlsx"))
        assert isinstance(res.result, dict)
        assert res.result.get("workflow_id") == "create_sales_report"


class TestMemoryUpdate:
    def test_create_sets_active_sheet(self, make_agent, files_dir):
        make_agent.handle_action(
            {"type": "create_excel", "filename": "s.xlsx"},
            filename="s.xlsx",
        )
        assert memory.get_active_sheet() == "Sheet1"

    def test_command_recorded(self, make_agent, files_dir):
        action = {"type": "create_excel", "filename": "r.xlsx"}
        make_agent.handle_action(action, filename="r.xlsx")
        cmds = memory.get_recent_commands(5)
        assert len(cmds) >= 1
        assert cmds[-1].get("type") == "create_excel"


# ── Integration with Flask (via test client) ─────────────────────

class TestFlaskIntegration:
    @pytest.fixture
    def client(self):
        memory.clear()
        from app import app
        app.config['TESTING'] = True
        with app.test_client() as c:
            yield c
        memory.clear()

    def test_workflows_endpoint(self, client):
        resp = client.get('/api/workflows')
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['success']
        assert len(body['workflows']) > 0

    def test_execute_with_agent_creates_file(self, client):
        resp = client.post('/api/execute', json={
            'action': {'type': 'create_excel', 'filename': 'agent_test.xlsx'},
            'filename': 'agent_test.xlsx',
        })
        body = resp.get_json()
        assert body['success'], body
        assert body.get('executed')
        assert body.get('workflow_id') == 'create_excel_file'
        assert body.get('download_url')

    def test_execute_precondition_fail(self, client):
        """add_row without active file → precondition error."""
        resp = client.post('/api/execute', json={
            'action': {'type': 'add_row', 'values': ['a']},
        })
        body = resp.get_json()
        assert resp.status_code == 400
        assert "No active file" in body.get('error', '')

    def test_execute_destructive_confirmation(self, client):
        # create first
        client.post('/api/execute', json={
            'action': {'type': 'create_excel', 'filename': 'dc.xlsx'},
            'filename': 'dc.xlsx',
        })
        # attempt delete_column
        resp = client.post('/api/execute', json={
            'action': {'type': 'delete_column', 'column': 'A'},
        })
        body = resp.get_json()
        assert body['success']
        assert body.get('requires_confirmation')
        assert body.get('confirmation_id')
