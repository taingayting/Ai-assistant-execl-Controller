"""Tests for workflow_router module."""
import os
import pytest
import workflow_router


# ── Registry tests ──────────────────────────────────────────────

class TestRegistry:
    def test_all_workflows_have_required_keys(self):
        for wf in workflow_router.WORKFLOWS:
            assert "id" in wf
            assert "category" in wf
            assert "action_types" in wf
            assert isinstance(wf["action_types"], set)
            assert "requires_file" in wf
            assert "destructive" in wf

    def test_workflow_ids_are_unique(self):
        ids = [wf["id"] for wf in workflow_router.WORKFLOWS]
        assert len(ids) == len(set(ids))

    def test_no_action_type_in_multiple_workflows(self):
        seen = {}
        for wf in workflow_router.WORKFLOWS:
            for at in wf["action_types"]:
                assert at not in seen, f"action_type '{at}' in both '{seen[at]}' and '{wf['id']}'"
                seen[at] = wf["id"]


# ── Lookup tests ────────────────────────────────────────────────

class TestLookup:
    def test_get_by_action_type_known(self):
        wf = workflow_router.get_workflow_by_action_type("create_excel")
        assert wf is not None
        assert wf["id"] == "create_excel_file"

    def test_get_by_action_type_unknown(self):
        assert workflow_router.get_workflow_by_action_type("nonexistent") is None

    def test_get_by_id(self):
        wf = workflow_router.get_workflow_by_id("insert_data")
        assert wf is not None
        assert "add_row" in wf["action_types"]

    def test_get_by_id_unknown(self):
        assert workflow_router.get_workflow_by_id("no_such_wf") is None

    def test_list_workflows(self):
        all_wf = workflow_router.list_workflows()
        assert len(all_wf) == len(workflow_router.WORKFLOWS)
        assert all_wf is not workflow_router.WORKFLOWS  # returns a copy


# ── Routing tests ───────────────────────────────────────────────

class TestRoute:
    def test_route_simple_action(self):
        action = {"type": "add_row", "values": ["a", "b"]}
        wf = workflow_router.route(action)
        assert wf is not None
        assert wf["id"] == "insert_data"

    def test_route_multi_step(self):
        action = {"steps": [{"type": "create_excel"}, {"type": "add_row"}]}
        wf = workflow_router.route(action)
        assert wf is not None
        assert wf["id"] == "create_excel_file"

    def test_route_unknown_type(self):
        assert workflow_router.route({"type": "unknown"}) is None

    def test_route_empty_dict(self):
        assert workflow_router.route({}) is None

    def test_route_non_dict(self):
        assert workflow_router.route("not a dict") is None


# ── Precondition tests ──────────────────────────────────────────

class TestPreconditions:
    def test_requires_file_no_active_file(self):
        wf = workflow_router.get_workflow_by_action_type("add_row")
        errors = workflow_router.check_preconditions(wf, {"active_file": None})
        assert len(errors) == 1
        assert "No active file" in errors[0]

    def test_requires_file_with_active_file(self):
        wf = workflow_router.get_workflow_by_action_type("add_row")
        errors = workflow_router.check_preconditions(
            wf, {"active_file": "test.xlsx"}, filepath=None
        )
        assert errors == []

    def test_no_requires_file(self):
        wf = workflow_router.get_workflow_by_action_type("create_excel")
        errors = workflow_router.check_preconditions(wf, {"active_file": None})
        assert errors == []

    def test_file_not_found_on_disk(self, tmp_path):
        wf = workflow_router.get_workflow_by_action_type("format_cells")
        fp = str(tmp_path / "nope.xlsx")
        errors = workflow_router.check_preconditions(
            wf, {"active_file": "nope.xlsx"}, filepath=fp
        )
        assert any("not found" in e for e in errors)


# ── Destructive detection ───────────────────────────────────────

class TestDestructive:
    def test_delete_column_destructive(self):
        assert workflow_router.is_destructive({"type": "delete_column"})

    def test_delete_row_destructive(self):
        assert workflow_router.is_destructive({"type": "delete_row"})

    def test_add_row_not_destructive(self):
        assert not workflow_router.is_destructive({"type": "add_row"})

    def test_multi_step_destructive(self):
        action = {"steps": [{"type": "add_row"}, {"type": "delete_column"}]}
        assert workflow_router.is_destructive(action)

    def test_multi_step_not_destructive(self):
        action = {"steps": [{"type": "add_row"}, {"type": "write_cell"}]}
        assert not workflow_router.is_destructive(action)

    def test_non_dict(self):
        assert not workflow_router.is_destructive("hello")
