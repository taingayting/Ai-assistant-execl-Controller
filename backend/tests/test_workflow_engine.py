"""Tests for backend.workflow_engine."""

import os

import workflow_engine


class TestWorkflowLoading:
    def test_load_workflows(self):
        workflows = workflow_engine.load_workflows()
        assert isinstance(workflows, dict)
        assert "create_sales_report" in workflows

    def test_get_workflow(self):
        wf = workflow_engine.get_workflow("create_sales_report")
        assert wf is not None
        assert wf["workflow_id"] == "create_sales_report"
        assert wf["category"] == "reporting"
        assert isinstance(wf["preconditions"], list)
        assert isinstance(wf["steps"], list)
        assert wf["tool_calls"]


class TestWorkflowExecution:
    def test_execute_workflow_create_sales_report(self, tmp_path):
        filepath = str(tmp_path / "sales.xlsx")

        result = workflow_engine.execute_workflow(
            "create_sales_report",
            {
                "filepath": filepath,
                "headers": ["Product", "Qty", "Price", "Total", "Date"],
            },
        )

        assert result.get("message") == "workflow_executed"
        assert result.get("workflow_id") == "create_sales_report"
        assert isinstance(result.get("steps"), list)
        assert len(result["steps"]) >= 1
        assert os.path.exists(filepath)

    def test_execute_workflow_missing(self):
        result = workflow_engine.execute_workflow("no_such_workflow", {"filepath": "x.xlsx"})
        assert result.get("error") == "workflow_not_found"

    def test_execute_workflow_requires_filepath(self):
        result = workflow_engine.execute_workflow("create_sales_report", {})
        assert result.get("error") == "filepath_required"
