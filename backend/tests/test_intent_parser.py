"""Tests for intent.intent_parser — IntentParser.

All LLM calls are mocked; tests exercise both the LLM-normalisation path
and the pure rule-based fallback.
"""
from __future__ import annotations

import os
import sys

import pytest

# ── path fixup ──────────────────────────────────────────────────
_BACKEND_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.insert(0, os.path.abspath(_BACKEND_DIR))

from intent.intent_parser import (
    IntentParser,
    SUPPORTED_ACTIONS,
    _coerce_value,
    _normalise_operator,
    _extract_sheet_name,
)


# ───────────────────────────────────────────────────────────────
# Helper: parser with LLM disabled (pure rule-based)
# ───────────────────────────────────────────────────────────────

@pytest.fixture
def parser():
    """Return a rule-only parser (no LLM calls)."""
    return IntentParser(use_llm=False)


# ───────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────

class TestConstants:
    def test_supported_actions_count(self):
        assert len(SUPPORTED_ACTIONS) == 6

    def test_supported_actions_contains_all(self):
        expected = {"create_sheet", "write_cell", "read_cell",
                    "create_chart", "sum_column", "filter_table"}
        assert set(SUPPORTED_ACTIONS) == expected


# ───────────────────────────────────────────────────────────────
# create_sheet
# ───────────────────────────────────────────────────────────────

class TestCreateSheet:
    def test_basic(self, parser):
        r = parser.parse("Create a new sheet called Sales")
        assert r["action"] == "create_sheet"
        assert r["sheet_name"] == "Sales"

    def test_make_sheet(self, parser):
        r = parser.parse("make a sheet named Report")
        assert r["action"] == "create_sheet"
        assert r["sheet_name"] == "Report"

    def test_add_worksheet(self, parser):
        r = parser.parse("add worksheet Data")
        assert r["action"] == "create_sheet"
        assert r["sheet_name"] == "Data"

    def test_new_tab(self, parser):
        r = parser.parse("new tab Summary")
        assert r["action"] == "create_sheet"
        assert r["sheet_name"] == "Summary"

    def test_create_sheet_simple(self, parser):
        r = parser.parse("create sheet Inventory")
        assert r["action"] == "create_sheet"
        assert r["sheet_name"] == "Inventory"


# ───────────────────────────────────────────────────────────────
# write_cell
# ───────────────────────────────────────────────────────────────

class TestWriteCell:
    def test_write_value_to_cell(self, parser):
        r = parser.parse("write 100 to A1")
        assert r["action"] == "write_cell"
        assert r["cell"] == "A1"
        assert r["value"] == 100

    def test_put_string_in_cell(self, parser):
        r = parser.parse("put hello in B2")
        assert r["action"] == "write_cell"
        assert r["cell"] == "B2"
        assert r["value"] == "hello"

    def test_set_cell_to_value(self, parser):
        r = parser.parse("set A5 to 42")
        assert r["action"] == "write_cell"
        assert r["cell"] == "A5"
        assert r["value"] == 42

    def test_enter_float(self, parser):
        r = parser.parse("enter 3.14 into C3")
        assert r["action"] == "write_cell"
        assert r["cell"] == "C3"
        assert r["value"] == 3.14

    def test_write_to_cell_keyword(self, parser):
        r = parser.parse("write hello to cell D4")
        assert r["action"] == "write_cell"
        assert r["cell"] == "D4"


# ───────────────────────────────────────────────────────────────
# read_cell
# ───────────────────────────────────────────────────────────────

class TestReadCell:
    def test_read_basic(self, parser):
        r = parser.parse("read A1")
        assert r["action"] == "read_cell"
        assert r["cell"] == "A1"

    def test_get_cell(self, parser):
        r = parser.parse("get cell B5")
        assert r["action"] == "read_cell"
        assert r["cell"] == "B5"

    def test_show_cell(self, parser):
        r = parser.parse("show C3")
        assert r["action"] == "read_cell"
        assert r["cell"] == "C3"

    def test_whats_in_cell(self, parser):
        r = parser.parse("what's in A1")
        assert r["action"] == "read_cell"
        assert r["cell"] == "A1"

    def test_what_is_in_cell(self, parser):
        r = parser.parse("what is in B2")
        assert r["action"] == "read_cell"
        assert r["cell"] == "B2"

    def test_value_of_cell(self, parser):
        r = parser.parse("value of cell D4")
        assert r["action"] == "read_cell"
        assert r["cell"] == "D4"


# ───────────────────────────────────────────────────────────────
# create_chart
# ───────────────────────────────────────────────────────────────

class TestCreateChart:
    def test_basic_chart(self, parser):
        r = parser.parse("create a chart")
        assert r["action"] == "create_chart"

    def test_bar_chart(self, parser):
        r = parser.parse("make a bar chart")
        assert r["action"] == "create_chart"
        assert r["chart_type"] == "bar"

    def test_pie_chart_with_range(self, parser):
        r = parser.parse("create a pie chart from A1:C10")
        assert r["action"] == "create_chart"
        assert r["chart_type"] == "pie"
        assert r["data_range"] == "A1:C10"

    def test_line_chart(self, parser):
        r = parser.parse("draw a line chart")
        assert r["action"] == "create_chart"
        assert r["chart_type"] == "line"


# ───────────────────────────────────────────────────────────────
# sum_column
# ───────────────────────────────────────────────────────────────

class TestSumColumn:
    def test_sum_column_b(self, parser):
        r = parser.parse("sum column B")
        assert r["action"] == "sum_column"
        assert r["column"] == "B"

    def test_total_of_column_a(self, parser):
        r = parser.parse("total of column A")
        assert r["action"] == "sum_column"
        assert r["column"] == "A"

    def test_calculate_sum(self, parser):
        r = parser.parse("calculate the sum of column C")
        assert r["action"] == "sum_column"
        assert r["column"] == "C"

    def test_add_up_column(self, parser):
        r = parser.parse("add up column D")
        assert r["action"] == "sum_column"
        assert r["column"] == "D"


# ───────────────────────────────────────────────────────────────
# filter_table
# ───────────────────────────────────────────────────────────────

class TestFilterTable:
    def test_filter_equals(self, parser):
        r = parser.parse("filter Status = Active")
        assert r["action"] == "filter_table"
        assert r["column"] == "Status"
        assert r["operator"] == "="
        assert r["value"] == "Active"

    def test_filter_by_column(self, parser):
        r = parser.parse("filter by column Age > 30")
        assert r["action"] == "filter_table"
        assert r["column"] == "Age"
        assert r["operator"] == ">"
        assert r["value"] == 30

    def test_show_rows_where(self, parser):
        r = parser.parse("show rows where Score >= 90")
        assert r["action"] == "filter_table"
        assert r["column"] == "Score"
        assert r["operator"] == ">="
        assert r["value"] == 90

    def test_filter_not_equal(self, parser):
        r = parser.parse("filter Status != Inactive")
        assert r["action"] == "filter_table"
        assert r["column"] == "Status"
        assert r["operator"] == "!="
        assert r["value"] == "Inactive"

    def test_find_rows(self, parser):
        r = parser.parse("find rows where Price < 50")
        assert r["action"] == "filter_table"
        assert r["column"] == "Price"
        assert r["operator"] == "<"
        assert r["value"] == 50


# ───────────────────────────────────────────────────────────────
# unknown / edge cases
# ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string(self, parser):
        r = parser.parse("")
        assert r["action"] == "unknown"

    def test_none(self, parser):
        r = parser.parse(None)
        assert r["action"] == "unknown"

    def test_gibberish(self, parser):
        r = parser.parse("asdfghjkl")
        assert r["action"] == "unknown"

    def test_whitespace_only(self, parser):
        r = parser.parse("   ")
        assert r["action"] == "unknown"


# ───────────────────────────────────────────────────────────────
# LLM normalisation path (mocked)
# ───────────────────────────────────────────────────────────────

class TestLLMPath:
    def test_llm_create_sheet(self):
        from unittest.mock import patch
        parser = IntentParser(use_llm=True)
        with patch("llm_router.route_llm",
                   return_value={"type": "create_sheet", "sheet_name": "Budget"}):
            r = parser.parse("Create a new sheet called Budget")
        assert r["action"] == "create_sheet"
        assert r["sheet_name"] == "Budget"

    def test_llm_write_cell(self):
        from unittest.mock import patch
        parser = IntentParser(use_llm=True)
        with patch("llm_router.route_llm",
                   return_value={"type": "write_cell", "cell": "A1", "value": 100}):
            r = parser.parse("Write 100 to A1")
        assert r["action"] == "write_cell"
        assert r["cell"] == "A1"
        assert r["value"] == 100

    def test_llm_create_chart(self):
        from unittest.mock import patch
        parser = IntentParser(use_llm=True)
        with patch("llm_router.route_llm",
                   return_value={"type": "create_chart", "chart_type": "bar",
                                 "data_range": "A1:C10"}):
            r = parser.parse("create a bar chart from A1:C10")
        assert r["action"] == "create_chart"
        assert r["chart_type"] == "bar"
        assert r["data_range"] == "A1:C10"

    def test_llm_returns_steps(self):
        from unittest.mock import patch
        parser = IntentParser(use_llm=True)
        with patch("llm_router.route_llm",
                   return_value={"steps": [
                       {"type": "create_excel", "headers": ["Name", "Score"]},
                       {"type": "add_row", "values": ["Tom", 90]},
                   ]}):
            r = parser.parse("create score sheet and add Tom 90")
        # Should map first meaningful step (create_excel → create_sheet)
        assert r["action"] == "create_sheet"

    def test_llm_unknown_falls_through(self):
        from unittest.mock import patch
        parser = IntentParser(use_llm=True)
        with patch("llm_router.route_llm",
                   return_value={"type": "unknown", "raw": "gibberish"}):
            # LLM returns unknown → falls to rule-based → returns unknown
            r = parser.parse("gibberish")
        assert r["action"] == "unknown"

    def test_llm_exception_falls_to_rules(self):
        from unittest.mock import patch
        parser = IntentParser(use_llm=True)
        with patch("llm_router.route_llm",
                   side_effect=RuntimeError("network error")):
            r = parser.parse("create sheet Sales")
        # Falls back to rule-based
        assert r["action"] == "create_sheet"
        assert r["sheet_name"] == "Sales"


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────

class TestHelpers:
    def test_coerce_int(self):
        assert _coerce_value("42") == 42

    def test_coerce_float(self):
        assert _coerce_value("3.14") == 3.14

    def test_coerce_string(self):
        assert _coerce_value("hello") == "hello"

    def test_normalise_equals(self):
        assert _normalise_operator("is") == "="
        assert _normalise_operator("==") == "="
        assert _normalise_operator("equals") == "="

    def test_normalise_not_equal(self):
        assert _normalise_operator("!=") == "!="
        assert _normalise_operator("<>") == "!="

    def test_normalise_comparison(self):
        assert _normalise_operator(">") == ">"
        assert _normalise_operator("greater than") == ">"
        assert _normalise_operator("<") == "<"
        assert _normalise_operator("less than") == "<"

    def test_normalise_contains(self):
        assert _normalise_operator("contains") == "contains"

    def test_extract_sheet_name_called(self):
        assert _extract_sheet_name("create sheet called Budget") == "Budget"

    def test_extract_sheet_name_named(self):
        assert _extract_sheet_name("make sheet named Sales") == "Sales"

    def test_extract_sheet_name_none(self):
        assert _extract_sheet_name("hello world") is None
