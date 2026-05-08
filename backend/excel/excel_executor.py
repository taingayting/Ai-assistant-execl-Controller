"""Excel Execution Engine for Sara

Executes structured intent JSON actions using openpyxl.

Supported operations:
- open workbook
- create sheet
- write value to cell
- read cell
- create charts
- calculate sums

Input: intent dict (see intent_parser)
Output: dict with 'success' (bool), 'result' (any), and 'error' (str, optional)
"""
from __future__ import annotations

import os
import logging
from typing import Any

from openpyxl import load_workbook, Workbook
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference

log = logging.getLogger("excel.executor")

class ExcelExecutor:
    def __init__(self, file_path: str | None = None):
        self.file_path = file_path
        self.wb = None
        self.active_sheet = None

    def open_workbook(self, file_path: str) -> dict:
        try:
            if os.path.exists(file_path):
                self.wb = load_workbook(file_path)
            else:
                self.wb = Workbook()
            self.file_path = file_path
            self.active_sheet = self.wb.active
            return {"success": True, "result": f"Opened {file_path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_sheet(self, sheet_name: str) -> dict:
        try:
            if self.wb is None:
                self.wb = Workbook()
            if sheet_name in self.wb.sheetnames:
                ws = self.wb[sheet_name]
            else:
                ws = self.wb.create_sheet(sheet_name)
            self.active_sheet = ws
            return {"success": True, "result": f"Created sheet {sheet_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def write_cell(self, cell: str, value: Any, sheet: str | None = None) -> dict:
        try:
            ws = self._get_sheet(sheet)
            ws[cell] = value
            return {"success": True, "result": f"Wrote {value} to {cell}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_cell(self, cell: str, sheet: str | None = None) -> dict:
        try:
            ws = self._get_sheet(sheet)
            value = ws[cell].value
            return {"success": True, "result": value}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_chart(self, data_range: str, chart_type: str = "bar", title: str = "Chart", sheet: str | None = None) -> dict:
        try:
            ws = self._get_sheet(sheet)
            min_col, min_row, max_col, max_row = self._parse_range(data_range)
            data = Reference(ws, min_col=min_col, min_row=min_row, max_col=max_col, max_row=max_row)
            chart = BarChart()
            chart.title = title
            chart.add_data(data, titles_from_data=True)
            ws.add_chart(chart, f"{get_column_letter(max_col+2)}{min_row}")
            return {"success": True, "result": f"Chart created: {title}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def sum_column(self, column: str, sheet: str | None = None) -> dict:
        try:
            ws = self._get_sheet(sheet)
            col_idx = self._col_letter_to_index(column)
            values = [cell.value for cell in ws.iter_cols(min_col=col_idx, max_col=col_idx, min_row=2) for cell in cell if isinstance(cell.value, (int, float))]
            total = sum(values)
            return {"success": True, "result": total}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save(self, file_path: str | None = None) -> dict:
        try:
            path = file_path or self.file_path
            if not path:
                return {"success": False, "error": "No file path specified"}
            self.wb.save(path)
            return {"success": True, "result": f"Saved to {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute(self, intent: dict) -> dict:
        action = intent.get("action")
        try:
            if action == "open_workbook":
                return self.open_workbook(intent["file_path"])
            elif action == "create_sheet":
                return self.create_sheet(intent["sheet_name"])
            elif action == "write_cell":
                return self.write_cell(intent["cell"], intent["value"], intent.get("sheet"))
            elif action == "read_cell":
                return self.read_cell(intent["cell"], intent.get("sheet"))
            elif action == "create_chart":
                return self.create_chart(intent["data_range"], intent.get("chart_type", "bar"), intent.get("title", "Chart"), intent.get("sheet"))
            elif action == "sum_column":
                return self.sum_column(intent["column"], intent.get("sheet"))
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_sheet(self, sheet: str | None = None):
        if self.wb is None:
            self.wb = Workbook()
        if sheet:
            if sheet in self.wb.sheetnames:
                return self.wb[sheet]
            else:
                return self.wb.create_sheet(sheet)
        if self.active_sheet:
            return self.active_sheet
        return self.wb.active

    @staticmethod
    def _col_letter_to_index(col: str) -> int:
        from openpyxl.utils import column_index_from_string
        return column_index_from_string(col)

    @staticmethod
    def _parse_range(cell_range: str):
        # e.g. "A1:C10" -> (1, 1, 3, 10)
        from openpyxl.utils import range_boundaries
        min_col, min_row, max_col, max_row = range_boundaries(cell_range)
        return min_col, min_row, max_col, max_row
