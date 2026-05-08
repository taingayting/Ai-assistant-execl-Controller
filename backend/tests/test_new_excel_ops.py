"""Tests for the four new Excel controller operations:
    open_excel, calculate_column_average, format_cells, create_chart
"""
import os
import pytest
from openpyxl import load_workbook

from excel_controller import (
    create_excel, add_row, open_excel, calculate_column_average,
    format_cells, create_chart, execute_action,
)

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'files')


@pytest.fixture
def xlsx_path(tmp_path):
    return str(tmp_path / 'test.xlsx')


@pytest.fixture
def sample_workbook(xlsx_path):
    """Create a small workbook with headers + 3 data rows."""
    create_excel(xlsx_path, headers=['Product', 'Price', 'Qty'])
    add_row(xlsx_path, ['Apple', 3.0, 10])
    add_row(xlsx_path, ['Banana', 1.5, 20])
    add_row(xlsx_path, ['Cherry', 5.0, 15])
    return xlsx_path


# ========================== open_excel ==========================

class TestOpenExcel:
    def test_open_existing_file(self, sample_workbook):
        result = open_excel(sample_workbook)
        assert result['message'] == 'file_opened'
        assert len(result['sheets']) == 1
        assert result['active_sheet'] == result['sheets'][0]
        assert result['rows'] == 4  # 1 header + 3 data
        assert result['columns'] == 3
        assert 'speech_text' in result

    def test_open_nonexistent(self, xlsx_path):
        result = open_excel(xlsx_path)
        assert result['error'] == 'file_not_found'

    def test_open_via_execute_action(self, sample_workbook):
        result = execute_action({'type': 'open_file'}, sample_workbook)
        assert result['message'] == 'file_opened'


# ===================== calculate_column_average =====================

class TestCalculateAverage:
    def test_average_numeric_column(self, sample_workbook):
        result = calculate_column_average(sample_workbook, 'B')
        assert result['message'] == 'column_average'
        assert abs(result['average'] - 3.1666666) < 0.01  # (3+1.5+5)/3
        assert result['count'] == 3

    def test_average_by_index(self, sample_workbook):
        result = calculate_column_average(sample_workbook, 3)  # Qty column
        assert result['message'] == 'column_average'
        assert result['average'] == 15.0  # (10+20+15)/3
        assert result['count'] == 3

    def test_average_non_numeric_column(self, sample_workbook):
        result = calculate_column_average(sample_workbook, 'A')  # Product = strings
        assert result['average'] == 0.0
        assert result['count'] == 0

    def test_average_column_out_of_range(self, sample_workbook):
        result = calculate_column_average(sample_workbook, 99)
        assert 'error' in result

    def test_average_none_column(self, sample_workbook):
        result = calculate_column_average(sample_workbook, None)
        assert 'error' in result

    def test_average_invalid_column(self, sample_workbook):
        result = calculate_column_average(sample_workbook, 'ZZZ_bad')
        assert 'error' in result

    def test_average_via_execute_action(self, sample_workbook):
        result = execute_action({'type': 'calculate_average', 'column': 'B'}, sample_workbook)
        assert result['message'] == 'column_average'
        assert result['count'] == 3


# ========================== format_cells ==========================

class TestFormatCells:
    def test_format_bold(self, sample_workbook):
        result = format_cells(sample_workbook, 'A1:C1', bold=True)
        assert result['message'] == 'cells_formatted'
        assert result['cells_affected'] == 3
        wb = load_workbook(sample_workbook)
        assert wb.active['A1'].font.bold is True

    def test_format_fill_color(self, sample_workbook):
        result = format_cells(sample_workbook, 'A1:C1', fill_color='4472C4')
        assert result['message'] == 'cells_formatted'
        wb = load_workbook(sample_workbook)
        assert wb.active['A1'].fill.start_color.rgb == '004472C4'

    def test_format_font_size(self, sample_workbook):
        result = format_cells(sample_workbook, 'A1', font_size=16)
        assert result['message'] == 'cells_formatted'
        wb = load_workbook(sample_workbook)
        assert wb.active['A1'].font.size == 16

    def test_format_num_format(self, sample_workbook):
        result = format_cells(sample_workbook, 'B2:B4', num_format='$#,##0.00')
        assert result['message'] == 'cells_formatted'
        wb = load_workbook(sample_workbook)
        assert wb.active['B2'].number_format == '$#,##0.00'

    def test_format_no_range(self, sample_workbook):
        result = format_cells(sample_workbook, None)
        assert 'error' in result

    def test_format_bad_font_size(self, sample_workbook):
        result = format_cells(sample_workbook, 'A1', font_size='abc')
        assert 'error' in result

    def test_format_via_execute_action(self, sample_workbook):
        result = execute_action({
            'type': 'format_cells',
            'range': 'A1:C1',
            'bold': True,
            'fill_color': 'FF0000',
        }, sample_workbook)
        assert result['message'] == 'cells_formatted'


# ========================== create_chart ==========================

class TestCreateChart:
    def test_create_bar_chart(self, sample_workbook):
        result = create_chart(sample_workbook, chart_type='bar',
                              data_range='A1:C4', title='Test Chart')
        assert result['message'] == 'chart_created'
        assert result['chart_type'] == 'bar'
        wb = load_workbook(sample_workbook)
        assert len(wb.active._charts) == 1

    def test_create_line_chart(self, sample_workbook):
        result = create_chart(sample_workbook, chart_type='line',
                              data_range='A1:B4')
        assert result['message'] == 'chart_created'

    def test_create_pie_chart(self, sample_workbook):
        result = create_chart(sample_workbook, chart_type='pie',
                              data_range='A1:B4')
        assert result['message'] == 'chart_created'

    def test_unsupported_chart_type(self, sample_workbook):
        result = create_chart(sample_workbook, chart_type='radar',
                              data_range='A1:B4')
        assert 'error' in result
        assert 'allowed' in result

    def test_no_data_range(self, sample_workbook):
        result = create_chart(sample_workbook, chart_type='bar')
        assert 'error' in result

    def test_invalid_data_range(self, sample_workbook):
        result = create_chart(sample_workbook, data_range='not_a_range')
        assert 'error' in result

    def test_chart_via_execute_action(self, sample_workbook):
        result = execute_action({
            'type': 'create_chart',
            'chart_type': 'bar',
            'data_range': 'A1:C4',
            'title': 'Sales',
            'position': 'F2',
        }, sample_workbook)
        assert result['message'] == 'chart_created'
        assert result['position'] == 'F2'
