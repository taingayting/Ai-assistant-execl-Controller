import os
import pytest

from excel_controller import (
    create_excel,
    add_row,
    write_cell,
    calculate_row_sum,
    delete_column,
)


def test_write_and_row_sum(tmp_path):
    p = tmp_path / 'test.xlsx'
    # create workbook with headers
    create_excel(str(p), headers=['A', 'B'])
    # write numeric values to row 2
    r1 = write_cell(str(p), 'A2', 10)
    r2 = write_cell(str(p), 'B2', '20')
    assert r1['message'] == 'cell_written'
    assert r2['message'] == 'cell_written'
    res = calculate_row_sum(str(p), 2)
    assert res['message'] == 'row_sum'
    assert float(res['sum']) == 30.0


def test_delete_column_and_counts(tmp_path):
    p = tmp_path / 'test2.xlsx'
    create_excel(str(p), headers=['H1', 'H2', 'H3'])
    # add some rows
    add_row(str(p), ['a', 'b', 'c'])
    add_row(str(p), ['1', '2', '3'])
    # delete column 2 (B)
    out = delete_column(str(p), 'B')
    assert out['message'] == 'column_deleted'
    assert 'rows_affected' in out


def test_write_invalid_cell(tmp_path):
    p = tmp_path / 'test3.xlsx'
    create_excel(str(p))
    out = write_cell(str(p), 'ZZZ', 5)
    assert 'error' in out
