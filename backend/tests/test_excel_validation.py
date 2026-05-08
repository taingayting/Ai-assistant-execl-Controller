"""Tests for excel_controller validation paths + execute_action dispatch."""
from excel_controller import (
    create_excel, add_row, delete_row, delete_column,
    add_column, write_cell, calculate_row_sum, save,
    execute_action,
)


# ---- create_excel ----

def test_create_excel_headers_not_list(tmp_path):
    p = str(tmp_path / 'bad.xlsx')
    r = create_excel(p, headers="NotAList")
    assert 'error' in r


# ---- add_row ----

def test_add_row_values_not_list(tmp_path):
    p = str(tmp_path / 'ar.xlsx')
    create_excel(p)
    r = add_row(p, "not a list")
    assert 'error' in r


def test_add_row_happy(tmp_path):
    p = str(tmp_path / 'ar2.xlsx')
    create_excel(p, headers=['A'])
    r = add_row(p, ['val'])
    assert r['message'] == 'row_added'


# ---- delete_row ----

def test_delete_row_non_integer_index(tmp_path):
    p = str(tmp_path / 'dr.xlsx')
    create_excel(p, headers=['A'])
    r = delete_row(p, "abc")
    assert 'error' in r


def test_delete_row_out_of_range_low(tmp_path):
    p = str(tmp_path / 'dr2.xlsx')
    create_excel(p, headers=['A'])
    r = delete_row(p, 0)
    assert 'error' in r
    assert r['error'] == 'index_out_of_range'


def test_delete_row_out_of_range_high(tmp_path):
    p = str(tmp_path / 'dr3.xlsx')
    create_excel(p, headers=['A'])
    r = delete_row(p, 99)
    assert 'error' in r


def test_delete_row_happy(tmp_path):
    p = str(tmp_path / 'dr4.xlsx')
    create_excel(p, headers=['A'])
    add_row(p, ['v1'])
    r = delete_row(p, 2)
    assert r['message'] == 'row_deleted'


# ---- delete_column ----

def test_delete_column_none(tmp_path):
    p = str(tmp_path / 'dc.xlsx')
    create_excel(p, headers=['A', 'B'])
    r = delete_column(p, None)
    assert 'error' in r


def test_delete_column_out_of_range(tmp_path):
    p = str(tmp_path / 'dc2.xlsx')
    create_excel(p, headers=['A'])
    r = delete_column(p, 'Z')
    assert r['error'] == 'column_out_of_range'


def test_delete_column_invalid_string(tmp_path):
    p = str(tmp_path / 'dc3.xlsx')
    create_excel(p, headers=['A'])
    r = delete_column(p, '!!')
    assert 'error' in r


# ---- add_column ----

def test_add_column_empty_name(tmp_path):
    p = str(tmp_path / 'ac.xlsx')
    create_excel(p)
    r = add_column(p, '')
    assert 'error' in r


def test_add_column_non_string_name(tmp_path):
    p = str(tmp_path / 'ac2.xlsx')
    create_excel(p)
    r = add_column(p, 123)
    assert 'error' in r


def test_add_column_happy(tmp_path):
    p = str(tmp_path / 'ac3.xlsx')
    create_excel(p, headers=['A'])
    r = add_column(p, 'B')
    assert r['message'] == 'column_added'


def test_add_column_with_default(tmp_path):
    p = str(tmp_path / 'ac4.xlsx')
    create_excel(p, headers=['A'])
    add_row(p, ['v1'])
    r = add_column(p, 'B', default_value=0)
    assert r['message'] == 'column_added'


# ---- write_cell ----

def test_write_cell_empty(tmp_path):
    p = str(tmp_path / 'wc.xlsx')
    create_excel(p)
    r = write_cell(p, '', 5)
    assert 'error' in r


def test_write_cell_none(tmp_path):
    p = str(tmp_path / 'wc2.xlsx')
    create_excel(p)
    r = write_cell(p, None, 5)
    assert 'error' in r


# ---- calculate_row_sum ----

def test_row_sum_non_integer_index(tmp_path):
    p = str(tmp_path / 'rs.xlsx')
    create_excel(p, headers=['A'])
    r = calculate_row_sum(p, "abc")
    assert 'error' in r


def test_row_sum_out_of_range(tmp_path):
    p = str(tmp_path / 'rs2.xlsx')
    create_excel(p, headers=['A'])
    r = calculate_row_sum(p, 99)
    assert r['error'] == 'row_out_of_range'


def test_row_sum_no_numeric(tmp_path):
    p = str(tmp_path / 'rs3.xlsx')
    create_excel(p, headers=['Name'])
    add_row(p, ['Alice'])
    r = calculate_row_sum(p, 2)
    assert r['sum'] == 0.0
    assert 'no numeric' in r['speech_text'].lower()


# ---- save ----

def test_save_noop(tmp_path):
    p = str(tmp_path / 's.xlsx')
    r = save(p)
    assert r['message'] == 'saved'


# ---- execute_action dispatch ----

def test_execute_action_invalid_action(tmp_path):
    p = str(tmp_path / 'ea.xlsx')
    r = execute_action("not a dict", p)
    assert 'error' in r


def test_execute_action_no_type(tmp_path):
    p = str(tmp_path / 'ea2.xlsx')
    r = execute_action({"values": [1]}, p)
    assert 'error' in r


def test_execute_action_unknown_type(tmp_path):
    p = str(tmp_path / 'ea3.xlsx')
    r = execute_action({"type": "dance"}, p)
    assert r['error'] == 'unknown_action'


def test_execute_action_save_dispatch(tmp_path):
    p = str(tmp_path / 'ea4.xlsx')
    r = execute_action({"type": "save"}, p)
    assert r['message'] == 'saved'


def test_execute_action_write_cell_dispatch(tmp_path):
    p = str(tmp_path / 'ea5.xlsx')
    create_excel(p)
    r = execute_action({"type": "write_cell", "cell": "A1", "value": 10}, p)
    assert r['message'] == 'cell_written'


def test_execute_action_insert_cell_alias(tmp_path):
    p = str(tmp_path / 'ea6.xlsx')
    create_excel(p)
    r = execute_action({"type": "insert_cell", "cell": "B1", "value": "hi"}, p)
    assert r['message'] == 'cell_written'


def test_execute_action_row_sum_dispatch(tmp_path):
    p = str(tmp_path / 'ea7.xlsx')
    create_excel(p, headers=['A'])
    add_row(p, [42])
    r = execute_action({"type": "calculate_row_sum", "row": 2}, p)
    assert r['message'] == 'row_sum'
