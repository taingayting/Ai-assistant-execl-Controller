import os

import app as app_module
from excel_controller import create_excel, add_row
from openpyxl import load_workbook


def test_execute_requires_confirmation_and_confirm_then_undo(client, tmp_path):
    files_dir = app_module.FILES_DIR
    os.makedirs(files_dir, exist_ok=True)
    fname = 'itest_confirm.xlsx'
    fpath = os.path.join(files_dir, fname)

    # create workbook with 3 columns and a couple rows
    create_excel(fpath, headers=['H1', 'H2', 'H3'])
    add_row(fpath, ['a', 'b', 'c'])
    add_row(fpath, ['1', '2', '3'])

    # Send execute request for delete_column without confirming
    action = {"type": "delete_column", "column": "B"}
    resp = client.post('/api/execute', json={"action": action, "filename": fname})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('requires_confirmation') is True
    cid = data.get('confirmation_id')
    assert cid

    # Confirm the action
    resp2 = client.post('/api/confirm', json={"confirmation_id": cid, "confirm": True})
    assert resp2.status_code == 200
    d2 = resp2.get_json()
    assert d2.get('success') is True
    # result should indicate column_deleted
    assert isinstance(d2.get('result'), dict)
    assert d2['result'].get('message') == 'column_deleted'

    # Check workbook column count decreased (was 3 -> now 2)
    wb = load_workbook(fpath)
    ws = wb.active
    assert ws.max_column == 2

    # Now call undo endpoint to restore from .bak
    resp3 = client.post('/api/undo', json={"filename": fname})
    assert resp3.status_code == 200
    d3 = resp3.get_json()
    assert d3.get('success') is True

    # After undo, workbook should have original 3 columns
    wb2 = load_workbook(fpath)
    ws2 = wb2.active
    assert ws2.max_column == 3
