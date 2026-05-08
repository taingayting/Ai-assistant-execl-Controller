"""Tests for POST /api/undo — validation, missing backup, multi-step."""
import os

import app as app_module
from excel_controller import create_excel, add_row
from openpyxl import load_workbook


# ---- validation ----

def test_undo_non_integer_steps(client):
    resp = client.post('/api/undo', json={"filename": "f.xlsx", "steps": "abc"})
    assert resp.status_code == 400
    assert 'integer' in resp.get_json()['error'].lower()


def test_undo_steps_zero(client):
    resp = client.post('/api/undo', json={"filename": "f.xlsx", "steps": 0})
    assert resp.status_code == 400


def test_undo_steps_too_large(client):
    resp = client.post('/api/undo', json={"filename": "f.xlsx", "steps": 999})
    assert resp.status_code == 400


def test_undo_missing_backup(client):
    fname = 'test_undo_missing.xlsx'
    fpath = os.path.join(app_module.FILES_DIR, fname)
    create_excel(fpath)

    resp = client.post('/api/undo', json={"filename": fname, "steps": 3})
    assert resp.status_code == 404
    assert 'checkpoint' in resp.get_json()['error'].lower() or 'No' in resp.get_json()['error']

    if os.path.exists(fpath):
        os.remove(fpath)


# ---- happy path with multi-step ----

def test_undo_multi_step(client):
    """Execute two mutations, then undo to step 2 to get original state."""
    fname = 'test_undo_multi.xlsx'
    fpath = os.path.join(app_module.FILES_DIR, fname)
    create_excel(fpath, headers=['H1'])

    # First mutation: add row → creates bak.1 (original state)
    client.post('/api/execute', json={
        "action": {"type": "add_row", "values": ["first"]},
        "filename": fname,
    })

    # Second mutation: add another row → rotates bak.1→bak.2, creates new bak.1
    client.post('/api/execute', json={
        "action": {"type": "add_row", "values": ["second"]},
        "filename": fname,
    })

    # Undo step 2 → should restore original (just headers)
    resp = client.post('/api/undo', json={"filename": fname, "steps": 2})
    j = resp.get_json()
    assert resp.status_code == 200
    assert j['success'] is True
    assert 'checkpoint 2' in j['message']

    wb = load_workbook(fpath)
    ws = wb.active
    assert ws.max_row == 1  # only header row

    # cleanup
    for i in range(6):
        for f in [fpath, f"{fpath}.bak", f"{fpath}.bak.{i}"]:
            if os.path.exists(f):
                os.remove(f)
