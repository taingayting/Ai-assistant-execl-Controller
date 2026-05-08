"""Tests for POST /api/execute — validation, immediate execution, confirmation flow."""
import os
import pytest

import app as app_module
from excel_controller import create_excel, add_row
from memory import memory


@pytest.fixture(autouse=True)
def _clear_memory():
    memory.clear()
    yield
    memory.clear()


# ---- validation ----

def test_execute_missing_action(client):
    resp = client.post('/api/execute', json={"filename": "f.xlsx"})
    assert resp.status_code == 400
    assert 'action' in resp.get_json()['error'].lower()


def test_execute_action_not_dict(client):
    resp = client.post('/api/execute', json={"action": "string"})
    assert resp.status_code == 400


def test_execute_action_missing_type(client):
    resp = client.post('/api/execute', json={"action": {"values": [1]}})
    assert resp.status_code == 400
    assert 'type' in resp.get_json()['error'].lower()


def test_execute_disallowed_extension(client):
    resp = client.post('/api/execute', json={
        "action": {"type": "add_row", "values": ["a"]},
        "filename": "data.txt",
    })
    assert resp.status_code == 400
    assert 'not allowed' in resp.get_json()['error']


def test_execute_disallowed_extension_pdf(client):
    resp = client.post('/api/execute', json={
        "action": {"type": "add_row", "values": ["a"]},
        "filename": "data.pdf",
    })
    assert resp.status_code == 400


# ---- happy paths ----

def test_execute_non_destructive_immediate(client):
    fname = 'test_exec_imm.xlsx'
    fpath = os.path.join(app_module.FILES_DIR, fname)
    create_excel(fpath, headers=['A', 'B'])

    resp = client.post('/api/execute', json={
        "action": {"type": "add_row", "values": ["x", "y"]},
        "filename": fname,
    })
    j = resp.get_json()
    assert resp.status_code == 200
    assert j['success'] is True
    assert j['result']['message'] == 'row_added'
    assert 'download_url' in j
    # should NOT have requires_confirmation
    assert 'requires_confirmation' not in j

    # cleanup
    for f in [fpath] + [f"{fpath}.bak", f"{fpath}.bak.1"]:
        if os.path.exists(f):
            os.remove(f)


def test_execute_destructive_needs_confirmation(client):
    fname = 'test_exec_destr.xlsx'
    fpath = os.path.join(app_module.FILES_DIR, fname)
    create_excel(fpath, headers=['H1', 'H2'])
    add_row(fpath, ['a', 'b'])

    resp = client.post('/api/execute', json={
        "action": {"type": "delete_row", "index": 1},
        "filename": fname,
    })
    j = resp.get_json()
    assert j['requires_confirmation'] is True
    assert 'confirmation_id' in j
    assert 'message' in j

    # cleanup
    for f in [fpath] + [f"{fpath}.bak", f"{fpath}.bak.1"]:
        if os.path.exists(f):
            os.remove(f)


def test_execute_destructive_with_confirmed_flag(client):
    fname = 'test_exec_conf.xlsx'
    fpath = os.path.join(app_module.FILES_DIR, fname)
    create_excel(fpath, headers=['H1', 'H2'])
    add_row(fpath, ['a', 'b'])

    resp = client.post('/api/execute', json={
        "action": {"type": "delete_row", "index": 2},
        "filename": fname,
        "confirmed": True,
    })
    j = resp.get_json()
    assert resp.status_code == 200
    assert j['success'] is True
    assert j['result']['message'] == 'row_deleted'
    # no confirmation asked
    assert 'requires_confirmation' not in j

    # cleanup
    for f in [fpath] + [f"{fpath}.bak", f"{fpath}.bak.1"]:
        if os.path.exists(f):
            os.remove(f)


def test_execute_create_excel(client):
    fname = 'test_exec_create.xlsx'
    fpath = os.path.join(app_module.FILES_DIR, fname)

    resp = client.post('/api/execute', json={
        "action": {"type": "create_excel", "headers": ["Name", "Age"]},
        "filename": fname,
    })
    j = resp.get_json()
    assert j['success'] is True
    assert j['result']['message'] == 'created'

    if os.path.exists(fpath):
        os.remove(fpath)


def test_execute_speech_text_propagated(client):
    """write_cell returns speech_text; verify it appears in API response."""
    fname = 'test_exec_speech.xlsx'
    fpath = os.path.join(app_module.FILES_DIR, fname)
    create_excel(fpath)

    resp = client.post('/api/execute', json={
        "action": {"type": "write_cell", "cell": "A1", "value": 42},
        "filename": fname,
    })
    j = resp.get_json()
    assert 'speech_text' in j

    for f in [fpath] + [f"{fpath}.bak", f"{fpath}.bak.1"]:
        if os.path.exists(f):
            os.remove(f)


def test_execute_precondition_recovery_hint(client):
    resp = client.post('/api/execute', json={
        "action": {"type": "add_column", "name": "price"},
    })
    j = resp.get_json()
    assert resp.status_code == 400
    assert "No active file" in j.get('error', '')
    assert j.get('speech_text') == "No Excel file is open. Would you like me to create a new file?"
    assert 'recovery' in j


def test_execute_auto_correct_missing_file(client):
    fname = 'data.xlsx'
    fpath = os.path.join(app_module.FILES_DIR, fname)
    if os.path.exists(fpath):
        os.remove(fpath)

    resp = client.post('/api/execute', json={
        "action": {"type": "add_column", "name": "price"},
        "auto_correct": True,
    })
    j = resp.get_json()
    assert resp.status_code == 200
    assert j.get('success') is True
    assert j.get('executed') is True
    assert j.get('auto_corrected') is True
    assert os.path.exists(fpath)
    if os.path.exists(fpath):
        os.remove(fpath)
