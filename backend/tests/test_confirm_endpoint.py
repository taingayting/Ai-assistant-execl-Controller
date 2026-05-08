"""Tests for POST /api/confirm — validation, cancel, expired, happy path."""
import os

import app as app_module
from excel_controller import create_excel, add_row


# ---- validation ----

def test_confirm_missing_cid(client):
    resp = client.post('/api/confirm', json={})
    assert resp.status_code == 400
    assert 'confirmation_id' in resp.get_json()['error'].lower()


def test_confirm_non_string_cid(client):
    resp = client.post('/api/confirm', json={"confirmation_id": 123})
    assert resp.status_code == 400


def test_confirm_bad_uuid_format(client):
    resp = client.post('/api/confirm', json={"confirmation_id": "not-a-uuid"})
    assert resp.status_code == 400
    assert 'Invalid' in resp.get_json()['error']


def test_confirm_path_traversal_blocked(client):
    """A crafted id with path chars must be rejected by UUID format check."""
    resp = client.post('/api/confirm', json={"confirmation_id": "../../../etc/passwd" + "x" * 20})
    assert resp.status_code == 400


def test_confirm_unknown_cid(client):
    resp = client.post('/api/confirm', json={
        "confirmation_id": "00000000-0000-0000-0000-000000000000",
        "confirm": True,
    })
    assert resp.status_code == 404
    assert 'expired' in resp.get_json()['error'].lower() or 'unknown' in resp.get_json()['error'].lower()


# ---- cancel ----

def test_confirm_cancel(client):
    fname = 'test_conf_cancel.xlsx'
    fpath = os.path.join(app_module.FILES_DIR, fname)
    create_excel(fpath, headers=['A', 'B'])
    add_row(fpath, ['1', '2'])

    # trigger confirmation
    resp = client.post('/api/execute', json={
        "action": {"type": "delete_row", "index": 2},
        "filename": fname,
    })
    cid = resp.get_json()['confirmation_id']

    # cancel
    resp2 = client.post('/api/confirm', json={"confirmation_id": cid, "confirm": False})
    j = resp2.get_json()
    assert resp2.status_code == 200
    assert j['canceled'] is True

    # cleanup
    for f in [fpath, f"{fpath}.bak", f"{fpath}.bak.1"]:
        if os.path.exists(f):
            os.remove(f)
