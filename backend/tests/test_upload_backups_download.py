"""Tests for POST /api/upload-audio validation + GET /api/backups + file download."""
import io
import os
from unittest.mock import patch

import app as app_module
from excel_controller import create_excel


# ---- /api/upload-audio validation ----

def test_upload_audio_no_file(client):
    resp = client.post('/api/upload-audio', data={}, content_type='multipart/form-data')
    assert resp.status_code == 400
    assert 'No file' in resp.get_json()['error']


def test_upload_audio_bad_extension(client):
    resp = client.post('/api/upload-audio', data={
        'file': (io.BytesIO(b'data'), 'malicious.exe'),
    }, content_type='multipart/form-data')
    assert resp.status_code == 400
    assert 'not supported' in resp.get_json()['error']


def test_upload_audio_bad_extension_py(client):
    resp = client.post('/api/upload-audio', data={
        'file': (io.BytesIO(b'data'), 'script.py'),
    }, content_type='multipart/form-data')
    assert resp.status_code == 400


def test_upload_audio_unknown_action_includes_suggestions(client):
    with patch('app.transcribe_audio_file', return_value='do something weird and random'), \
         patch('agent.route_llm', return_value={'type': 'unknown', 'raw': 'do something weird and random'}):
        resp = client.post('/api/upload-audio', data={
            'file': (io.BytesIO(b'fake'), 'test.wav'),
            'use_llm': 'false',
            'auto_execute': 'false',
        }, content_type='multipart/form-data')
        j = resp.get_json()
        assert resp.status_code == 200
        assert j['action']['type'] == 'unknown'
        assert 'suggestions' in j


# ---- GET /api/backups ----

def test_backups_empty(client):
    resp = client.get('/api/backups?filename=nonexistent.xlsx')
    j = resp.get_json()
    assert resp.status_code == 200
    assert j['checkpoints'] == []
    assert j['total_bytes'] == 0
    assert 'budget_bytes' in j


def test_backups_with_checkpoints(client):
    fname = 'test_bak_info.xlsx'
    fpath = os.path.join(app_module.FILES_DIR, fname)
    create_excel(fpath, headers=['A'])

    # execute twice to create backups
    client.post('/api/execute', json={
        "action": {"type": "add_row", "values": ["x"]},
        "filename": fname,
    })
    client.post('/api/execute', json={
        "action": {"type": "add_row", "values": ["y"]},
        "filename": fname,
    })

    resp = client.get(f'/api/backups?filename={fname}')
    j = resp.get_json()
    assert j['success'] is True
    assert len(j['checkpoints']) >= 1
    assert j['total_bytes'] > 0

    # cleanup
    for i in range(6):
        for f in [fpath, f"{fpath}.bak", f"{fpath}.bak.{i}"]:
            if os.path.exists(f):
                os.remove(f)


def test_backups_default_filename(client):
    """When no filename param, default 'data.xlsx' is used."""
    resp = client.get('/api/backups')
    j = resp.get_json()
    assert j['filename'] == 'data.xlsx'


# ---- GET /files/<filename> ----

def test_download_existing_file(client):
    fname = 'test_dl.xlsx'
    fpath = os.path.join(app_module.FILES_DIR, fname)
    create_excel(fpath)

    resp = client.get(f'/files/{fname}')
    assert resp.status_code == 200
    assert len(resp.data) > 0
    # skip cleanup to avoid Windows file-lock errors; file is in FILES_DIR


def test_download_missing_file(client):
    resp = client.get('/files/does_not_exist_12345.xlsx')
    assert resp.status_code == 404
