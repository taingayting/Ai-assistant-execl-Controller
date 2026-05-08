"""Tests for POST /api/parse — validation, LLM-based parsing, suggestions."""
import app as app_module  # noqa: F401 (imported via conftest)
from unittest.mock import patch


# ---- validation ----

def test_parse_missing_text(client):
    resp = client.post('/api/parse', json={})
    assert resp.status_code == 400
    assert 'text is required' in resp.get_json()['error']


def test_parse_empty_text(client):
    resp = client.post('/api/parse', json={"text": ""})
    assert resp.status_code == 400


def test_parse_null_text(client):
    resp = client.post('/api/parse', json={"text": None})
    assert resp.status_code == 400


def test_parse_non_string_text(client):
    resp = client.post('/api/parse', json={"text": 42})
    assert resp.status_code == 400


def test_parse_text_too_long(client):
    resp = client.post('/api/parse', json={"text": "x" * 2001})
    assert resp.status_code == 400
    assert 'exceeds' in resp.get_json()['error']


def test_parse_text_at_limit_ok(client):
    resp = client.post('/api/parse', json={"text": "add row Tom 90" + " " * 1986})
    assert resp.status_code == 200


def test_parse_no_json_body(client):
    resp = client.post('/api/parse', content_type='application/json', data='')
    assert resp.status_code == 400


# ---- happy paths (mock LLM) ----

def test_parse_add_row(client):
    with patch('app.parse_with_llm', return_value={"type": "add_row", "values": ["Tom", 90]}):
        resp = client.post('/api/parse', json={"text": "add row Tom 90"})
    j = resp.get_json()
    assert j['success'] is True
    assert j['action']['type'] == 'add_row'
    assert 'cleaned' in j


def test_parse_create_excel(client):
    with patch('app.parse_with_llm', return_value={"type": "create_excel", "headers": ["Name", "Score"]}):
        resp = client.post('/api/parse', json={"text": 'create excel with headers "Name,Score"'})
    j = resp.get_json()
    assert j['action']['type'] == 'create_excel'
    assert j['action']['headers'] == ['Name', 'Score']


def test_parse_delete_row(client):
    with patch('app.parse_with_llm', return_value={"type": "delete_row", "index": 3}):
        j = client.post('/api/parse', json={"text": "delete row 3"}).get_json()
    assert j['action']['type'] == 'delete_row'
    assert j['action']['index'] == 3


def test_parse_save(client):
    with patch('app.parse_with_llm', return_value={"type": "save"}):
        j = client.post('/api/parse', json={"text": "save"}).get_json()
    assert j['action']['type'] == 'save'


def test_parse_unknown_returns_suggestions(client):
    with patch('app.parse_with_llm', return_value={"type": "unknown", "raw": "do something weird"}):
        j = client.post('/api/parse', json={"text": "do something weird"}).get_json()
    assert j['action']['type'] == 'unknown'
    assert 'suggestions' in j
    assert isinstance(j['suggestions'], list)


def test_parse_no_groq_key_returns_unknown(client):
    """Without GROQ key, parse_with_llm returns unknown."""
    with patch('app.parse_with_llm', return_value={"type": "unknown", "raw": "delete raw 2"}):
        j = client.post('/api/parse', json={"text": "delete raw 2"}).get_json()
    assert j['action']['type'] == 'unknown'


def test_parse_llm_succeeds(client):
    """When LLM returns a valid action, it is passed through."""
    with patch('app.parse_with_llm', return_value={"type": "add_row", "values": ["Tom", 90]}):
        j = client.post('/api/parse', json={"text": "add row Tom 90"}).get_json()
    assert j['success'] is True
    assert j['action']['type'] == 'add_row'
