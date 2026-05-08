"""Tests for llm_parser: _parse_groq_direct, parse_with_llm delegation + generate_suggestions heuristics."""
from unittest.mock import patch, MagicMock
from llm_parser import (
    parse_with_llm,
    _parse_groq_direct,
    generate_suggestions,
    build_intermediate_plan,
    validate_execution_plan,
)


# ---- parse_with_llm delegates to router ----

def test_parse_with_llm_delegates_to_router():
    """parse_with_llm should delegate to llm_router.route_llm."""
    with patch('llm_router.route_llm', return_value={"type": "add_row", "values": ["X"]}) as mock:
        r = parse_with_llm('add row X')
    mock.assert_called_once_with('add row X')
    assert r == {"type": "add_row", "values": ["X"]}


# ---- _parse_groq_direct (Groq-specific logic) ----

def test_parse_groq_direct_no_key_returns_unknown():
    """Without GROQ_API_KEY, should return unknown action."""
    with patch('llm_parser.GROQ_API_KEY', None):
        r = _parse_groq_direct('add row Tom 90')
    assert r['type'] == 'unknown'


def test_parse_groq_direct_success():
    """With a key, a mocked Groq 200 with valid JSON should be extracted."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"type":"add_row","values":["Sara",95]}'}}]
    }

    with patch('llm_parser.GROQ_API_KEY', 'fake-key'), \
         patch('llm_parser.requests.post', return_value=mock_resp):
        r = _parse_groq_direct('add row Sara 95')
    assert 'steps' in r
    assert r['steps'][0]['type'] == 'create_excel'
    assert r['steps'][1]['type'] == 'add_row'
    assert r['steps'][1]['values'] == ['Sara', 95]


def test_parse_groq_direct_failure_returns_unknown():
    """If Groq request throws, should return unknown action."""
    with patch('llm_parser.GROQ_API_KEY', 'fake-key'), \
         patch('llm_parser.requests.post', side_effect=Exception('timeout')):
        r = _parse_groq_direct('delete row 3')
    assert r['type'] == 'unknown'


def test_parse_groq_direct_bad_json_returns_unknown():
    """If Groq returns non-JSON content, should return unknown."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "I cannot help with that."}}]
    }
    mock_resp.text = "I cannot help with that."

    with patch('llm_parser.GROQ_API_KEY', 'fake-key'), \
         patch('llm_parser.requests.post', return_value=mock_resp):
        r = _parse_groq_direct('add column score')
    assert r['type'] == 'unknown'


# ---- generate_suggestions ----

def test_suggestions_empty_text():
    r = generate_suggestions('')
    assert isinstance(r, list)
    assert len(r) >= 3
    assert 'create excel' in r


def test_suggestions_with_create_keyword():
    r = generate_suggestions('create something')
    assert 'create excel' in r


def test_suggestions_with_column_keyword():
    r = generate_suggestions('add column name')
    assert any('column' in s for s in r)


def test_suggestions_with_row_keyword():
    r = generate_suggestions('add row')
    assert any('row' in s for s in r)


def test_suggestions_max_length():
    r = generate_suggestions('some random text')
    assert len(r) <= 5


def test_build_intermediate_plan_for_score_rows():
    action = {
        'steps': [
            {'type': 'create_excel', 'headers': ['Name', 'Score']},
            {'type': 'add_row', 'values': ['Tom', 90]},
            {'type': 'add_row', 'values': ['Alex', 85]},
        ]
    }
    plan = build_intermediate_plan(action, 'Create a score table and add Tom 90 and Alex 85')
    assert plan['domain'] == 'student_scores'
    assert plan['headers'] == ['Name', 'Score']
    assert plan['rows'] == [['Tom', 90], ['Alex', 85]]


def test_validate_execution_plan_inserts_create_before_add():
    plan = {
        'domain': 'student_scores',
        'steps': [
            {'type': 'add_row', 'values': ['Tom', 90]},
        ],
    }
    repaired = validate_execution_plan(plan)
    assert repaired['steps'][0] == {'type': 'create_excel', 'headers': ['Name', 'Score']}
    assert repaired['steps'][1] == {'type': 'add_row', 'values': ['Tom', 90]}


def test_validate_execution_plan_normalizes_numeric_and_missing_values():
    plan = {
        'domain': 'budget_tracking',
        'headers': ['Category', 'Budget', 'Actual'],
        'steps': [
            {'type': 'create_excel', 'headers': ['Category', 'Budget', 'Actual']},
            {'type': 'add_row', 'values': ['Marketing', '5000']},
            {'type': 'add_row', 'values': ['Ops', 'oops', None]},
        ],
    }
    repaired = validate_execution_plan(plan)
    assert repaired['steps'][1]['values'] == ['Marketing', 5000, 0]
    assert repaired['steps'][2]['values'] == ['Ops', 0, 0]


def test_validate_execution_plan_deduplicates_create_steps():
    plan = {
        'domain': 'inventory',
        'steps': [
            {'type': 'create_excel', 'headers': ['Item', 'Quantity']},
            {'type': 'create_excel', 'headers': ['Item', 'Quantity']},
            {'type': 'add_row', 'values': ['Keyboard', 10]},
        ],
    }
    repaired = validate_execution_plan(plan)
    create_steps = [s for s in repaired['steps'] if s.get('type') == 'create_excel']
    assert len(create_steps) == 1
