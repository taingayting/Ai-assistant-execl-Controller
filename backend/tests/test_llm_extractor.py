import pytest
from llm_parser import RobustJsonExtractor


def test_extract_fenced_json():
    raw = 'Here is the result:\n```json\n{"action":"add_row","values":["Tom",90]}\n```'
    out = RobustJsonExtractor.extract(raw)
    assert out == {"action": "add_row", "values": ["Tom", 90]}


def test_extract_inline_braces():
    raw = 'Response: {"action":"delete_column","column":"B"} -- done'
    out = RobustJsonExtractor.extract(raw)
    assert out["action"] == "delete_column"


def test_fix_python_literals_and_trailing_commas():
    raw = '```json\n{\n  "action": "calculate",\n  "value": True,\n}\n```'
    out = RobustJsonExtractor.extract(raw)
    # True in Python becomes true in JSON and loads back to True
    assert out["value"] is True


def test_single_quotes_coercion():
    raw = "{'action':'add_row','values':['Alice', '10']}"
    out = RobustJsonExtractor.extract(raw)
    assert out["action"] == "add_row"


def test_no_json_raises():
    with pytest.raises(ValueError):
        RobustJsonExtractor.extract('nothing to see here')
