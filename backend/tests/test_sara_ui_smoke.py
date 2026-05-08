"""Minimal smoke tests for desktop Sara UI.

These tests are intentionally lightweight and skip automatically when PyQt6
is not installed in the active environment.
"""

import pytest
import importlib.util

HAS_PYQT6 = importlib.util.find_spec("PyQt6") is not None

pytestmark = pytest.mark.skipif(not HAS_PYQT6, reason="PyQt6 not installed")


def test_ui_constants_present():
    from desktop import sara_ui
    assert sara_ui.MAX_HISTORY_ITEMS == 10


def test_upload_result_dataclass_defaults():
    from desktop import sara_ui
    result = sara_ui.UploadResult(success=True)
    assert result.success is True
    assert result.transcript == ""
    assert result.ai_message == ""
