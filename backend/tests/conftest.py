"""Shared fixtures and path setup for all backend tests."""
import os
import sys
import pathlib

import pytest

# ---- sys.path configuration ----
# Add backend/ so top-level imports (command_parser, llm_parser, etc.) resolve.
HERE = pathlib.Path(__file__).resolve().parent
BACKEND_DIR = str(HERE.parent)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
# Add project root so wakeword/ package is importable.
PROJECT_ROOT = str(HERE.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app as app_module  # noqa: E402


# ---- shared fixtures ----

@pytest.fixture
def client():
    """Flask test client used by all endpoint tests."""
    app_module.app.testing = True
    yield app_module.app.test_client()


@pytest.fixture(autouse=True)
def _ensure_runtime_dirs():
    """Guarantee FILES_DIR and CONF_DIR exist before every test."""
    os.makedirs(app_module.FILES_DIR, exist_ok=True)
    os.makedirs(app_module.CONF_DIR, exist_ok=True)
