"""Web UI tests for Sara using Playwright (sync API).

Tests the frontend served by Flask — verifying page load, DOM elements,
action preview, suggestions panel, workflow grid, and execute flow via
programmatic interactions.

Automatically skipped when ``playwright`` is not installed or when the Flask
server cannot be reached.

Prerequisites:
    pip install playwright pytest-playwright
    playwright install chromium

Run:
    pytest backend/tests/test_ui_web.py -v -m ui
"""
from __future__ import annotations

import importlib.util
import os
import sys
import pathlib
import time
import threading

import pytest

# ── path setup ──────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent
BACKEND_DIR = str(HERE.parent)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
PROJECT_ROOT = str(HERE.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HAS_PLAYWRIGHT = importlib.util.find_spec("playwright") is not None

pytestmark = [
    pytest.mark.ui,
    pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed"),
]

# ── Flask test server fixture ───────────────────────────────────

@pytest.fixture(scope="module")
def live_server():
    """Start a Flask development server in a background thread for Playwright tests.

    Yields the base URL (e.g. ``http://127.0.0.1:5099``).
    """
    import app as app_module

    port = 5099
    app_module.app.testing = True

    server_thread = threading.Thread(
        target=lambda: app_module.app.run(
            host="127.0.0.1", port=port, use_reloader=False, threaded=True,
        ),
        daemon=True,
    )
    server_thread.start()

    # Wait for server to be ready
    import requests as req
    base = f"http://127.0.0.1:{port}"
    for _ in range(40):
        try:
            resp = req.get(base, timeout=1)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.25)
    else:
        pytest.skip("Could not start Flask test server")

    yield base


@pytest.fixture
def browser_page(live_server):
    """Launch a headless Chromium page pointing at the live server."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(live_server, wait_until="networkidle")
        yield page
        context.close()
        browser.close()


# ── Page load tests ─────────────────────────────────────────────

class TestPageLoad:

    def test_title(self, browser_page):
        assert "Smart Office Assistant" in browser_page.title()

    def test_header_visible(self, browser_page):
        header = browser_page.locator(".site-header")
        assert header.is_visible()
        assert "Smart Office Assistant" in header.inner_text()


# ── DOM element existence ───────────────────────────────────────

class TestDomElements:

    def test_start_rec_button(self, browser_page):
        btn = browser_page.locator("#startRec")
        assert btn.is_visible()
        assert btn.is_enabled()

    def test_stop_rec_button_disabled_initially(self, browser_page):
        btn = browser_page.locator("#stopRec")
        assert btn.is_visible()
        assert btn.is_disabled()

    def test_transcript_textarea(self, browser_page):
        ta = browser_page.locator("#transcriptText")
        assert ta.is_visible()

    def test_send_transcript_button(self, browser_page):
        btn = browser_page.locator("#sendTranscript")
        assert btn.is_visible()

    def test_execute_button_disabled(self, browser_page):
        btn = browser_page.locator("#executeParsed")
        assert btn.is_disabled()

    def test_suggestion_section(self, browser_page):
        el = browser_page.locator("#suggestionsList")
        assert el.is_visible()
        assert "No suggestions" in el.inner_text()

    def test_action_preview(self, browser_page):
        el = browser_page.locator("#actionPreview")
        assert el.is_visible()

    def test_workflow_grid(self, browser_page):
        el = browser_page.locator("#workflowGrid")
        assert el.is_visible()

    def test_metrics_section(self, browser_page):
        el = browser_page.locator("#metricsSection")
        assert el.is_visible()

    def test_undo_button(self, browser_page):
        btn = browser_page.locator("#undoBtn")
        assert btn.is_visible()

    def test_confirm_modal_hidden(self, browser_page):
        modal = browser_page.locator("#confirmModal")
        assert modal.get_attribute("aria-hidden") == "true"

    def test_language_selector(self, browser_page):
        sel = browser_page.locator("#speechLang")
        assert sel.is_visible()
        # Should have multiple language options
        options = sel.locator("option").count()
        assert options >= 5

    def test_filename_input(self, browser_page):
        inp = browser_page.locator("#recFilename")
        assert inp.input_value() == "data.xlsx"


# ── API interaction via JS ──────────────────────────────────────

class TestApiInteraction:
    """Test frontend interactions that trigger API calls."""

    def test_parse_via_transcript(self, browser_page):
        """Type a command, click Parse Transcript, verify action preview updates."""
        ta = browser_page.locator("#transcriptText")
        ta.fill("add row Tom 90")

        # Click Parse Transcript — this triggers confirm dialog first
        # We need to handle the confirm modal
        parse_btn = browser_page.locator("#sendTranscript")

        # The sendTranscript handler shows a confirm modal; auto-confirm it
        browser_page.locator("#sendTranscript").click()
        # Wait for confirm modal to appear
        modal = browser_page.locator("#confirmModal")
        try:
            modal.wait_for(state="visible", timeout=3000)
            browser_page.locator("#modalConfirm").click()
        except Exception:
            pass  # modal may not appear in all configurations

        # Wait for action preview to be updated
        time.sleep(2)
        preview = browser_page.locator("#actionPreview")
        text = preview.inner_text()
        # Should no longer be the default text
        assert text != "Speak or type a command, then click Send." or "add" in text.lower()

    def test_workflow_grid_populated(self, browser_page):
        """The workflow grid should load cards from the backend."""
        # Wait for workflows to load
        time.sleep(2)
        grid = browser_page.locator("#workflowGrid")
        cards = grid.locator(".wf-card")
        # Should have at least a few workflow cards
        count = cards.count()
        assert count >= 1, f"Expected at least 1 workflow card, got {count}"

    def test_workflow_card_click_fills_transcript(self, browser_page):
        """Clicking a workflow card should fill the transcript textarea."""
        time.sleep(1)
        cards = browser_page.locator(".wf-card")
        if cards.count() > 0:
            cards.first.click()
            time.sleep(0.5)
            ta = browser_page.locator("#transcriptText")
            val = ta.input_value()
            assert len(val) > 0, "Transcript should be filled after clicking a workflow card"

    def test_metrics_refresh(self, browser_page):
        """Clicking Refresh on metrics should not crash."""
        btn = browser_page.locator("#refreshMetrics")
        btn.click()
        time.sleep(1)
        # At least the commands count should be a number
        cmd_el = browser_page.locator("#mCommands")
        text = cmd_el.inner_text()
        assert text != "-" or text.isdigit(), f"Metrics didn't load: {text}"

    def test_clear_history(self, browser_page):
        """Clicking Clear History should work without error."""
        btn = browser_page.locator("#clearHistory")
        btn.click()
        time.sleep(0.5)
        hist = browser_page.locator("#commandHistory")
        assert "No history" in hist.inner_text()


# ── Responsive layout checks ───────────────────────────────────

class TestResponsiveLayout:

    def test_mobile_viewport(self, live_server):
        """Page should render without JS errors at mobile width."""
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 375, "height": 812})
            page = context.new_page()

            errors = []
            page.on("pageerror", lambda err: errors.append(str(err)))

            page.goto(live_server, wait_until="networkidle")
            time.sleep(1)

            assert len(errors) == 0, f"JS errors on mobile viewport: {errors}"
            # Header should still be visible
            assert page.locator(".site-header").is_visible()
            context.close()
            browser.close()

    def test_no_console_errors_on_load(self, live_server):
        """Full page load should produce zero uncaught JS exceptions."""
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            errors = []
            page.on("pageerror", lambda err: errors.append(str(err)))

            page.goto(live_server, wait_until="networkidle")
            time.sleep(2)

            assert len(errors) == 0, f"JS page errors: {errors}"
            browser.close()
