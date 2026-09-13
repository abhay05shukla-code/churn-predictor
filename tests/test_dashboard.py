"""Integration test: boot the real API, then drive the Streamlit app headlessly."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

from tests.conftest import requires_model

ROOT = Path(__file__).resolve().parent.parent
pytestmark = requires_model


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def api_url():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if requests.get(f"{url}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.2)
        else:
            pytest.fail("API did not start")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_dashboard_renders_and_predicts_without_exceptions(api_url, monkeypatch):
    monkeypatch.setenv("CHURN_API_URL", api_url)
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]

    # Tab 2 scored the whole customer base and rendered the ranked table.
    assert any("High-value retention risk list" in h.value for h in at.subheader)
    assert len(at.dataframe) >= 1

    # Submit the prediction form on tab 1 and make sure the results render.
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    labels = [m.label for m in at.metric]
    assert "Risk tier" in labels
    assert "Monthly revenue at risk" in labels
    assert any("Retention levers" in h.value for h in at.subheader)


def test_dashboard_shows_clear_error_when_api_is_down(monkeypatch):
    monkeypatch.setenv("CHURN_API_URL", f"http://127.0.0.1:{_free_port()}")
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("API unreachable" in e.value for e in at.error)
