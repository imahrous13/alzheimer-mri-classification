from pathlib import Path

import pytest


def test_streamlit_app_runs_without_exception():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = streamlit_testing.AppTest.from_file(str(app_path), default_timeout=240)
    app.run()
    assert not app.exception
    assert any("Alzheimer MRI" in title.value for title in app.title)
    assert len(app.tabs) == 4
