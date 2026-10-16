import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_app_startup_and_rendering():
    """Verify that app.py compiles, boots up, and loads default UI elements without exception."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception
    # When no documents are loaded initially, it displays the welcome info
    assert len(at.info) > 0 or len(at.markdown) > 0


def test_app_with_demo_document():
    """Verify that app.py processes demo document when button is clicked."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception

    # Find the Load Demo Paper 1 button and click it
    demo_button = None
    for b in at.button:
        if "Demo Paper 1" in b.label:
            demo_button = b
            break

    if demo_button is not None:
        demo_button.click().run()
        assert not at.exception
        # Now document should be loaded and tabs rendered
        assert len(at.tabs) > 0

