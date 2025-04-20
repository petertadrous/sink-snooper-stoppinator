import pytest
from loguru import logger
from io import StringIO


@pytest.fixture
def loguru_capture():
    """Fixture to capture loguru logs."""
    log_stream = StringIO()
    logger.add(log_stream, format="{message}")
    yield log_stream
    logger.remove()


def test_logger_debug(loguru_capture):
    """Test logger debug level."""
    logger.debug("Debug message")
    loguru_capture.seek(0)
    assert "Debug message" in loguru_capture.getvalue()


def test_logger_error(loguru_capture):
    """Test logger error level."""
    logger.error("Error message")
    loguru_capture.seek(0)
    assert "Error message" in loguru_capture.getvalue()
