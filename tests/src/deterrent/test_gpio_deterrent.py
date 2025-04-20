import pytest
from unittest.mock import patch, MagicMock
import sys
import types
from src.deterrent.gpio_deterrent import GpioDeterrent
import src.deterrent.gpio_deterrent as gpio_mod


@pytest.fixture
def gpio_deterrent():
    return GpioDeterrent()


def test_gpio_deterrent_setup(gpio_deterrent):
    """Test the setup method of GpioDeterrent."""
    mock_gpio = MagicMock()
    mock_time = MagicMock()
    sys.modules["RPi"] = types.SimpleNamespace(GPIO=mock_gpio)  # type: ignore
    sys.modules["RPi.GPIO"] = mock_gpio
    sys.modules["time"] = mock_time
    with patch("src.deterrent.gpio_deterrent.IS_PI", True):
        setattr(gpio_mod, "GPIO", mock_gpio)
        gpio_deterrent.setup()
        mock_gpio.setmode.assert_called_once()
        mock_gpio.setup.assert_called_once()


def test_gpio_deterrent_activate(gpio_deterrent):
    """Test the activate method of GpioDeterrent."""
    mock_gpio = MagicMock()
    mock_time = MagicMock()
    sys.modules["RPi"] = types.SimpleNamespace(GPIO=mock_gpio)  # type: ignore
    sys.modules["RPi.GPIO"] = mock_gpio
    sys.modules["time"] = mock_time
    with patch("src.deterrent.gpio_deterrent.IS_PI", True):
        setattr(gpio_mod, "GPIO", mock_gpio)
        setattr(gpio_mod, "time", mock_time)
        gpio_deterrent.activate(duration=1.5)
        mock_gpio.output.assert_any_call(gpio_deterrent.pin, mock_gpio.HIGH)
        mock_time.sleep.assert_called_once_with(1.5)
        mock_gpio.output.assert_any_call(gpio_deterrent.pin, mock_gpio.LOW)


def test_gpio_deterrent_cleanup(gpio_deterrent):
    """Test the cleanup method of GpioDeterrent."""
    mock_gpio = MagicMock()
    sys.modules["RPi"] = types.SimpleNamespace(GPIO=mock_gpio)  # type: ignore
    sys.modules["RPi.GPIO"] = mock_gpio
    with patch("src.deterrent.gpio_deterrent.IS_PI", True):
        setattr(gpio_mod, "GPIO", mock_gpio)
        gpio_deterrent.cleanup()
        mock_gpio.cleanup.assert_called_once()


def test_gpio_deterrent_setup_not_pi(gpio_deterrent):
    with patch("src.deterrent.gpio_deterrent.IS_PI", False):
        gpio_deterrent.setup()  # Should just log and return


def test_gpio_deterrent_activate_not_pi(gpio_deterrent):
    with patch("src.deterrent.gpio_deterrent.IS_PI", False):
        gpio_deterrent.activate(1.0)  # Should just log and return


def test_gpio_deterrent_cleanup_not_pi(gpio_deterrent):
    with patch("src.deterrent.gpio_deterrent.IS_PI", False):
        gpio_deterrent.cleanup()  # Should do nothing
