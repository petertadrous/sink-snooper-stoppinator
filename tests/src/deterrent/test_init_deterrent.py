import pytest
from src.deterrent import get_deterrent
from src.deterrent.gpio_deterrent import GpioDeterrent
from src.deterrent.audio_deterrent import AudioDeterrent
from src.deterrent.speech_deterrent import SpeechDeterrent


def test_get_deterrent_gpio():
    d = get_deterrent("gpio")
    assert isinstance(d, GpioDeterrent)


def test_get_deterrent_gunshots():
    d = get_deterrent("gunshots")
    assert isinstance(d, AudioDeterrent)
    assert d.audio_name == "gunshots"


def test_get_deterrent_speech():
    d = get_deterrent("speech")
    assert isinstance(d, SpeechDeterrent)
    assert not d.creative


def test_get_deterrent_llm():
    d = get_deterrent("llm")
    assert isinstance(d, SpeechDeterrent)
    assert d.creative


def test_get_deterrent_invalid():
    with pytest.raises(ValueError):
        get_deterrent("invalid")
