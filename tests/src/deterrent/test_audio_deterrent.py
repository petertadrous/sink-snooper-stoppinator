import pytest
from unittest.mock import patch, MagicMock

from src.deterrent.audio_deterrent import AudioDeterrent


@pytest.fixture
def audio_deterrent():
    return AudioDeterrent()


def test_audio_deterrent_init(audio_deterrent):
    """Test initialization of AudioDeterrent."""
    assert audio_deterrent.audio_name == "gunshots"
    assert audio_deterrent.audio is None


def test_audio_deterrent_setup(audio_deterrent):
    """Test the setup method of AudioDeterrent."""
    with patch("pydub.AudioSegment.from_mp3") as mock_from_mp3:
        mock_audio = MagicMock()
        mock_audio.__len__ = lambda _: 1000  # 1 second in milliseconds
        mock_from_mp3.return_value = mock_audio
        audio_deterrent.setup()
        assert audio_deterrent.audio is mock_audio


def test_audio_deterrent_activate(audio_deterrent):
    """Test the activate method of AudioDeterrent."""
    with patch("src.deterrent.audio_deterrent.play") as mock_play:
        mock_audio = MagicMock()
        mock_audio.__len__ = lambda _: 1000  # 1 second in milliseconds
        mock_audio.__mul__ = lambda _, x: mock_audio  # Mock multiplication
        audio_deterrent.audio = mock_audio
        audio_deterrent.activate(duration=1.5)
        # Give thread time to start
        import time

        time.sleep(0.1)
        # Verify audio was played
        mock_play.assert_called_once()


def test_audio_deterrent_cleanup(audio_deterrent):
    """Test the cleanup method of AudioDeterrent."""
    mock_audio = MagicMock()
    audio_deterrent.audio = mock_audio
    audio_deterrent.cleanup()
    assert audio_deterrent.audio is None


def test_loop_gunshots_longer_duration(audio_deterrent):
    """Test looping gunshots for longer duration."""
    mock_audio = MagicMock()
    mock_audio.__len__ = lambda _: 1000  # 1 second in milliseconds
    mock_audio.__mul__ = lambda _, x: mock_audio  # Mock multiplication
    audio_deterrent.audio = mock_audio

    result = audio_deterrent._loop_gunshots(duration=2.0)
    # Should loop twice for 2 second duration
    assert result is mock_audio  # Since we mocked multiplication to return mock_audio


def test_audio_deterrent_invalid_audio(audio_deterrent):
    """Test AudioDeterrent with invalid audio name."""
    with pytest.raises(ValueError):
        audio_deterrent.audio_name = "invalid"
        audio_deterrent.setup()


def test_audio_deterrent_no_init(audio_deterrent):
    """Test activating AudioDeterrent without initialization."""
    with pytest.raises(RuntimeError):
        audio_deterrent.activate(duration=1.0)
