import pytest
from unittest.mock import patch, MagicMock
from src.deterrent.audio_deterrent import AudioDeterrent
from pydub import AudioSegment


@pytest.fixture
def audio_deterrent():
    return AudioDeterrent(audio_name="gunshots")


def test_audio_deterrent_setup(audio_deterrent):
    """Test the setup method of AudioDeterrent."""
    with patch("src.deterrent.audio_deterrent.AudioSegment.from_mp3") as mock_from_mp3:
        mock_from_mp3.return_value = MagicMock()
        audio_deterrent.setup()
        mock_from_mp3.assert_called_once()


def test_audio_deterrent_activate(audio_deterrent):
    """Test the activate method of AudioDeterrent."""
    with patch("src.deterrent.audio_deterrent.play") as mock_play:
        mock_audio = MagicMock()
        mock_audio.duration_seconds = 2.0  # Mock required attribute
        audio_deterrent.audio = mock_audio
        audio_deterrent.activate(duration=1.5)
        mock_play.assert_called_once()


def test_audio_deterrent_cleanup(audio_deterrent):
    """Test the cleanup method of AudioDeterrent."""
    audio_deterrent.audio = MagicMock()
    audio_deterrent.cleanup()
    assert audio_deterrent.audio is None


def test_audio_deterrent_setup_invalid():
    d = AudioDeterrent(audio_name="invalid")
    with pytest.raises(ValueError):
        d.setup()


def test_audio_deterrent_activate_invalid():
    d = AudioDeterrent(audio_name="invalid")
    d.audio = MagicMock()
    with pytest.raises(ValueError):
        d.activate(1.0)


def test_audio_deterrent_activate_play_error(audio_deterrent):
    with patch("src.deterrent.audio_deterrent.play", side_effect=Exception("fail")):
        mock_audio = MagicMock()
        mock_audio.duration_seconds = 2.0
        audio_deterrent.audio = mock_audio
        with pytest.raises(Exception):
            audio_deterrent.activate(duration=1.5)


def test_splice_and_loop_mp3():
    audio = MagicMock(spec=AudioSegment)
    segment = MagicMock(spec=AudioSegment)
    audio.__getitem__.return_value = segment
    segment.append.return_value = segment
    audio.__add__.return_value = audio
    result = AudioDeterrent._splice_and_loop_mp3(audio, 0, 1000, 2, cross_fade=100)
    assert result is not None


def test_loop_gunshots_longer_duration(audio_deterrent):
    mock_audio = MagicMock()
    mock_audio.duration_seconds = 1.0
    mock_audio.__add__.return_value = mock_audio
    audio_deterrent.audio = mock_audio
    with patch.object(
        AudioDeterrent, "_splice_and_loop_mp3", return_value="spliced"
    ) as mock_splice:
        result = audio_deterrent._loop_gunshots(duration=2.0)
        assert result == "spliced"
        mock_splice.assert_called()
