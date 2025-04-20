from unittest.mock import patch
import numpy as np
import pytest
import cv2

from src.detection.video_recorder import VideoRecorder
from src.config import (
    VIDEO_PRE_DETECTION_BUFFER,
    VIDEO_POST_DETECTION_BUFFER,
    FREQUENCY,
)


@pytest.fixture
def video_recorder(tmp_path):
    with patch("src.detection.video_recorder.VIDEO_OUTPUT_DIR", str(tmp_path)):
        recorder = VideoRecorder()
        yield recorder


@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_init(video_recorder):
    """Test VideoRecorder initialization."""
    assert not video_recorder.is_recording
    assert video_recorder.detection_time is None
    assert video_recorder.last_detection_time is None
    assert len(video_recorder.frame_buffer) == 0
    assert len(video_recorder.recording_buffer) == 0


def test_buffer_size(video_recorder):
    """Test that frame buffer maintains correct size."""
    expected_size = int(VIDEO_PRE_DETECTION_BUFFER / FREQUENCY)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add more frames than buffer size
    for _ in range(expected_size + 5):
        video_recorder.add_frame(frame, False)

    assert len(video_recorder.frame_buffer) == expected_size


def test_recording_cycle(video_recorder, dummy_frame, tmp_path):
    """Test complete recording cycle with detection."""
    with patch("time.time") as mock_time:
        # Start at t=0
        mock_time.return_value = 0

        # Add some pre-detection frames
        for i in range(5):
            mock_time.return_value = i * FREQUENCY
            video_recorder.add_frame(dummy_frame, False)

        # Add detection frame at t=0.5s
        mock_time.return_value = 5 * FREQUENCY
        video_recorder.add_frame(dummy_frame, True)
        assert video_recorder.is_recording

        # Add post-detection frames until past VIDEO_POST_DETECTION_BUFFER
        # Need to go past t=0.5s + VIDEO_POST_DETECTION_BUFFER
        max_time = (5 * FREQUENCY) + VIDEO_POST_DETECTION_BUFFER + FREQUENCY
        frame_count = int(max_time / FREQUENCY)

        for i in range(6, frame_count + 1):
            mock_time.return_value = i * FREQUENCY
            video_recorder.add_frame(dummy_frame, False)

        # Verify recording stopped
        assert not video_recorder.is_recording

        # Check video was saved
        video_files = list(tmp_path.glob("*.mp4"))
        assert len(video_files) == 1


def test_overlapping_detections(video_recorder, dummy_frame, tmp_path):
    """Test that overlapping detections extend the recording."""
    with patch("time.time") as mock_time:
        # Start at t=0
        mock_time.return_value = 0

        # Add initial detection
        video_recorder.add_frame(dummy_frame, True)
        initial_detection_time = mock_time.return_value

        # Add overlapping detection during post-buffer period
        mock_time.return_value = VIDEO_POST_DETECTION_BUFFER / 2
        video_recorder.add_frame(dummy_frame, True)
        second_detection_time = mock_time.return_value

        # Add frames until after second post-buffer
        mock_time.return_value = (
            second_detection_time + VIDEO_POST_DETECTION_BUFFER + 0.1
        )
        video_recorder.add_frame(dummy_frame, False)

        # Verify single video was created with extended duration
        video_files = list(tmp_path.glob("*.mp4"))
        assert len(video_files) == 1
        filename = video_files[0].name
        # Duration should be included in filename
        expected_duration = round(
            second_detection_time
            + VIDEO_POST_DETECTION_BUFFER
            - initial_detection_time,
            1,
        )
        assert f"{expected_duration}s.mp4" in filename


def test_video_saving(video_recorder, dummy_frame, tmp_path):
    """Test that video is properly saved with correct format."""
    with patch("time.time") as mock_time:
        mock_time.return_value = 0

        # Trigger recording
        video_recorder.add_frame(dummy_frame, True)
        mock_time.return_value = VIDEO_POST_DETECTION_BUFFER + 0.1
        video_recorder.add_frame(dummy_frame, False)

        # Check that video file exists and is readable
        video_files = list(tmp_path.glob("*.mp4"))
        assert len(video_files) == 1

        cap = cv2.VideoCapture(str(video_files[0]))
        try:
            ret, frame = cap.read()
            assert ret
            assert frame.shape == dummy_frame.shape
        finally:
            cap.release()


def test_multiple_detection_sequences(video_recorder, dummy_frame, tmp_path):
    """Test multiple separate detection sequences create separate videos."""
    with patch("time.time") as mock_time:
        # First detection sequence
        mock_time.return_value = 0
        video_recorder.add_frame(dummy_frame, True)
        mock_time.return_value = VIDEO_POST_DETECTION_BUFFER + 0.1
        video_recorder.add_frame(dummy_frame, False)

        # Second detection sequence
        mock_time.return_value = VIDEO_POST_DETECTION_BUFFER + 1.0
        video_recorder.add_frame(dummy_frame, True)
        mock_time.return_value = VIDEO_POST_DETECTION_BUFFER * 2 + 1.1
        video_recorder.add_frame(dummy_frame, False)

        # Verify two separate videos were created
        video_files = list(tmp_path.glob("*.mp4"))
        assert len(video_files) == 2
