import pytest
from unittest.mock import MagicMock
from src.detection.camera import get_camera, read_frame
import cv2


@pytest.fixture
def mock_camera():
    cap = MagicMock(spec=cv2.VideoCapture)
    cap.isOpened.return_value = True
    cap.read.return_value = (True, "mock_frame")
    return cap


def test_get_camera(mock_camera, monkeypatch):
    """Test if the camera can be initialized."""
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: mock_camera)
    cap = get_camera()
    assert isinstance(cap, MagicMock)
    assert cap.isOpened()


def test_get_camera_error(monkeypatch):
    """Test get_camera raises IOError if camera cannot be opened."""
    cap = MagicMock(spec=cv2.VideoCapture)
    cap.isOpened.return_value = False
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: cap)
    with pytest.raises(IOError):
        get_camera()


def test_read_frame(mock_camera, monkeypatch):
    """Test if a frame can be read from the camera."""
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: mock_camera)
    cap = get_camera()
    frame = read_frame(cap, input_size=640, preprocess=False)
    assert frame == "mock_frame"


def test_read_frame_error(mock_camera):
    """Test read_frame raises RuntimeError if frame cannot be read."""
    mock_camera.read.return_value = (False, None)
    with pytest.raises(RuntimeError):
        read_frame(mock_camera)


def test_read_frame_preprocess(monkeypatch, mock_camera):
    """Test read_frame with preprocess=True calls letterbox_image."""
    # Use a mock frame with a .shape attribute
    mock_frame = MagicMock()
    mock_frame.shape = (480, 640, 3)
    mock_camera.read.return_value = (True, mock_frame)

    def fake_letterbox_image(frame, input_size, mode):
        assert hasattr(frame, "shape")
        return ("processed_frame", None, None, None)

    monkeypatch.setattr("src.detection.camera.letterbox_image", fake_letterbox_image)
    frame = read_frame(mock_camera, input_size=640, preprocess=True)
    assert frame == "processed_frame"
