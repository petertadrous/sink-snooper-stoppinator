import pytest
import numpy as np
from src.detection.preprocessing import letterbox_image


@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_letterbox_image_pad(dummy_frame):
    """Test letterbox_image with padding mode."""
    processed, scale, pad_w, pad_h = letterbox_image(
        dummy_frame, input_size=640, mode="pad"
    )
    assert processed.shape == (640, 640, 3)
    assert scale > 0
    assert pad_w >= 0
    assert pad_h >= 0


def test_letterbox_image_crop(dummy_frame):
    """Test letterbox_image with cropping mode."""
    processed, scale, _, _ = letterbox_image(dummy_frame, input_size=640, mode="crop")
    assert processed.shape == (640, 640, 3)
    assert scale > 0


def test_invalid_mode(dummy_frame):
    """Test letterbox_image with an invalid mode."""
    with pytest.raises(ValueError):
        letterbox_image(dummy_frame, input_size=640, mode="invalid")
