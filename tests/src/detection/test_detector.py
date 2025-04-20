import pytest
import numpy as np
from unittest.mock import MagicMock
from src.detection.detector import detect_objects, detect_cat, debug_draw


@pytest.fixture
def dummy_frame():
    return np.zeros((640, 640, 3), dtype=np.uint8)


def test_detect_objects(dummy_frame):
    """Test object detection on a dummy frame."""
    detections = detect_objects(dummy_frame)
    assert isinstance(detections, list)


def test_detect_cat(dummy_frame):
    """Test cat detection on a dummy frame."""
    result = detect_cat(dummy_frame, debug=True, show_all=True)
    assert "detected" in result
    assert "detections" in result


def test_detect_objects_with_detection(dummy_frame, monkeypatch):
    # Patch model and outputs to simulate a detection
    import src.detection.detector as detector_mod

    fake_model = MagicMock()
    # Simulate YOLO output: shape (num_features, num_detections) after .T
    # 3 classes, so detection[4:] is [0.1, 0.2, 0.9] and [0.1, 0.2, 0.3]
    # Each column is a detection
    yolo_out = np.array(
        [
            [10, 30],  # cx
            [10, 30],  # cy
            [20, 40],  # w
            [20, 40],  # h
            [0.1, 0.1],  # class 0 score
            [0.2, 0.2],  # class 1 score
            [0.9, 0.3],  # class 2 score
        ]
    )
    fake_model.forward.return_value = [yolo_out]
    monkeypatch.setattr(detector_mod, "model", fake_model)
    monkeypatch.setattr(detector_mod, "CLASS_NAMES", ["cat", "dog", "car"])
    monkeypatch.setattr(detector_mod, "CONFIDENCE_THRESHOLD", 0.5)
    monkeypatch.setattr(detector_mod, "SCORE_THRESHOLD", 0.1)
    fake_cv2 = MagicMock()
    fake_cv2.dnn.NMSBoxes.return_value = [[0]]  # Only the first detection is kept
    monkeypatch.setattr(detector_mod, "cv2", fake_cv2)
    detections = detect_objects(dummy_frame)
    assert isinstance(detections, list)
    assert len(detections) == 1
    assert detections[0]["label"] == "car"


def test_detect_cat_no_cats(dummy_frame, monkeypatch):
    # Patch detect_objects to return only non-interested detections
    monkeypatch.setattr(
        "src.detection.detector.detect_objects",
        lambda frame: [
            {"bbox": (0, 0, 1, 1), "class_id": 1, "label": "car", "score": 0.9}
        ],
    )
    result = detect_cat(dummy_frame, debug=True, show_all=False)
    assert result["detected"] is False
    assert result["detections"] == []


def test_detect_cat_with_cat(dummy_frame, monkeypatch):
    # Patch detect_objects to return a cat detection
    monkeypatch.setattr(
        "src.detection.detector.detect_objects",
        lambda frame: [
            {"bbox": (0, 0, 1, 1), "class_id": 0, "label": "cat", "score": 0.95}
        ],
    )
    result = detect_cat(dummy_frame, debug=True, show_all=False)
    assert result["detected"] is True
    assert result["detections"] == [
        {"bbox": (0, 0, 1, 1), "class_id": 0, "label": "cat", "score": 0.95}
    ]


def test_debug_draw_empty(monkeypatch, dummy_frame):
    # Should not raise with empty detections
    monkeypatch.setattr("cv2.imshow", lambda *a, **k: None)
    debug_draw(dummy_frame, {"detections": []})


def test_debug_draw_with_detection(monkeypatch, dummy_frame):
    # Should call cv2 drawing functions
    called = {}

    def fake_rectangle(*a, **k):
        called["rectangle"] = True

    def fake_putText(*a, **k):
        called["putText"] = True

    def fake_getTextSize(*a, **k):
        return ((10, 10), None)

    monkeypatch.setattr("cv2.rectangle", fake_rectangle)
    monkeypatch.setattr("cv2.putText", fake_putText)
    monkeypatch.setattr("cv2.getTextSize", fake_getTextSize)
    monkeypatch.setattr("cv2.imshow", lambda *a, **k: None)
    debug_draw(
        dummy_frame,
        {"detections": [{"bbox": (1, 2, 3, 4), "label": "cat", "score": 0.9}]},
    )
    assert called["rectangle"] and called["putText"]
