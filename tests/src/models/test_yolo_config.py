import pytest
from unittest.mock import patch, MagicMock
from src.models import yolo_config
from src.models.yolo_config import load_class_names, get_class_id, load_model


def test_get_class_id():
    """Test getting a class ID by name."""
    class_names = load_class_names()
    class_id = get_class_id("cat", class_names)
    assert isinstance(class_id, int)


def test_load_model():
    """Test loading the YOLO model."""
    model = load_model()
    assert model is not None


def test_get_class_id_found():
    class_names = {0: "cat", 1: "dog"}
    assert yolo_config.get_class_id("cat", class_names) == 0


def test_get_class_id_not_found():
    class_names = {0: "cat", 1: "dog"}
    with pytest.raises(RuntimeError):
        yolo_config.get_class_id("car", class_names)


def test_load_class_names(tmp_path, monkeypatch):
    labels_path = tmp_path / "labels.names"
    labels_path.write_text("cat\ndog\n")
    monkeypatch.setattr(yolo_config, "LABELS_PATH", str(labels_path))
    names = yolo_config.load_class_names()
    assert names == {0: "cat", 1: "dog"}


def test_load_model_existing(monkeypatch):
    monkeypatch.setattr(yolo_config, "MODEL_PATH", "assets/yolov8n.onnx")
    monkeypatch.setattr(yolo_config.Path, "is_file", lambda self: True)
    with patch("cv2.dnn.readNetFromONNX") as mock_read:
        mock_read.return_value = "net"
        net = yolo_config.load_model()
        assert net == "net"


def test_load_model_export(monkeypatch, tmp_path):
    # Simulate model export path
    model_path = tmp_path / "yolov8n.onnx"
    labels_path = tmp_path / "coco.names"
    monkeypatch.setattr(yolo_config, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(yolo_config, "LABELS_PATH", str(labels_path))
    monkeypatch.setattr(yolo_config.Path, "is_file", lambda self: False)
    fake_yolo = MagicMock()
    fake_yolo.export.return_value = str(model_path)
    fake_yolo.names = {0: "cat", 1: "dog"}
    with patch("ultralytics.YOLO", return_value=fake_yolo):
        with (
            patch("cv2.dnn.readNetFromONNX") as mock_read,
            patch("shutil.move") as mock_move,
        ):
            mock_read.return_value = "net"
            net = yolo_config.load_model()
            assert net == "net"
            mock_move.assert_called()
            assert labels_path.read_text() == "cat\ndog\n"
