import cv2
from pathlib import Path

MODEL_PATH = "assets/yolov8n.onnx"
LABELS_PATH = "assets/coco.names"


def load_class_names() -> dict[int, str]:
    """
    Loads the class names from the specified path.
    """
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    return {i: name for i, name in enumerate(class_names)}


def get_class_id(class_name: str, class_names: dict[int, str]) -> int:
    """
    Returns the ID of the class name in the class dictionary.
    """
    for key in class_names:
        if class_names[key] == class_name:
            return key
    else:
        raise RuntimeError(f"Class '{class_name}' not found in class mapping.")


def load_model() -> cv2.dnn.Net:
    """
    Loads the YOLO model from the specified path.
    """
    if not Path(MODEL_PATH).is_file():
        from ultralytics import YOLO
        import shutil

        model = YOLO(MODEL_PATH.split("/")[-1].split(".")[0] + ".pt")
        result = model.export(format="onnx")
        shutil.move(result, MODEL_PATH)
        labels = dict(model.names).values()
        with open(LABELS_PATH, "w+") as f:
            for label in labels:
                f.write(f"{label}\n")

    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    return net
