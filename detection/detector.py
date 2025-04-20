import cv2
import numpy as np

from models.yolo_config import load_model, load_class_names, get_class_id
from config import CONFIDENCE_THRESHOLD, SCORE_THRESHOLD, INTERESTED_CLASSES
from utils.logger import logger


model = load_model()
CLASS_NAMES = load_class_names()
CAT_CLASS_ID = get_class_id("cat", CLASS_NAMES)


def detect_objects(
    frame: cv2.typing.MatLike,
) -> list[dict]:
    """
    Runs YOLOv8 object detection on the given frame.

    Args:
        frame: Original BGR image (H, W, C)

    Returns:
        List of detections, each a dict:
            {
                'bbox': (x1, y1, x2, y2),
                'class_id': int,
                'label': str,
                'score': float
            }
    """
    height, width = frame.shape[:2]
    # Create input blob and run inference
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (height, width), swapRB=True, crop=False)
    model.setInput(blob)
    # Transposed: shape (8400, 84) → (84, 8400) → transpose to (8400, 84)
    outputs = model.forward()[0].T
    boxes, confidences, class_ids = [], [], []

    for detection in outputs:
        class_scores = detection[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence > CONFIDENCE_THRESHOLD:
            cx, cy, w, h = detection[:4]

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            boxes.append([x1, y1, x2 - x1, y2 - y1])  # x, y, w, h
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD)

    detections = []
    for i in indices:
        i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
        x, y, w, h = boxes[i]
        x1, y1, x2, y2 = x, y, x + w, y + h

        label = CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else "object"
        logger.debug(f"Detected {label} (conf: {confidences[i]:.2f}) at {x1}, {y1}, {x2}, {y2}")
        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "class_id": class_ids[i],
                "label": label,
                "score": confidences[i],
            }
        )

    return detections


def detect_cat(
    frame: cv2.typing.MatLike,
    debug: bool = False,
    show_all: bool = True,
) -> dict:
    """
    Detects all cats in the frame using YOLOv8 Nano.
    Returns a dictionary with "detected": bool and "detections": list of cat boxes.
    """
    all_detections = detect_objects(frame)
    cat_detections = [d for d in all_detections if d["label"] in INTERESTED_CLASSES]

    result: dict = {"detected": len(cat_detections) > 0}

    if debug:
        result["detections"] = all_detections if show_all else cat_detections

    return result


def debug_draw(
    frame: cv2.typing.MatLike,
    detections: dict,
) -> None:
    """
    Draws detection bounding boxes and labels on the frame.

    Args:
        frame: The original image to draw on.
        detections: List of detection dicts with 'bbox', 'label', and 'score'.
        window_name: Window title for the display.
        box_thickness: Thickness of the box outlines.
        font_scale: Scale of the label text.
    """
    box_thickness: int = 2
    font_scale: float = 0.6
    detections = detections.get("detections", [])

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det.get("label", "object")
        score = det.get("score", 0.0)

        # Color: green for 'cat', light gray for others
        color = (0, 255, 0) if label == "cat" else (180, 180, 180)
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        # Prepare label text
        text = f"{label} ({score:.2f})"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w, y1), color, -1)  # background
        cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

    cv2.imshow("Detection Debug View", frame)
