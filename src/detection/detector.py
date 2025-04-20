import cv2
import numpy as np
from src.models.yolo_config import load_model, load_class_names, get_class_id
from src.config import CONFIDENCE_THRESHOLD, SCORE_THRESHOLD, INTERESTED_CLASSES
from src.utils.logger import logger

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
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (height, width), swapRB=True, crop=False
    )
    model.setInput(blob)
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

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD
    )

    detections = []
    for i in indices:
        i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
        x, y, w, h = boxes[i]
        x1, y1, x2, y2 = x, y, x + w, y + h

        label = (
            CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else "object"
        )
        logger.debug(
            f"Detected {label} (conf: {confidences[i]:.2f}) at {x1}, {y1}, {x2}, {y2}"
        )
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
    deterrent_active: bool = False,
    show_preview: bool = True,
) -> cv2.typing.MatLike:
    """
    Draws detection bounding boxes, labels, and deterrent status on the frame.

    Args:
        frame: The original image to draw on.
        detections: List of detection dicts with 'bbox', 'label', and 'score'.
        deterrent_active: Whether the deterrent is currently active.
        show_preview: Whether to show the preview window.

    Returns:
        The annotated frame.
    """
    annotated_frame = frame.copy()
    box_thickness: int = 2
    font_scale: float = 0.6
    detections = detections.get("detections", [])

    # Draw detection boxes and labels
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det.get("label", "object")
        score = det.get("score", 0.0)

        color = (0, 255, 0) if label in INTERESTED_CLASSES else (180, 180, 180)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, box_thickness)

        text = f"{label} ({score:.2f})"
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(
            annotated_frame, (x1, y1 - text_h - 6), (x1 + text_w, y1), color, -1
        )
        cv2.putText(
            annotated_frame,
            text,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
        )

    # Draw deterrent status indicator
    if deterrent_active:
        status_text = "DETERRENT ACTIVE"
        color = (0, 0, 255)  # Red for active deterrent
    else:
        status_text = "monitoring"
        color = (255, 255, 255)  # White for normal monitoring

    # Position at top-right corner
    (text_w, text_h), _ = cv2.getTextSize(
        status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )
    margin = 10
    x = annotated_frame.shape[1] - text_w - margin
    y = text_h + margin

    # Draw status text with background
    cv2.rectangle(
        annotated_frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), color, -1
    )
    cv2.putText(
        annotated_frame,
        status_text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        2,
    )

    if show_preview:
        cv2.imshow("Detection Debug View", annotated_frame)

    return annotated_frame
