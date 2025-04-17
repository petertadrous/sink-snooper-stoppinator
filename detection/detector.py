from utils.logger import logger
import random
import cv2


def detect_cat(frame: cv2.typing.MatLike, debug: bool = False) -> dict:
    """
    Detects if a cat is present in the frame.
    Returns a dictionary with "detected" and "bbox" keys.
    """
    # Simulate detection (replace later with real model)
    detected = random.random() < 0.1

    if detected:
        logger.info("Cat detected!")
        # Fake bounding box + label
        h, w, _ = frame.shape
        x1, y1 = int(w * 0.3), int(h * 0.3)
        x2, y2 = int(w * 0.6), int(h * 0.6)
        label = "cat"

        if debug:
            return {
                "detected": True,
                "bbox": (x1, y1, x2, y2),
                "label": label,
            }
        else:
            return {"detected": True}

    return {"detected": False}


def debug_draw(frame: cv2.typing.MatLike, detection: dict) -> None:
    """
    Draws a bounding box and label on the frame if a cat is detected.
    """
    if detection.get("detected") and "bbox" in detection:
        (x1, y1, x2, y2) = detection["bbox"]
        label = detection.get("label", "object")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Cat Sink Debug View", frame)
