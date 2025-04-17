from utils.logger import logger
import random

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cv2


def detect_cat(frame: "cv2.typing.MatLike") -> bool:
    # Simulate detection (replace later with real model)
    detected = random.random() < 0.1
    if detected:
        logger.info("Cat detected!")
    return detected
