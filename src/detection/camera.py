import cv2
from src.utils.logger import logger
from src.config import PREPROCESS_MODE
from src.detection.preprocessing import letterbox_image


def get_camera(index: int = 0) -> cv2.VideoCapture:
    """
    Returns a cv2.VideoCapture object for the specified camera index.
    If the camera cannot be opened, raises an IOError.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        raise IOError("Cannot open webcam")
    logger.debug("Webcam initialized")
    return cap


def read_frame(
    cap: cv2.VideoCapture, input_size: int = 640, preprocess: bool = True
) -> cv2.typing.MatLike:
    """
    Reads a frame from the specified cv2.VideoCapture object.
    If the frame cannot be read, raises a RuntimeError.
    """
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to read from camera")
        raise RuntimeError("Failed to read from camera")

    if preprocess:
        frame, _, _, _ = letterbox_image(
            frame, input_size=input_size, mode=PREPROCESS_MODE
        )
    return frame
