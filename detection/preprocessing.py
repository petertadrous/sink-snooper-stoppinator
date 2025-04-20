import cv2
import numpy as np


def letterbox_image(frame: cv2.typing.MatLike, input_size: int = 640, mode: str = "pad"):
    """
    Resizes the image to fit into input_size x input_size.

    Args:
        frame: Input image (H, W, C)
        input_size: Target square size (default 640)
        mode: 'pad' (default) for aspect-preserving padding, or 'crop' for center crop

    Returns:
        processed image (input_size, input_size, 3)
        scale used to resize original image
        pad_w and pad_h: only if mode='pad', else (0, 0)
    """
    original_h, original_w = frame.shape[:2]

    if mode == "pad":
        # Maintain aspect ratio and pad
        scale = min(input_size / original_w, input_size / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (input_size - new_w) // 2
        pad_h = (input_size - new_h) // 2

        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        return padded, scale, pad_w, pad_h

    elif mode == "crop":
        # Resize the image so the smaller side == input_size
        scale = input_size / min(original_w, original_h)
        resized_w, resized_h = int(original_w * scale), int(original_h * scale)
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        # Center crop to input_size x input_size
        x_start = (resized_w - input_size) // 2
        y_start = (resized_h - input_size) // 2
        cropped = resized[y_start : y_start + input_size, x_start : x_start + input_size]

        return cropped, scale, 0, 0

    else:
        raise ValueError(f"Invalid mode '{mode}'. Use 'pad' or 'crop'.")
