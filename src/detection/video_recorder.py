import os
import time
from pathlib import Path
from collections import deque
from typing import Optional, Deque, Tuple, Dict, Any

import cv2
import numpy as np

from src.config import (
    VIDEO_PRE_DETECTION_BUFFER,
    VIDEO_POST_DETECTION_BUFFER,
    VIDEO_OUTPUT_DIR,
    VIDEO_DEBUG_OVERLAY,
    FREQUENCY,
)
from src.utils.logger import logger
from src.detection.detector import debug_draw


class VideoRecorder:
    def __init__(self):
        """Initialize the video recorder with a circular buffer for frames."""
        self.frame_buffer: Deque[Tuple[float, np.ndarray, Dict[str, Any], bool]] = (
            deque(maxlen=int(VIDEO_PRE_DETECTION_BUFFER / FREQUENCY))
        )
        self.recording_buffer: list[Tuple[float, np.ndarray, Dict[str, Any], bool]] = []
        self.is_recording = False
        self.detection_time: Optional[float] = None
        self.last_detection_time: Optional[float] = None
        self._setup_output_dir()

    def _setup_output_dir(self) -> None:
        """Create the output directory if it doesn't exist."""
        Path(VIDEO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def add_frame(
        self,
        frame: np.ndarray,
        detected: bool,
        detection_info: Optional[Dict[str, Any]] = None,
        deterrent_active: bool = False,
    ) -> None:
        """
        Add a frame to the buffer and handle recording state.

        Args:
            frame: The current video frame
            detected: Whether an interested object was detected in this frame
            detection_info: Optional detection information for debug overlay
            deterrent_active: Whether the deterrent is currently active
        """
        current_time = time.time()
        frame_data = (
            current_time,
            frame.copy(),
            detection_info or {},
            deterrent_active,
        )
        self.frame_buffer.append(frame_data)

        if detected:
            self.last_detection_time = current_time
            if not self.is_recording:
                # Start new recording
                self.is_recording = True
                self.detection_time = current_time
                # Copy pre-detection buffer
                self.recording_buffer = list(self.frame_buffer)
                logger.info("Started recording detection event")
            else:
                # Add frame and extend recording duration
                self.recording_buffer.append(frame_data)
                logger.debug("Extended recording due to new detection")

        elif self.is_recording:
            # Add frame to recording
            self.recording_buffer.append(frame_data)

            # Stop recording if enough time has passed since last detection
            if self.last_detection_time is not None:
                elapsed = current_time - self.last_detection_time
                if elapsed >= VIDEO_POST_DETECTION_BUFFER:  # Changed from > to >=
                    self._save_recording()
                    self.is_recording = False
                    self.detection_time = None
                    self.last_detection_time = None
                    self.recording_buffer = []
                    logger.info(
                        f"Finished recording detection event after {elapsed:.1f}s"
                    )

    def _save_recording(self) -> None:
        """Save the current recording buffer to a video file."""
        if (
            not self.recording_buffer
            or self.detection_time is None
            or self.last_detection_time is None
        ):
            return

        # Get video properties from first frame
        first_frame = self.recording_buffer[0][1]
        height, width = first_frame.shape[:2]

        # Create output filename with timestamp and duration
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        start_time = self.detection_time
        end_time = self.last_detection_time + VIDEO_POST_DETECTION_BUFFER
        duration = round(end_time - start_time, 1)

        # Add unique identifier to filename to prevent overwriting
        base_name = f"detection_{timestamp}_{duration}s.mp4"
        output_path = os.path.join(VIDEO_OUTPUT_DIR, base_name)
        counter = 1
        while os.path.exists(output_path):
            base_name = f"detection_{timestamp}_{duration}s_{counter}.mp4"
            output_path = os.path.join(VIDEO_OUTPUT_DIR, base_name)
            counter += 1

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # type: ignore
        fps = 1.0 / FREQUENCY  # Convert frequency to FPS
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            # Write frames
            for (
                _,
                frame,
                detection_info,
                deterrent_active,
            ) in self.recording_buffer:
                if VIDEO_DEBUG_OVERLAY and detection_info:
                    # Add debug overlay to frame
                    frame = debug_draw(
                        frame,
                        detection_info,
                        deterrent_active=deterrent_active,
                        show_preview=False,
                    )
                out.write(frame)

            logger.info(
                f"Saved detection video to {output_path} (duration: {duration}s)"
            )
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
        finally:
            out.release()
