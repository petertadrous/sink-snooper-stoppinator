import time
import argparse
import traceback
from typing import Optional

import cv2

from src.detection.detector import detect_cat, debug_draw
from src.detection.camera import get_camera, read_frame
from src.detection.video_recorder import VideoRecorder
from src.deterrent import get_deterrent
from src.config import (
    DETERRENT_DURATION,
    FREQUENCY,
    CAMERA_INDEX,
    DETECTION_HOLD_TIME,
    DETERRENT_TYPE,
)
from src.utils.logger import logger


def main():
    args = parse_args()
    debug_mode = args.debug

    logger.info("Starting Sink Snooper Stoppinator...")
    deterrent = get_deterrent(deterrent_type=DETERRENT_TYPE)
    deterrent.setup()
    cap = get_camera(index=CAMERA_INDEX)
    video_recorder = VideoRecorder()

    # State
    cat_detected_since: Optional[float] = None
    deterrent_active = False
    deterrent_start_time: Optional[float] = None

    try:
        while True:
            now = time.time()
            frame = read_frame(cap)
            detection = detect_cat(
                frame, debug=True
            )  # Always get detection info for recording

            # Check if deterrent should be deactivated
            if deterrent_active and deterrent_start_time is not None:
                if now - deterrent_start_time >= DETERRENT_DURATION:
                    deterrent_active = False
                    deterrent_start_time = None
                    cat_detected_since = None
                    logger.info("Deterrent deactivated")

            # Add frame to video buffer with detection info and deterrent status
            video_recorder.add_frame(
                frame,
                detection["detected"],
                detection_info=detection,
                deterrent_active=deterrent_active,
            )

            if detection["detected"]:
                if cat_detected_since is None:
                    cat_detected_since = now
                elif now - cat_detected_since >= DETECTION_HOLD_TIME:
                    if not deterrent_active:
                        deterrent_active = True
                        deterrent_start_time = now
                        logger.info(f"Deterrent activated for {DETERRENT_DURATION}s")
                        # Start deterrent in non-blocking way
                        deterrent.activate(DETERRENT_DURATION)
            else:
                cat_detected_since = None

            if debug_mode:
                debug_draw(frame, detection, deterrent_active=deterrent_active)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Exiting debug mode")
                    break

            time.sleep(FREQUENCY)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
    finally:
        deterrent.cleanup()
        cap.release()
        if debug_mode:
            cv2.destroyAllWindows()
        logger.info("System shut down cleanly")


def parse_args():
    parser = argparse.ArgumentParser(description="Sink Snooper Stoppinator")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug webcam view with detection overlay",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
