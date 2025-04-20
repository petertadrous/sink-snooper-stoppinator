import time
import argparse
import traceback

import cv2

from src.detection.detector import detect_cat, debug_draw
from src.detection.camera import get_camera, read_frame
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

    # State
    cat_detected_since = None
    deterrent_active = False

    try:
        while True:
            frame = read_frame(cap)
            detection = detect_cat(frame, debug=debug_mode)

            if detection["detected"]:
                now = time.time()
                if cat_detected_since is None:
                    cat_detected_since = now
                elif now - cat_detected_since >= DETECTION_HOLD_TIME:
                    if not deterrent_active:
                        deterrent_active = True
                        logger.info(f"Deterrent activated for {DETERRENT_DURATION}s")
                        deterrent.activate(DETERRENT_DURATION)
                    else:
                        if now - cat_detected_since >= DETERRENT_DURATION:
                            cat_detected_since = None
                            deterrent_active = False
            else:
                cat_detected_since = None
                deterrent_active = False

            if debug_mode:
                debug_draw(frame, detection)
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
