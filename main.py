import time
import argparse

import cv2

from detection.detector import detect_cat, debug_draw
from hardware.camera import get_camera, read_frame
from hardware.deterrent import setup, activate_deterrent, cleanup
from config import DETERRENT_DURATION
from utils.logger import logger


def main():
    args = parse_args()
    debug_mode = args.debug

    logger.info("Starting Sink Snooper Stoppinator...")
    setup()
    cap = get_camera()

    try:
        while True:
            frame = read_frame(cap)
            detection = detect_cat(frame, debug=debug_mode)

            if detection["detected"]:
                activate_deterrent(DETERRENT_DURATION)

            if debug_mode:
                debug_draw(frame, detection)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Exiting debug mode")
                    break

            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        cleanup()
        cap.release()
        if debug_mode:
            cv2.destroyAllWindows()
        logger.info("System shut down cleanly")


def parse_args():
    parser = argparse.ArgumentParser(description="Sink Snooper Stoppinator")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug webcam view with detection overlay"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
