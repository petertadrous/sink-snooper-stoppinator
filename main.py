from detection.detector import detect_cat
from hardware.camera import get_camera, read_frame
from hardware.deterrent import setup, activate_deterrent, cleanup
from config import DETERRENT_DURATION
from utils.logger import logger
import time


def main():
    logger.info("Starting Cat Sink Guard...")
    setup()
    cap = get_camera()

    try:
        while True:
            frame = read_frame(cap)
            if detect_cat(frame):
                activate_deterrent(DETERRENT_DURATION)
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        cleanup()
        cap.release()
        logger.info("System shut down cleanly")


if __name__ == "__main__":
    main()
