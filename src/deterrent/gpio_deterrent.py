import threading
import time
from typing import Optional

from src.deterrent._deterrent import Deterrent
from src.utils.logger import logger

try:
    import RPi.GPIO as GPIO  # type: ignore

    IS_PI = True
    logger.debug("Running on Pi")
except ImportError:
    IS_PI = False
    logger.debug("Not running on Pi")

PIN = 17  # GPIO pin


class GpioDeterrent(Deterrent):
    def __init__(self, pin: int = PIN) -> None:
        self.pin = pin
        self.deterrent_thread: Optional[threading.Thread] = None

    def setup(self):
        """Sets up the GPIO pin for the deterrent."""
        if not IS_PI:
            logger.debug("Skipping GPIO setup (not on Pi)")
            return
        GPIO.setmode(GPIO.BCM)  # type: ignore
        GPIO.setup(self.pin, GPIO.OUT)  # type: ignore
        logger.debug("GPIO setup complete")

    def _run_deterrent(self, duration: float) -> None:
        """Runs the deterrent for the specified duration in a separate thread."""
        if not IS_PI:
            logger.info(f"Simulated deterrent activated for {duration}s")
            time.sleep(duration)  # type: ignore
            return

        try:
            GPIO.output(self.pin, GPIO.HIGH)  # type: ignore
            time.sleep(duration)  # type: ignore
        finally:
            if IS_PI:
                GPIO.output(self.pin, GPIO.LOW)  # type: ignore
            logger.debug(f"Deterrent completed after {duration}s")

    def activate(self, duration: float) -> None:
        """Activates the deterrent for the specified duration."""
        self.deterrent_thread = threading.Thread(
            target=self._run_deterrent, args=(duration,)
        )
        self.deterrent_thread.daemon = True
        self.deterrent_thread.start()

    def cleanup(self) -> None:
        """Cleans up the GPIO pin."""
        if IS_PI:
            GPIO.cleanup(self.pin)  # type: ignore
            logger.debug("GPIO cleanup complete")
