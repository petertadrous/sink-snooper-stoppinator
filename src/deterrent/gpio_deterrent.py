from src.deterrent._deterrent import Deterrent
from src.utils.logger import logger

try:
    import RPi.GPIO as GPIO  # type: ignore
    import time

    IS_PI = True
except ImportError:
    IS_PI = False

PIN = 17  # GPIO pin


class GpioDeterrent(Deterrent):
    def __init__(self, pin: int = PIN) -> None:
        self.pin = pin

    def setup(self):
        """
        Sets up the GPIO pin for the deterrent.
        """
        if not IS_PI:
            logger.debug("Skipping GPIO setup (not on Pi)")
            return
        GPIO.setmode(GPIO.BCM)  # type: ignore
        GPIO.setup(self.pin, GPIO.OUT)  # type: ignore
        logger.debug("GPIO setup complete")

    def activate(self, duration: float) -> None:
        """
        Activates the deterrent for the specified duration.
        """
        if not IS_PI:
            logger.info(f"Simulated deterrent activated for {duration}s")
            return
        GPIO.output(self.pin, GPIO.HIGH)  # type: ignore
        time.sleep(duration)  # type: ignore
        GPIO.output(self.pin, GPIO.LOW)  # type: ignore
        logger.debug(f"Deterrent activated for {duration}s")

    def cleanup(self) -> None:
        """
        Cleans up the GPIO pin.
        """
        if IS_PI:
            GPIO.cleanup()  # type: ignore
            logger.debug("GPIO cleanup complete")
