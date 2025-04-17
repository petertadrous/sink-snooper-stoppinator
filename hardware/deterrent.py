from utils.logger import logger

try:
    import RPi.GPIO as GPIO  # type: ignore
    import time

    IS_PI = True
except ImportError:
    IS_PI = False

PIN = 17  # GPIO pin


def setup() -> None:
    """
    Sets up the GPIO pin for the deterrent.
    """
    if not IS_PI:
        logger.debug("Skipping GPIO setup (not on Pi)")
        return
    GPIO.setmode(GPIO.BCM)  # type: ignore
    GPIO.setup(PIN, GPIO.OUT)  # type: ignore
    logger.debug("GPIO setup complete")


def activate_deterrent(duration: float = 1) -> None:
    """
    Activates the deterrent for the specified duration.
    """
    if not IS_PI:
        logger.info(f"Simulated deterrent activated for {duration}s")
        return
    GPIO.output(PIN, GPIO.HIGH)  # type: ignore
    time.sleep(duration)  # type: ignore
    GPIO.output(PIN, GPIO.LOW)  # type: ignore
    logger.debug(f"Deterrent activated for {duration}s")


def cleanup() -> None:
    """
    Cleans up the GPIO pin.
    """
    if IS_PI:
        GPIO.cleanup()  # type: ignore
        logger.debug("GPIO cleanup complete")
