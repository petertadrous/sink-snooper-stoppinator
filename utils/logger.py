from loguru import logger
import sys

logger.remove()

# Info and below -> stdout
logger.add(
    sys.stdout,
    level="DEBUG",
    filter=lambda record: record["level"].no < 40,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
)

# Errors and above -> stderr
logger.add(
    sys.stderr, level="ERROR", format="<red>{time:HH:mm:ss}</red> | <level>{message}</level>"
)
