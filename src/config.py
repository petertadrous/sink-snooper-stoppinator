CAMERA_INDEX: int = 1
INTERESTED_CLASSES: list[str] = [
    "cat",
    "dog",
    "teddy bear",
    "remote",
    "cell phone",
]
CONFIDENCE_THRESHOLD: float = 0.5
SCORE_THRESHOLD: float = 0.4
DETERRENT_DURATION: float = 1.5  # Seconds
FREQUENCY: float = 0.1  # Seconds
PREPROCESS_MODE = "pad"
DETECTION_HOLD_TIME: float = 1.0
# Options: "gpio", "gunshots", "speech", "llm"
DETERRENT_TYPE: str = "speech"

# Video recording settings
VIDEO_PRE_DETECTION_BUFFER: float = 3.0  # Seconds of video to save before detection
VIDEO_POST_DETECTION_BUFFER: float = 2.0  # Seconds of video to save after detection
VIDEO_OUTPUT_DIR: str = "recordings"  # Directory to save video recordings

# Debug settings
VIDEO_DEBUG_OVERLAY: bool = (
    True  # Include detection boxes and deterrent status in saved videos
)
