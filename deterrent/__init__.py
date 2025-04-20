from deterrent._deterrent import Deterrent
from deterrent.gpio_deterrent import GpioDeterrent
from deterrent.audio_deterrent import AudioDeterrent


def get_deterrent(deterrent_type: str) -> Deterrent:
    if deterrent_type == "gpio":
        return GpioDeterrent()
    if deterrent_type == "gunshots":
        return AudioDeterrent(audio_name="gunshots")

    raise ValueError(f"Unknown deterrent type: {deterrent_type}")
