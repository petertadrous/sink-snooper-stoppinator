from src.deterrent._deterrent import Deterrent
from src.deterrent.gpio_deterrent import GpioDeterrent
from src.deterrent.audio_deterrent import AudioDeterrent
from src.deterrent.speech_deterrent import SpeechDeterrent


def get_deterrent(deterrent_type: str) -> Deterrent:
    if deterrent_type == "gpio":
        return GpioDeterrent()
    if deterrent_type == "gunshots":
        return AudioDeterrent(audio_name="gunshots")
    if deterrent_type == "speech":
        return SpeechDeterrent(creative=False)
    if deterrent_type == "llm":
        return SpeechDeterrent(creative=True)

    raise ValueError(f"Unknown deterrent type: {deterrent_type}")
