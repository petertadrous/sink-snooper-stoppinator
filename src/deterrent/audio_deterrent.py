import threading
from typing import Optional

import traceback

from pydub import AudioSegment
from pydub.playback import play

from src.utils.logger import logger
from src.deterrent._deterrent import Deterrent


class AudioDeterrent(Deterrent):
    def __init__(self, audio_name: str = "gunshots") -> None:
        self.audio_name = audio_name
        self.audio: Optional[AudioSegment] = None
        self.play_thread: Optional[threading.Thread] = None

    def setup(self):
        if self.audio_name == "gunshots":
            self.input_file = "assets/clean-machine-gun-burst-98224.mp3"
            self.audio = AudioSegment.from_mp3(self.input_file)
            return
        raise ValueError(f"Unknown audio name: {self.audio_name}")

    def _play_audio(self, audio: AudioSegment):
        try:
            play(audio)
        except KeyboardInterrupt:
            # Do nothing
            pass
        except Exception as e:
            logger.error(f"Playback failed: {e}")
            logger.debug(traceback.format_exc())

    def activate(self, duration: float) -> None:
        if self.audio_name == "gunshots":
            if self.audio is None:
                raise RuntimeError("Audio not initialized")
            audio = self._loop_gunshots(duration)
        else:
            raise ValueError(f"Unknown audio name: {self.audio_name}")

        logger.debug(f"Playing audio for {duration} seconds")
        # Start audio playback in a separate thread
        self.play_thread = threading.Thread(target=self._play_audio, args=(audio,))
        self.play_thread.daemon = True  # Thread will be killed when main program exits
        self.play_thread.start()

    def cleanup(self):
        if self.play_thread and self.play_thread.is_alive():
            # Can't really stop pydub playback, but clear the reference
            self.play_thread = None
        self.audio = None

    @staticmethod
    def _splice_and_loop_mp3(
        audio: AudioSegment,
        start_time: float,
        end_time: float,
        loop_times: int,
        cross_fade: Optional[int] = 100,
    ) -> AudioSegment:
        """
        Splices an audio segment and loops a segment for a specified number of times.
        """
        segment: AudioSegment = audio[start_time:end_time]  # type: ignore[misc]
        looped_segment: AudioSegment = segment
        for _ in range(loop_times - 1):
            if cross_fade is not None:
                looped_segment = looped_segment.append(segment, crossfade=cross_fade)
            else:
                looped_segment += segment
        looped_segment += audio[end_time:]  # type: ignore[misc]

        return looped_segment

    def _loop_gunshots(self, duration: float = 1.5) -> AudioSegment:
        """
        Loops gunshot audio for a specified duration.
        Audio is free from https://pixabay.com/sound-effects/clean-machine-gun-burst-98224/
        """
        if not self.audio:
            raise RuntimeError("Audio not initialized")

        # Get audio length in milliseconds
        audio_length = len(self.audio)
        if audio_length == 0:
            raise RuntimeError("Audio file has zero length")

        # Calculate how many times to loop
        loop_count = max(1, int((duration * 1000) / audio_length))
        return self.audio * loop_count
