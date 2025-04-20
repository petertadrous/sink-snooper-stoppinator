from typing import Optional

import traceback

from pydub import AudioSegment
from pydub.playback import play

from utils.logger import logger

from deterrent._deterrent import Deterrent


class AudioDeterrent(Deterrent):
    def __init__(self, audio_name: str = "gunshots") -> None:
        self.audio_name = audio_name

    def setup(self):
        if self.audio_name != "gunshots":
            self.input_file = "assets/clean-machine-gun-burst-98224.mp3"
            self.audio: AudioSegment = AudioSegment.from_mp3(self.input_file)
            return
        raise ValueError(f"Unknown audio name: {self.audio_name}")

    def activate(self, duration: float) -> None:
        if self.audio_name == "gunshots":
            audio = self._loop_gunshots(duration)
        else:
            raise ValueError(f"Unknown audio name: {self.audio_name}")

        try:
            logger.debug(f"Playing audio for {duration} seconds")
            play(audio)
        except KeyboardInterrupt:
            # Do nothing
            raise
        except Exception as e:
            logger.error(f"Playback failed: {e}")
            logger.debug(traceback.format_exc())
            raise e
        else:
            logger.debug("Playback finished successfully")

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
        segment: AudioSegment = audio[start_time:end_time]  # type: ignore
        looped_segment: AudioSegment = segment
        for _ in range(loop_times - 1):
            if cross_fade is not None:
                looped_segment = looped_segment.append(segment, crossfade=cross_fade)
            else:
                looped_segment += segment
        looped_segment += audio[end_time:]

        return looped_segment

    def _loop_gunshots(self, duration: float = 1.5) -> AudioSegment:
        """
        Plays gunshot audio for a specified duration.
        Audio is free from https://pixabay.com/sound-effects/clean-machine-gun-burst-98224/
        """
        audio = self.audio
        original_duration = audio.duration_seconds
        audio: AudioSegment = audio + AudioSegment.silent(duration=500)
        if duration <= original_duration:
            return audio
        end_time = 1.43  # seconds
        tail_time = original_duration - end_time

        loop_times = int((duration - tail_time) / end_time)

        return self._splice_and_loop_mp3(
            audio, start_time=0, end_time=end_time * 1000, loop_times=loop_times
        )

    def cleanup(self):
        self.audio = None  # type: ignore
