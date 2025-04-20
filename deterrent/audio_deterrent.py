from typing import Optional

import traceback

from pydub import AudioSegment
from pydub.playback import play

from utils.logger import logger


def splice_and_loop_mp3(
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


def _play_gunshots(duration: float = 1.5) -> AudioSegment:
    """
    Plays gunshot audio for a specified duration.
    Audio is free from https://pixabay.com/sound-effects/clean-machine-gun-burst-98224/
    """
    input_file = "assets/clean-machine-gun-burst-98224.mp3"
    audio: AudioSegment = AudioSegment.from_mp3(input_file)
    original_duration = audio.duration_seconds
    audio: AudioSegment = audio + AudioSegment.silent(duration=500)
    if duration <= original_duration:
        return audio
    end_time = 1.43  # seconds
    tail_time = original_duration - end_time

    loop_times = int((duration - tail_time) / end_time)

    return splice_and_loop_mp3(audio, start_time=0, end_time=end_time * 1000, loop_times=loop_times)


def play_audio(audio_name: str, duration: float = 1.5) -> None:
    if audio_name == "gunshots":
        audio = _play_gunshots(duration)
    else:
        raise ValueError(f"Unknown audio name: {audio_name}")

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
