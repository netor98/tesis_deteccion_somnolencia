"""
Audio handling module for drowsiness detection system.

Handles audio frame processing and alarm playback.
"""

import av
import numpy as np
from pydub import AudioSegment


class AudioFrameHandler:
    """Handles audio frame processing and alarm playback based on detection events."""

    def __init__(self, sound_file_path: str = ""):
        """Initialize the audio handler.

        Args:
            sound_file_path: Path to the alarm sound file (WAV format)
        """
        self.custom_audio = AudioSegment.from_file(file=sound_file_path, format="wav")
        self.custom_audio_len = len(self.custom_audio)

        self.ms_per_audio_segment: int = 20
        self.audio_segment_shape: tuple = None

        self.play_state_tracker: dict = {"curr_segment": -1}  # Currently playing segment
        self.audio_segments_created: bool = False
        self.audio_segments: list = []

    def prepare_audio(self, frame: av.AudioFrame):
        """Prepare audio segments based on the input frame format.

        Args:
            frame: AudioFrame from the stream
        """
        raw_samples = frame.to_ndarray()
        sound = AudioSegment(
            data=raw_samples.tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels),
        )

        self.ms_per_audio_segment = len(sound)
        self.audio_segment_shape = raw_samples.shape

        self.custom_audio = self.custom_audio.set_channels(sound.channels)
        self.custom_audio = self.custom_audio.set_frame_rate(sound.frame_rate)
        self.custom_audio = self.custom_audio.set_sample_width(sound.sample_width)

        self.audio_segments = [
            self.custom_audio[i : i + self.ms_per_audio_segment]
            for i in range(0, self.custom_audio_len - self.custom_audio_len % self.ms_per_audio_segment,
                          self.ms_per_audio_segment)
        ]
        self.total_segments = len(self.audio_segments) - 1  # -1 because we start from 0.

        self.audio_segments_created = True

    def process(self, frame: av.AudioFrame, play_sound: bool = False):
        """Process audio frame and play alarm if requested.

        Takes in the current input audio frame and based on play_sound boolean value
        either starts sending the custom audio frame or dampens the frame wave to emulate silence.

        Args:
            frame: Input audio frame
            play_sound: Boolean indicating whether to play the alarm sound

        Returns:
            av.AudioFrame: Processed audio frame
        """
        if not self.audio_segments_created:
            self.prepare_audio(frame)

        raw_samples = frame.to_ndarray()
        _curr_segment = self.play_state_tracker["curr_segment"]

        if play_sound:
            if _curr_segment < self.total_segments:
                _curr_segment += 1
            else:
                _curr_segment = 0

            sound = self.audio_segments[_curr_segment]

        else:
            if -1 < _curr_segment < self.total_segments:
                _curr_segment += 1
                sound = self.audio_segments[_curr_segment]
            else:
                _curr_segment = -1
                sound = AudioSegment(
                    data=raw_samples.tobytes(),
                    sample_width=frame.format.bytes,
                    frame_rate=frame.sample_rate,
                    channels=len(frame.layout.channels),
                )
                sound = sound.apply_gain(-100)

        self.play_state_tracker["curr_segment"] = _curr_segment

        channel_sounds = sound.split_to_mono()
        channel_samples = [s.get_array_of_samples() for s in channel_sounds]

        new_samples = np.array(channel_samples).T

        new_samples = new_samples.reshape(self.audio_segment_shape)
        new_frame = av.AudioFrame.from_ndarray(new_samples, layout=frame.layout.name)
        new_frame.sample_rate = frame.sample_rate

        return new_frame


def play_alarm_sound(sound_file_path: str, duration: float = 2.0):
    """Play alarm sound directly (for standalone mode without WebRTC).

    Args:
        sound_file_path: Path to the alarm WAV file
        duration: Duration to play the sound in seconds
    """
    try:
        import subprocess
        import platform

        system = platform.system()

        if system == "Linux":
            # Try multiple players in order of preference
            players = ["aplay", "paplay", "ffplay"]

            for player in players:
                try:
                    if player == "ffplay":
                        # ffplay from ffmpeg
                        subprocess.run(
                            [player, "-nodisp", "-autoexit", "-t", str(duration), sound_file_path],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=duration + 1
                        )
                    else:
                        # aplay or paplay
                        subprocess.run(
                            [player, sound_file_path],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=duration + 1
                        )
                    return  # Success
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    continue

            # If no player worked, print warning
            print(f"⚠️  No audio player found (tried: {', '.join(players)})")

        elif system == "Darwin":  # macOS
            subprocess.run(
                ["afplay", sound_file_path],
                timeout=duration + 1
            )
        elif system == "Windows":
            import winsound
            winsound.PlaySound(sound_file_path, winsound.SND_FILENAME)
        else:
            print(f"⚠️  Audio playback not supported on {system}")

    except Exception as e:
        print(f"⚠️  Error playing alarm sound: {e}")

