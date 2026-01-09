# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading
from typing import Literal

import numpy as np

from arduino.app_utils import Logger, brick
from arduino.app_peripherals.speaker import Speaker, BaseSpeaker, ALSASpeaker

logger = Logger("WaveGenerator")

type WaveType = Literal["sine", "square", "sawtooth", "triangle"]


@brick
class WaveGenerator:
    """
    Continuous wave generator brick for audio synthesis.

    This brick generates continuous audio waveforms (sine, square, sawtooth, triangle)
    and streams them to a Speaker in real-time. It provides smooth transitions
    between frequency and amplitude changes using configurable envelope parameters.

    The generator runs continuously in a background thread, producing audio blocks
    with minimal latency.
    """

    def __init__(
        self,
        speaker: BaseSpeaker | None = None,
        wave_type: WaveType = "sine",
        attack: float = 0.01,
        release: float = 0.03,
        glide: float = 0.02,
    ):
        """
        Initialize the WaveGenerator brick.

        Args:
            speaker (BaseSpeaker): Pre-configured Speaker instance. If None, WaveGenerator
                will create an internal Speaker optimized for real-time synthesis. If provided,
                ensure it uses np.float32 format and appropriate latency settings.
            wave_type (WaveType): Initial waveform type (default: "sine").
            attack (float): Attack time for amplitude envelope in seconds (default: 0.01).
            release (float): Release time for amplitude envelope in seconds (default: 0.03).
            glide (float): Frequency glide time (portamento) in seconds (default: 0.02).

            Example external Speaker configuration:
            ```python
            speaker = Speaker(
                device="plughw:CARD=MyCard,DEV=0",
                sample_rate=Speaker.RATE_48K,
                channels=2,
                format=np.float32,
            )
            ```

        Raises:
            SpeakerException: If no USB speaker is found or device is busy.
        """
        if speaker is None:
            # Create internal Speaker instance optimized for real-time synthesis
            self._speaker = ALSASpeaker(
                device=Speaker.USB_SPEAKER_1,
                sample_rate=Speaker.RATE_48K,
                channels=Speaker.CHANNELS_MONO,
                format=np.float32,
                buffer_size=Speaker.BUFFER_SIZE_REALTIME,
                shared=False,
            )
        else:
            if speaker.format != np.float32:
                raise ValueError("Provided Speaker must use np.float32 format for real-time synthesis")
            self._speaker = speaker

        self.volume = 100  # Set default volume to max

        # Target state (set by user)
        self._wave_type: WaveType = wave_type
        self._frequency = 440.0
        self._amplitude = 0.0
        self._attack = float(attack)
        self._release = float(release)
        self._glide = float(glide)

        # Internal audio state (set by audio thread)
        self._prev_frequency = self._frequency
        self._prev_amplitude = self._amplitude
        self._prev_phase = 0.0
        self._amp_ramp_start = self._amplitude  # Amplitude at start of current ramp
        self._amp_ramp_target = self._amplitude  # Target amplitude for current ramp
        self._amp_ramp_duration = 0.0  # Total duration of current ramp
        self._amp_ramp_elapsed = 0.0  # Time elapsed in current ramp
        self._freq_glide_start = self._frequency  # Frequency at start of current glide
        self._freq_glide_target = self._frequency  # Target frequency for current glide
        self._freq_glide_elapsed = 0.0  # Time elapsed in current glide

        # Number of ALSA frames to generate for each audio block produced
        self._block_frame_count = max(32, min(self._speaker.buffer_size, self._speaker.buffer_size // 4))
        self._block_duration = self._block_frame_count / float(self.sample_rate)

        # Pre-allocate buffers during instance initialization
        # Holds the static ramp vector (0.0 to 1.0) for interpolation
        self._ramp_vec = np.linspace(0.0, 1.0, self._block_frame_count, dtype=np.float32)
        # Holds the phase angle (rad) for every sample in the current block
        self._buf_phases = np.zeros(self._block_frame_count, dtype=np.float32)
        # Holds the volume multiplier for every sample in the current block used in attack/release
        self._buf_envelope = np.zeros(self._block_frame_count, dtype=np.float32)
        # Holds the output samples for the current block
        self._buf_samples = np.zeros(self._block_frame_count, dtype=np.float32)

        self._two_pi = np.float32(2.0 * np.pi)

        self._running = threading.Event()

    @property
    def wave_type(self) -> WaveType:
        """
        Get or set the current waveform type.

        Args:
            wave_type (WaveType): One of "sine", "square", "sawtooth", "triangle".

        Returns:
            WaveType: Current waveform type ("sine", "square", "sawtooth", "triangle").
        """
        return self._wave_type

    @wave_type.setter
    def wave_type(self, wave_type: WaveType):
        valid_types = ("sine", "square", "sawtooth", "triangle")
        if wave_type not in valid_types:
            raise ValueError(f"Invalid wave_type '{wave_type}'. Must be one of {valid_types}")

        self._wave_type = wave_type

    @property
    def sample_rate(self) -> int:
        """
        Get the audio sample rate in Hz.

        Returns:
            int: Sample rate in Hz.

        Raises:
            RuntimeError: If no speaker is configured.
        """
        if self._speaker is None:
            raise RuntimeError("Speaker is not configured")

        return self._speaker.sample_rate

    @property
    def frequency(self) -> float:
        """
        Get or set the current output frequency in Hz.

        The frequency will smoothly transition to the new value over the
        configured glide time.

        Args:
            frequency (float): Target frequency in Hz (typically 20-8000 Hz).

        Returns:
            float: Current output frequency in Hz.

        Raises:
            ValueError: If the frequency is negative.
        """
        return self._frequency

    @frequency.setter
    def frequency(self, freq: float):
        if freq < 0.0:
            raise ValueError(f"Invalid frequency '{freq}'. Must be non-negative")

        self._frequency = freq

    @property
    def amplitude(self) -> float:
        """
        Get or set the current output amplitude.

        The amplitude will smoothly transition to the new value over the
        configured attack/release time.

        Args:
            amplitude (float): Target amplitude in range [0.0, 1.0].

        Returns:
            float: Current output amplitude (0.0-1.0).

        Raises:
            ValueError: If the amplitude is not in range [0.0, 1.0].
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, amp: float):
        if amp < 0.0 or amp > 1.0:
            raise ValueError(f"Invalid amplitude '{amp}'. Must be in range [0.0, 1.0]")

        self._amplitude = amp

    @property
    def attack(self) -> float:
        """
        Get or set the current attack time in seconds.

        Attack time controls how quickly the amplitude rises to the target value.

        Args:
            attack (float): Attack time in seconds.

        Returns:
            float: Current attack time in seconds.

        Raises:
            ValueError: If the attack time is negative.
        """
        return self._attack

    @attack.setter
    def attack(self, attack: float):
        if attack < 0.0:
            raise ValueError(f"Invalid attack time '{attack}'. Must be non-negative")

        self._attack = attack

    @property
    def release(self) -> float:
        """
        Get or set the current release time in seconds.

        Release time controls how quickly the amplitude falls to the target value.

        Args:
            release (float): Release time in seconds.

        Returns:
            float: Current release time in seconds.

        Raises:
            ValueError: If the release time is negative.
        """
        return self._release

    @release.setter
    def release(self, release: float):
        if release < 0.0:
            raise ValueError(f"Invalid release time '{release}'. Must be non-negative")

        self._release = release

    @property
    def glide(self) -> float:
        """
        Get the current frequency glide time in seconds (portamento).

        Glide time controls how quickly the frequency transitions to the target value.

        Args:
            glide (float): Frequency glide time in seconds.

        Returns:
            float: Current frequency glide time in seconds.

        Raises:
            ValueError: If the glide time is negative.
        """
        return self._glide

    @glide.setter
    def glide(self, glide: float):
        if glide < 0.0:
            raise ValueError(f"Invalid glide time '{glide}'. Must be non-negative")

        self._glide = glide

    @property
    def volume(self) -> int | None:
        """
        Get or set the wave generator volume level.

        Args:
            volume (int): Hardware volume level (0-100).

        Returns:
            int: Current volume level (0-100).

        Raises:
            ValueError: If the volume is not in range [0, 100].
        """
        return self._speaker.volume

    @volume.setter
    def volume(self, volume: int):
        self._speaker.volume = volume

    @property
    def state(self) -> dict:
        """
        Get current generator state.

        Returns:
            dict: Dictionary containing current frequency, amplitude, wave type, etc.
        """
        return {
            "amplitude": self._amplitude,
            "frequency": self._frequency,
            "wave_type": self._wave_type,
            "attack": self._attack,
            "release": self._release,
            "glide": self._glide,
            "volume": self.volume,
        }

    def start(self):
        """
        Start the wave generator and audio output.

        This starts the speaker device too.
        """
        if self._running.is_set():
            logger.warning("WaveGenerator is already running")
            return

        logger.info("Starting WaveGenerator...")
        self._speaker.start()
        self._running.set()
        logger.info("WaveGenerator started")

    def stop(self):
        """
        Stop the wave generator and audio output.

        This stops the speaker device too.
        """
        if not self._running.is_set():
            logger.warning("WaveGenerator is not running")
            return

        logger.info("Stopping WaveGenerator...")
        self._running.clear()
        self._speaker.stop()
        logger.info("WaveGenerator stopped")

    @brick.execute
    def _wave_generator_loop(self):
        logger.debug(f"Generator loop started. Block frame size: {self._block_frame_count}, Rate: {self.sample_rate}.")

        while self._running.is_set():
            try:
                buf_samples = self._generate_audio_block()
                # We rely on speaker.play() to block if the hardware buffer is full.
                # This maintains synchronization with the audio device.
                self._speaker.play(buf_samples)
            except Exception as e:
                logger.error(f"Failed to generate audio block: {e}")

    def _generate_audio_block(self) -> np.ndarray:
        """
        Generate a single audio block.

        Returns:
            numpy.ndarray: The generated audio block
        """
        # INITIALIZATION
        # Localize variables to stack for speed
        block_frame_count = self._block_frame_count
        block_duration = self._block_duration
        sample_rate = float(self.sample_rate)
        two_pi = self._two_pi
        ramp_vec = self._ramp_vec

        # Local buffers
        buf_phases = self._buf_phases
        buf_samples = self._buf_samples
        buf_envelope = self._buf_envelope

        # Target parameters
        frequency = self._frequency
        amplitude = self._amplitude
        wave_type = self._wave_type
        glide = self._glide
        attack = self._attack
        release = self._release

        # FREQUENCY & PHASE CALCULATION
        current_freq = self._prev_frequency
        if current_freq == frequency:
            # Frequency is constant
            # phases = (arange * inc) + start_phase
            inc = (frequency * two_pi) / sample_rate
            np.multiply(np.arange(1, block_frame_count + 1, dtype=np.float32), inc, out=buf_phases)
            np.add(buf_phases, self._prev_phase, out=buf_phases)

            # Update state for next block
            self._prev_phase = buf_phases[-1] % two_pi
        else:
            # Frequency is changing
            # Check if this is a new glide (target changed)
            if frequency != self._freq_glide_target:
                # Start a new glide
                self._freq_glide_start = current_freq
                self._freq_glide_target = frequency
                self._freq_glide_elapsed = 0.0

            if glide <= 0.0:
                # Gliding is disabled, jump immediately
                inc = (frequency * two_pi) / sample_rate
                np.multiply(np.arange(1, block_frame_count + 1, dtype=np.float32), inc, out=buf_phases)
                np.add(buf_phases, self._prev_phase, out=buf_phases)
                current_freq = frequency
                self._freq_glide_elapsed = 0.0
            else:
                # Gliding is enabled, linear interpolation based on time
                glide_start = self._freq_glide_start
                glide_target = self._freq_glide_target
                elapsed = self._freq_glide_elapsed

                # Calculate progress through the glide
                progress_start = min(elapsed / glide, 1.0)
                progress_end = min((elapsed + block_duration) / glide, 1.0)

                freq_start = glide_start + (glide_target - glide_start) * progress_start
                freq_end = glide_start + (glide_target - glide_start) * progress_end

                # buf_phases temporarily holds the frequencies
                # freq[i] = freq_start + (freq_end - freq_start) * ramp[i]
                np.subtract(freq_end, freq_start, out=buf_phases)  # delta
                np.multiply(buf_phases, ramp_vec, out=buf_phases)  # delta * ramp
                np.add(buf_phases, freq_start, out=buf_phases)  # start + delta*ramp

                # Convert Freq to Phase Increment: inc = freq * 2pi / rate
                np.multiply(buf_phases, two_pi / sample_rate, out=buf_phases)

                # Accumulate Phase
                np.cumsum(buf_phases, out=buf_phases)
                np.add(buf_phases, self._prev_phase, out=buf_phases)

                current_freq = freq_end
                self._freq_glide_elapsed += block_duration

            self._prev_frequency = current_freq
            self._prev_phase = buf_phases[-1] % two_pi

        # Wrap phases to [0, 2pi) to maintain floating point alignment
        # avoid accumulating floating point errors over time
        np.mod(buf_phases, two_pi, out=buf_phases)

        # AMPLITUDE ENVELOPE CALCULATION
        prev_amp = self._prev_amplitude
        if prev_amp == amplitude:
            # Already at target amplitude
            amp_start = amplitude
            amp_end = amplitude
        else:
            # Check if this is a new ramp (target changed)
            if amplitude != self._amp_ramp_target:
                # Start a new ramp
                self._amp_ramp_start = prev_amp
                self._amp_ramp_target = amplitude
                self._amp_ramp_elapsed = 0.0
                self._amp_ramp_duration = attack if amplitude > prev_amp else release

            ramp_duration = self._amp_ramp_duration
            if ramp_duration <= 0.0:
                # Ramp disabled, instant change
                amp_start = amplitude
                amp_end = amplitude
                self._amp_ramp_elapsed = 0.0
            else:
                # Ramp enabled, calculate progress
                ramp_start = self._amp_ramp_start
                ramp_target = self._amp_ramp_target
                elapsed = self._amp_ramp_elapsed

                # Linear interpolation based on time
                progress_start = min(elapsed / ramp_duration, 1.0)
                progress_end = min((elapsed + block_duration) / ramp_duration, 1.0)

                amp_start = ramp_start + (ramp_target - ramp_start) * progress_start
                amp_end = ramp_start + (ramp_target - ramp_start) * progress_end
                self._amp_ramp_elapsed += block_duration

        if amp_start == 0.0 and amp_end == 0.0:
            # Entire block is silent, skip waveform generation
            buf_samples.fill(0.0)
            self._prev_amplitude = amp_end
            return buf_samples

        # WAVEFORM GENERATION
        if wave_type == "sine":
            np.sin(buf_phases, out=buf_samples)
        elif wave_type == "square":
            # np.sign(sin(x)) gives -1 or 1
            np.sin(buf_phases, out=buf_samples)
            np.sign(buf_samples, out=buf_samples)
        elif wave_type == "sawtooth":
            # (phase / 2pi) * 2 - 1
            np.multiply(buf_phases, 1.0 / two_pi, out=buf_samples)  # 0..1
            np.multiply(buf_samples, 2.0, out=buf_samples)  # 0..2
            np.subtract(buf_samples, 1.0, out=buf_samples)  # -1..1
        elif wave_type == "triangle":
            # 2 * abs(2 * (phase/2pi - 0.5)) - 1  ... approx
            # Let's use the saw based approach: abs(saw) * 2 - 1
            np.multiply(buf_phases, 1.0 / two_pi, out=buf_samples)  # 0..1
            np.subtract(buf_samples, 0.5, out=buf_samples)  # -0.5..0.5
            np.abs(buf_samples, out=buf_samples)  # 0..0.5
            np.multiply(buf_samples, 4.0, out=buf_samples)  # 0..2
            np.subtract(buf_samples, 1.0, out=buf_samples)  # -1..1
        else:
            np.sin(buf_phases, out=buf_samples)

        # APPLY AMPLITUDE ENVELOPE
        if amp_start == amp_end:
            # Constant amplitude
            if amp_start != 1.0:
                np.multiply(buf_samples, amp_start, out=buf_samples)
        else:
            # Variable amplitude
            np.subtract(amp_end, amp_start, out=buf_envelope)
            np.multiply(buf_envelope, ramp_vec, out=buf_envelope)
            np.add(buf_envelope, amp_start, out=buf_envelope)
            np.multiply(buf_samples, buf_envelope, out=buf_samples)

        self._prev_amplitude = amp_end

        return buf_samples
