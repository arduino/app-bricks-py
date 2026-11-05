# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import io
import math
import os
import wave
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    # numpy is imported where it is used: arduino.app_utils imports this module, a module-level import would slow down every app start
    import numpy as np
    from numpy.typing import NDArray


class SineGenerator:
    """Generate sine-wave audio blocks with amplitude envelope smoothing.

    This class produces contiguous sine-wave blocks while maintaining internal
    state (phase, last amplitude, last frequency) so consecutive calls do not
    introduce discontinuities. It uses preallocated NumPy buffers and performs
    in-place operations for efficiency.

    Attributes:
        sample_rate (int): Audio sample rate in Hz.
        attack (float): Attack time for amplitude smoothing in seconds.
        release (float): Release time for amplitude smoothing in seconds.
        glide (float): Glide time for frequency smoothing in seconds.
    """

    def __init__(self, sample_rate: int) -> None:
        """Create a new SineGenerator.

        Args:
            sample_rate (int): The playback sample rate (Hz) used to compute
                phase increments and buffer sizes.
        """
        import numpy as np

        self.sample_rate = int(sample_rate)
        # envelope parameters (attack/release in seconds)
        self.attack = 0.01
        self.release = 0.03
        self.glide = 0.02

        # reusable buffers
        # grown on demand by generate_block, see there
        self._buf_N = 0
        self._buf_phase_incs: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._buf_phases: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._buf_envelope: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._buf_samples: NDArray[np.float32] = np.empty(0, dtype=np.float32)

        # runtime state
        self._amp_current = 0.0
        self._freq_last = 440.0
        self._phase = 0.0

    def reset(self) -> None:
        """Reset internal generator state.

        Resets phase, last frequency and current amplitude to silence. Useful
        when reinitializing playback or ensuring a known baseline before
        tests.
        """
        self._amp_current = 0.0
        self._freq_last = 440.0
        self._phase = 0.0

    def get_state(self) -> dict[str, float]:
        """Return a snapshot of internal generator state.

        Returns a small dict containing ``phase``, ``amp_current`` and
        ``freq_last`` suitable for serialization or for later restoration via
        :meth:`set_state`.

        Returns:
            dict: {'phase': float, 'amp_current': float, 'freq_last': float}
        """
        return {
            "phase": float(self._phase),
            "amp_current": float(self._amp_current),
            "freq_last": float(self._freq_last),
        }

    def set_state(self, state: dict[str, float]) -> None:
        """Restore internal generator state from a snapshot.

        Args:
            state (dict): State dict with keys ``phase``, ``amp_current`` and
                ``freq_last``. Missing keys are ignored and current values are
                preserved.
        """
        if not state:
            return
        if "phase" in state:
            self._phase = float(state["phase"])
        if "amp_current" in state:
            self._amp_current = float(state["amp_current"])
        if "freq_last" in state:
            self._freq_last = float(state["freq_last"])

    def set_envelope_params(self, attack: float, release: float, glide: float) -> None:
        """Update attack and release envelope parameters.

        Args:
            attack (float): Attack time in seconds (time to rise to target
                amplitude when increasing amplitude).
            release (float): Release time in seconds (time to fall to target
                amplitude when decreasing amplitude).
            glide (float): Glide time in seconds (time to reach target frequency).
        """
        self.attack = float(max(0.0, attack))
        self.release = float(max(0.0, release))
        self.glide = float(max(0.0, glide))

    def generate_block(self, freq: float, amp_target: float, block_dur: float, master_volume: float) -> np.ndarray:
        """Generate a block of float32 audio samples.

        The generator keeps internal phase continuity across calls. Amplitude is
        smoothed between the previous amplitude and ``amp_target`` using the
        configured ``attack`` and ``release`` times. Returned buffer is a
        NumPy view (float32) into an internal preallocated array and is valid
        until the next call to this method.

        Args:
            freq (float): Target frequency in Hz for this block.
            amp_target (float): Target amplitude in range [0.0, 1.0].
            block_dur (float): Duration of the requested block in seconds.
            master_volume (float, optional): Global gain multiplier. Defaults
                to 1.0.

        Returns:
            numpy.ndarray: A 1-D float32 NumPy array containing the generated
            audio samples for the requested block.
        """
        import numpy as np

        N = max(1, int(self.sample_rate * block_dur))
        if N > self._buf_N:
            self._buf_N = N
            self._buf_phase_incs = np.empty(self._buf_N, dtype=np.float32)
            self._buf_phases = np.empty(self._buf_N, dtype=np.float32)
            self._buf_envelope = np.empty(self._buf_N, dtype=np.float32)
            self._buf_samples = np.empty(self._buf_N, dtype=np.float32)

        phases = self._buf_phases[:N]
        envelope = self._buf_envelope[:N]
        samples = self._buf_samples[:N]

        # amplitude smoothing (use instance params)
        attack = float(self.attack)
        release = float(self.release)
        amp_target = float(max(0.0, min(1.0, amp_target)))
        amp_current = float(self._amp_current)
        if amp_target == amp_current or (attack <= 0.0 and release <= 0.0):
            envelope.fill(amp_target)
        else:
            ramp = attack if amp_target > amp_current else release
            if ramp <= 0.0:
                envelope.fill(amp_target)
            else:
                frac = min(1.0, block_dur / float(ramp))
                next_amp = amp_current + (amp_target - amp_current) * frac
                envelope[:] = np.linspace(amp_current, next_amp, N, dtype=np.float32)
                amp_current = float(envelope[-1])

        # frequency glide (portamento)
        freq_current = float(self._freq_last)
        freq_target = float(freq)
        glide = float(self.glide)
        phase_incs = self._buf_phase_incs[:N]

        if glide > 0.0 and freq_current != freq_target:
            # Apply glide smoothing over time
            frac = min(1.0, block_dur / glide)
            next_freq = freq_current + (freq_target - freq_current) * frac

            # Linear interpolation within block
            freq_ramp = np.linspace(freq_current, next_freq, N, dtype=np.float32)
            phase_incs[:] = 2.0 * math.pi * freq_ramp / float(self.sample_rate)

            freq_current = float(next_freq)
        else:
            # No glide or already at target
            phase_incr = 2.0 * math.pi * freq_target / float(self.sample_rate)
            phase_incs.fill(phase_incr)
            freq_current = freq_target

        # oscillator (phase accumulation)
        np.cumsum(phase_incs, dtype=np.float32, out=phases)
        phases += self._phase
        self._phase = float(phases[-1] % (2.0 * math.pi))

        # compute sine
        np.sin(phases, out=samples)

        # apply envelope and gain
        np.multiply(samples, envelope, out=samples)
        mg = float(master_volume)
        if mg != 1.0:
            np.multiply(samples, mg, out=samples)

        # update state
        self._amp_current = amp_current
        self._freq_last = freq_current

        return samples


AudioFormat = Literal["wav", "mp3"]

# The value of a `type` alias is evaluated lazily, so numpy is only needed for type checking.
type AudioInput = str | bytes | np.ndarray[Any, Any]
"""An audio clip: a WAV/MP3 file path, WAV/MP3 file bytes, or a numpy array from the Microphone peripheral."""

_EXTENSION_FORMATS: dict[str, AudioFormat] = {".wav": "wav", ".mp3": "mp3"}


@dataclass(frozen=True)
class AudioClip:
    """An audio clip encoded as a complete WAV or MP3 file.

    The constructors accept the forms an app has audio in: a file path, encoded file
    bytes, or the numpy arrays returned by the Microphone peripheral.

    Attributes:
        data (bytes): The encoded file content.
        format (AudioFormat): "wav" or "mp3".
    """

    # A whole file in repr() would flood logs and tracebacks.
    data: bytes = field(repr=False)
    format: AudioFormat

    @staticmethod
    def detect_format(data: bytes) -> AudioFormat | None:
        """Detects whether encoded audio bytes are a WAV or an MP3 file.

        Args:
            data (bytes): The encoded file content.

        Returns:
            AudioFormat | None: "wav" or "mp3", or None when the header is neither.
        """
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return "wav"
        # An ID3 tag, or a bare MPEG audio frame (11-bit frame sync).
        if data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
            return "mp3"
        return None

    @classmethod
    def from_file(cls, path: str) -> AudioClip:
        """Reads a .wav or .mp3 file.

        Args:
            path (str): The file path. Its extension tells the format.

        Returns:
            AudioClip: The file content.

        Raises:
            ValueError: If the extension is not .wav or .mp3.
            FileNotFoundError: If the file does not exist.
        """
        audio_format = _EXTENSION_FORMATS.get(os.path.splitext(path)[1].lower())
        if audio_format is None:
            raise ValueError(f"Unsupported audio file '{path}': only .wav and .mp3 files are supported.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        with open(path, "rb") as f:
            return cls(f.read(), audio_format)

    @classmethod
    def from_bytes(cls, data: bytes) -> AudioClip:
        """Wraps the content of a WAV or MP3 file, telling the format from its header.

        Args:
            data (bytes): The encoded file content.

        Returns:
            AudioClip: The file content.

        Raises:
            ValueError: If the header is neither WAV nor MP3.
        """
        audio_format = cls.detect_format(data)
        if audio_format is None:
            raise ValueError("Unrecognized audio bytes: pass the content of a WAV or MP3 file.")
        return cls(data, audio_format)

    @classmethod
    def from_pcm(cls, samples: np.ndarray[Any, Any], sample_rate: int = 16000, channels: int = 1) -> AudioClip:
        """Wraps raw PCM samples in a 16-bit WAV file.

        Args:
            samples (np.ndarray): int16 samples, or float32/float64 samples normalized to [-1.0, 1.0].
                Multi-channel audio is either interleaved in a flat array or shaped (frames, channels).
            sample_rate (int): Sample rate in Hz. Defaults to 16000, the Microphone default.
            channels (int): Channel count of a flat array. Defaults to 1 (mono).

        Returns:
            AudioClip: A WAV file.

        Raises:
            ValueError: If the array shape or the sample type is not supported.
        """
        import numpy as np

        if samples.ndim == 2:
            channels = int(samples.shape[1])
        elif samples.ndim != 1:
            raise ValueError(f"PCM audio must be a 1-D or (frames, channels) array, got shape {samples.shape}.")

        if samples.dtype.kind == "f":
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        elif samples.dtype.kind == "i" and samples.dtype.itemsize == 2:
            pcm = samples.astype("<i2", copy=False)
        else:
            raise ValueError(
                f"Unsupported PCM sample type {samples.dtype}: pass int16 or float samples, or the WAV file returned by Microphone.record_wav()."
            )

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return cls(buffer.getvalue(), "wav")

    @classmethod
    def from_input(cls, audio: AudioInput) -> AudioClip:
        """Encodes an audio clip given in any of the supported forms.

        Args:
            audio (AudioInput): A .wav/.mp3 file path, WAV/MP3 file bytes, the uint8 WAV array
                returned by ``Microphone.record_wav()``, or raw PCM samples (int16 or float in
                [-1.0, 1.0]) as returned by ``Microphone.record_pcm()``. Raw PCM is assumed to be
                16 kHz mono, the Microphone defaults; for other rates or channel counts use
                ``from_pcm()``, or record with ``record_wav()``, which stores them in the WAV header.

        Returns:
            AudioClip: The encoded clip.

        Raises:
            FileNotFoundError: If a file path does not exist.
            ValueError: If the file type, header or sample type is not supported.
            TypeError: If the input is none of the supported types.
        """
        if isinstance(audio, str):
            return cls.from_file(audio)

        if isinstance(audio, (bytes, bytearray, memoryview)):
            return cls.from_bytes(bytes(audio))

        dtype = getattr(audio, "dtype", None)
        if dtype is None:
            raise TypeError(f"Unsupported audio input of type {type(audio).__name__}: pass a file path, WAV/MP3 bytes or a numpy array.")

        # Microphone.record_wav() returns the whole WAV file as a uint8 array.
        if dtype.kind == "u" and dtype.itemsize == 1:
            data = audio.tobytes()
            audio_format = cls.detect_format(data)
            if audio_format is None:
                raise ValueError("Unrecognized uint8 audio array: pass the WAV returned by Microphone.record_wav(), or int16/float PCM samples.")
            return cls(data, audio_format)

        # Anything else is raw PCM, as returned by Microphone.record_pcm() / capture() / stream().
        return cls.from_pcm(audio)

    @classmethod
    def from_inputs(cls, audio: AudioInput | Sequence[AudioInput] | None) -> list[AudioClip]:
        """Encodes one clip or a sequence of clips, each as ``from_input()`` does.

        A single clip is accepted as well as a sequence of them, so that the array returned
        by ``Microphone.record_pcm()`` is one clip, not a sequence of samples.

        Args:
            audio (AudioInput | Sequence[AudioInput] | None): One clip, a sequence of clips, or None.

        Returns:
            list[AudioClip]: The encoded clips, empty when ``audio`` is None.

        Raises:
            FileNotFoundError: If a file path does not exist.
            ValueError: If a file type, header or sample type is not supported.
            TypeError: If a clip is none of the supported types.
        """
        if audio is None:
            return []
        if isinstance(audio, (str, bytes, bytearray, memoryview)) or hasattr(audio, "dtype"):
            return [cls.from_input(audio)]  # pyright: ignore[reportArgumentType]
        return [cls.from_input(clip) for clip in audio]
