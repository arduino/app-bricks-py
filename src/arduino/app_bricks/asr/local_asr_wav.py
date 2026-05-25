# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import io
import wave

import numpy as np

from arduino.app_utils import brick

from .local_asr import (
    ASREvent,
    AudioSourceExhausted,
    TranscriptionStream,
    BaseASR,
)


class InMemoryAudioSource:
    """
    Audio source wrapping WAV bytes or a raw PCM ndarray.

    Exposes only the subset of BaseMicrophone attributes/methods that ASR uses,
    so it can be used uniformly. ``capture()`` raises ``AudioSourceExhausted``
    when the underlying buffer is drained.
    """

    _DEFAULT_SAMPLING_RATE = 16000
    _DEFAULT_CHANNELS = 1
    _DEFAULT_BUFFER_SIZE = 1024

    def __init__(self, samples: bytes | np.ndarray):
        if isinstance(samples, (bytes, bytearray)):
            with wave.open(io.BytesIO(bytes(samples)), "rb") as wf:
                self.sample_rate = wf.getframerate()
                self.channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            # Derive numpy dtype from WAV sample width (signed int, little-endian — WAV convention)
            dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
            if sample_width not in dtype_map:
                raise ValueError(f"Unsupported WAV sample width: {sample_width}")
            self.format = np.dtype(dtype_map[sample_width])
            self._samples = np.frombuffer(frames, dtype=self.format)
        elif isinstance(samples, np.ndarray):
            self.sample_rate = self._DEFAULT_SAMPLING_RATE
            self.channels = self._DEFAULT_CHANNELS
            self.format = samples.dtype
            self._samples = samples
        else:
            raise TypeError(f"Unsupported in-memory audio source type: {type(samples)!r}")

        self.format_is_packed = False
        self.buffer_size = self._DEFAULT_BUFFER_SIZE
        self._started = True  # It's started by default, this is not a real device
        self._cursor = 0

    def is_started(self) -> bool:
        return self._started

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def capture(self) -> np.ndarray:
        step = self.buffer_size * self.channels
        if self._cursor >= len(self._samples):
            raise AudioSourceExhausted()

        chunk = self._samples[self._cursor : self._cursor + step]
        self._cursor += step

        return chunk


@brick
class WAVAutomaticSpeechRecognition(BaseASR):
    """ASR brick for offline transcription of in-memory audio."""

    def __init__(
        self,
        wav: np.ndarray | bytes,
        language: str | None = None,
    ):
        """
        ASR brick that transcribes a finite in-memory audio buffer.

        Args:
            wav: WAV data to be used for transcription. One of:
                np.ndarray: treated as raw PCM samples at 16 kHz mono.
                bytes: treated as a WAV container.
            language (str): Language code for the ASR model (e.g. "en" for
                English). This is typically auto-detected by the model,
                but can be overridden here if needed. It is exposed as
                the public ``language`` attribute and may be reassigned at
                runtime; the new value takes effect on the next session.

        Note:
            Only one transcription can be active at a time.
        """
        super().__init__(source=wav, language=language)  # type: ignore[arg-type]

    def _build_source(self, source) -> tuple:
        if not isinstance(source, (np.ndarray, bytes, bytearray)):
            raise TypeError(f"Unsupported source type: {type(source)!r}")
        return InMemoryAudioSource(source), False

    def transcribe(self) -> str:
        """
        Consume the WAV to completion and return the transcribed text.

        Returns:
            str: The transcribed text, or an empty string if no speech was detected.

        Raises:
            ASRBusyError: If this instance already has an active session.
            ASRServiceBusyError: If no more concurrent sessions are available.
            ASRUnavailableError: If the inference service is unreachable or the
                connection drops mid-session.
        """
        return self._collect_transcription(self.transcribe_stream())

    def transcribe_stream(self) -> TranscriptionStream[ASREvent]:
        """
        Consume the WAV to completion and yield transcription events.

        Yields:
            ASREvent: objects representing transcription events.

        Raises:
            ASRBusyError: If this instance already has an active session.
            ASRServiceBusyError: If no more concurrent sessions are available.
            ASRUnavailableError: If the inference service is unreachable or the
                connection drops mid-session.
        """
        self._ensure_source_started()
        return TranscriptionStream(self._transcribe_stream(duration=0))
