# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Encoding of audio inputs into chat-completions ``input_audio`` content parts.

OpenAI-compatible servers (llama.cpp included) take audio as a base64 WAV or MP3 file
inside the user message: ``{"type": "input_audio", "input_audio": {"data": ..., "format": "wav"}}``.
The helpers here accept what an app naturally has at hand (a file path, encoded file
bytes, or the numpy arrays returned by the Microphone peripheral) and turn it into that
part. numpy is imported lazily: it ships with the microphone extra, not with the LLM bricks.
"""

import base64
import io
import os
import wave
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

AudioFormat = Literal["wav", "mp3"]

# The value of a `type` alias is evaluated lazily, so numpy is only needed for type checking.
type AudioInput = str | bytes | np.ndarray[Any, Any]
"""An audio clip: a WAV/MP3 file path, WAV/MP3 file bytes, or a numpy array from the Microphone peripheral."""

# Raw PCM arrays carry no header: they are assumed to use the Microphone defaults.
PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1

_EXTENSION_FORMATS: dict[str, AudioFormat] = {".wav": "wav", ".mp3": "mp3"}


def detect_audio_format(data: bytes) -> AudioFormat | None:
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


def pcm_to_wav(samples: "np.ndarray[Any, Any]", sample_rate: int = PCM_SAMPLE_RATE, channels: int = PCM_CHANNELS) -> bytes:
    """Wraps raw PCM samples in a 16-bit WAV file.

    Args:
        samples (np.ndarray): int16 samples, or float32/float64 samples normalized to [-1.0, 1.0].
            Multi-channel audio is either interleaved in a flat array or shaped (frames, channels).
        sample_rate (int): Sample rate in Hz. Defaults to 16000, the Microphone default.
        channels (int): Channel count of a flat array. Defaults to 1 (mono).

    Returns:
        bytes: A complete WAV file.

    Raises:
        ValueError: If the sample type is not int16 or float.
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
    return buffer.getvalue()


def _encoded_audio(audio: AudioInput) -> tuple[bytes, AudioFormat]:
    """Resolves an audio input to encoded file bytes and their format."""
    if isinstance(audio, str):
        extension = os.path.splitext(audio)[1].lower()
        audio_format = _EXTENSION_FORMATS.get(extension)
        if audio_format is None:
            raise ValueError(f"Unsupported audio file '{audio}': only .wav and .mp3 files are supported.")
        if not os.path.isfile(audio):
            raise FileNotFoundError(f"Audio file not found: {audio}")
        with open(audio, "rb") as f:
            return f.read(), audio_format

    if isinstance(audio, (bytes, bytearray, memoryview)):
        data = bytes(audio)
        audio_format = detect_audio_format(data)
        if audio_format is None:
            raise ValueError("Unrecognized audio bytes: pass the content of a WAV or MP3 file.")
        return data, audio_format

    dtype = getattr(audio, "dtype", None)
    if dtype is None:
        raise TypeError(f"Unsupported audio input of type {type(audio).__name__}: pass a file path, WAV/MP3 bytes or a numpy array.")

    # Microphone.record_wav() returns the whole WAV file as a uint8 array.
    if dtype.kind == "u" and dtype.itemsize == 1:
        data = audio.tobytes()
        audio_format = detect_audio_format(data)
        if audio_format is None:
            raise ValueError("Unrecognized uint8 audio array: pass the WAV returned by Microphone.record_wav(), or int16/float PCM samples.")
        return data, audio_format

    # Anything else is raw PCM, as returned by Microphone.record_pcm() / capture() / stream().
    return pcm_to_wav(audio), "wav"


def as_audio_list(audio: "AudioInput | Sequence[AudioInput] | None") -> list[AudioInput]:
    """Normalizes the ``audio`` argument of the chat methods to a list of clips.

    A single clip is accepted as well as a sequence of them, so that passing the array
    returned by ``Microphone.record_pcm()`` directly does not iterate over its samples.

    Args:
        audio (AudioInput | Sequence[AudioInput] | None): One clip, a sequence of clips, or None.

    Returns:
        list[AudioInput]: The clips, empty when ``audio`` is None.
    """
    if audio is None:
        return []
    if isinstance(audio, (str, bytes, bytearray, memoryview)) or hasattr(audio, "dtype"):
        return [audio]  # pyright: ignore[reportReturnType]
    return list(audio)  # pyright: ignore[reportUnknownArgumentType]


def audio_to_content_part(audio: AudioInput) -> dict[str, Any]:
    """Builds the chat-completions ``input_audio`` content part for an audio input.

    Args:
        audio (AudioInput): A .wav/.mp3 file path, WAV/MP3 file bytes, the uint8 WAV array
            returned by ``Microphone.record_wav()``, or raw PCM samples (int16 or float in
            [-1.0, 1.0]) as returned by ``Microphone.record_pcm()``. Raw PCM is assumed to be
            16 kHz mono, the Microphone defaults; record other rates or channel counts with
            ``record_wav()``, which stores them in the WAV header.

    Returns:
        dict[str, Any]: The content part, ready to be placed in a user message.

    Raises:
        FileNotFoundError: If a file path does not exist.
        ValueError: If the file type, header or sample type is not supported.
        TypeError: If the input is none of the supported types.
    """
    data, audio_format = _encoded_audio(audio)
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": audio_format}}
