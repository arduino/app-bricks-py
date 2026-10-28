# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Conversion of every supported audio input into an ``input_audio`` content part.

The inputs mirror what an app has at hand: a file, file bytes, or the arrays the
Microphone peripheral returns (``record_wav()`` gives a uint8 WAV file, ``record_pcm()``
raw samples). A raw PCM array must come out as a WAV file llama.cpp can decode.
"""

import base64
import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from arduino.app_bricks.cloud_llm.audio import as_audio_list, audio_to_content_part, detect_audio_format
from arduino.app_peripherals.microphone.base_microphone import BaseMicrophone

MP3_FRAME = bytes([0xFF, 0xFB, 0x90, 0x64]) + b"\x00" * 60


def _wav_bytes(samples: np.ndarray, rate: int = 16000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()


def _decode_part(part: dict) -> tuple[str, bytes]:
    assert part["type"] == "input_audio"
    return part["input_audio"]["format"], base64.b64decode(part["input_audio"]["data"])


def _read_wav(data: bytes) -> tuple[int, int, int, np.ndarray]:
    with wave.open(io.BytesIO(data), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        return wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth(), np.frombuffer(frames, dtype="<i2")


def test_wav_file_path(tmp_path):
    data = _wav_bytes(np.arange(100, dtype=np.int16))
    path = tmp_path / "clip.WAV"
    path.write_bytes(data)

    assert _decode_part(audio_to_content_part(str(path))) == ("wav", data)


def test_mp3_file_path(tmp_path):
    path = tmp_path / "clip.mp3"
    path.write_bytes(MP3_FRAME)

    assert _decode_part(audio_to_content_part(str(path))) == ("mp3", MP3_FRAME)


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "clip.ogg"
    path.write_bytes(b"OggS")

    with pytest.raises(ValueError, match=r"\.wav and \.mp3"):
        audio_to_content_part(str(path))


def test_missing_file_is_reported():
    with pytest.raises(FileNotFoundError):
        audio_to_content_part("does-not-exist.wav")


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(_wav_bytes(np.zeros(10, dtype=np.int16)), "wav", id="wav"),
        pytest.param(b"ID3\x04\x00" + b"\x00" * 20, "mp3", id="mp3-id3"),
        pytest.param(MP3_FRAME, "mp3", id="mp3-frame"),
    ],
)
def test_encoded_bytes_are_detected(data, expected):
    assert detect_audio_format(data) == expected
    assert _decode_part(audio_to_content_part(data)) == (expected, data)


def test_unknown_bytes_are_rejected():
    with pytest.raises(ValueError, match="WAV or MP3"):
        audio_to_content_part(b"not audio at all")


def test_record_wav_output_is_sent_as_is():
    """``Microphone.record_wav()`` returns the WAV file as a uint8 array: it must not be re-wrapped as PCM."""
    mic = SimpleNamespace(sample_rate=22050, channels=2, format_is_packed=False)
    samples = np.arange(-50, 50, dtype=np.int16)
    record_wav_output = BaseMicrophone._pcm_to_wav(mic, samples)
    assert record_wav_output.dtype == np.uint8

    audio_format, data = _decode_part(audio_to_content_part(record_wav_output))

    assert audio_format == "wav"
    assert data == record_wav_output.tobytes()
    rate, channels, _, decoded = _read_wav(data)
    assert (rate, channels) == (22050, 2)
    np.testing.assert_array_equal(decoded, samples)


def test_int16_pcm_is_wrapped_as_16khz_mono_wav():
    """``Microphone.record_pcm()`` returns flat int16 samples at the Microphone defaults."""
    samples = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)

    audio_format, data = _decode_part(audio_to_content_part(samples))

    assert audio_format == "wav"
    rate, channels, width, decoded = _read_wav(data)
    assert (rate, channels, width) == (16000, 1, 2)
    np.testing.assert_array_equal(decoded, samples)


def test_float_pcm_is_clipped_and_scaled_to_int16():
    samples = np.array([0.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float32)

    _, data = _decode_part(audio_to_content_part(samples))

    _, _, width, decoded = _read_wav(data)
    assert width == 2
    np.testing.assert_array_equal(decoded, [0, 16383, -16383, 32767, -32767])


def test_frames_by_channels_pcm_keeps_its_channel_count():
    samples = np.zeros((8, 2), dtype=np.int16)

    _, data = _decode_part(audio_to_content_part(samples))

    _, channels, _, decoded = _read_wav(data)
    assert channels == 2
    assert decoded.size == 16


def test_unsupported_pcm_type_points_to_record_wav():
    with pytest.raises(ValueError, match="record_wav"):
        audio_to_content_part(np.zeros(10, dtype=np.int32))


def test_raw_uint8_without_header_is_rejected():
    with pytest.raises(ValueError, match="record_wav"):
        audio_to_content_part(np.zeros(10, dtype=np.uint8))


def test_unsupported_type_is_rejected():
    with pytest.raises(TypeError):
        audio_to_content_part(12345)


def test_single_clip_is_not_iterated():
    """Passing ``audio=mic.record_pcm(5)`` directly must send one clip, not one clip per sample."""
    samples = np.zeros(100, dtype=np.int16)

    assert len(as_audio_list(samples)) == 1
    assert as_audio_list("clip.wav") == ["clip.wav"]
    assert as_audio_list(b"RIFF") == [b"RIFF"]
    assert len(as_audio_list([samples, "clip.wav"])) == 2
    assert as_audio_list(None) == []
    assert as_audio_list([]) == []
