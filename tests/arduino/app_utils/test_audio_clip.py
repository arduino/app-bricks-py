# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Encoding of every supported audio input into a WAV or MP3 file.

The inputs mirror what an app has at hand: a file, file bytes, or the arrays the
Microphone peripheral returns (``record_wav()`` gives a uint8 WAV file, ``record_pcm()``
raw samples). A raw PCM array must come out as a WAV file a decoder can read.
"""

import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from arduino.app_peripherals.microphone.base_microphone import BaseMicrophone
from arduino.app_utils import AudioClip

MP3_FRAME = bytes([0xFF, 0xFB, 0x90, 0x64]) + b"\x00" * 60


def _wav_bytes(samples: np.ndarray, rate: int = 16000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()


def _read_wav(data: bytes) -> tuple[int, int, int, np.ndarray]:
    with wave.open(io.BytesIO(data), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        return wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth(), np.frombuffer(frames, dtype="<i2")


def test_wav_file_path(tmp_path):
    data = _wav_bytes(np.arange(100, dtype=np.int16))
    path = tmp_path / "clip.WAV"
    path.write_bytes(data)

    assert AudioClip.from_input(str(path)) == AudioClip(data, "wav")


def test_mp3_file_path(tmp_path):
    path = tmp_path / "clip.mp3"
    path.write_bytes(MP3_FRAME)

    assert AudioClip.from_input(str(path)) == AudioClip(MP3_FRAME, "mp3")


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "clip.ogg"
    path.write_bytes(b"OggS")

    with pytest.raises(ValueError, match=r"\.wav and \.mp3"):
        AudioClip.from_input(str(path))


def test_missing_file_is_reported():
    with pytest.raises(FileNotFoundError):
        AudioClip.from_input("does-not-exist.wav")


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(_wav_bytes(np.zeros(10, dtype=np.int16)), "wav", id="wav"),
        pytest.param(b"ID3\x04\x00" + b"\x00" * 20, "mp3", id="mp3-id3"),
        pytest.param(MP3_FRAME, "mp3", id="mp3-frame"),
    ],
)
def test_encoded_bytes_are_detected(data, expected):
    assert AudioClip.detect_format(data) == expected
    assert AudioClip.from_input(data) == AudioClip(data, expected)


def test_unknown_bytes_are_rejected():
    with pytest.raises(ValueError, match="WAV or MP3"):
        AudioClip.from_input(b"not audio at all")


def test_record_wav_output_is_kept_as_is():
    """``Microphone.record_wav()`` returns the WAV file as a uint8 array: it must not be re-wrapped as PCM."""
    mic = SimpleNamespace(sample_rate=22050, channels=2, format_is_packed=False)
    samples = np.arange(-50, 50, dtype=np.int16)
    record_wav_output = BaseMicrophone._pcm_to_wav(mic, samples)
    assert record_wav_output.dtype == np.uint8

    clip = AudioClip.from_input(record_wav_output)

    assert clip == AudioClip(record_wav_output.tobytes(), "wav")
    rate, channels, _, decoded = _read_wav(clip.data)
    assert (rate, channels) == (22050, 2)
    np.testing.assert_array_equal(decoded, samples)


def test_int16_pcm_is_wrapped_as_16khz_mono_wav():
    """``Microphone.record_pcm()`` returns flat int16 samples at the Microphone defaults."""
    samples = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)

    clip = AudioClip.from_input(samples)

    assert clip.format == "wav"
    rate, channels, width, decoded = _read_wav(clip.data)
    assert (rate, channels, width) == (16000, 1, 2)
    np.testing.assert_array_equal(decoded, samples)


def test_pcm_with_explicit_rate_and_channels():
    samples = np.arange(-8, 8, dtype=np.int16)

    clip = AudioClip.from_pcm(samples, sample_rate=22050, channels=2)

    rate, channels, _, decoded = _read_wav(clip.data)
    assert (rate, channels) == (22050, 2)
    np.testing.assert_array_equal(decoded, samples)


def test_float_pcm_is_clipped_and_scaled_to_int16():
    samples = np.array([0.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float32)

    clip = AudioClip.from_input(samples)

    _, _, width, decoded = _read_wav(clip.data)
    assert width == 2
    np.testing.assert_array_equal(decoded, [0, 16383, -16383, 32767, -32767])


def test_frames_by_channels_pcm_keeps_its_channel_count():
    samples = np.zeros((8, 2), dtype=np.int16)

    clip = AudioClip.from_input(samples)

    _, channels, _, decoded = _read_wav(clip.data)
    assert channels == 2
    assert decoded.size == 16


def test_unsupported_pcm_type_points_to_record_wav():
    with pytest.raises(ValueError, match="record_wav"):
        AudioClip.from_input(np.zeros(10, dtype=np.int32))


def test_raw_uint8_without_header_is_rejected():
    with pytest.raises(ValueError, match="record_wav"):
        AudioClip.from_input(np.zeros(10, dtype=np.uint8))


def test_unsupported_type_is_rejected():
    with pytest.raises(TypeError):
        AudioClip.from_input(12345)


def test_single_clip_is_not_iterated(tmp_path):
    """Passing ``mic.record_pcm(5)`` directly must give one clip, not one clip per sample."""
    samples = np.zeros(100, dtype=np.int16)
    wav = _wav_bytes(samples)
    path = tmp_path / "clip.wav"
    path.write_bytes(wav)

    assert len(AudioClip.from_inputs(samples)) == 1
    assert AudioClip.from_inputs(str(path)) == [AudioClip(wav, "wav")]
    assert AudioClip.from_inputs(wav) == [AudioClip(wav, "wav")]
    assert len(AudioClip.from_inputs([samples, str(path)])) == 2
    assert AudioClip.from_inputs(None) == []
    assert AudioClip.from_inputs([]) == []
