# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import wave

import numpy as np
import pytest

from arduino.app_bricks.sound_generator import SoundGenerator


class DummySpeaker:
    sample_rate = 8000
    buffer_size = 4096
    shared = True

    def __init__(self):
        self.played = []

    def play_wav(self, wav_audio):
        self.played.append(wav_audio)


def _write_wav(path, samples, sample_rate=8000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def test_play_wav_delegates_wav_file_to_output_device(tmp_path):
    wav_path = tmp_path / "tone.wav"
    samples = (np.sin(np.linspace(0, 2 * np.pi * 440, 8000)) * 32767).astype(np.int16)
    _write_wav(wav_path, samples)

    speaker = DummySpeaker()
    generator = SoundGenerator(output_device=speaker)
    generator.play_wav(str(wav_path))

    assert len(speaker.played) == 1
    wav_audio = speaker.played[0]
    assert isinstance(wav_audio, np.ndarray)
    assert wav_audio.dtype == np.uint8
    assert wav_audio.tobytes() == wav_path.read_bytes()


def test_play_wav_missing_file_raises(tmp_path):
    speaker = DummySpeaker()
    generator = SoundGenerator(output_device=speaker)

    with pytest.raises(FileNotFoundError):
        generator.play_wav(str(tmp_path / "missing.wav"))

    assert speaker.played == []
