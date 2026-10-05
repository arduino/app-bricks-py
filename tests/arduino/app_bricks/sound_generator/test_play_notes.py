# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from arduino.app_bricks.sound_generator import SoundGeneratorStreamer


@pytest.fixture
def gen() -> SoundGeneratorStreamer:
    return SoundGeneratorStreamer()


def test_play_returns_float32_block_for_the_note_duration(gen):
    block = gen.play("A4", 1 / 4)
    assert block.dtype == np.float32
    # 120 bpm, 4/4: a quarter note lasts half a second
    assert len(block) == gen._sample_rate // 2


def test_one_note_chord_lasts_as_long_as_the_note(gen):
    single = gen.play("A4", 1 / 4)
    chord = gen.play_chord(["A4"], 1 / 4)
    assert len(chord) == len(single)


def test_chord_lasts_the_note_duration(gen):
    chord = gen.play_chord(["A4", "C#5", "E5"], 1 / 4)
    assert chord.dtype == np.float32
    assert len(chord) == gen._sample_rate // 2


def test_chord_skips_unknown_notes_but_plays_the_known_ones(gen):
    chord = gen.play_chord(["A4", "H9"], 1 / 4)
    assert len(chord) == gen._sample_rate // 2


def test_unknown_notes_raise(gen):
    with pytest.raises(ValueError):
        gen.play("H9", 1 / 4)
    with pytest.raises(ValueError):
        gen.play_tone("H9", 0.25)
    with pytest.raises(ValueError):
        gen.play_chord(["H9", "X0"], 1 / 4)
    with pytest.raises(ValueError):
        gen.play_polyphonic([[("H9", 1 / 4)]])


def test_non_positive_tone_duration_raises(gen):
    with pytest.raises(ValueError):
        gen.play_tone("A4", 0.0)


def test_polyphonic_mixes_tracks_to_the_longest_one(gen):
    block, duration = gen.play_polyphonic([[("A4", 1 / 4), ("C5", 1 / 4)], [("E5", 1 / 2)]])
    assert block.dtype == np.float32
    assert duration == pytest.approx(0.5)
    assert len(block) == gen._sample_rate  # two quarters or one half: one second at 120 bpm
