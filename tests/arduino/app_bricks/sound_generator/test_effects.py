# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_bricks.sound_generator.effects import SoundEffect
from arduino.app_bricks.sound_generator import SoundGeneratorStreamer, WaveSamplesBuilder


def test_adsr_effect():
    generator = WaveSamplesBuilder(sample_rate=16000, wave_form="square")
    adsr = SoundEffect.adsr()
    blk = generator.generate_block(440.0, 1 / 8, 1.0)  # Generate a block to initialize
    assert adsr is not None

    # Apply ADSR effect
    processed = adsr.apply(blk)

    assert processed is not None
    assert len(processed) == len(blk)


def test_available_notes():
    note_sequence = [
        ("E5", 0.125),
        ("E5", 0.125),
        ("REST", 0.125),
        ("E5", 0.125),
        ("REST", 0.125),
        ("C5", 0.125),
        ("E5", 0.125),
        ("REST", 0.125),
        ("G5", 0.25),
        ("REST", 0.25),
        ("G4", 0.25),
        ("REST", 0.25),
        ("C5", 0.25),
        ("REST", 0.125),
        ("G4", 0.25),
        ("REST", 0.125),
        ("E4", 0.25),
        ("REST", 0.125),
        ("A4", 0.25),
        ("B4", 0.25),
        ("Bb4", 0.125),
        ("A4", 0.25),
        ("G4", 0.125),
        ("E5", 0.125),
        ("G5", 0.125),
        ("A5", 0.25),
        ("F5", 0.125),
        ("G5", 0.125),
        ("REST", 0.125),
        ("E5", 0.25),
        ("C5", 0.125),
        ("D5", 0.125),
        ("B4", 0.25),
    ]

    generator = SoundGeneratorStreamer()
    for note, duration in note_sequence:
        frequency = generator._get_note(note)
        if "REST" != note:
            assert frequency is not None and frequency > 0.0
        else:
            assert frequency is not None and frequency == 0.0
