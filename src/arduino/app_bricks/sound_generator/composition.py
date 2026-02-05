# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from dataclasses import dataclass, field
from .effects import SoundEffect


@dataclass
class MusicComposition:
    """
    A structured representation of a musical composition for SoundGenerator.

    This class encapsulates all the parameters needed to play a polifonic step sequence,
    making it easy to save, load, and share musical sequences.

    Attributes:
        composition (list[list[tuple[str, float]]]): Polyphonic sequence as a list of tracks.
            Each track is a list of tuples (note, duration).
            Duration is in note fractions (1/4 = quarter note, 1/8 = eighth note).
            Example: [[("C4", 0.25), ("E4", 0.25)], [("G4", 0.5)]]
        bpm (int): Tempo in beats per minute. Default: 120.
        waveform (str): Wave form type ("sine", "square", "triangle", "sawtooth"). Default: "sine".
        volume (float): Master volume level (0.0 to 1.0). Default: 0.8.
        effects (list): List of SoundEffect instances to apply. Default: [SoundEffect.adsr()].

    Example:
        ```python
        from arduino.app_bricks.sound_generator import MusicComposition, SoundGenerator, SoundEffect

        # Create a composition
        comp = MusicComposition(
            composition=[
                [("B4", 1 / 16), ("A4", 1 / 16), ("G4", 1 / 16)],  # Step 0
                [("A#4", 1 / 16), ("G#4", 1 / 16)],  # Step 1
                [("A4", 1 / 16)],  # Step 2
                [],  # Step 3 - REST
                [("A4", 1 / 16)],  # Step 4
            ],
            bpm=120,
            waveform="square",
            volume=0.8,
            effects=[SoundEffect.adsr(), SoundEffect.tremolo(depth=0.5, rate=5.0)],
        )

        # Configure and play with SoundGenerator
        gen = SoundGenerator()
        gen.start()
        gen.play_composition(comp, block=True)

        # Alternatively, set parameters manually and play
        gen = SoundGenerator()
        gen.start()
        gen.set_bpm(comp.bpm)
        gen.set_wave_form(comp.waveform)
        gen.set_master_volume(comp.volume)
        gen.set_effects(comp.effects)
        gen.play_polyphonic(comp.composition, volume=comp.volume, block=True)
        ```
    """

    composition: list[list[tuple[str, float]]]
    bpm: int = 120
    waveform: str = "sine"
    volume: float = 0.8
    effects: list = field(default_factory=lambda: [SoundEffect.adsr()])
