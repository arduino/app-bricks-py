# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np


class WaveSamplesBuilder:
    """Generate wave audio blocks.

    This class produces wave blocks as NumPy buffers.

    Attributes:
        sample_rate (int): Audio sample rate in Hz.
    """

    def __init__(self, wave_form: str = "sine", sample_rate: int = 16000):
        """Create a new WaveGenerator.

        Args:
            wave_form (str): The type of wave form to generate. Supported values
                are "sine", "square", "triangle", "white_noise" and "sawtooth".
            sample_rate (int): The playback sample rate (Hz) used to compute
                phase increments and buffer sizes.
        """
        self.wave_form = wave_form.lower()
        self.sample_rate = int(sample_rate)

    def generate_block(self, freq: float, block_dur: float, master_volume: float = 1.0):
        """Generate a block of float32 audio samples.

        Returned buffer is a NumPy view (float32) into an internal preallocated array and is valid
        until the next call to this method.

        Args:
            freq (float): Target frequency in Hz for this block.
            block_dur (float): Duration of the requested block in seconds.
            master_volume (float, optional): Global gain multiplier. Defaults
                to 1.0.

        Returns:
            numpy.ndarray: A 1-D float32 NumPy array containing the generated
            audio samples for the requested block.
        """
        N = max(1, int(self.sample_rate * block_dur))

        # compute wave form based on selected type
        t = np.arange(N, dtype=np.float32) / self.sample_rate

        match self.wave_form:
            case "square":
                samples = 0.5 * (1 + np.sign(np.sin(2.0 * np.pi * freq * t)))
            case "triangle":
                samples = 2.0 * np.abs(2.0 * (freq * t % 1) - 1.0) - 1.0
            case "sawtooth":
                samples = 2.0 * (freq * t % 1.0) - 1.0
            case "white_noise":
                samples = np.random.uniform(-1.0, 1.0, size=N).astype(np.float32)
            case _:  # "sine" e default
                samples = np.sin(2.0 * np.pi * freq * t)

        samples = samples.astype(np.float32)

        # apply gain
        mg = float(master_volume)
        if mg != 1.0:
            np.multiply(samples, mg, out=samples)

        return samples
