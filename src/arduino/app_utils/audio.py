# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import math
import numpy as np


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

    def __init__(self, sample_rate: int):
        """Create a new SineGenerator.

        Args:
            sample_rate (int): The playback sample rate (Hz) used to compute
                phase increments and buffer sizes.
        """
        self.sample_rate = int(sample_rate)
        # envelope parameters (attack/release in seconds)
        self.attack = 0.01
        self.release = 0.03
        self.glide = 0.02

        # reusable buffers
        self._buf_N = 0
        self._buf_phase_incs = None
        self._buf_phases = None
        self._buf_envelope = None
        self._buf_samples = None

        # runtime state
        self._amp_current = 0.0
        self._freq_last = 440.0
        self._phase = 0.0

    def reset(self):
        """Reset internal generator state.

        Resets phase, last frequency and current amplitude to silence. Useful
        when reinitializing playback or ensuring a known baseline before
        tests.
        """
        self._amp_current = 0.0
        self._freq_last = 440.0
        self._phase = 0.0

    def get_state(self) -> dict:
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

    def set_state(self, state: dict) -> None:
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

    def generate_block(self, freq: float, amp_target: float, block_dur: float, master_volume: float):
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
