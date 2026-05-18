# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np


class SoundEffect:
    @staticmethod
    def overdrive(drive: float = 100.0) -> np.ndarray:
        """
        Apply overdrive effect to the audio signal.
        Args:
            signal (np.ndarray): Input audio signal.
            drive (float): Overdrive intensity factor.
        Returns:
            np.ndarray: Processed audio signal with overdrive effect.
        """

        class SoundEffectOverdrive:
            def __init__(self, drive: float = 1.0):
                pass

            def apply(self, signal: np.ndarray) -> np.ndarray:
                signal = signal * drive
                # soft clipping
                return (2 / 3) * np.tanh(signal)

        return SoundEffectOverdrive(drive)

    @staticmethod
    def chorus(depth_ms=10, rate_hz: float = 0.25, mix: float = 0.5) -> np.ndarray:
        """
        Apply chorus effect to the audio signal.
        Args:
            signal (np.ndarray): Input audio signal.
            depth_ms (float): Depth of the chorus effect in milliseconds.
            rate_hz (float): Rate of the LFO in Hz.
            mix (float): Mix ratio between dry and wet signals (0.0 to 1.0).
        Returns:
            np.ndarray: Processed audio signal with chorus effect.
        """

        class SoundEffectChorus:
            def __init__(self, depth_ms: int = 10, rate_hz: float = 0.25, mix: float = 0.5):
                self.fs = 16000  # sample rate
                self.depth_ms = depth_ms
                self.rate_hz = rate_hz
                self.mix = mix
                pass

            def apply(self, signal: np.ndarray) -> np.ndarray:
                n = len(signal)
                depth = (self.depth_ms / 1000.0) * self.fs  # in samples
                t = np.arange(n)

                lfo = (np.sin(2 * np.pi * self.rate_hz * t / self.fs) + 1) / 2  # [0..1]
                delay = (lfo * depth).astype(int)

                out = np.zeros_like(signal)
                for i in range(n):
                    d = delay[i]
                    if i - d >= 0:
                        out[i] = signal[i - d]

                # mix dry/wet
                return ((1 - self.mix) * signal + self.mix * out).astype(np.float32)

        return SoundEffectChorus(depth_ms, rate_hz, mix)

    @staticmethod
    def adsr(attack: float = 0.015, decay: float = 0.2, sustain: float = 0.5, release: float = 0.35):
        """
        Apply ADSR (attack/decay/sustain/release) envelope to the audio signal.
        Args:
            attack (float): Attack time in seconds.
            decay (float): Decay time in seconds.
            sustain (float): Sustain level (0.0 to 1.0).
            release (float): Release time in seconds.
        """

        class SoundEffectADSR:
            def __init__(self, attack: float = 0.015, decay: float = 0.2, sustain: float = 0.5, release: float = 0.35):
                """
                Initialize ADSR effect.
                Args:
                    attack (float): Attack time in seconds.
                    decay (float): Decay time in seconds.
                    sustain (float): Sustain level (0.0 to 1.0).
                    release (float): Release time in seconds.
                """
                self.attack = attack
                self.decay = decay
                self.sustain = sustain
                self.release = release

            def apply(self, signal: np.ndarray) -> np.ndarray:
                """
                Apply ADSR filter on signal.
                Args:
                    signal: np.ndarray float32 (audio)
                """
                n = len(signal)
                env = np.zeros(n)

                a = int(n * self.attack)
                d = int(n * self.decay)
                r = int(n * self.release)

                s = max(0, n - (a + d + r))

                env[:a] = np.linspace(0, 1, a, endpoint=False)  # Attack
                env[a : a + d] = np.linspace(1, self.sustain, d, endpoint=False)  # Decay
                env[a + d : a + d + s] = self.sustain  # Sustain
                env[a + d + s :] = np.linspace(self.sustain, 0, n - (a + d + s), endpoint=False)  # Release

                return (signal * env).astype(np.float32)

        return SoundEffectADSR(attack, decay, sustain, release)

    @staticmethod
    def tremolo(depth: float = 0.5, rate: float = 5.0):
        class SoundEffectTremolo:
            def __init__(self, depth: float = 0.5, rate: float = 5.0):
                """
                Tremolo effect block-local.
                Args:
                    depth (float): modulation depth (0=no effect, 1=full)
                    rate (float): rate in cycles per block
                """
                self.depth = np.clip(depth, 0.0, 1.0)
                self.rate = rate  # cicli di tremolo per blocco

            def apply(self, signal: np.ndarray) -> np.ndarray:
                """
                Apply tremolo to a block of audio.
                Args:
                    signal (np.ndarray): input block
                """
                n = len(signal)
                t = np.linspace(0, 1, n, endpoint=False)  # normalizzato al blocco
                lfo = (1 - self.depth) + self.depth * np.sin(2 * np.pi * self.rate * t)
                return (signal * lfo).astype(np.float32)

        return SoundEffectTremolo(depth, rate)

    @staticmethod
    def vibrato(depth: float = 0.02, rate: float = 0.5):
        class SoundEffectVibrato:
            def __init__(self, depth: float = 0.02, rate: float = 2.0):
                """
                Vibrato effect
                Args:
                    depth (float): max deviation (0=no effect, 0.5=max)
                    rate (float): number of cycles per block
                """
                self.depth = np.clip(depth, 0.0, 0.5)
                self.rate = rate

            def apply(self, signal: np.ndarray) -> np.ndarray:
                n = len(signal)
                t = np.linspace(0, 1, n, endpoint=False)
                lfo = self.depth * n * np.sin(2 * np.pi * self.rate * t)
                indices = np.arange(n) + lfo
                indices = np.clip(indices, 0, n - 1.001)
                i0 = np.floor(indices).astype(int)
                i1 = np.ceil(indices).astype(int)
                frac = indices - i0
                output = (1 - frac) * signal[i0] + frac * signal[i1]
                return output.astype(np.float32)

        return SoundEffectVibrato(depth=depth, rate=rate)

    @staticmethod
    def bitcrusher(bits: int = 4, reduction: int = 6):
        class SoundEffectBitcrusher:
            def __init__(self, bits: int = 4, reduction: int = 4):
                """
                Bitcrusher effect.
                Args:
                    bits (int): Bit depth for quantization (1-16).
                    reduction (int): Redeuction factor for downsampling (>=1).
                """
                self.bit_depth = np.clip(bits, 1, 16)
                self.reduction = max(1, reduction)

            def apply(self, signal: np.ndarray) -> np.ndarray:
                # Downsampling
                reduced = signal[:: self.reduction]
                expanded = np.repeat(reduced, self.reduction)
                expanded = expanded[: len(signal)]  # taglia se serve

                # Quantization
                levels = 2**self.bit_depth
                crushed = np.round(expanded * (levels / 2)) / (levels / 2)
                crushed = np.clip(crushed, -1.0, 1.0)
                return crushed.astype(np.float32)

        return SoundEffectBitcrusher(bits, reduction)

    @staticmethod
    def octaver(oct_up: bool = True, oct_down: bool = False):
        class SoundEffectOctaver:
            def __init__(self, oct_up: bool = True, oct_down: bool = True):
                """
                Octaver effect.
                Args:
                    oct_up (bool): Add one octave above the original signal.
                    oct_down (bool): Add one octave below the original signal.
                """
                self.oct_up = oct_up
                self.oct_down = oct_down

            def apply(self, signal: np.ndarray) -> np.ndarray:
                """
                Apply the octaver effect to a mono audio signal.
                signal: numpy array with float values in range [-1, 1]
                """
                output = signal.astype(np.float32)
                n = len(signal)

                # Upper octave
                if self.oct_up:
                    up = np.zeros(n, dtype=np.float32)
                    up[: n // 2] = signal[::2]
                    output += up

                # Lower octave
                if self.oct_down:
                    down = np.zeros(n, dtype=np.float32)
                    down[::2] = signal[: n // 2]
                    output += down

                # Normalize to prevent clipping
                max_val = np.max(np.abs(output))
                if max_val > 1.0:
                    output /= max_val

                return output

        return SoundEffectOctaver(oct_up, oct_down)
