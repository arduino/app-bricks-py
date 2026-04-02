# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import brick, Logger
from arduino.app_peripherals.speaker import Speaker
import threading
from typing import Iterable
import numpy as np
import time
from pathlib import Path
from collections import OrderedDict
import math

from .generator import WaveSamplesBuilder
from .effects import *
from .loaders import ABCNotationLoader
from .composition import MusicComposition as MusicComposition

logger = Logger("SoundGenerator")


class LRUDict(OrderedDict):
    """A dictionary-like object with a fixed size that evicts the least recently used items."""

    def __init__(self, maxsize=128, *args, **kwargs):
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)

        super().__setitem__(key, value)

        if len(self) > self.maxsize:
            # Evict the least recently used item (the first item)
            self.popitem(last=False)


@brick
class SoundGeneratorStreamer:
    # Default sample rate fallback; prefer using the `Speaker` constants when possible.
    SAMPLE_RATE = Speaker.RATE_16K
    A4_FREQUENCY = 440.0

    # Semitone mapping for the 12 notes (0 = C, 11 = B).
    # This is used to determine the relative position within an octave.
    SEMITONE_MAP = {
        "C": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
    }

    NOTE_DURATTION = {
        "W": 1.0,  # Whole
        "H": 0.5,  # Half
        "Q": 0.25,  # Quarter
        "E": 0.125,  # Eighth
        "S": 0.0625,  # Sixteenth
        "T": 0.03125,  # Thirty-second
        "X": 0.015625,  # Sixty-fourth
    }

    # The reference point in the overall semitone count from C0. A4 is (4 * 12) + 9 semitones from C0.
    A4_SEMITONE_INDEX = (4 * 12) + 9

    def __init__(
        self,
        bpm: int = 120,
        time_signature: tuple = (4, 4),
        octaves: int = 8,
        wave_form: str = "sine",
        master_volume: float = 1.0,
        sound_effects: list = None,
    ):
        """Initialize the SoundGeneratorStreamer. Generates sound blocks for streaming, without internal playback.
        Args:
            bpm (int): The tempo in beats per minute for note duration calculations.
            time_signature (tuple): The time signature as (numerator, denominator).
            octaves (int): Number of octaves to generate notes for (starting from octave
                0 up to octaves-1).
            wave_form (str): The type of wave form to generate. Supported values
                are "sine" (default), "square", "triangle" and "sawtooth".
            master_volume (float): The master volume level (0.0 to 1.0).
            sound_effects (list, optional): List of sound effect instances to apply to the audio
                signal (e.g., [SoundEffect.adsr()]). See SoundEffect class for available effects.
        """

        self._cfg_lock = threading.Lock()
        # instance sample rate. Prefer speaker defaults but allow re-init later
        self._sample_rate = int(self.SAMPLE_RATE)
        self._init_wave_generator(wave_form, sample_rate=self._sample_rate)

        self._bpm = bpm
        self.time_signature = time_signature
        self._master_volume = master_volume
        self._sound_effects = sound_effects

        self._notes = {}
        for octave in range(octaves):
            notes = self._fill_node_frequencies(octave)
            self._notes.update(notes)

        self._wav_cache = LRUDict(maxsize=10)

    def start(self):
        pass

    def stop(self):
        pass

    def _init_wave_generator(self, wave_form: str, sample_rate: int | None = None):
        """Initialize the WaveSamplesBuilder with the given sample rate.

        If `sample_rate` is None, uses `self._sample_rate`.
        This allows the SoundGenerator subclass to reinitialize the generator after
        creating the actual output `Speaker` so both sides agree on sample rate.
        """
        with self._cfg_lock:
            sr = int(sample_rate) if sample_rate is not None else self._sample_rate
            self._wave_gen = WaveSamplesBuilder(sample_rate=sr, wave_form=wave_form)
            self._sample_rate = int(sr)

    def set_wave_form(self, wave_form: str):
        """
        Set the wave form type for sound generation.
        Args:
            wave_form (str): The type of wave form to generate. Supported values
                are "sine", "square", "triangle" and "sawtooth".
        """
        self._init_wave_generator(wave_form)

    def set_master_volume(self, volume: float):
        """
        Set the master volume level.
        Args:
            volume (float): Volume level (0.0 to 1.0).
        """
        self._master_volume = max(0.0, min(1.0, volume))

    def set_bpm(self, bpm: int):
        """
        Set the tempo in beats per minute.
        Args:
            bpm (int): Tempo in beats per minute.
        """
        with self._cfg_lock:
            self._bpm = bpm
        logger.debug(f"BPM updated to {bpm}")

    def set_effects(self, effects: list):
        """
        Set the list of sound effects to apply to the audio signal.
        Args:
            effects (list): List of sound effect instances (e.g., [SoundEffect.adsr()]).
        """
        with self._cfg_lock:
            self._sound_effects = effects

    def _fill_node_frequencies(self, octave: int) -> dict:
        """Generate note-name-to-frequency mappings for a given octave.

        Args:
            octave (int): The octave number to generate frequencies for.

        Returns:
            dict: Mapping of note names (e.g., 'C4', 'A#3') to frequencies in Hz.
        """
        notes = {}

        notes[f"REST"] = 0.0  # Rest note

        # Generate frequencies for all notes in the given octave
        for note_name in self.SEMITONE_MAP:
            frequency = self._note_to_frequency(note_name, octave)
            notes[f"{note_name}{octave}"] = frequency

        return notes

    def _note_to_frequency(self, note_name: str, octave: int) -> float:
        """
        Calculates the frequency (in Hz) of a musical note based on its name and octave.

        It uses the standard 12-tone equal temperament formula: f = f0 * 2^(n/12),
        where f0 is the reference frequency (A4=440Hz) and n is the number of
        semitones from the reference note.

        Args:
            note_name: The name of the note (e.g., 'A', 'C#', 'Bb', case-insensitive).
            octave: The octave number (e.g., 4 for A4, 5 for C5).

        Returns:
            The frequency in Hertz (float).
        """
        # 1. Normalize the note name for lookup
        normalized_note = note_name.strip().upper()
        if len(normalized_note) > 1 and normalized_note[1] == "#":
            # Ensure sharps are treated correctly (e.g., 'C#' is fine)
            pass
        elif len(normalized_note) > 1 and normalized_note[1].lower() == "b":
            # Replace 'B' (flat) with 'B' for consistent dictionary key
            normalized_note = normalized_note[0] + "B"

        # 2. Look up the semitone count within the octave
        if normalized_note not in self.SEMITONE_MAP:
            raise ValueError(f"Invalid note name: {note_name}. Please use notes like 'A', 'C#', 'Eb', etc.")

        semitones_in_octave = self.SEMITONE_MAP[normalized_note]

        # 3. Calculate the absolute semitone index (from C0)
        # Total semitones = (octave number * 12) + semitones_from_C_in_octave
        target_semitone_index = (octave * 12) + semitones_in_octave

        # 4. Calculate 'n', the number of semitones from the reference pitch (A4)
        # A4 is the reference, so n is the distance from A4.
        semitones_from_a4 = target_semitone_index - self.A4_SEMITONE_INDEX

        # 5. Calculate the frequency
        # f = 440 * 2^(n/12)
        frequency_hz = self.A4_FREQUENCY * (2.0 ** (semitones_from_a4 / 12.0))

        return frequency_hz

    def _note_duration(self, symbol: str | float | int) -> float:
        """
        Decode a note duration symbol into its corresponding fractional value.
        Args:
            symbol (str | float | int): Note duration symbol (e.g., 'W', 'H', 'Q', etc.) or a float/int value.
        Returns:
            float: Corresponding fractional duration value or the float itself if provided.
        """

        if isinstance(symbol, float) or isinstance(symbol, int):
            return self._compute_time_duration(symbol)

        duration = self.NOTE_DURATTION.get(symbol.upper(), None)
        if duration is not None:
            return self._compute_time_duration(duration)

        return self._compute_time_duration(1 / 4)  # Default to quarter note

    def _compute_time_duration(self, note_fraction: float) -> float:
        """Compute the time duration in seconds for a given note fraction.

        Uses the instance's time_signature and bpm to calculate the result.

        Args:
            note_fraction (float): The fraction of the note (e.g., 1.0 for whole, 0.5 for half).

        Returns:
            float: Duration in seconds.
        """

        numerator, denominator = self.time_signature

        # For compound time signatures (6/8, 9/8, 12/8), the beat is the dotted quarter note (3/8)
        if denominator == 8 and numerator % 3 == 0:
            beat_value = 3 / 8
        else:
            beat_value = 1 / denominator  # es. 1/4 in 4/4

        # Calculate the duration of a single beat in seconds
        beat_duration = 60.0 / self._bpm

        # Compute the total duration
        return beat_duration * (note_fraction / beat_value)

    def _apply_sound_effects(self, signal: np.ndarray, frequency: float) -> np.ndarray:
        """
        Apply the configured sound effects to the audio signal.
        Args:
            signal (np.ndarray): Input audio signal.
            frequency (float): Frequency of the note being played.
        Returns:
            np.ndarray: Processed audio signal with sound effects applied.
        """
        with self._cfg_lock:
            if self._sound_effects is None:
                return signal

            processed_signal = signal
            for effect in self._sound_effects:
                if hasattr(effect, "apply_with_tone"):
                    processed_signal = effect.apply_with_tone(processed_signal, frequency)
                else:
                    processed_signal = effect.apply(processed_signal)

            return processed_signal

    def _get_note(self, note: str) -> float | None:
        if note is None:
            return None
        return self._notes.get(note.strip().upper())

    def play_polyphonic(self, notes: list[list[tuple[str, float]]], as_tone: bool = False, volume: float = None) -> tuple[bytes, float]:
        """Generate audio for multiple note sequences mixed together (polyphony).

        Produces multi-track audio by mixing a list of sequences, where each
        sequence is a list of (note, duration) tuples.

        Args:
            notes (list[list[tuple[str, float]]]): List of sequences, each a list of (note, duration) tuples.
            as_tone (bool): If True, interpret duration values as seconds instead of note fractions.
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.

        Returns:
            tuple[np.ndarray, float]: The mixed audio block (float32) and its duration in seconds.
        """
        if volume is None:
            volume = self._master_volume

        # Multi track mixing
        sequences_data = []
        base_frequency = None
        max_duration = 0.0
        for sequence in notes:
            sequence_waves = []
            sequence_duration = 0.0
            for note, duration in sequence:
                sequence_duration += duration
                frequency = self._get_note(note)
                if frequency >= 0.0:
                    if base_frequency is None:
                        base_frequency = frequency
                    if not as_tone:
                        duration = self._note_duration(duration)
                    data = self._wave_gen.generate_block(float(frequency), duration, volume)
                    sequence_waves.append(data)
                else:
                    continue

            if len(sequence_waves) > 0:
                single_track_data = np.concatenate(sequence_waves)
                sequences_data.append(single_track_data)
                if sequence_duration > max_duration:
                    max_duration = sequence_duration

        if len(sequences_data) == 0:
            return

        # Mix sequences - align lengths
        max_length = max(len(seq) for seq in sequences_data)
        # Pad shorter sequences with zeros
        for i in range(len(sequences_data)):
            seq = sequences_data[i]
            if len(seq) < max_length:
                padding = np.zeros(max_length - len(seq), dtype=np.float32)
                sequences_data[i] = np.concatenate((seq, padding))

        # Sum all sequences
        mixed = np.sum(sequences_data, axis=0, dtype=np.float32)
        mixed /= np.max(np.abs(mixed))  # Normalize to prevent clipping
        blk = mixed.astype(np.float32)
        blk = self._apply_sound_effects(blk, base_frequency)
        return (blk, max_duration)

    def play_chord(self, notes: list[str], note_duration: float | str = 1 / 4, volume: float = None) -> bytes:
        """Generate audio for a chord of simultaneous notes.

        Args:
            notes (list[str]): List of musical notes (e.g., ['A4', 'C#5', 'E5']).
            note_duration (float | str): Duration as a note fraction (like 1/4, 1/8) or symbol ('W', 'H', 'Q', etc.).
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.

        Returns:
            np.ndarray: The audio block of the chord (float32).
        """
        duration = self._note_duration(note_duration)
        logger.debug(f"play_chord: notes={notes}, note_duration={note_duration}, duration={duration}s, volume={volume}")
        if len(notes) == 1:
            self.play(notes[0], duration, volume)
            return

        waves = []
        base_frequency = None
        for note in notes:
            frequency = self._get_note(note)
            if frequency:
                if base_frequency is None:
                    base_frequency = frequency
                if volume is None:
                    volume = self._master_volume
                data = self._wave_gen.generate_block(float(frequency), duration, volume)
                waves.append(data)
                logger.debug(f"  Generated wave for {note} @ {frequency}Hz, {len(data)} samples")
            else:
                continue
        if len(waves) == 0:
            return
        chord = np.sum(waves, axis=0, dtype=np.float32)
        chord /= np.max(np.abs(chord))  # Normalize to prevent clipping
        blk = chord.astype(np.float32)
        blk = self._apply_sound_effects(blk, base_frequency)
        logger.debug(f"  Chord generated: {len(blk)} samples")
        return blk

    def play(self, note: str, note_duration: float | str = 1 / 4, volume: float = None) -> bytes:
        """Generate audio samples for a single musical note.

        Args:
            note (str): The musical note to generate (e.g., 'A4', 'C#5', 'REST').
            note_duration (float | str): Duration as a note fraction (like 1/4, 1/8) or symbol ('W', 'H', 'Q', etc.).
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.

        Returns:
            np.ndarray: The audio block (float32), or None if the note is invalid.
        """
        duration = self._note_duration(note_duration)
        frequency = self._get_note(note)
        logger.debug(f"play: note={note}, note_duration={note_duration}, duration={duration}s, frequency={frequency}Hz, volume={volume}")

        # Treat REST (mapped to frequency == 0.0) as explicit silence:
        # return a zero-filled float32 buffer for the requested duration so that
        # the playback loop can enqueue it and maintain proper timing.
        if frequency is not None and float(frequency) == 0.0:
            frames = int(duration * self._sample_rate)
            silent = np.zeros(frames, dtype=np.float32)
            silent = self._apply_sound_effects(silent, float(frequency))
            logger.debug(f"  Generated silence: {len(silent)} samples (expected {frames} @ {self._sample_rate}Hz, duration={duration}s)")
            return silent

        if frequency is not None and frequency >= 0.0:
            if volume is None:
                volume = self._master_volume
            data = self._wave_gen.generate_block(float(frequency), duration, volume)
            data = self._apply_sound_effects(data, frequency)
            # diagnostic: log expected frames and actual length
            expected_frames = int(duration * self._sample_rate)
            logger.debug(f"  Generated audio: {len(data)} samples (expected {expected_frames} @ {self._sample_rate}Hz, duration={duration}s)")
            return data

    def play_tone(self, note: str, duration: float = 0.25, volume: float = None) -> bytes:
        """Generate audio samples for a note with duration in seconds.

        Unlike ``play()`` which interprets duration as a musical note fraction,
        this method takes the duration directly in seconds.

        Args:
            note (str): The musical note to generate (e.g., 'A4', 'C#5', 'REST').
            duration (float): Duration in seconds (default 0.25).
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.

        Returns:
            np.ndarray: The audio block (float32), or None if the note is invalid.
        """
        frequency = self._get_note(note)
        if frequency is not None and frequency >= 0.0 and duration > 0.0:
            if volume is None:
                volume = self._master_volume
            data = self._wave_gen.generate_block(float(frequency), duration, volume)
            data = self._apply_sound_effects(data, frequency)
            return data

    def play_abc(self, abc_string: str, volume: float = None) -> Iterable[tuple[bytes, float]]:
        """Generate audio samples from an ABC notation string.

        Yields one audio block per note in the parsed ABC sequence.  The parser
        is ABC 2.1 standard compliant (key signatures, accidentals, tuplets,
        broken rhythm, multimeasure rests, etc.).  See
        :class:`ABCNotationLoader` for the full feature list and limitations.

        Args:
            abc_string (str): ABC notation string defining the sequence of notes.
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.

        Yields:
            tuple[np.ndarray, float]: Audio block (float32) and its duration in seconds.
        """
        if not abc_string or abc_string.strip() == "":
            return
        if volume is None:
            volume = self._master_volume
        metadata, notes = ABCNotationLoader.parse_abc_notation(abc_string)
        for note, duration in notes:
            frequency = self._get_note(note)
            if frequency is not None and frequency >= 0.0:
                data = self._wave_gen.generate_block(float(frequency), duration, volume)
                data = self._apply_sound_effects(data, frequency)
                yield (data, duration)

    def play_wav(self, wav_file: str) -> tuple[bytes, float]:
        """Load a WAV file and return its raw PCM data.

        Results are cached (up to 250 KB total) for repeated playback.

        Args:
            wav_file (str): The WAV audio file path.

        Returns:
            tuple[bytes, float]: Raw PCM audio data and its duration in seconds.
        """
        import wave

        file_path = Path(wav_file)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"WAV file not found: {wav_file}")

        if wav_file in self._wav_cache:
            return self._wav_cache[wav_file]

        with wave.open(wav_file, "rb") as wav:
            # Read all frames (raw PCM data)
            duration = wav.getnframes() / wav.getframerate()
            wav_data = wav.readframes(wav.getnframes())
            if len(self._wav_cache) < 250 * 1024:  # 250 KB cache limit
                self._wav_cache[wav_file] = (wav_data, duration)
            return (wav_data, duration)

        return (None, None)


@brick
class SoundGenerator(SoundGeneratorStreamer):
    def __init__(
        self,
        output_device: Speaker = None,
        bpm: int = 120,
        time_signature: tuple = (4, 4),
        octaves: int = 8,
        wave_form: str = "sine",
        master_volume: float = 1.0,
        sound_effects: list = None,
    ):
        """Initialize the SoundGenerator.

        Args:
            output_device (Speaker, optional): The output device to play sound through.
                When omitted, SoundGenerator creates an internal shared speaker so
                multiple instances can overlap playback on the same device.
            bpm (int): The tempo in beats per minute for note duration calculations.
            time_signature (tuple): The time signature as (numerator, denominator).
            octaves (int): Number of octaves to generate notes for (starting from octave
                0 up to octaves-1).
            wave_form (str): The type of wave form to generate. Supported values
                are "sine" (default), "square", "triangle" and "sawtooth".
            master_volume (float): The master volume level (0.0 to 1.0).
            sound_effects (list, optional): List of sound effect instances to apply to the audio
                signal (e.g., [SoundEffect.adsr()]). See SoundEffect class for available effects.
        """

        super().__init__(
            bpm=bpm,
            time_signature=time_signature,
            octaves=octaves,
            wave_form=wave_form,
            master_volume=master_volume,
            sound_effects=sound_effects,
        )

        self._started = threading.Event()
        if output_device is None:
            self.external_speaker = False
            # Use shared mode by default so multiple SoundGenerator instances can
            # overlap playback on the same speaker.
            self._output_device = Speaker(sample_rate=Speaker.RATE_32K, format=np.float32, buffer_size=Speaker.BUFFER_SIZE_SAFE, shared=True)
        else:
            self.external_speaker = True
            self._output_device = output_device

        # Ensure wave generator sample rate matches the actual output device
        dev_sr = self._output_device.sample_rate
        if dev_sr and dev_sr != self._sample_rate:
            self._sample_rate = dev_sr
            self._init_wave_generator(wave_form, sample_rate=dev_sr)

        # Step sequencer state
        self._sequence_thread = None
        self._sequence_stop_event = threading.Event()
        self._sequence_lock = threading.Lock()
        self._playback_session_id = 0  # Incremented each playback to invalidate stale threads

    def start(self):
        """Start the sound generator and its internal speaker (if not external)."""
        if self._started.is_set():
            return
        if not self.external_speaker:
            self._output_device.start()
        # After starting the device, query its actual sample rate and
        # reinitialize the wave generator to match the device. This avoids
        # mismatches where the requested sample rate is adapted by the ALSA
        # driver and the generator would otherwise produce buffers with the
        # wrong number of samples (leading to drift).
        self._sync_sample_rate()
        self._started.set()

    def stop(self):
        """Stop playback, halt any running sequence, and close the internal speaker."""
        self.stop_sequence()
        if not self.external_speaker:
            self._output_device.stop()
        self._started.clear()

    def _sync_sample_rate(self):
        """Synchronize the wave generator sample rate with the actual output device."""
        actual_sr = self._output_device.sample_rate
        if actual_sr and actual_sr != self._sample_rate:
            self._sample_rate = actual_sr
            self._init_wave_generator(self._wave_gen.wave_form, sample_rate=actual_sr)
            logger.debug(f"Synced wave_gen sample_rate to {actual_sr}")

    def _ensure_speaker_ready(self):
        """Ensure the internal speaker is started and ready for playback.

        Auto-starts the speaker on the first play call so users don't need to
        call ``start()`` explicitly.  Also transparently reopens it after
        ``stop_sequence()`` which closes the speaker to drop pending audio.
        """
        if not self._started.is_set():
            self.start()
            return
        if not self.external_speaker and not self._output_device.is_started():
            self._output_device.start()
            self._sync_sample_rate()

    def _estimate_output_drain_time(self) -> float:
        """Estimate a small drain time so blocking playback does not cut the tail."""
        sample_rate = self._output_device.sample_rate
        buffer_size = self._output_device.buffer_size
        if not sample_rate or not buffer_size:
            return 0.0
        return min((float(buffer_size) / float(sample_rate)) * 2.0, 0.25)

    def _wait_for_playback_session_end(self, session_id: int):
        """Wait until the given playback session is no longer active."""
        while True:
            with self._sequence_lock:
                current_session_id = self._playback_session_id
                current_thread = self._sequence_thread
            if current_session_id != session_id or current_thread is None or not current_thread.is_alive():
                return
            time.sleep(0.01)

    def _schedule_sequence_stop(self, session_id: int, delay: float) -> threading.Event:
        """Schedule a timed stop for the currently running playback session."""
        stop_done = threading.Event()

        def stop_after_delay():
            deadline = time.monotonic() + delay
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(0.1, remaining))
                with self._sequence_lock:
                    current_session_id = self._playback_session_id
                    current_thread = self._sequence_thread
                if current_session_id != session_id or current_thread is None or not current_thread.is_alive():
                    stop_done.set()
                    return

            with self._sequence_lock:
                current_session_id = self._playback_session_id
                current_thread = self._sequence_thread
            if current_session_id == session_id and current_thread is not None and current_thread.is_alive():
                self.stop_sequence()
            stop_done.set()

        threading.Thread(target=stop_after_delay, daemon=True, name="SoundGen-CompStopTimer").start()
        return stop_done

    def set_master_volume(self, volume: float):
        """
        Set the master volume level.
        Args:
            volume (float): Volume level (0.0 to 1.0).
        """
        super().set_master_volume(volume)

    def set_effects(self, effects: list):
        """
        Set the list of sound effects to apply to the audio signal.
        Args:
            effects (list): List of sound effect instances (e.g., [SoundEffect.adsr()]).
        """
        super().set_effects(effects)

    def play_polyphonic(self, notes: list[list[tuple[str, float]]], as_tone: bool = False, volume: float = None, block: bool = False):
        """
        Play multiple sequences of musical notes simultaneously (poliphony).
        It is possible to play multi track music by providing a list of sequences,
        where each sequence is a list of tuples (note, duration).
        Duration is in notes fractions (e.g., 1/4 for quarter note).
        Args:
            notes (list[list[tuple[str, float]]]): List of sequences, each sequence is a list of tuples (note, duration).
            as_tone (bool): If True, play as tones, considering duration in seconds
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.
            block (bool): If True, block until the entire sequence has been played.
        """
        self._ensure_speaker_ready()
        blk, duration = super().play_polyphonic(notes, as_tone, volume)
        self._output_device.play(blk)
        if block and duration > 0.0:
            time.sleep(duration)

    def play_composition(
        self,
        composition: "MusicComposition",
        block: bool | None = None,
        loop: bool = False,
        play_for: float | None = None,
    ):
        """
        Play a MusicComposition object.

        Configures the SoundGenerator with the composition's settings and plays
        the sequence using play_step_sequence.

        The composition format is interpreted as a list of steps, where each step
        is a list of (note, duration) tuples to play simultaneously.

        Args:
            composition (MusicComposition): The composition to play.
            block (bool | None): Controls whether this call waits for playback.
                - True: wait until the current playback session ends. When
                  ``loop=True`` and ``play_for`` is not set, this may block
                  indefinitely until ``stop_sequence()`` or ``stop()`` is called
                  from another thread.
                - False: start playback and return immediately.
                - None: choose automatically based on the playback mode. Finite
                  playback blocks, infinite looping returns immediately, and
                  timed looping (``loop=True`` with ``play_for`` set) blocks
                  until the timed stop completes. This is the recommended
                  default for most scripts and examples.
            loop (bool): If True, loop the composition until ``stop_sequence()``
                is called or until ``play_for`` expires.
            play_for (float | None): When looping, stop automatically after the
                given number of seconds. Requires ``loop=True``.
        """
        if play_for is not None:
            play_for = float(play_for)
            if play_for <= 0.0:
                raise ValueError("play_for must be greater than 0.")
            if not loop:
                raise ValueError("play_for requires loop=True.")

        if block is None:
            block = (not loop) or (play_for is not None)

        # Configure the generator with composition settings
        self.set_bpm(composition.bpm)
        self.set_wave_form(composition.waveform)
        self.set_master_volume(composition.volume)
        self.set_effects(composition.effects)

        sequence = []
        step_duration = None

        for step_data in composition.composition:
            step_notes = []
            for note, duration in step_data:
                if step_duration is None:
                    step_duration = duration  # Use first note's duration as step duration
                if note.upper() != "REST":
                    step_notes.append(note)
            sequence.append(step_notes)

        if step_duration is None:
            step_duration = 1 / 16  # Default fallback

        playback_done = threading.Event()
        on_complete_callback = None
        if not loop:

            def on_complete():
                playback_done.set()

            on_complete_callback = on_complete

        self.play_step_sequence(
            sequence=sequence,
            note_duration=step_duration,
            bpm=composition.bpm,
            loop=loop,
            on_complete_callback=on_complete_callback,
            volume=composition.volume,
        )

        with self._sequence_lock:
            session_id = self._playback_session_id

        timed_stop_done = None
        if loop and play_for is not None:
            timed_stop_done = self._schedule_sequence_stop(session_id, play_for)

        if not block:
            return

        if not loop:
            playback_done.wait()
            self._wait_for_playback_session_end(session_id)
            drain_time = self._estimate_output_drain_time()
            if drain_time > 0.0:
                time.sleep(drain_time)
            return

        if timed_stop_done is not None:
            timed_stop_done.wait()
        self._wait_for_playback_session_end(session_id)

    def play_chord(self, notes: list[str], note_duration: float | str = 1 / 4, volume: float = None, block: bool = False):
        """
        Play a chord consisting of multiple musical notes simultaneously for a specified duration and volume.
        Args:
            notes (list[str]): List of musical notes to play (e.g., ['A4', 'C#5', 'E5']).
            note_duration (float | str): Duration of the chord as a float (like 1/4, 1/8) or a symbol ('W', 'H', 'Q', etc.).
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.
            block (bool): If True, block until the entire chord has been played.
        """
        self._ensure_speaker_ready()
        logger.debug(f"SoundGenerator.play_chord: notes={notes}")
        blk = super().play_chord(notes, note_duration, volume)
        self._output_device.play(blk)
        if block:
            duration = self._note_duration(note_duration)
            if duration > 0.0:
                time.sleep(duration)

    def play(self, note: str, note_duration: float | str = 1 / 4, volume: float = None, block: bool = False):
        """
        Play a musical note for a specified duration and volume.
        Args:
            note (str): The musical note to play (e.g., 'A4', 'C#5', 'REST').
            note_duration (float | str): Duration of the note as a float (like 1/4, 1/8) or a symbol ('W', 'H', 'Q', etc.).
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.
            block (bool): If True, block until the entire note has been played.
        """
        self._ensure_speaker_ready()
        logger.debug(f"SoundGenerator.play: note={note}")
        data = super().play(note, note_duration, volume)
        self._output_device.play(data)
        if block:
            duration = self._note_duration(note_duration)
            if duration > 0.0:
                time.sleep(duration)

    def play_tone(self, note: str, duration: float = 0.25, volume: float = None, block: bool = False):
        """Play a musical note with duration specified in seconds.

        Unlike ``play()`` which interprets duration as a musical note fraction,
        this method takes the duration directly in seconds.

        Args:
            note (str): The musical note to play (e.g., 'A4', 'C#5', 'REST').
            duration (float): Duration in seconds (default 0.25).
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.
            block (bool): If True, block until the entire note has been played.
        """
        self._ensure_speaker_ready()
        data = super().play_tone(note, duration, volume)
        self._output_device.play(data)
        if block and duration > 0.0:
            time.sleep(duration)

    def play_abc(self, abc_string: str, volume: float = None, block: bool = False):
        """Play a sequence of musical notes defined in ABC notation.

        The parser is ABC 2.1 standard compliant (key signatures, accidentals,
        tuplets, broken rhythm, multimeasure rests, etc.).  See
        :class:`ABCNotationLoader` for the full feature list and limitations.

        Args:
            abc_string (str): ABC notation string defining the sequence of notes.
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.
            block (bool): If True, block until the entire sequence has been played.
        """
        if not abc_string or abc_string.strip() == "":
            return
        self._ensure_speaker_ready()
        player = super().play_abc(abc_string, volume)
        overall_duration = 0.0
        for data, duration in player:
            self._output_device.play(data)
            overall_duration += duration
        if block:
            time.sleep(overall_duration)

    def play_wav(self, wav_file: str, block: bool = False):
        """Play a WAV audio file through the output device.

        Args:
            wav_file (str): The WAV audio file path.
            block (bool): If True, block until the entire WAV file has been played.
        """
        self._ensure_speaker_ready()
        to_play, duration = super().play_wav(wav_file)
        self._output_device.play(to_play)
        if block and duration > 0.0:
            time.sleep(duration)

    def play_step_sequence(
        self,
        sequence: list[list[str]],
        note_duration: float | str = 1 / 16,
        bpm: int = None,
        loop: bool = False,
        on_step_callback: callable = None,
        on_complete_callback: callable = None,
        volume: float = None,
    ):
        """
        Play a step sequence with automatic timing.
        This method handles all the complexity of buffer management internally,
        allowing the app to simply provide the sequence and let the brick manage playback.

        Args:
            sequence (list[list[str]]): List of steps, where each step is a list of notes.
                Empty list or None means REST (silence) for that step.
                Example: [['C4'], ['E4', 'G4'], [], ['C5']]
            note_duration (float | str): Duration of each step as a float (like 1/16) or symbol ('E', 'Q', etc.).
            bpm (int, optional): Tempo in beats per minute. If None, uses instance BPM.
            loop (bool): If True, the sequence will loop indefinitely until stop_sequence() is called.
            on_step_callback (callable, optional): Callback function called for each step.
                Signature: on_step_callback(current_step: int, total_steps: int)
            on_complete_callback (callable, optional): Callback function called when sequence completes (only if loop=False).
                Signature: on_complete_callback()
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.

        Returns:
            None: Returns immediately after starting playback thread.

        Example:
            ```python
            # Simple melody with chords
            sequence = [
                ["C4"],  # Step 0: Single note
                ["E4", "G4"],  # Step 1: Chord
                [],  # Step 2: REST
                ["C5"],  # Step 3: High note
            ]
            sound_gen.play_step_sequence(sequence, note_duration=1 / 16, bpm=120)
            ```
        """
        # Stop any existing sequence
        self.stop_sequence()

        # Use instance BPM if not specified
        if bpm is None:
            bpm = self._bpm

        # Validate sequence
        if not sequence or len(sequence) == 0:
            logger.warning("play_step_sequence: Empty sequence provided")
            return

        # Ensure speaker is ready before starting the sequence thread
        self._ensure_speaker_ready()

        # Start a non-daemon thread so queued playback can keep the process
        # alive until the sequence finishes or is explicitly stopped.
        self._sequence_stop_event.clear()
        self._playback_session_id += 1
        session_id = self._playback_session_id
        self._sequence_thread = threading.Thread(
            target=self._playback_sequence_thread,
            args=(sequence, note_duration, bpm, loop, on_step_callback, on_complete_callback, volume, session_id),
            daemon=False,
            name="SoundGen-StepSeq",
        )
        self._sequence_thread.start()
        logger.info(f"Step sequence started: {len(sequence)} steps at {bpm} BPM (session {session_id})")

    def stop_sequence(self):
        """
        Stop the currently playing step sequence.

        Signals the playback thread to stop and closes the internal speaker to
        immediately drop any pending audio in the ALSA buffer.  The speaker is
        transparently restarted on the next play call via _ensure_speaker_ready.
        """
        logger.debug("stop_sequence() called")
        should_stop_speaker = False
        with self._sequence_lock:
            if self._sequence_thread and self._sequence_thread.is_alive():
                logger.info("Stopping step sequence playback")
                self._playback_session_id += 1
                self._sequence_stop_event.set()
                self._sequence_thread = None
                should_stop_speaker = True
            else:
                logger.debug("stop_sequence called but no active sequence thread")

        # Stop the speaker outside the lock.  Closing the ALSA PCM device
        # immediately drops all pending audio and causes any in-flight
        # speaker.play() in the sequence thread to finish, after which the
        # next play() call will raise SpeakerWriteError and the thread exits.
        if should_stop_speaker and not self.external_speaker:
            self._output_device.stop()

    def is_sequence_playing(self) -> bool:
        """
        Check if a step sequence is currently playing.

        Returns:
            bool: True if a sequence is playing, False otherwise.
        """
        with self._sequence_lock:
            return self._sequence_thread is not None and self._sequence_thread.is_alive()

    def _render_sequence_step(self, notes: list[str], note_duration: float | str, volume: float):
        """Render a single sequence step to a float32 audio buffer."""
        if notes and len(notes) > 0:
            if len(notes) == 1:
                return super(SoundGenerator, self).play(notes[0], note_duration, volume)
            return super(SoundGenerator, self).play_chord(notes, note_duration, volume)
        return super(SoundGenerator, self).play("REST", note_duration, volume)

    def _playback_sequence_thread(
        self,
        sequence: list[list[str]],
        note_duration: float | str,
        bpm: int,
        loop: bool,
        on_step_callback: callable,
        on_complete_callback: callable,
        volume: float,
        session_id: int,
    ):
        """Internal thread for step sequence playback.

        Uses a simple generate-play loop.  Each ``speaker.play()`` call writes
        directly to the ALSA PCM device; when the hardware buffer is full the
        call blocks, providing natural back-pressure and timing.

        Stopping is handled by ``stop_sequence()`` which closes the speaker.
        The resulting ``SpeakerWriteError`` (or similar exception) on the next
        ``play()`` call is caught here and the thread exits cleanly.
        """
        from itertools import cycle

        try:
            duration = self._note_duration(note_duration)
            total_steps = len(sequence)

            logger.info(f"Starting sequence: {total_steps} steps at {bpm} BPM")
            speaker_buffer = float(self._output_device.buffer_size or 0)
            speaker_rate = float(self._sample_rate or self._output_device.sample_rate or 0)
            shared_prequeue_lead = (
                (speaker_buffer / speaker_rate) if self._output_device.shared and speaker_buffer > 0.0 and speaker_rate > 0.0 else 0.0
            )
            prequeue_future_steps = max(1, int(math.ceil(shared_prequeue_lead / duration))) if shared_prequeue_lead > 0.0 and duration > 0.0 else 0
            render_ahead_steps = max(1, prequeue_future_steps)

            step_iterator = cycle(enumerate(sequence)) if loop else enumerate(sequence)
            current_step = next(step_iterator, None)
            if current_step is None:
                return

            current_step_index, current_notes = current_step
            current_data = self._render_sequence_step(current_notes, note_duration, volume)
            current_data_prequeued = False
            future_steps = []

            processed_steps = 0
            while current_step is not None:
                step_start = time.monotonic()

                if self._sequence_stop_event.is_set():
                    logger.debug(f"Sequence stopped at step {current_step_index}")
                    break

                # --- Send audio to speaker ---
                if current_data is not None and not current_data_prequeued:
                    try:
                        self._output_device.play(current_data)
                    except Exception:
                        # Speaker was closed by stop_sequence() — exit gracefully
                        if self._sequence_stop_event.is_set():
                            break
                        raise

                # --- Step callback ---
                if on_step_callback:
                    try:
                        on_step_callback(current_step_index, total_steps)
                    except Exception as e:
                        logger.error(f"Error in step callback: {e}")

                processed_steps += 1

                while not self._sequence_stop_event.is_set() and len(future_steps) < render_ahead_steps:
                    if not loop and processed_steps + len(future_steps) >= total_steps:
                        break
                    next_step = next(step_iterator, None)
                    if next_step is None:
                        break

                    next_step_index, next_notes = next_step
                    next_data = self._render_sequence_step(next_notes, note_duration, volume)
                    next_data_prequeued = len(future_steps) < prequeue_future_steps
                    if next_data is not None and next_data_prequeued:
                        try:
                            self._output_device.play(next_data)
                        except Exception:
                            if self._sequence_stop_event.is_set():
                                break
                            raise
                    future_steps.append((next_step_index, next_notes, next_data, next_data_prequeued))

                # --- Wait for the remaining step duration ----
                # Rendering the next step happens before this sleep so the next
                # write can be issued immediately at the step boundary.
                elapsed = time.monotonic() - step_start
                remaining = duration - elapsed
                if remaining > 0.0:
                    deadline = step_start + duration
                    while True:
                        left = deadline - time.monotonic()
                        if left <= 0:
                            break
                        if self._sequence_stop_event.is_set():
                            break
                        time.sleep(min(0.01, left))

                if not future_steps:
                    break

                current_step_index, current_notes, current_data, current_data_prequeued = future_steps.pop(0)
            logger.info("Sequence playback ended")

            if not self._sequence_stop_event.is_set() and not loop and on_complete_callback:
                try:
                    on_complete_callback()
                except Exception as e:
                    logger.error(f"Error in complete callback: {e}")

        except Exception as e:
            logger.error(f"Error in sequence playback: {e}", exc_info=True)
        finally:
            with self._sequence_lock:
                self._sequence_thread = None
