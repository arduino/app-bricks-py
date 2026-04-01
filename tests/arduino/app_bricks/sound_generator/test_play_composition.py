# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading

import numpy as np
import pytest

from arduino.app_bricks.sound_generator import MusicComposition, SoundGenerator
import arduino.app_bricks.sound_generator as sound_generator_module


class DummySpeaker:
    sample_rate = 32000
    buffer_size = 4096
    shared = True


def test_playback_sequence_thread_prequeues_enough_future_steps_to_cover_speaker_period(monkeypatch):
    events = []
    current_time = {"value": 0.0}

    class SequencerSpeaker:
        sample_rate = 10
        buffer_size = 1
        shared = True

        def play(self, data):
            events.append(("play", round(current_time["value"], 3), len(data)))

    generator = SoundGenerator(output_device=SequencerSpeaker(), bpm=180)

    def fake_render(notes, note_duration, volume):
        events.append(("render", round(current_time["value"], 3), tuple(notes)))
        current_time["value"] += 0.01
        return np.ones(8, dtype=np.float32)

    def fake_sleep(delay):
        events.append(("sleep", round(current_time["value"], 3), round(delay, 3)))
        current_time["value"] += delay

    monkeypatch.setattr(generator, "_render_sequence_step", fake_render)
    monkeypatch.setattr(sound_generator_module.time, "monotonic", lambda: current_time["value"])
    monkeypatch.setattr(sound_generator_module.time, "sleep", fake_sleep)

    generator._playback_sequence_thread(
        sequence=[["C4"], ["D4"], ["E4"]],
        note_duration=1 / 16,
        bpm=180,
        loop=False,
        on_step_callback=None,
        on_complete_callback=None,
        volume=None,
        session_id=1,
    )

    non_sleep_events = [event for event in events if event[0] != "sleep"]
    assert non_sleep_events[:6] == [
        ("render", 0.0, ("C4",)),
        ("play", 0.01, 8),
        ("render", 0.01, ("D4",)),
        ("play", 0.02, 8),
        ("render", 0.02, ("E4",)),
        ("play", 0.03, 8),
    ]


def test_sound_generator_default_speaker_is_shared(monkeypatch):
    captured = {}

    class FakeInternalSpeaker:
        sample_rate = 32000

    class FakeSpeakerFactory:
        RATE_32K = 32000
        BUFFER_SIZE_SAFE = 4096

        def __new__(cls, **kwargs):
            captured.update(kwargs)
            return FakeInternalSpeaker()

    monkeypatch.setattr(sound_generator_module, "Speaker", FakeSpeakerFactory)

    generator = SoundGenerator()

    assert generator.external_speaker is False
    assert isinstance(generator._output_device, FakeInternalSpeaker)
    assert captured == {
        "sample_rate": 32000,
        "format": np.float32,
        "buffer_size": 4096,
        "shared": True,
    }


def test_play_composition_passes_loop_to_step_sequence():
    generator = SoundGenerator(output_device=DummySpeaker())
    composition = MusicComposition(
        composition=[
            [("C4", 1 / 16)],
            [("REST", 1 / 16)],
            [("E4", 1 / 16), ("G4", 1 / 16)],
        ],
        bpm=140,
        waveform="square",
        volume=0.7,
        effects=[],
    )

    captured = {}

    def fake_play_step_sequence(*, sequence, note_duration, bpm, loop, volume, on_complete_callback=None, **kwargs):
        captured["sequence"] = sequence
        captured["note_duration"] = note_duration
        captured["bpm"] = bpm
        captured["loop"] = loop
        captured["volume"] = volume
        captured["on_complete_callback"] = on_complete_callback

    generator.play_step_sequence = fake_play_step_sequence

    generator.play_composition(composition, loop=True)

    assert captured == {
        "sequence": [["C4"], [], ["E4", "G4"]],
        "note_duration": 1 / 16,
        "bpm": 140,
        "loop": True,
        "volume": 0.7,
        "on_complete_callback": None,
    }


def test_play_composition_defaults_to_blocking_for_one_shot(monkeypatch):
    generator = SoundGenerator(output_device=DummySpeaker())
    composition = MusicComposition(
        composition=[[("C4", 1 / 16)]],
        effects=[],
    )
    captured = {}

    monkeypatch.setattr(sound_generator_module.time, "sleep", lambda _: None)

    def fake_play_step_sequence(*, on_complete_callback=None, **kwargs):
        captured["on_complete_callback"] = on_complete_callback
        if on_complete_callback is not None:
            on_complete_callback()

    generator.play_step_sequence = fake_play_step_sequence

    generator.play_composition(composition)

    assert captured["on_complete_callback"] is not None


def test_play_composition_defaults_to_blocking_for_timed_loop(monkeypatch):
    generator = SoundGenerator(output_device=DummySpeaker())
    composition = MusicComposition(
        composition=[[("C4", 1 / 16)]],
        effects=[],
    )
    captured = {}
    stop_done = threading.Event()
    stop_done.set()

    def fake_play_step_sequence(**kwargs):
        generator._playback_session_id = 7
        generator._sequence_thread = None
        captured["loop"] = kwargs["loop"]
        captured["on_complete_callback"] = kwargs.get("on_complete_callback")

    def fake_schedule_sequence_stop(session_id: int, delay: float):
        captured["scheduled_session_id"] = session_id
        captured["scheduled_delay"] = delay
        return stop_done

    def fake_wait_for_playback_session_end(session_id: int):
        captured["waited_session_id"] = session_id

    monkeypatch.setattr(generator, "play_step_sequence", fake_play_step_sequence)
    monkeypatch.setattr(generator, "_schedule_sequence_stop", fake_schedule_sequence_stop)
    monkeypatch.setattr(generator, "_wait_for_playback_session_end", fake_wait_for_playback_session_end)

    generator.play_composition(composition, loop=True, play_for=5.0)

    assert captured == {
        "loop": True,
        "on_complete_callback": None,
        "scheduled_session_id": 7,
        "scheduled_delay": 5.0,
        "waited_session_id": 7,
    }


def test_play_composition_rejects_invalid_play_for_without_loop():
    generator = SoundGenerator(output_device=DummySpeaker())
    composition = MusicComposition(
        composition=[[("C4", 1 / 16)]],
        effects=[],
    )

    with pytest.raises(ValueError, match="play_for requires loop=True"):
        generator.play_composition(composition, play_for=5.0)


def test_play_composition_rejects_non_positive_play_for():
    generator = SoundGenerator(output_device=DummySpeaker())
    composition = MusicComposition(
        composition=[[("C4", 1 / 16)]],
        effects=[],
    )

    with pytest.raises(ValueError, match="play_for must be greater than 0"):
        generator.play_composition(composition, loop=True, play_for=0.0)


def test_play_step_sequence_uses_non_daemon_thread(monkeypatch):
    generator = SoundGenerator(output_device=DummySpeaker())
    captured = {}

    class FakeThread:
        def __init__(self, *, target, args, daemon, name):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon
            captured["name"] = name
            self._alive = False

        def start(self):
            self._alive = True

        def is_alive(self):
            return self._alive

    monkeypatch.setattr(sound_generator_module.threading, "Thread", FakeThread)

    generator.play_step_sequence(sequence=[["C4"]], loop=False)

    assert captured["daemon"] is False
    assert captured["name"] == "SoundGen-StepSeq"
