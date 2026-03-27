import threading

import pytest

from arduino.app_bricks.sound_generator import MusicComposition, SoundGenerator
import arduino.app_bricks.sound_generator as sound_generator_module


class DummySpeaker:
    sample_rate = 16000


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
