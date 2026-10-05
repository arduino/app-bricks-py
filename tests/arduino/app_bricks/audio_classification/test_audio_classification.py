# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import io
import struct
import wave

import pytest

from arduino.app_bricks.audio_classification import AudioClassification


def _wav(sample_width: int = 2) -> io.BytesIO:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<4h", 0, 1, -1, 2) if sample_width == 2 else bytes(4 * sample_width))
    buf.seek(0)
    return buf


def _inference(scores: dict[str, float]) -> dict[str, object]:
    return {"result": {"classification": scores}}


def test_classify_from_file_returns_the_best_class(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(AudioClassification, "infer_from_features", classmethod(lambda cls, features: _inference({"glass": 0.9, "noise": 0.1})))
    assert AudioClassification.classify_from_file(_wav(), confidence=0.5) == {"class_name": "glass", "confidence": 90.0}


def test_classify_from_file_below_confidence_is_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(AudioClassification, "infer_from_features", classmethod(lambda cls, features: _inference({"glass": 0.3})))
    assert AudioClassification.classify_from_file(_wav(), confidence=0.5) is None


def test_classify_from_file_reads_24_bit_samples(monkeypatch: pytest.MonkeyPatch):
    seen: list[list[int]] = []

    def infer(cls: type, features: list[int]) -> dict[str, object]:
        seen.append(features)
        return _inference({"glass": 0.9})

    monkeypatch.setattr(AudioClassification, "infer_from_features", classmethod(infer))
    AudioClassification.classify_from_file(_wav(sample_width=3), confidence=0.5)
    assert seen == [[0, 0, 0, 0]]
