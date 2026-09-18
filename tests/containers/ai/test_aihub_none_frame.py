# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Metadata-only runners (e.g. OCR) return None instead of an annotated frame.

The aihub runner framework must keep working in that case: video-only sinks
skip the frame, sinks with a metadata channel still deliver the metadata.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

# Make the shared aihub package importable; it lives in its own source-only image, which
# both aihub-litert-models-runner and aihub-onnx-models-runner copy it from.
FRAMEWORK_DIR = Path(__file__).resolve().parents[3] / "containers" / "base" / "aihub-framework"
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

from aihub.app import AIHubApp  # noqa: E402
from aihub.websocket.output import WebSocketOutput  # noqa: E402


class RecordingSink:
    """Minimal output sink that records every send_frame call."""

    def __init__(self):
        self.calls: list[tuple[np.ndarray | None, dict]] = []

    def send_frame(self, frame: np.ndarray | None, metadata: dict) -> None:
        self.calls.append((frame, metadata))


def _import_mjpeg_output(monkeypatch):
    """Import MJPEGOutput, stubbing flask/waitress when they are not installed."""
    for name, attrs in (
        ("flask", {"Flask": lambda name: None, "Response": object}),
        ("waitress", {"serve": lambda *args, **kwargs: None}),
    ):
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            for attr, value in attrs.items():
                setattr(module, attr, value)
            monkeypatch.setitem(sys.modules, name, module)

    from aihub.mjpeg.output import MJPEGOutput

    return MJPEGOutput


def test_frame_callback_forwards_none_frame_to_sinks():
    metadata = {"text": "hello", "detections": []}
    app = AIHubApp(inference_cb=lambda frame: (None, metadata))
    sink = RecordingSink()
    app._output_sinks.append(sink)

    app._frame_callback(np.zeros((4, 4, 3), dtype=np.uint8))

    assert sink.calls == [(None, metadata)]


def test_websocket_output_encodes_metadata_only_message():
    output = WebSocketOutput()

    message = output._encode_frame(None, {"text": "hello"})

    assert json.loads(message) == {"frame": None, "metadata": {"text": "hello"}}


def test_websocket_output_still_encodes_real_frames():
    output = WebSocketOutput()

    message = output._encode_frame(np.zeros((2, 3, 3), dtype=np.uint8), {"text": "hello"})

    decoded = json.loads(message)
    assert decoded["frame"]
    assert decoded["width"] == 3
    assert decoded["height"] == 2
    assert decoded["metadata"] == {"text": "hello"}


def test_mjpeg_output_skips_none_frame(monkeypatch):
    MJPEGOutput = _import_mjpeg_output(monkeypatch)
    output = MJPEGOutput()

    output.send_frame(None, {"text": "hello"})

    assert output._latest_jpeg is None


def test_mjpeg_output_still_encodes_real_frames(monkeypatch):
    MJPEGOutput = _import_mjpeg_output(monkeypatch)
    output = MJPEGOutput()

    output.send_frame(np.zeros((2, 3, 3), dtype=np.uint8), {"text": "hello"})

    assert output._latest_jpeg is not None
