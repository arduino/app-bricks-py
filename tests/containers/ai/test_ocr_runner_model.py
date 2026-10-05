# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""End-to-end reading of reference images with the real EasyOCR models, on the CPU.

The runner is executed in a subprocess, from its own directory, exactly as the container
imports it (`import inference`), so its top-level `utils` package cannot clash with other
runners' and the models load from their committed paths. Only numpy, OpenCV and
onnxruntime are needed; without onnxruntime (not a dependency of this repository's test
environment) the tests are skipped. Install it with `pip install onnxruntime` to run them.

`hey-arduino.png` holds "Hey, Arduino!" over "Nice to meet you!"; `hey-arduino-cw90.png` is
the same text turned 90 degrees clockwise, running top to bottom.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("cv2")

REPO = Path(__file__).resolve().parents[3]
RUNNER_DIR = REPO / "containers" / "ai" / "ocr-runner"
FRAMEWORK_DIR = REPO / "containers" / "base" / "aihub-framework"
IMAGES = Path(__file__).resolve().parent / "ocr_runner_images"

_SCRIPT = """
import json, sys
import cv2
import inference

results = []
for path, rotation in json.loads(sys.argv[1]):
    inference.apply_config({"allowlist": "", "rotation": rotation})
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    _, metadata = inference.inference_callback(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    results.append({"shape": list(bgr.shape[:2]), **metadata})
print("@@RESULT@@" + json.dumps(results))
"""


def _read(*cases: tuple[str, list[int]]) -> list[dict]:
    env = {
        **os.environ,
        "EASYOCR_EP": "cpu",
        "PYTHONPATH": os.pathsep.join([str(RUNNER_DIR), str(FRAMEWORK_DIR)]),
    }
    arg = json.dumps([(str(IMAGES / name), rotation) for name, rotation in cases])
    proc = subprocess.run([sys.executable, "-c", _SCRIPT, arg], cwd=RUNNER_DIR, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(line for line in proc.stdout.splitlines() if line.startswith("@@RESULT@@"))
    return json.loads(line[len("@@RESULT@@") :])


@pytest.fixture(scope="module")
def readings() -> dict[str, dict]:
    upright, rotated, rotated_not_turned, upright_with_rotation = _read(
        ("hey-arduino.png", []),
        ("hey-arduino-cw90.png", [90]),
        ("hey-arduino-cw90.png", []),
        ("hey-arduino.png", [90, 180, 270]),
    )
    return {"upright": upright, "rotated": rotated, "rotated_not_turned": rotated_not_turned, "upright_with_rotation": upright_with_rotation}


def _words(metadata: dict) -> list[str]:
    return " ".join(d["text"] for d in metadata["detections"]).split()


def test_upright_image_reads_both_lines_in_order(readings):
    words = _words(readings["upright"])
    assert words[:5] == ["Hey,", "Arduino!", "Nice", "to", "meet"]
    # The final "you!" is read "youl" by this model: '!' after a word is often decoded as
    # 'l' (measured: 62% of '!' read correctly across 7 fonts). Pinning the known reading
    # would make an improvement fail the test, so only the word is checked.
    assert words[5].startswith("you")


def test_rotated_image_reads_like_the_upright_one(readings):
    """With rotation=[90] the page is read turned: lines merge and come out in reading
    order, instead of one interleaved box per word and a '2' for "you!"."""
    assert _words(readings["rotated"]) == _words(readings["upright"])


def test_rotated_image_positions_are_in_the_original_frame(readings):
    """The "Hey, Arduino!" line is the right-hand column of the 431x896 rotated image (x 208-326)."""
    rotated = readings["rotated"]
    height, width = rotated["shape"]
    first = rotated["detections"][0]
    x1, y1, x2, y2 = first["bounding_box_xyxy"]
    assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
    assert (x1 + x2) / 2 > width / 2  # the first line read is the right-hand column
    assert y2 - y1 > x2 - x1  # and it runs vertically
    # The polygon starts at the text's own top-left corner: top-right in the frame.
    start_x, start_y = first["polygon"][0]
    assert start_x == x2 and start_y == y1


def test_rotated_image_without_rotation_reads_garbage(readings):
    """Documents why `rotation` exists: read as is, vertical text is not read."""
    assert "Arduino!" not in _words(readings["rotated_not_turned"])


def test_rotation_setting_does_not_change_upright_text(readings):
    assert _words(readings["upright_with_rotation"]) == _words(readings["upright"])
    assert readings["upright_with_rotation"]["detections"][0]["bounding_box_xyxy"] == readings["upright"]["detections"][0]["bounding_box_xyxy"]
