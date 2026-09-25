# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from arduino.app_bricks.ocr import OCR, OcrError, OcrResult, TextDetection


def _jpeg_bytes(width: int = 8, height: int = 8, quality: int = 90) -> bytes:
    """A real, small JPEG (a gradient, so it is not trivially compressible)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    frame[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return encoded.tobytes()


_JPEG = _jpeg_bytes()


def _decode(b64: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runner_metadata() -> dict:
    """Metadata shaped exactly like the ocr-runner's build_metadata output."""
    return {
        "text": "HELLO\nWORLD",
        "detections": [
            {
                "text": "HELLO",
                "confidence": 0.91,
                "bounding_box_xyxy": [10, 20, 110, 60],
                "polygon": [[10, 20], [110, 20], [110, 60], [10, 60]],
                "type": "horizontal",
            },
            {
                "text": "WORLD",
                "confidence": 0.85,
                "bounding_box_xyxy": [12, 80, 118, 122],
                "polygon": [[14, 84], [118, 80], [116, 118], [12, 122]],
                "type": "free",
            },
        ],
    }


class FakeConnection:
    """Context-manager stand-in for websockets.sync.client.connect()."""

    def __init__(self, uri: str, messages: list[str] | None = None, recv_error: Exception | None = None):
        self.uri = uri
        self.sent: list[str] = []
        self._messages = list(messages or [])
        self._recv_error = recv_error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def send(self, message: str) -> None:
        self.sent.append(message)

    def recv(self, timeout: float | None = None) -> str:
        if self._recv_error is not None:
            raise self._recv_error
        return self._messages.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ocr(monkeypatch: pytest.MonkeyPatch, **kwargs) -> OCR:
    """Return an OCR instance with infrastructure mocked out."""
    fake_compose = {"services": {"ocr": {}}}
    monkeypatch.setattr("arduino.app_bricks.ocr.load_brick_compose_file", lambda cls: fake_compose)
    monkeypatch.setattr("arduino.app_bricks.ocr.resolve_address", lambda host: "127.0.0.1")
    return OCR(**kwargs)


@pytest.fixture()
def ocr(monkeypatch: pytest.MonkeyPatch) -> OCR:
    return _make_ocr(monkeypatch)


def _patch_connect(monkeypatch: pytest.MonkeyPatch, connections: dict[str, FakeConnection]):
    """Route module-level connect(uri) calls to the given fake connections by port."""

    def fake_connect(uri: str, **kwargs):
        port = uri.rsplit(":", 1)[1]
        return connections[port]

    monkeypatch.setattr("arduino.app_bricks.ocr.connect", fake_connect)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_resolves_runner_endpoints(ocr: OCR):
    assert ocr._ws_send_url == "ws://127.0.0.1:5000"
    assert ocr._ws_recv_url == "ws://127.0.0.1:5001"


def test_constructor_rejects_a_compose_without_services(monkeypatch: pytest.MonkeyPatch):
    """No service to talk to: the brick says so instead of failing later on a missing host."""
    monkeypatch.setattr("arduino.app_bricks.ocr.load_brick_compose_file", lambda cls: {"services": {}})
    monkeypatch.setattr("arduino.app_bricks.ocr.resolve_address", lambda host: host)
    with pytest.raises(RuntimeError):
        OCR()


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def test_encode_image_accepts_numpy_arrays():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    encoded, scale = OCR._encode_image(frame)
    data = base64.b64decode(encoded)
    assert data[:2] == b"\xff\xd8"  # JPEG magic number
    assert scale == (1.0, 1.0)


def test_encode_image_sends_small_encoded_bytes_untouched():
    encoded, scale = OCR._encode_image(_JPEG)
    assert base64.b64decode(encoded) == _JPEG  # no recompression when it already fits
    assert scale == (1.0, 1.0)


def test_encode_image_accepts_file_paths(tmp_path: Path):
    image_file = tmp_path / "image.jpg"
    image_file.write_bytes(_JPEG)
    assert base64.b64decode(OCR._encode_image(image_file)[0]) == _JPEG
    assert base64.b64decode(OCR._encode_image(str(image_file))[0]) == _JPEG


def test_encode_image_rejects_undecodable_bytes():
    with pytest.raises(ValueError):
        OCR._encode_image(b"\xff\xd8fake-jpeg-bytes")


def test_encode_image_downscales_large_arrays_to_the_max_side():
    frame = np.zeros((3024, 4032, 3), dtype=np.uint8)
    frame[..., 2] = np.linspace(0, 255, 4032, dtype=np.uint8)[None, :]

    encoded, (scale_x, scale_y) = OCR._encode_image(frame)

    sent = _decode(encoded)
    assert sent.shape[:2] == (1536, 2048)  # longest side capped, aspect ratio kept
    assert (scale_x, scale_y) == pytest.approx((2048 / 4032, 1536 / 3024))
    assert len(base64.b64decode(encoded)) * 4 / 3 <= 1_000_000


def test_encode_image_downscales_large_encoded_files(tmp_path: Path):
    big = _jpeg_bytes(width=4000, height=2000)
    image_file = tmp_path / "big.jpg"
    image_file.write_bytes(big)

    encoded, (scale_x, scale_y) = OCR._encode_image(image_file)

    assert base64.b64decode(encoded) != big
    assert _decode(encoded).shape[:2] == (1024, 2048)
    assert (scale_x, scale_y) == pytest.approx((2048 / 4000, 1024 / 2000))


def _png_bytes(arr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", arr)
    assert ok
    return encoded.tobytes()


def test_encode_image_handles_png_with_alpha_and_greyscale():
    small_rgba = _png_bytes(np.full((8, 8, 4), (10, 20, 30, 0), np.uint8))
    small_grey = _png_bytes(np.full((8, 8), 200, np.uint8))
    for data in (small_rgba, small_grey):
        encoded, scale = OCR._encode_image(data)
        assert base64.b64decode(encoded) == data  # small PNGs travel untouched
        assert scale == (1.0, 1.0)
        assert _decode(encoded).shape == (8, 8, 3)  # the runner decodes them as BGR

    big_rgba = _png_bytes(np.full((2200, 3000, 4), (10, 20, 30, 128), np.uint8))
    encoded, (scale_x, scale_y) = OCR._encode_image(big_rgba)
    assert base64.b64decode(encoded)[:2] == b"\xff\xd8"  # downscaled ones are re-encoded as JPEG
    assert _decode(encoded).shape[:2] == (1502, 2048)
    assert (scale_x, scale_y) == pytest.approx((2048 / 3000, 1502 / 2200))


def test_encode_image_accepts_greyscale_and_bgra_arrays():
    for frame in (np.full((8, 8), 200, np.uint8), np.full((8, 8, 4), (10, 20, 30, 255), np.uint8)):
        encoded, scale = OCR._encode_image(frame)
        assert _decode(encoded).shape == (8, 8, 3)
        assert scale == (1.0, 1.0)


def test_encode_image_keeps_the_payload_under_the_runner_limit():
    # White noise at 2048 px does not compress: the encoder has to lower quality, then size.
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(1536, 2048, 3), dtype=np.uint8)

    encoded, (scale_x, scale_y) = OCR._encode_image(frame)

    assert 4 * ((len(base64.b64decode(encoded)) + 2) // 3) <= 1_000_000
    sent = _decode(encoded)
    assert sent.shape[1] == round(2048 * scale_x) and sent.shape[0] == round(1536 * scale_y)


def test_encode_image_rejects_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        OCR._encode_image(tmp_path / "missing.jpg")


def test_encode_image_rejects_unsupported_types():
    with pytest.raises(TypeError):
        OCR._encode_image(12345)


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------


def test_parse_metadata_round_trip():
    result = OCR._parse_metadata(_runner_metadata())

    assert isinstance(result, OcrResult)
    assert result.text == "HELLO\nWORLD"
    assert str(result) == "HELLO\nWORLD"
    assert [d.text for d in result.detections] == ["HELLO", "WORLD"]

    first = result.detections[0]
    assert isinstance(first, TextDetection)
    assert first.confidence == pytest.approx(0.91)
    assert first.bounding_box_xyxy == (10, 20, 110, 60)
    assert first.polygon == [(10, 20), (110, 20), (110, 60), (10, 60)]


def test_parse_metadata_maps_positions_back_to_the_original_image():
    result = OCR._parse_metadata(_runner_metadata(), 0.0, scale=(0.5, 0.25))

    first = result.detections[0]
    assert first.bounding_box_xyxy == (20, 80, 220, 240)
    assert first.polygon == [(20, 80), (220, 80), (220, 240), (20, 240)]
    second = result.detections[1]
    assert second.polygon[0] == (28, 336)


def test_parse_metadata_handles_empty_payloads():
    result = OCR._parse_metadata({})
    assert result.text == ""
    assert result.detections == []


def test_parse_metadata_skips_malformed_detections_and_rebuilds_text():
    metadata = {
        "detections": [
            {"text": "OK", "confidence": 0.9, "bounding_box_xyxy": [0, 0, 5, 5], "polygon": [[0, 0], [5, 0], [5, 5], [0, 5]]},
            {"confidence": 0.9},  # missing every other field
        ]
    }
    result = OCR._parse_metadata(metadata)
    assert [d.text for d in result.detections] == ["OK"]
    assert result.text == "OK"  # no 'text' key: rebuilt from the detections


# ---------------------------------------------------------------------------
# extract_text request/response
# ---------------------------------------------------------------------------


def test_extract_text_sends_config_then_frame_and_parses_answer(ocr: OCR, monkeypatch: pytest.MonkeyPatch):
    answer = json.dumps({"frame": None, "metadata": _runner_metadata()})
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", messages=[answer])
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    result = ocr.extract_text(_JPEG)

    assert result.text == "HELLO WORLD"  # single_line is on by default
    assert len(send_ws.sent) == 2
    # The config travels before the frame, so the runner applies it to this image
    config = json.loads(send_ws.sent[0])
    assert config == {"config": {"allowlist": "", "rotation": []}}
    sent = json.loads(send_ws.sent[1])
    assert base64.b64decode(sent["frame"]) == _JPEG


def test_extract_text_filters_by_constructor_confidence(monkeypatch: pytest.MonkeyPatch):
    metadata = _runner_metadata()  # HELLO at 0.91, WORLD at 0.85
    answer = json.dumps({"frame": None, "metadata": metadata})
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", messages=[answer])
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    ocr = _make_ocr(monkeypatch, confidence=0.9)
    result = ocr.extract_text(_JPEG)

    assert [d.text for d in result.detections] == ["HELLO"]
    assert result.text == "HELLO"  # rebuilt from the kept detections


def test_confidence_is_validated():
    with pytest.raises(ValueError):
        OCR._validate_min_confidence(1.5)
    with pytest.raises(ValueError):
        OCR._validate_min_confidence(-0.1)
    with pytest.raises(ValueError):
        OCR._validate_min_confidence(True)
    with pytest.raises(ValueError):
        OCR._validate_min_confidence("0.5")


def test_extract_text_sends_constructor_allowlist(monkeypatch: pytest.MonkeyPatch):
    answer = json.dumps({"frame": None, "metadata": {}})
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", messages=[answer])
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    ocr = _make_ocr(monkeypatch, allowlist="0123456789")
    ocr.extract_text(_JPEG)

    assert json.loads(send_ws.sent[0]) == {"config": {"allowlist": "0123456789", "rotation": []}}


def test_extract_text_sends_constructor_rotation_and_call_override(monkeypatch: pytest.MonkeyPatch):
    answers = [json.dumps({"frame": None, "metadata": {}})] * 3
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", messages=answers)
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    ocr = _make_ocr(monkeypatch, rotation=[270, 90, 90])
    ocr.extract_text(_JPEG)
    ocr.extract_text(_JPEG, rotation=180)
    ocr.extract_text(_JPEG, rotation=[])  # upright only for this call

    configs = [json.loads(send_ws.sent[i])["config"]["rotation"] for i in (0, 2, 4)]
    assert configs == [[90, 270], [180], []]


def test_extract_text_single_line_joins_with_spaces(monkeypatch: pytest.MonkeyPatch):
    answers = [json.dumps({"frame": None, "metadata": _runner_metadata()})] * 5
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", messages=answers)
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    default = _make_ocr(monkeypatch)
    assert default.extract_text(_JPEG).text == "HELLO WORLD"  # on by default

    ocr = _make_ocr(monkeypatch, single_line=True)
    assert ocr.extract_text(_JPEG).text == "HELLO WORLD"
    assert ocr.extract_text(_JPEG, single_line=False).text == "HELLO\nWORLD"  # call override wins

    multiline = _make_ocr(monkeypatch, single_line=False)
    assert multiline.extract_text(_JPEG).text == "HELLO\nWORLD"
    assert multiline.extract_text(_JPEG, single_line=True).text == "HELLO WORLD"


def test_rotation_is_validated():
    assert OCR._validate_rotation(None) == []
    assert OCR._validate_rotation(90) == [90]
    assert OCR._validate_rotation((180, 90, 0, 450)) == [90, 180]  # 0 is upright, 450 wraps to 90
    for bad in ([45], "90", [True], [90.5], [None]):
        with pytest.raises(ValueError):
            OCR._validate_rotation(bad)


def test_extract_text_returns_positions_in_the_original_image(ocr: OCR, monkeypatch: pytest.MonkeyPatch):
    """A 4032x3024 frame is sent at 2048x1536; the runner's boxes come back scaled up again."""
    answer = json.dumps({"frame": None, "metadata": _runner_metadata()})
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", messages=[answer])
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    frame = np.zeros((3024, 4032, 3), dtype=np.uint8)
    result = ocr.extract_text(frame)

    assert _decode(json.loads(send_ws.sent[1])["frame"]).shape[:2] == (1536, 2048)
    factor = 4032 / 2048
    assert result.detections[0].bounding_box_xyxy == tuple(round(v * factor) for v in (10, 20, 110, 60))


def test_extract_text_raises_when_runner_is_unreachable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("arduino.app_bricks.ocr._RETRY_INTERVAL_SEC", 0.01)

    def refuse(uri: str, **kwargs):
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr("arduino.app_bricks.ocr.connect", refuse)

    ocr = _make_ocr(monkeypatch, timeout=0.05)
    with pytest.raises(OcrError):
        ocr.extract_text(_JPEG)


def test_extract_text_raises_when_runner_does_not_answer(ocr: OCR, monkeypatch: pytest.MonkeyPatch):
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", recv_error=TimeoutError())
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    with pytest.raises(OcrError):
        ocr.extract_text(_JPEG)
