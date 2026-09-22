# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import base64
import json
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from arduino.app_internal.core import load_brick_compose_file, resolve_address
from arduino.app_utils import AppError, Logger, brick
from arduino.app_utils.image.adjustments import compress_to_jpeg

logger = Logger("OCR")

_WS_SEND_PORT = 5000  # model runner websocket input (frames in)
_WS_RECV_PORT = 5001  # model runner websocket output (results out)
_RETRY_INTERVAL_SEC = 1.0

# Images are downscaled before being sent so that their longest side is at most this.
# The detector network works on an 800x608 letterbox of the whole image, so detection
# gains nothing above that; the recognizer reads each detected region resized to 64 px of
# height, so some extra resolution helps it read text the detector found. 2048 px keeps
# that margin while the payload stays well under the runner's 1 MiB websocket limit.
_MAX_IMAGE_SIDE = 2048
# Upper bound for the encoded image on the wire (the runner's websocket rejects messages
# above 1 MiB, base64 and JSON framing included).
_MAX_PAYLOAD_BYTES = 1_000_000
_JPEG_QUALITIES = (85, 70, 55)
_VALID_ROTATIONS = (90, 180, 270)


class OcrError(AppError):
    """Raised when the OCR model runner cannot be reached or does not answer in time."""


@dataclass
class TextDetection:
    """One piece of text found in the image.

    Attributes:
        text (str): The recognized text.
        confidence (float): Recognition confidence in [0.0, 1.0].
        bounding_box_xyxy (tuple[int, int, int, int]): (x1, y1, x2, y2) axis-aligned
            box enclosing the text, in pixel coordinates of the image passed to
            `extract_text`.
        polygon (list[tuple[int, int]]): The 4 (x, y) vertices of the detected text
            region, in pixel coordinates of the image passed to `extract_text`,
            ordered top-left, top-right, bottom-right, bottom-left. They differ from
            the bounding box when the text is slanted.
    """

    text: str
    confidence: float
    bounding_box_xyxy: tuple[int, int, int, int]
    polygon: list[tuple[int, int]]


@dataclass
class OcrResult:
    """The text extracted from one image.

    Converting the result to a string (`str(result)` or `print(result)`) yields
    the full extracted text.

    Attributes:
        text (str): Every recognized string joined in reading order (top to
            bottom, left to right), by newlines or, when `single_line` is set,
            by single spaces. Empty when no text was found.
        detections (list[TextDetection]): One entry per piece of text found, in
            reading order, each with its position and confidence.
    """

    text: str
    detections: list[TextDetection] = field(default_factory=list[TextDetection])

    def __str__(self) -> str:
        return self.text


@brick
class OCR:
    """Extracts text from images (OCR) using the EasyOCR model runner.

    The brick sends each image to the model runner over WebSocket and returns
    the recognized text with per-detection positions and confidences:

        ocr = OCR()
        result = ocr.extract_text(image)
        print(result.text)

    Images larger than 2048 px on their longest side are downscaled before being
    sent: the model reads the whole image at 800x608 anyway, so text has to be
    reasonably large in the frame to be found (roughly at least 1.5% of the image
    height). Positions in the result are always in the coordinates of the image
    you passed in.
    """

    def __init__(
        self,
        confidence: float = 0.3,
        allowlist: str | None = None,
        rotation: Iterable[int] | int | None = None,
        single_line: bool = False,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the OCR brick.

        Args:
            confidence (float): Minimum recognition confidence for a piece of
                text to be reported, in [0.0, 1.0]. Detections below it are dropped
                from the result. Default is 0.3; pass 0.0 to report everything the
                model finds.
            allowlist (str): Restrict recognition to these characters, e.g.
                "0123456789" to read only digits. Applied by the model runner while
                decoding, so it improves accuracy on constrained text rather than
                just filtering the output. Default is None (no restriction).
            rotation (Iterable[int] | int): Extra orientations to try when reading
                each detected piece of text, as angles in degrees among 90, 180 and
                270, e.g. `[90, 270]` for text running vertically or `180` for
                upside-down labels. A rotated reading replaces the upright one only
                when it is clearly more confident. Text is always read upright too;
                90 and 270 are only tried on regions taller than wide (vertical
                text), 180 on every region, and each applicable angle costs one
                more recognizer pass per region. Default is None (upright only).
                Can be overridden per call in `extract_text`.
            single_line (bool): Join every recognized piece of text with single
                spaces instead of newlines, so `result.text` is one line. Useful
                when the image holds one logical string split across regions, e.g.
                a plate or a serial number. Default is False. Can be overridden
                per call in `extract_text`.
            timeout (float): Maximum seconds `extract_text` waits for the model
                runner, connection retries included. Default is 30.

        Raises:
            ValueError: If `confidence` is not a number in [0.0, 1.0], or
                `rotation` contains an angle other than 90, 180 or 270.
            RuntimeError: If the model runner host address could not be resolved.
        """
        self._confidence = self._validate_min_confidence(confidence)
        self._allowlist = allowlist
        self._rotation = self._validate_rotation(rotation)
        self._single_line = bool(single_line)
        self._timeout = timeout
        # extract_text calls are serialized so each sent image matches its own answer
        self._lock = threading.Lock()

        infra: dict[str, Any] | None = load_brick_compose_file(self.__class__)
        if infra is None or "services" not in infra:
            raise RuntimeError("Infrastructure configuration could not be loaded.")
        services: dict[str, Any] = infra["services"]
        service = next(iter(services), "")  # Only one service is expected

        self._host = resolve_address(service)
        if not self._host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")

        self._ws_send_url = f"ws://{self._host}:{_WS_SEND_PORT}"
        self._ws_recv_url = f"ws://{self._host}:{_WS_RECV_PORT}"
        logger.info(f"[{self.__class__.__name__}] Host: {self._host}")

    def extract_text(
        self,
        image: np.ndarray | bytes | str | Path,
        rotation: Iterable[int] | int | None = None,
        single_line: bool | None = None,
    ) -> OcrResult:
        """Extract the text visible in an image.

        Blocks until the model runner answers. Concurrent calls are serialized,
        so each image is matched with its own result. Large images are downscaled
        to 2048 px on the longest side before being sent; positions in the result
        are mapped back to the coordinates of `image`.

        Args:
            image (np.ndarray | bytes | str | Path): The image to read: a numpy
                array in BGR channel order (as returned by `Camera.capture()`),
                the raw bytes of an encoded image file (e.g. JPEG or PNG), or a
                path to an image file.
            rotation (Iterable[int] | int): Override the constructor's `rotation`
                for this call only, e.g. `[90, 270]` for an image whose text runs
                vertically. None (default) uses the constructor value; pass `[]`
                to read upright only for this call.
            single_line (bool): Override the constructor's `single_line` for this
                call only: True joins every recognized piece of text with single
                spaces instead of newlines. None (default) uses the constructor
                value.

        Returns:
            OcrResult: The extracted text, with one `TextDetection` per piece of
                text found. `result.text` is empty when no text was recognized.

        Raises:
            TypeError: If `image` is not one of the supported types.
            ValueError: If `image` could not be decoded, or `rotation` contains
                an angle other than 90, 180 or 270.
            FileNotFoundError: If `image` is a path that does not exist.
            OcrError: If the model runner cannot be reached or does not answer
                within the configured timeout.
        """
        rotation = self._rotation if rotation is None else self._validate_rotation(rotation)
        single_line = self._single_line if single_line is None else bool(single_line)

        encoded, scale = self._encode_image(image)
        payload = json.dumps({"frame": encoded})
        # The runner keeps these settings across calls (and clients), so they are
        # restated on every request to make each call self-contained.
        config = json.dumps({"config": {"allowlist": self._allowlist or "", "rotation": rotation}})
        with self._lock:
            metadata = self._request(payload, config)
        return self._parse_metadata(metadata, self._confidence, scale, single_line)

    @staticmethod
    def _validate_min_confidence(value: object) -> float:
        """Normalize a minimum-confidence value, rejecting anything outside [0.0, 1.0]."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"confidence must be a number in [0.0, 1.0], got {value!r}")
        return float(value)

    @staticmethod
    def _validate_rotation(value: object) -> list[int]:
        """Normalize a rotation value into a sorted list of distinct angles among 90, 180, 270."""
        if value is None:
            return []
        items: list[object]
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            items = [value]
        else:
            items = list(cast(Iterable[object], value))
        angles: set[int] = set()
        for item in items:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or int(item) != item:
                raise ValueError(f"rotation must be angles in degrees among {_VALID_ROTATIONS}, got {item!r}")
            angle = int(item) % 360
            if angle == 0:
                continue  # upright is always read
            if angle not in _VALID_ROTATIONS:
                raise ValueError(f"rotation must be angles in degrees among {_VALID_ROTATIONS}, got {item!r}")
            angles.add(angle)
        return sorted(angles)

    @staticmethod
    def _encode_image(image: object) -> tuple[str, tuple[float, float]]:
        """Turn any supported image input into a base64-encoded image for the runner.

        Images whose longest side exceeds `_MAX_IMAGE_SIDE` are downscaled and JPEG
        encoded; the encoding quality (then the size) is lowered until the payload fits
        the runner's websocket limit. Encoded inputs that already fit are sent as they
        are, without recompression.

        Returns:
            tuple[str, tuple[float, float]]: The base64 payload and the (x, y) scale
                applied to the image, i.e. sent size / original size.
        """
        raw: bytes | None = None
        frame: np.ndarray | None
        # `image` is whatever the caller passed: the isinstance chain is the contract,
        # and the casts only restore the parameters narrowing an untyped value drops.
        if isinstance(image, np.ndarray):
            frame = cast(np.ndarray, image)
        elif isinstance(image, (bytes, bytearray, memoryview)):
            raw = bytes(cast(bytes | bytearray | memoryview, image))
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            raw = path.read_bytes()
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        else:
            raise TypeError(f"Unsupported image type {type(image).__name__}: pass a numpy array, encoded image bytes, or a file path.")

        if frame is None or frame.ndim not in (2, 3) or frame.size == 0:
            raise ValueError("The image could not be decoded: pass a valid JPEG/PNG or a numpy image array.")

        height, width = frame.shape[:2]
        if raw is not None and max(height, width) <= _MAX_IMAGE_SIDE and OCR._payload_size(raw) <= _MAX_PAYLOAD_BYTES:
            return base64.b64encode(raw).decode("utf-8"), (1.0, 1.0)

        scale = min(1.0, _MAX_IMAGE_SIDE / max(height, width))
        while True:
            resized, scales = OCR._downscale(frame, scale)
            for quality in _JPEG_QUALITIES:
                jpeg = compress_to_jpeg(resized, quality)
                if jpeg is None:
                    raise ValueError("The image could not be encoded to JPEG.")
                data = jpeg.tobytes()
                if OCR._payload_size(data) <= _MAX_PAYLOAD_BYTES:
                    return base64.b64encode(data).decode("utf-8"), scales
            # Extremely noisy image: shrink further until it fits.
            scale *= 0.8

    @staticmethod
    def _downscale(frame: np.ndarray, scale: float) -> tuple[np.ndarray, tuple[float, float]]:
        """Resize `frame` by `scale` (<= 1), returning the frame and the exact per-axis scales applied."""
        height, width = frame.shape[:2]
        if scale >= 1.0:
            return frame, (1.0, 1.0)
        new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, (new_width / width, new_height / height)

    @staticmethod
    def _payload_size(data: bytes) -> int:
        """Bytes the encoded image occupies on the wire (base64, without the JSON framing)."""
        return 4 * ((len(data) + 2) // 3)

    def _request(self, payload: str, config: str) -> dict[str, Any]:
        """Send one frame to the model runner and return the metadata it answers with.

        The configuration is sent before the frame on the same socket, so the runner
        applies it to this image. The result socket is connected before the frame is
        sent, so the runner's broadcast cannot be missed. Connection errors are
        retried until the deadline: the runner container may still be starting up.
        """
        deadline = time.monotonic() + self._timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with connect(self._ws_recv_url) as recv_ws, connect(self._ws_send_url) as send_ws:
                    send_ws.send(config)
                    send_ws.send(payload)
                    try:
                        message = recv_ws.recv(timeout=max(deadline - time.monotonic(), 0.1))
                    except TimeoutError as e:
                        raise OcrError(
                            f"The OCR model runner accepted the image but did not answer within {self._timeout:.0f}s.",
                            hint="Check the OCR container logs; if the board is under heavy load, retry with a larger `timeout`.",
                        ) from e
                    data = cast(dict[str, Any], json.loads(message))
                    metadata = data.get("metadata")
                    return cast(dict[str, Any], metadata) if isinstance(metadata, dict) else {}
            except (OSError, ConnectionClosed) as e:
                last_error = e
                logger.debug(f"OCR model runner not reachable yet ({e}); retrying...")
                time.sleep(_RETRY_INTERVAL_SEC)
        raise OcrError(
            f"Could not reach the OCR model runner at '{self._host}' within {self._timeout:.0f}s.",
            hint="Check that the OCR container is up and healthy, then try again.",
        ) from last_error

    @staticmethod
    def _parse_metadata(
        metadata: dict[str, Any],
        min_confidence: float = 0.0,
        scale: tuple[float, float] = (1.0, 1.0),
        single_line: bool = False,
    ) -> OcrResult:
        """Build an OcrResult out of the model runner's metadata payload.

        Detections below `min_confidence` are dropped, and the result text is
        rebuilt from the kept detections (they arrive in reading order), joined by
        newlines or, when `single_line` is set, by single spaces. Positions are
        divided by `scale` (the x, y downscale applied before sending) so they
        refer to the original image.
        """
        scale_x, scale_y = scale

        def to_original(x: float, y: float) -> tuple[int, int]:
            return round(float(x) / scale_x), round(float(y) / scale_y)

        detections: list[TextDetection] = []
        for det in metadata.get("detections", []):
            try:
                x1, y1, x2, y2 = det["bounding_box_xyxy"]
                detection = TextDetection(
                    text=str(det["text"]),
                    confidence=float(det["confidence"]),
                    bounding_box_xyxy=(*to_original(x1, y1), *to_original(x2, y2)),
                    polygon=[to_original(x, y) for x, y in det["polygon"]],
                )
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Skipping malformed detection {det!r}: {e}")
                continue
            if detection.confidence >= min_confidence:
                detections.append(detection)

        separator = " " if single_line else "\n"
        return OcrResult(text=separator.join(d.text for d in detections), detections=detections)


__all__ = [
    "OCR",
    "OcrError",
    "OcrResult",
    "TextDetection",
]
