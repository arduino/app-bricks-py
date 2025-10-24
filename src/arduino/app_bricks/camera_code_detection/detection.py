# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from dataclasses import dataclass
import threading
from typing import Callable

import cv2
from pyzbar.pyzbar import decode, ZBarSymbol, PyZbarError
import numpy as np
from PIL.Image import Image

from arduino.app_peripherals.camera import Camera
from arduino.app_utils import brick, Logger

logger = Logger("CameraCodeDetection")

barcodes_only = [s for s in ZBarSymbol if s not in (ZBarSymbol.QRCODE, ZBarSymbol.SQCODE)]
qrcodes_only = [ZBarSymbol.QRCODE, ZBarSymbol.SQCODE]


@dataclass
class Detection:
    """This class represents a single QR code or barcode detection result from a video frame.

    This data structure holds the decoded content, the type of code, and its location
    in the image as determined by the detection algorithm.

    Attributes:
        content (str): The decoded string extracted from the QR code or barcode.
        type (str): The type of code detected, typically "QRCODE" or "BARCODE".
        coords (np.ndarray): A NumPy array of shape (4, 2) representing the four corner
            points (x, y) of the detected code region in the image.
    """

    content: str
    type: str
    coords: np.ndarray


@brick
class CameraCodeDetection:
    """Scans a camera video feed for QR codes and/or barcodes.

    Args:
        camera (USBCamera): The USB camera instance. If None, a default camera will be initialized.
        detect_qr (bool): Whether to detect QR codes. Defaults to True.
        detect_barcode (bool): Whether to detect barcodes. Defaults to True.

    Raises:
        ValueError: If both detect_qr and detect_barcode are False.
        RuntimeError: If there is an error during initialization.
    """

    def __init__(
        self,
        camera: Camera = None,
        detect_qr: bool = True,
        detect_barcode: bool = True,
    ):
        """Initialize the CameraCodeDetection brick."""
        if detect_qr is False and detect_barcode is False:
            raise ValueError("At least one of 'detect_qr' or 'detect_barcode' must be True.")

        self._detect_qr = detect_qr
        self._detect_barcode = detect_barcode

        # These callbacks do not require locks as long as we're running on CPython
        self._on_frame_cb = None
        self._on_error_cb = None

        self._on_detect_cb = None
        self._on_detect_cb_expects_list = False
        self._on_detect_cb_lock = threading.Lock()  # Synchronizes access to both callback and bool flag

        self.already_seen_codes = set()

        self._camera = camera if camera else Camera()

    def start(self):
        """Start the detector and begin scanning for codes."""
        self._camera.start()

    def stop(self):
        """Stop the detector and release resources."""
        self._camera.stop()

    def on_detect(self, callback: Callable[[Image, list[Detection]], None] | Callable[[Image, Detection], None] | None):
        """Registers or removes a callback to be triggered on code detection.

        When a QR code or barcode is detected in the camera feed, the provided callback function will be invoked.
        The callback function should accept the Image frame and a list[Detection] or Detection objects. If the former
        is used, it will receive all detections at once. If the latter is used, it will be called once for each
        detection. If None is provided, the callback will be removed.

        Args:
            callback (Callable[[Image, list[Detection]], None]): A callback that will be called every time a detection
                                                                 is made with all the detections.
            callback (Callable[[Image, Detection], None]): A callback that will be called every time a detection is
                                                           made with a single detection.
            callback (None): To unregister the current callback, if any.

        Example:
            def on_code_detected(frame: Image, detection: Detection):
                print(f"Detected {detection.type} with content: {detection.content}")
                # Here you can add your code to process the detected code,
                # e.g., draw a bounding box, save it to a database or log it.

            detector.on_detect(on_code_detected)
        """
        with self._on_detect_cb_lock:
            self._on_detect_cb = callback
            self._on_detect_cb_expects_list = False
            if callback is not None:
                import inspect

                sig = inspect.signature(callback)
                params = list(sig.parameters.values())
                if len(params) >= 2 and params[1].annotation == list[Detection]:
                    self._on_detect_cb_expects_list = True

    def on_frame(self, callback: Callable[[Image], None] | None):
        """Registers a callback function to be called when a new camera frame is captured.

        The callback function should accept the Image frame.
        If None is provided, the callback is removed.

        Args:
            callback (Callable[[Image], None]): A callback that will be called with each captured frame.
            callback (None): Signals to remove the current callback, if any.
        """
        self._on_frame_cb = callback

    def on_error(self, callback: Callable[[Exception], None] | None):
        """Registers a callback function to be called when an error occurs in the detector.

        The callback function should accept the exception as an argument.
        If None is provided, the callback is removed.

        Args:
            callback (Callable): A callback that will be called with the exception raised in the detector.
            callback (None): Signals to remove the current callback, if any.
        """
        self._on_error_cb = callback

    def loop(self):
        """Main loop to capture frames and detect codes."""
        try:
            frame = self._camera.capture()
            if frame is None:
                return
        except Exception as e:
            self._on_error(e)
            return

        # Use grayscale for barcode/QR code detection
        gs_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        self._on_frame(frame)

        detections = self._scan_frame(gs_frame)
        self._on_detect(frame, detections)

    def _on_frame(self, frame: Image):
        if self._on_frame_cb:
            try:
                self._on_frame_cb(frame)
            except Exception as e:
                logger.error(f"Failed to run on_frame callback: {e}")
                self._on_error(e)

    def _scan_frame(self, frame: cv2.typing.MatLike) -> list[Detection]:
        """Scan the frame for a single barcode or QR code."""
        detections = []

        try:
            symbols = []
            if self._detect_qr:
                symbols += qrcodes_only
            if self._detect_barcode:
                symbols += barcodes_only

            codes = decode(frame, symbols=None if len(symbols) == 0 else symbols)
            for d in codes:
                content = d.data.decode("utf-8")
                if content is not None and content not in self.already_seen_codes:
                    self.already_seen_codes.add(content)
                    x, y, w, h = d.rect
                    points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
                    detections.append(Detection(content, d.type, points))
        except PyZbarError as p:
            logger.error(f"Failed to detect or decode: {p}")
            self._on_error(p)
        except Exception as e:
            logger.error(f"Unknown error while detecting: {e}")
            self._on_error(e)

        return detections

    def _on_detect(self, frame: Image, detections: list[Detection]):
        with self._on_detect_cb_lock:
            if self._on_detect_cb and len(detections) > 0:
                try:
                    if self._on_detect_cb_expects_list:
                        self._on_detect_cb(frame, detections)
                    else:
                        for detection in detections:
                            self._on_detect_cb(frame, detection)
                except Exception as e:
                    logger.error(f"Failed to run on_detect callback: {e}")
                    self._on_error(e)

    def _on_error(self, exception: Exception):
        if self._on_error_cb:
            try:
                self._on_error_cb(exception)
            except Exception as e:
                logger.exception(f"Failed to run on_error callback: {e}")
