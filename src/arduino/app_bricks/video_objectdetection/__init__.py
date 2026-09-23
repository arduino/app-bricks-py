# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import time

import numpy as np

from arduino.app_internal.edge_impulse import AllDetectionsCallback, BoxStabilizer, DetectionCallback, LabelColors, VideoInference, draw_detections
from arduino.app_internal.ei_inference import Result
from arduino.app_peripherals.camera import BaseCamera
from arduino.app_utils import Logger, brick

logger = Logger("VideoObjectDetection")

MODEL_VARIABLE = "EI_V_OBJ_DETECTION_MODEL"
STREAM_PORT = VideoInference.STREAM_PORT

__all__ = ["AllDetectionsCallback", "DetectionCallback", "VideoObjectDetection"]


@brick
class VideoObjectDetection(VideoInference):
    """Module for object detection on a **live video stream** using a specified machine learning model.

    This brick:
      - Streams the camera frames to the Edge Impulse inference service over its Unix socket.
      - Receives the bounding boxes reaching the confidence, in the coordinates of the camera frame.
      - Debounces repeated triggers of the same label.
      - Invokes per-label callbacks and/or a catch-all callback.
      - Streams the video with the bounding boxes on port 4912, for browsers and embedded iframes: every camera
        frame is drawn with the boxes of the latest inference, steadied across results.
    """

    MODEL_VARIABLE = MODEL_VARIABLE

    def __init__(
        self,
        camera: BaseCamera | None = None,
        confidence: float = 0.3,
        debounce_sec: float = 0.0,
        camera_preview: bool = False,
        stream_port: int | None = STREAM_PORT,
    ) -> None:
        """Initialize the VideoObjectDetection class.

        Args:
            camera (BaseCamera): The camera instance to use for capturing video. If None, a default camera will be initialized.
            confidence (float): Confidence level for detection. Default is 0.3 (30%).
            debounce_sec (float): Minimum seconds between repeated detections of the same object. Default is 0 seconds.
            camera_preview (bool): Receive current camera frame on callback invocation.
                Frame is a raw jpeg-encoded image without bounding boxes applied on it. Default is False.
            stream_port (int | None): Port of the MJPEG stream of the video with the bounding boxes, the one
                external viewers embed. Default is 4912, None disables the stream.

        Raises:
            RuntimeError: If no model is configured.
        """
        super().__init__(camera=camera, confidence=confidence, debounce_sec=debounce_sec, camera_preview=camera_preview, stream_port=stream_port)
        self._colors = LabelColors()
        self._boxes = BoxStabilizer()  # what the video shows: the boxes of the results, steadied across them

    def on_detect(self, object: str, callback: DetectionCallback) -> None:  # noqa: A002
        """Register a callback invoked when a **specific label** is detected.

        Args:
            object (str): The label of the object to check for in the classification results.
            callback (DetectionCallback): A plain function taking either no parameters, or one
                parameter receiving the detection details dict
                `{"confidence": float, "bounding_box_xyxy": (x1, y1, x2, y2)}`. A function that also
                declares a `frame` parameter receives the current camera frame as raw JPEG bytes
                (or None when no preview frame is available, see `camera_preview`).

        Raises:
            TypeError: If `callback` is not a function.
        """
        super().on_detect(object, callback)

    def on_detect_all(self, callback: AllDetectionsCallback) -> None:
        """Register a callback invoked for **every detection event**.

        This is useful to receive a consolidated dictionary of detections for each frame.

        Args:
            callback (AllDetectionsCallback): A plain function taking one dict argument mapping
                each detected label to the list of its detections, with the shape
                `{label: [{"confidence": float, "bounding_box_xyxy": (x1, y1, x2, y2)}, ...], ...}`.
                A function that also declares a `frame` parameter receives the current camera
                frame as raw JPEG bytes (or None when no preview frame is available, see
                `camera_preview`).

        Raises:
            TypeError: If `callback` is not a function.
        """
        super().on_detect_all(callback)

    def start(self) -> None:
        """Start the video object detection process."""
        super().start()

    def stop(self) -> None:
        """Stop the video object detection process and release resources, the service releases the model."""
        super().stop()

    def _process_result(self, result: Result) -> None:
        """Turn the boxes of one frame into detections, feed the video boxes and invoke the handlers."""
        if not result.ok:
            logger.warning(f"Inference failed ({result.error_code}): {result.error}")
            return

        detections: dict[str, list[dict]] = {}
        for box in result.boxes:
            xyxy_bbox = (round(box.x), round(box.y), round(box.x + box.w), round(box.y + box.h))
            detections.setdefault(box.label, []).append({"confidence": box.score, "bounding_box_xyxy": xyxy_bbox})
        self._boxes.update(result.boxes, (time.monotonic_ns() - result.ts_ns) / 1e9)
        if not detections:
            return

        preview = self._encode_preview(result.frame)
        for label, label_detections in detections.items():
            for detection_details in label_detections:
                self._execute_handler(key=label, payload=detection_details, frame=preview)
        self._execute_handler(key=self.ALL_HANDLERS_KEY, payload=detections, frame=preview)

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        """A copy of the frame with the steadied boxes and their labels drawn on it."""
        return draw_detections(frame, self._boxes.visible(), self._colors)
