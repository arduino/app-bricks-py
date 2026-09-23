# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import inspect
import math
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from arduino.app_bricks.video_objectdetection import STREAM_PORT, AllDetectionsCallback, DetectionCallback, VideoObjectDetection
from arduino.app_internal.edge_impulse import draw_caption, draw_crossing_line
from arduino.app_internal.ei_inference import InferenceClient, Result, ServerError
from arduino.app_peripherals.camera import BaseCamera
from arduino.app_utils import AppError, Logger, LRUDict, brick

logger = Logger("VideoObjectTracking")

MODEL_VARIABLE = "EI_V_OBJ_TRACKING_MODEL"
STARTUP_TIMEOUT = 10.0  # seconds the constructor waits for the service to open the model, as long as the service gives a .eim to start
CENTROID_MODEL_TYPE = "constrained_object_detection"  # FOMO reports centroids, matched by distance instead of overlap
TRACKING_BLOCK = "object_tracking"  # the threshold block holding the tracker knobs

type LineCrossingCallback = Callable[[dict[str, Any]], None]
"""Callback accepted by `on_line_crossing`: one dict argument, `{"label": str, "object_id": int, "direction": str}`."""


class VideoObjectTrackingError(AppError):
    """Base class for video object tracking errors."""


@brick
class VideoObjectTracking(VideoObjectDetection):
    """Module for object tracking on a **live video stream** using a specified machine learning model.

    This brick:
      - Streams the camera frames to the Edge Impulse inference service, as VideoObjectDetection does.
      - Receives the objects the model tracks at the confidence of the brick, each with an identity that stays
        the same while it is in view.
      - Counts the distinct objects per label and the ones crossing a line, and follows their direction.
      - Invokes per-label callbacks and/or a catch-all callback with the tracked objects.
      - Streams the video with the tracked boxes, labelled with their identities, on port 4912.
    """

    MODEL_VARIABLE = MODEL_VARIABLE

    def __init__(
        self,
        camera: BaseCamera | None = None,
        confidence: float = 0.4,
        keep_grace: int = 3,
        min_detections: int = 3,
        iou_threshold: float = 0.1,
        euclidean_distance_threshold: int = 50,
        debounce_sec: float = 0.0,
        labels_to_track: list[str] | None = None,
        min_movement_threshold: int = 10,
        stream_port: int | None = STREAM_PORT,
    ) -> None:
        """Initialize the VideoObjectTracking class.

        Args:
            camera (BaseCamera): The camera instance to use for capturing video. If None, a default camera will be initialized.
            confidence (float): Confidence level for detection. Default is 0.4 (40%).
            keep_grace (int): Number of frames to keep an object if it disappears. Default is 3.
            min_detections (int): How many times an object must be detected before the tracker reports it as a
                track of its own. Higher values delay the first report but discard more spurious detections. Default is 3.
            iou_threshold (float): Intersection over Union threshold for tracking. Default is 0.1. This is used in case of object detection models.
            euclidean_distance_threshold (int): Maximum distance in pixels. Default is 50 (px). This is used in case of centroids models, like FOMO.
            debounce_sec (float): Minimum seconds between repeated detections of the same object. Default is 0 seconds.
            labels_to_track (list[str], optional): List of labels to track. If None, all labels are tracked.
            min_movement_threshold(int): Minimum distance in pixels to consider a movement significant for
                direction tracking. Default is 10.
            stream_port (int | None): Port of the MJPEG stream of the video with the tracked boxes, the one
                external viewers embed. Default is 4912, None disables the stream.

        Raises:
            RuntimeError: If no model is configured.
            VideoObjectTrackingError: If the model has no object tracking block.
        """
        super().__init__(camera=camera, confidence=confidence, debounce_sec=debounce_sec, stream_port=stream_port)
        self._labels_to_track = labels_to_track
        # The knobs of the tracking block as the model names them: keep_grace is max_age, min_detections is min_hits,
        # iou_threshold matches the boxes of a detection model and threshold the centroids of a FOMO one
        self._tracker: dict[str, float] = {
            "max_age": keep_grace,
            "min_hits": min_detections,
            "iou_threshold": iou_threshold,
            "threshold": euclidean_distance_threshold,
        }

        self._counter_lock = threading.RLock()
        self._object_counters: Counter[str] = Counter()  # distinct objects seen, per label
        self._recent_objects: LRUDict[int, tuple[int, int]] = LRUDict(maxsize=150)  # last seen position (x, y) of the recent object ids
        self._line_coordinates: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2 of the crossing line
        self._crossing_line_object: dict[str, Counter[str]] = {}  # crossings of the line, per label and direction
        self._line_crossing_handler: LineCrossingCallback | None = None
        self._object_directions: dict[int, list[str]] = {}  # direction history, per object id
        self._min_movement_threshold = min_movement_threshold

        self._require_object_tracking()

    def _require_object_tracking(self) -> None:
        """Open the model while the app is starting, to refuse one that can never report a tracked object.

        The connection is kept for the tracking loop. When the service is not reachable or refuses the model,
        the check is skipped and the loop keeps trying once started.

        Raises:
            VideoObjectTrackingError: If the model has no object tracking block.
        """
        try:
            client = self._open(timeout=STARTUP_TIMEOUT)
        except ServerError as e:
            logger.warning(f"The inference service refused model '{self._model}' ({e.code}): {e}. Skipping the object tracking check.")
            return
        except OSError as e:  # unreachable, or not ready within the timeout
            logger.warning(f"Could not ask the inference service about model '{self._model}' ({e}): skipping the object tracking check.")
            return
        if not client.object_tracking:
            self._close()
            raise VideoObjectTrackingError(
                "This model has no object tracking block, so it can never report a tracked object.",
                hint="Enable object tracking in the Edge Impulse project and export the model again, or pick a model that already has the block.",
            )

    def _configure(self, client: InferenceClient) -> None:
        """Set the tracker knobs on the model every time it is opened; the identifiers seen so far are forgotten,
        the tracker numbers its tracks from zero on each run."""
        self._forget_tracks()
        if client.object_tracking:
            self._apply_tracker(client)

    def _connect(self, running: threading.Event) -> InferenceClient | None:
        client = super()._connect(running)
        if client is not None and not client.object_tracking:
            logger.error(
                "This model has no object tracking block, so no object will ever be reported. "
                "Enable object tracking in the Edge Impulse project and export the model again."
            )
        return client

    def _apply_tracker(self, client: InferenceClient) -> None:
        """Set the tracker knobs on the model, those its tracking block exposes."""
        try:
            tracking = client.threshold_block(TRACKING_BLOCK)
            if tracking is not None:
                knobs = ("max_age", "min_hits", self._matching_knob(client))
                values = {knob: self._tracker[knob] for knob in knobs if knob in tracking}
                if values:
                    client.configure(tracking["id"], **values)
        except (ServerError, TimeoutError, ConnectionError) as e:
            logger.error(f"Failed to configure the tracker: {e}")

    @staticmethod
    def _matching_knob(client: InferenceClient) -> str:
        """The knob matching the objects between frames: a distance for centroid models, an overlap for the others."""
        return "threshold" if client.info.get("model_type") == CENTROID_MODEL_TYPE else "iou_threshold"

    def _is_label_enabled(self, label: str) -> bool:
        """Check if a label is enabled for tracking.

        Args:
            label (str): The label to check.
        Returns:
            bool: True if the label is enabled for tracking, False otherwise.
        """
        if self._labels_to_track is None:
            return True
        return label in self._labels_to_track

    def _record_object(self, detected_object_label: str, object_id: int, x: int, y: int) -> None:
        """
        Record that an object with a specific label and ID has been seen.

        Args:
            detected_object_label (str): The label of the detected object.
            object_id (int): The unique ID of the detected object.
            x (int): The x-coordinate of the object reference point.
            y (int): The y-coordinate of the object reference point.
        """

        with self._counter_lock:
            if object_id not in self._recent_objects:
                self._object_counters[detected_object_label] += 1

        if object_id in self._recent_objects:
            last_x, last_y = self._recent_objects[object_id]
            if last_x == x and last_y == y:
                # No movement detected; skip further processing
                return
            self._record_line_crossing(detected_object_label, object_id, x, y)
            self._record_object_direction(detected_object_label, object_id, x, y)
        # Update the last seen position
        self._recent_objects[object_id] = (x, y)

    def _record_line_crossing(self, detected_object_label: str, object_id: int, x: int, y: int) -> None:
        """Count a crossing of the line by the object, under the direction it moved in.

        Args:
            detected_object_label (str): The label of the detected object.
            object_id (int): The unique ID of the detected object.
            x (int): The x-coordinate of the object reference point.
            y (int): The y-coordinate of the object reference point.
        """
        with self._counter_lock:
            if self._line_coordinates is None or object_id not in self._recent_objects:
                return
            last_x, last_y = self._recent_objects[object_id]
            x1, y1, x2, y2 = self._line_coordinates
            dx, dy = x2 - x1, y2 - y1
            before = (last_x - x1) * dy - (last_y - y1) * dx
            after = (x - x1) * dy - (y - y1) * dx
            if before == 0 or (after != 0 and (after > 0) == (before > 0)):
                return
            direction = _crossing_direction(dx, dy, before)
            self._crossing_line_object.setdefault(detected_object_label, Counter())[direction] += 1
            handler = self._line_crossing_handler
        logger.debug(f"Object ID {object_id} crossed the line from ({last_x}, {last_y}) to ({x}, {y}) moving {direction}")
        if handler is not None:
            crossing = {"label": detected_object_label, "object_id": object_id, "direction": direction}
            try:
                self._executor.submit(handler, crossing)
            except RuntimeError:  # the executor was shut down before the task could be submitted
                pass

    def _record_object_direction(self, detected_object_label: str, object_id: int, x: int, y: int) -> None:
        """
        Record the movement direction of an object with a specific label and ID.

        Args:
            detected_object_label (str): The label of the detected object.
            object_id (int): The unique ID of the detected object.
            x (int): The x-coordinate of the object reference point.
            y (int): The y-coordinate of the object reference point.
        """
        with self._counter_lock:
            if object_id in self._recent_objects:
                last_x, last_y = self._recent_objects[object_id]
                direction = _get_direction(last_x, last_y, x, y, self._min_movement_threshold)
                if direction is None:
                    return
                # check if last direction is different from current direction to avoid duplicates
                if object_id in self._object_directions:
                    if len(self._object_directions[object_id]) > 0 and self._object_directions[object_id][-1] == direction:
                        return
                else:
                    # Initialize the list if no object ID entry exists
                    self._object_directions[object_id] = []
                self._object_directions[object_id].append(direction)
                logger.debug(f"Object ID {object_id} moved {direction} from ({last_x}, {last_y}) to ({x}, {y})")

    def _forget_tracks(self) -> None:
        """Forget the identifiers seen so far, keeping the counts: the tracker numbers tracks from zero on each run."""
        with self._counter_lock:
            self._recent_objects.clear()
            self._object_directions.clear()

    def get_unique_objects_count(self) -> dict[str, int]:
        """
        Get all identified object types and their counts since the last reset.
            This includes all distinguished objects sees, based on their unique IDs.

        Returns:
            dict: A dictionary with labels as keys and their respective counts as values.
        """
        with self._counter_lock:
            return dict(self._object_counters)

    def get_line_crossing_counts(self) -> dict[str, dict[str, int]]:
        """
        Get the crossings of the defined line since the last reset, per label and direction.

        Returns:
            dict: `{label: {direction: count}}`, where the direction is one of `up`, `down`, `left`, `right`,
                `up-left`, `up-right`, `down-left`, `down-right`, as seen on the screen; only the directions seen appear.
        """
        with self._counter_lock:
            return {label: dict(counts) for label, counts in self._crossing_line_object.items()}

    def get_objects_directions(self) -> dict[int, list[str]]:
        """
        Get the last known movement directions of tracked objects.

        Returns:
            dict: A dictionary with object IDs as keys and their respective movement directions as values.
        """
        with self._counter_lock:
            return dict(self._object_directions)

    def set_crossing_line_coordinates(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """
        Set the coordinates of the line for counting objects crossing it.

        Args:
            x1 (int): The x-coordinate of the first point of the line.
            y1 (int): The y-coordinate of the first point of the line.
            x2 (int): The x-coordinate of the second point of the line.
            y2 (int): The y-coordinate of the second point of the line.
        """
        with self._counter_lock:
            self._line_coordinates = (x1, y1, x2, y2)

    def set_horizontal_crossing_line(self, y: int) -> None:
        """
        Set a horizontal line for counting objects crossing it.

        Args:
            y (int): The y-coordinate of the horizontal line.
        """
        self.set_crossing_line_coordinates(0, y, 480, y)

    def set_vertical_crossing_line(self, x: int) -> None:
        """
        Set a vertical line for counting objects crossing it.

        Args:
            x (int): The x-coordinate of the vertical line.
        """
        self.set_crossing_line_coordinates(x, 0, x, 480)

    def reset_counters(self) -> None:
        """Reset the counts of tracked objects."""
        with self._counter_lock:
            self._object_counters.clear()
            self._recent_objects.clear()
            self._crossing_line_object.clear()

    def on_detect(self, object: str, callback: DetectionCallback) -> None:  # noqa: A002
        """Register a callback invoked when a **specific label** is tracked.

        Args:
            object (str): The label of the object to check for in the tracking results.
            callback (DetectionCallback): A plain function taking either no parameters, or one parameter receiving
                the tracking details dict `{"object_id": int, "confidence": float, "bounding_box_xyxy": (x1, y1, x2, y2)}`,
                once per object of that label in the frame.

        Raises:
            TypeError: If `callback` is not a function.
        """
        super().on_detect(object, callback)

    def on_line_crossing(self, callback: LineCrossingCallback) -> None:
        """Register a callback invoked **every time a tracked object crosses the line**.

        Args:
            callback (LineCrossingCallback): A plain function taking one dict argument,
                `{"label": str, "object_id": int, "direction": str}`, where `direction` is the screen direction of
                the crossing, as `get_line_crossing_counts()` names it.

        Raises:
            TypeError: If `callback` is not a function.
        """
        if not inspect.isfunction(callback):
            raise TypeError("Callback must be a callable function.")
        with self._counter_lock:
            self._line_crossing_handler = callback

    def on_detect_all(self, callback: AllDetectionsCallback) -> None:
        """Register a callback invoked for **every frame with tracked objects**.

        This is useful to receive a consolidated dictionary of the tracked objects for each frame.

        Args:
            callback (AllDetectionsCallback): A plain function taking one dict argument mapping each tracked
                label to the list of its objects, with the shape
                `{label: [{"object_id": int, "confidence": float, "bounding_box_xyxy": (x1, y1, x2, y2)}, ...], ...}`.

        Raises:
            TypeError: If `callback` is not a function.
        """
        super().on_detect_all(callback)

    def _process_result(self, result: Result) -> None:
        """Turn the tracks of one frame into detections with their ids, update the counters and invoke the handlers."""
        if not result.ok:
            logger.warning(f"Inference failed ({result.error_code}): {result.error}")
            return

        tracked = [(track, track.id) for track in result.tracks if track.id is not None and self._is_label_enabled(track.label)]
        detections: dict[str, list[dict[str, Any]]] = {}
        for track, object_id in tracked:
            x1, y1, x2, y2 = round(track.x), round(track.y), round(track.x + track.w), round(track.y + track.h)
            details = {"object_id": object_id, "confidence": track.score, "bounding_box_xyxy": (x1, y1, x2, y2)}
            detections.setdefault(track.label, []).append(details)
            self._record_object(track.label, object_id, (x1 + x2) // 2, (y1 + y2) // 2)
            self._execute_handler(key=track.label, payload=details)
        # The video shows every tracked object under its label and id
        self._boxes.update(
            [replace(track, label=f"{track.label} #{object_id}") for track, object_id in tracked], (time.monotonic_ns() - result.ts_ns) / 1e9
        )
        if detections:
            self._execute_handler(key=self.ALL_HANDLERS_KEY, payload=detections)

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        """A copy of the frame with the tracked boxes, the crossing line when one is set, and the counters per label."""
        annotated = super()._annotate(frame)
        with self._counter_lock:
            line = self._line_coordinates
        if line is not None:
            draw_crossing_line(annotated, line)
        return draw_caption(annotated, self._captions())

    def _captions(self) -> list[str]:
        """One line per label seen: the distinct objects and, with a line set, the crossings in each direction."""
        with self._counter_lock:
            line = self._line_coordinates
            seen = dict(self._object_counters)
            crossed = {label: dict(counts) for label, counts in self._crossing_line_object.items()}
        captions = []
        for label, count in sorted(seen.items()):
            parts = [f"{count} seen"]
            if line is not None:
                parts += [f"{n} {direction}" for direction, n in sorted(crossed.get(label, {}).items())]
            captions.append(f"{label}: " + ", ".join(parts))
        return captions

    def override_keep_grace(self, keep_grace: int) -> None:
        """Override keep grace for object tracking model.
            Keep Grace: how many frames an object is kept if it disappears.

        Args:
            keep_grace (int): The new value for the keep grace.

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the tracking block of the model has no such knob or the service refuses the value.
        """
        self._override("max_age", keep_grace)

    def override_min_detections(self, min_detections: int) -> None:
        """Override the number of detections a track needs for the object tracking model to report it.

        Args:
            min_detections (int): The new value for the minimum number of detections.

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the tracking block of the model has no such knob or the service refuses the value.
        """
        self._override("min_hits", min_detections)

    def override_iou_threshold(self, iou_threshold: float) -> None:
        """Override IoU threshold for object tracking model.
            This is valid for bounding box object detection based models, like Yolo.
            IOU Threshold: Intersection over Union threshold for tracking.

        Args:
            iou_threshold (float): The new value for the IoU threshold.

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the tracking block of the model has no such knob or the service refuses the value.
        """
        if self._model_type() == CENTROID_MODEL_TYPE:
            logger.debug("This model reports centroids. Use 'override_euclidean_distance_threshold' instead.")
            return
        self._override("iou_threshold", iou_threshold)

    def override_euclidean_distance_threshold(self, euclidean_distance_threshold: float) -> None:
        """Override euclidean distance threshold for object tracking model.
            This is valid for centroids based detection models, like FOMO.
            Euclidean Distance Threshold: Maximum distance in pixels to consider two detections as the same object.

        Args:
            euclidean_distance_threshold (float): The new value for the euclidean distance threshold, in pixels.

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the tracking block of the model has no such knob or the service refuses the value.
        """
        if self._model_type() not in (None, CENTROID_MODEL_TYPE):
            logger.debug("This model reports bounding boxes. Use 'override_iou_threshold' instead.")
            return
        self._override("threshold", euclidean_distance_threshold)

    def _override(self, knob: str, value: object) -> None:
        """Set a knob of the tracking block on the model, and keep it for the next connections."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("Invalid types for value.")
        self._push(knob, value)
        self._tracker[knob] = value

    def _push(self, knob: str, value: float) -> None:
        """Send a knob of the tracking block to the model when connected, otherwise it is set when the model is opened.

        Raises:
            RuntimeError: If the model has no tracking block, the block has no such knob or the service refuses the value.
        """
        client = self._connected()
        if client is None:
            logger.debug(f"Not connected: {knob}={value} will be set when the model is opened")
            return
        block = client.threshold_block(TRACKING_BLOCK)
        if block is None:
            available = ", ".join(str(b.get("type")) for b in client.thresholds) or "nothing"
            raise RuntimeError(f"This model has no {TRACKING_BLOCK} block, it only exposes: {available}.")
        if knob not in block:
            knobs = ", ".join(key for key in block if key not in ("id", "type", "min_score"))
            raise RuntimeError(f"The {TRACKING_BLOCK} block of this model exposes {knobs}, not '{knob}'.")
        try:
            client.configure(block["id"], **{knob: value})
        except (ServerError, TimeoutError, ConnectionError) as e:
            raise RuntimeError(f"The inference service refused {knob}={value}: {e}") from e
        logger.debug(f"Set {knob}={value} on the tracking block of the model")

    def _model_type(self) -> str | None:
        """The type of the model as the service reports it, None while not connected."""
        client = self._connected()
        return client.info.get("model_type") if client is not None else None


COMPASS = ("right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right")  # clockwise on the screen, y down


def _compass(dx: float, dy: float) -> str:
    """The screen direction of the vector (dx, dy), y growing downwards, rounded to the nearest of the eight.

    Each name covers 45 degrees: a vector within 22.5 degrees of an axis is `right`, `down`, `left` or `up`.
    """
    angle = math.degrees(math.atan2(dy, dx))
    return COMPASS[math.floor(angle / 45 + 0.5) % 8]


def _crossing_direction(dx: int, dy: int, before: int) -> str:
    """The screen direction of a crossing of the line of direction (dx, dy): perpendicular to it, towards the side
    reached, given `before`, the signed side of the starting point, `(x - x1) * dy - (y - y1) * dx`.
    """
    towards = -1 if before > 0 else 1
    return _compass(towards * dy, -towards * dx)


def _get_direction(last_x: int, last_y: int, x: int, y: int, min_movement_threshold: int = 10) -> str | None:
    """
    Determine the movement direction based on the change in coordinates.
    Note: The directions are mirrored both horizontally and vertically.

    Args:
        last_x (int): The previous x-coordinate.
        last_y (int): The previous y-coordinate.
        x (int): The current x-coordinate.
        y (int): The current y-coordinate.
        min_movement_threshold (int): Minimum distance in pixels to consider a movement significant. Default is 10.

    Returns:
        str: The movement direction ('up', 'down', 'left', 'right', 'up-left', 'up-right', 'down-left', 'down-right').
    """
    logger.debug(f"Calculating direction change from ({last_x}, {last_y}) to ({x}, {y})")
    dx = x - last_x
    dy = y - last_y

    # Check for minimal movement to avoid noise detection as direction change
    if abs(dx) < min_movement_threshold and abs(dy) < min_movement_threshold:
        return None

    direction = None
    if abs(dx) == abs(dy):
        logger.debug("Diagonal movement detected.")
        if dx > 0 and dy > 0:
            direction = "down-left"  # up-right becomes down-left
        elif dx > 0 > dy:
            direction = "up-left"  # down-right becomes up-left
        elif dx < 0 < dy:
            direction = "down-right"  # up-left becomes down-right ok
        elif dx < 0 and dy < 0:
            direction = "up-right"  # down-left becomes up-right ok
    elif abs(dx) > abs(dy):
        logger.debug("Horizontal movement detected.")
        if dx > 0:
            direction = "left"  # right becomes left
        else:
            direction = "right"  # left becomes right
    else:
        logger.debug("Vertical movement detected.")
        if dy > 0:
            direction = "down"  # up becomes down
        else:
            direction = "up"  # down becomes up

    return direction
