# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import brick, Logger, LRUDict
from arduino.app_bricks.video_objectdetection import AllDetectionsCallback, DetectionCallback, VideoObjectDetection
from arduino.app_internal.core import EdgeImpulseRunnerFacade
from arduino.app_peripherals.camera import BaseCamera
from websockets.sync.client import connect
from websockets.sync.connection import Connection
import json
from collections import Counter
import threading

logger = Logger("VideoObjectTracking")


@brick
class VideoObjectTracking(VideoObjectDetection):
    """Module for object tracking on a **live video stream** using a specified machine learning model.

    This brick:
      - Connects to a model runner over WebSocket.
      - Parses incoming classification messages with bounding boxes.
      - Filters detections by a configurable confidence threshold.
      - Debounces repeated triggers of the same label.
      - Invokes per-label callbacks and/or a catch-all callback.
    """

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
    ) -> None:
        """Initialize the VideoObjectTracking class.

        Args:
            camera (BaseCamera): The camera instance to use for capturing video. If None, a default camera will be initialized.
            confidence (float): Confidence level for detection. Default is 0.4 (40%).
            debounce_sec (float): Minimum seconds between repeated detections of the same object. Default is 0 seconds.
            keep_grace (int): Number of frames to keep an object if it disappears. Default is 3.
            min_detections (int): How many times an object must be detected before the tracker reports it as a
                track of its own. Higher values delay the first report but discard more spurious detections. Default is 3.
            iou_threshold (float): Intersection over Union threshold for tracking. Default is 0.1. This is used in case of object detection models.
            euclidean_distance_threshold (int): Maximum distance in pixels. Default is 50 (px). This is used in case of centroids models, like FOMO.
            labels_to_track (list[str], optional): List of labels to track. If None, all labels are tracked.
            min_movement_threshold(int): Minimum distance in pixels to consider a movement significant for
                direction tracking. Default is 10.

        Raises:
            RuntimeError: If the host address could not be resolved.
        """
        super().__init__(camera=camera, confidence=confidence, debounce_sec=debounce_sec)
        self._labels_to_track = labels_to_track
        self._min_detections = min_detections
        self._keep_grace = keep_grace
        self._iou_threshold = iou_threshold
        self._euclidean_distance_threshold = euclidean_distance_threshold

        # Counter for tracked objects
        self._counter_lock = threading.RLock()
        self._object_counters = Counter()
        # Map of recent object IDs to their last seen positions (x, y)
        self._recent_objects = LRUDict(maxsize=150)  # To track recent object IDs and their labels

        # Crossing line coordinates
        self._line_coordinates = None  # x1, y1, x2, y2
        self._crossing_line_object = Counter()

        # Directions tracking dict
        self._object_directions = {}
        self._min_movement_threshold = min_movement_threshold

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

    def _record_object(self, detected_object_label: str, object_id: float, x: int, y: int) -> None:
        """
        Record that an object with a specific label and ID has been seen.

        Args:
            detected_object_label (str): The label of the detected object.
            object_id (float): The unique ID of the detected object.
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

    def _record_line_crossing(self, detected_object_label: str, object_id: float, x: int, y: int) -> None:
        """
        Record that an object with a specific label and ID has crossed the line.

        Args:
            detected_object_label (str): The label of the detected object.
            object_id (float): The unique ID of the detected object.
            x (int): The x-coordinate of the object reference point.
            y (int): The y-coordinate of the object reference point.
        """
        if self._line_coordinates is None:
            return

        with self._counter_lock:
            if object_id in self._recent_objects:
                last_x, last_y = self._recent_objects[object_id]
                x1, y1, x2, y2 = self._line_coordinates
                logger.debug(
                    f"Checking line crossing for object ID {object_id} from ({last_x}, {last_y}) to ({x}, {y}) "
                    f"against line ({x1}, {y1}) to ({x2}, {y2})"
                )

                if y1 == y2:
                    # Horizontal line set so check for crossing horizontally only
                    if (last_y < y1 <= y) or (last_y > y1 >= y):
                        logger.debug(f"Object ID {object_id} crossed the horizontal line from y={last_y} to y={y}")
                        self._crossing_line_object[detected_object_label] += 1
                elif x1 == x2:
                    # Vertical line set so check for crossing vertically only
                    if (last_x < x1 <= x) or (last_x > x1 >= x):
                        logger.debug(f"Object ID {object_id} crossed the vertical line from x={last_x} to x={x}")
                        self._crossing_line_object[detected_object_label] += 1
                else:
                    # Diagonal line
                    if (x2 - x1) == 0:
                        return
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    line_y_at_last_x = slope * last_x + intercept
                    line_y_at_current_x = slope * x + intercept
                    crossed_up = last_y < line_y_at_last_x and y >= line_y_at_current_x
                    crossed_down = last_y > line_y_at_last_x and y <= line_y_at_current_x
                    if crossed_up or crossed_down:
                        logger.debug(f"Object ID {object_id} crossed the diagonal line from ({last_x}, {last_y}) to ({x}, {y})")
                        self._crossing_line_object[detected_object_label] += 1

    def _record_object_direction(self, detected_object_label: str, object_id: float, x: int, y: int) -> None:
        """
        Record the movement direction of an object with a specific label and ID.

        Args:
            detected_object_label (str): The label of the detected object.
            object_id (float): The unique ID of the detected object.
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

    def get_line_crossing_counts(self) -> dict[str, int]:
        """
        Get the count of objects that have crossed the defined line since the last reset.
            This includes all distinguished objects sees, based on their unique IDs.

        Returns:
            dict: A dictionary with labels as keys and their respective counts as values.
        """
        with self._counter_lock:
            return dict(self._crossing_line_object)

    def get_objects_directions(self) -> dict[float, list[str]]:
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
        """Register a callback invoked when a **specific label** is detected.

        Args:
            object (str): The label of the object to check for in the classification results.
            callback (DetectionCallback): A function with **no parameters**.

        Raises:
            TypeError: If `callback` is not a function.
            ValueError: If `callback` accepts any parameters.
        """
        super().on_detect(object, callback)

    def on_detect_all(self, callback: AllDetectionsCallback) -> None:
        """Register a callback invoked for **every detection event**.

        This is useful to receive a consolidated dictionary of detections for each frame.

        Args:
            callback (AllDetectionsCallback): A function that accepts **one dict argument** mapping
                each tracked label to the list of its objects, with the shape
                `{label: [{"object_id": int, "confidence": float, "bounding_box_xyxy": (x1, y1, x2, y2)}, ...], ...}`.

        Raises:
            TypeError: If `callback` is not a function.
            ValueError: If `callback` does not accept exactly one argument.
        """
        super().on_detect_all(callback)

    def start(self) -> None:
        """Start the video object detection process."""
        super().start()

    def stop(self) -> None:
        """Stop the video object detection process."""
        super().stop()

    def _process_message(self, ws: Connection, message: str) -> None:
        jmsg = json.loads(message)
        if jmsg.get("type") == "hello":
            # Parse hello message to extract model info if needed
            logger.debug(f"Connected to model runner: {jmsg}")
            try:
                self._model_info = EdgeImpulseRunnerFacade.parse_model_info_message(jmsg)
            except Exception as e:
                logger.error(f"Error parsing WS hello message: {e}")
                return

            self._forget_tracks()

            if not jmsg.get("modelParameters", {}).get("has_object_tracking", False):
                logger.error(
                    "This model has no object tracking block, so no object will ever be reported. "
                    "Enable object tracking in the Edge Impulse project and export the model again."
                )
                return

            if self._model_info and self._model_info.thresholds is not None:
                try:
                    self._set_thresholds()
                except Exception as e:
                    logger.error(f"Failed to configure the tracker: {e}")
            return

        elif jmsg.get("type") == "handling-message-success":
            # Ignore handling-message-success messages
            return

        elif jmsg.get("type") == "classification":
            result = jmsg.get("result", {})
            if not isinstance(result, dict):
                return

            tracked_objects = result.get("object_tracking", [])
            if not tracked_objects:
                return

            detections = {}
            for box in tracked_objects:
                detected_object = box.get("label")
                if detected_object is None:
                    continue

                object_id = box.get("object_id")
                if object_id is None:
                    continue

                if not self._is_label_enabled(detected_object):
                    continue

                x, y = box.get("x", 0), box.get("y", 0)
                width, height = box.get("width", 0), box.get("height", 0)

                detection_details = {
                    "object_id": object_id,
                    "confidence": box.get("value", 0.0),
                    "bounding_box_xyxy": (x, y, x + width, y + height),
                }
                detections.setdefault(detected_object, []).append(detection_details)

                self._record_object(
                    detected_object_label=detected_object,
                    object_id=object_id,
                    x=x + width // 2,
                    y=y + height // 2,
                )

                super()._execute_handler(key=detected_object, payload=detection_details)

            if len(detections) > 0:
                super()._execute_handler(key=self.ALL_HANDLERS_KEY, payload=detections)

        else:
            # Leave logging for unknown message types for debugging purposes
            logger.warning(f"Unknown message type: {jmsg.get('type')}")

    def _set_thresholds(self) -> None:
        """Set the thresholds for the object tracking model."""
        self.override_confidence(self._confidence)
        self.override_keep_grace(self._keep_grace)
        self.override_min_detections(self._min_detections)
        self.override_iou_threshold(self._iou_threshold)
        self.override_euclidean_distance_threshold(self._euclidean_distance_threshold)

    def override_confidence(self, confidence: float) -> None:
        """Override the confidence threshold for object detection model.

        Args:
            confidence (float): The new value for the confidence threshold in the range [0.0, 1.0].

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the model information is not available or does not support threshold override.
        """
        super().override_threshold(confidence)

    def override_keep_grace(self, keep_grace: int) -> None:
        """Override keep grace for object tracking model.
            Keep Grace: how many frames an object is kept if it disappears.

        Args:
            keep_grace (int): The new value for the keep grace.

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the model information is not available or does not support threshold override.
        """
        try:
            with connect(self._uri) as ws:
                self._override_config_value(ws, "max_age", keep_grace)
            self._keep_grace = keep_grace
        except Exception as e:
            logger.error(f"Failed to override keep grace: {e}")
            raise

    def override_min_detections(self, min_detections: int) -> None:
        """Override the number of detections a track needs for the object tracking model to report it.

        Args:
            min_detections (int): The new value for the minimum number of detections.

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the model information is not available or does not support threshold override.
        """
        try:
            with connect(self._uri) as ws:
                self._override_config_value(ws, "min_hits", min_detections)
            self._min_detections = min_detections
        except Exception as e:
            logger.error(f"Failed to override the minimum number of detections: {e}")
            raise

    def override_iou_threshold(self, iou_threshold: float) -> None:
        """Override IoU threshold for object tracking model.
            This is valid for bounding box object detection based models, like Yolo.
            IOU Threshold: Intersection over Union threshold for tracking.

        Args:
            iou_threshold (float): The new value for the IoU threshold.

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the model information is not available or does not support threshold override.
        """

        if self._model_info is not None and self._model_info.model_type is not None:
            if self._model_info.model_type == "constrained_object_detection":
                logger.debug("This model reports centroids. Use 'override_euclidean_distance_threshold' instead.")
                return

        try:
            with connect(self._uri) as ws:
                self._override_config_value(ws, "iou_threshold", iou_threshold)
            self._iou_threshold = iou_threshold
        except Exception as e:
            logger.error(f"Failed to override IoU threshold: {e}")
            raise

    def override_euclidean_distance_threshold(self, euclidean_distance_threshold: float) -> None:
        """Override euclidean distance threshold for object tracking model.
            This is valid for centroids based detection models, like FOMO.
            Euclidean Distance Threshold: Maximum distance in pixels to consider two detections as the same object.

        Args:
            euclidean_distance_threshold (float): The new value for the euclidean distance threshold, in pixels.

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the model information is not available or does not support threshold override.
        """

        if self._model_info is not None and self._model_info.model_type is not None:
            if self._model_info.model_type == "object_detection":
                logger.debug("This model reports bounding boxes. Use 'override_iou_threshold' instead.")
                return

        try:
            with connect(self._uri) as ws:
                self._override_config_value(ws, "threshold", euclidean_distance_threshold)
            self._euclidean_distance_threshold = euclidean_distance_threshold
        except Exception as e:
            logger.error(f"Failed to override the euclidean distance threshold: {e}")
            raise

    def _override_config_value(self, ws: Connection, key: str, value: float | int) -> None:
        """Override a specific configuration value for the object tracking model.

        Args:
            ws (Connection): The WebSocket connection to send the message through.
            key (str): The configuration key to override.
            value (float | int): The new value for the configuration.

        Raises:
            RuntimeError: If the model has no object tracking block, or that block has no such key.
            TypeError: If the value is not a number.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Invalid types for value.")

        if self._model_info is None or not self._model_info.thresholds:
            raise RuntimeError("Model information is not available or does not support threshold override.")

        block = next((th for th in self._model_info.thresholds if th.get("type") == "object_tracking"), None)
        if block is None:
            available = ", ".join(str(th.get("type")) for th in self._model_info.thresholds)
            raise RuntimeError(f"This model has no object tracking block, it only exposes: {available}.")

        if key not in block:
            knobs = ", ".join(k for k in block if k not in ("id", "type"))
            raise RuntimeError(f"The object tracking block of this model exposes {knobs}, not '{key}'.")

        logger.debug(f"Overriding {key} to {value} on block {block['id']}")
        ws.send(json.dumps({"type": "threshold-override", "id": block["id"], "key": key, "value": value}))


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
        logger.debug(f"Diagonal movement detected.")
        if dx > 0 and dy > 0:
            direction = "down-left"  # up-right becomes down-left
        elif dx > 0 > dy:
            direction = "up-left"  # down-right becomes up-left
        elif dx < 0 < dy:
            direction = "down-right"  # up-left becomes down-right ok
        elif dx < 0 and dy < 0:
            direction = "up-right"  # down-left becomes up-right ok
    elif abs(dx) > abs(dy):
        logger.debug(f"Horizontal movement detected.")
        if dx > 0:
            direction = "left"  # right becomes left
        else:
            direction = "right"  # left becomes right
    else:
        logger.debug(f"Vertical movement detected.")
        if dy > 0:
            direction = "down"  # up becomes down
        else:
            direction = "up"  # down becomes up

    return direction
