# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import base64
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import websockets

from arduino.app_peripherals.camera import BaseCamera, Camera
from arduino.app_utils import brick, Logger
from arduino.app_utils.image.adjustments import compress_to_jpeg
from arduino.app_internal.core.module import load_brick_compose_file, resolve_address

logger = Logger("PoseEstimation")

_RUNNER_MIN_POSE_SCORE = 0.25

"""Names of the 17 body keypoints detected for each person, in model output order."""
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass
class Keypoint:
    """One of the 17 body keypoints of a detected person.

    Attributes:
        name (str): Keypoint name, one of `KEYPOINT_NAMES`.
        x (int): Horizontal pixel coordinate in the camera frame.
        y (int): Vertical pixel coordinate in the camera frame.
        score (float): Confidence score in [0.0, 1.0] for this keypoint.
    """

    name: str
    x: int
    y: int
    score: float


@dataclass
class Person:
    """A person detected in a frame.

    Attributes:
        keypoints (dict[str, Keypoint]): The person's 17 keypoints, keyed by
            keypoint name (see `KEYPOINT_NAMES`). Low-confidence keypoints are
            included; filter by their score.
        bounding_box_xyxy (tuple[int, int, int, int]): (x1, y1, x2, y2) box
            enclosing the person's confident keypoints, in frame coordinates.
    """

    keypoints: dict[str, Keypoint]
    bounding_box_xyxy: tuple[int, int, int, int]


@dataclass
class Pose:
    """A pose classification event for a single person.

    Delivered by `on_pose` callbacks. Not implemented yet: pose classification
    with built-in pose names will be added in a future version of this brick.

    Attributes:
        name (str): Built-in pose name, e.g. "arms_up".
        event (Literal["enter", "exit"]): "enter" when the person assumes the
            pose, "exit" when they leave it.
        confidence (float): Classification confidence in [0.0, 1.0] at the event edge.
        keypoints (dict[str, Keypoint]): The person's 17 keypoints, keyed by
            keypoint name (see `KEYPOINT_NAMES`).
        bounding_box_xyxy (tuple[int, int, int, int]): (x1, y1, x2, y2) box
            enclosing the person's confident keypoints, in frame coordinates.
    """

    name: str
    event: Literal["enter", "exit"]
    confidence: float
    keypoints: dict[str, Keypoint]
    bounding_box_xyxy: tuple[int, int, int, int]


@brick
class PoseEstimation:
    def __init__(
        self,
        camera: BaseCamera | None = None,
        confidence: float = 0.25,
        debounce_sec: float = 0.0,
    ):
        """Initialize the PoseEstimation brick.

        Args:
            camera (BaseCamera): The camera instance to use for capturing video. If None, a default
                camera will be initialized. Pass the same instance shared with other bricks to reuse
                a single camera.
            confidence (float): Minimum pose score for a detected person to be reported. Default is 0.25.
                Values below 0.25 have no additional effect: the model runner never emits poses
                scoring less than that.
            debounce_sec (float): Minimum seconds a presence or people-count change must be stable
                before `on_enter`/`on_exit`/`on_count_change` fire again. Filters out detection
                flicker. Default is 0 (no debounce).

        Raises:
            RuntimeError: If the model runner host address could not be resolved.
        """
        self._camera = camera if camera else Camera(fps=30)
        self._confidence = confidence
        self._debounce_sec = debounce_sec

        if confidence < _RUNNER_MIN_POSE_SCORE:
            logger.warning(
                f"confidence={confidence} is below the model runner's decode floor ({_RUNNER_MIN_POSE_SCORE}): "
                f"poses scoring less are never emitted, so this setting behaves like {_RUNNER_MIN_POSE_SCORE}"
            )

        # Callbacks
        self._callbacks: dict[str, Callable] = {}
        self._callbacks_lock = threading.Lock()

        # State tracking
        self._person_present = False
        self._presence_change_ts = 0.0
        self._person_count = 0
        self._count_change_ts = 0.0
        self._is_running = False

        self._camera_frame_queue = queue.Queue(maxsize=2)

        # Callback executor and per-callback in-progress locks
        self._executor: ThreadPoolExecutor | None = None
        self._callback_locks: dict[str, threading.Lock] = {}

        # WebSocket endpoints
        infra = load_brick_compose_file(self.__class__)
        if infra is None or "services" not in infra:
            raise RuntimeError("Infrastructure configuration could not be loaded.")
        for k, _ in infra["services"].items():
            self._host = k
            break  # Only one service is expected

        self._host = resolve_address(self._host)
        if not self._host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")

        self._ws_send_url = f"ws://{self._host}:5000"
        self._ws_recv_url = f"ws://{self._host}:5001"

    def start(self):
        """Start the capture thread and asyncio event loop."""
        self._executor = ThreadPoolExecutor()
        self._camera.start()
        self._is_running = True

    def stop(self):
        """Stop all tracking and close connections."""
        self._is_running = False
        self._camera.stop()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def on_keypoints(self, callback: Callable[[Person], None] | None):
        """Register a callback invoked once per detected person, for every processed frame.

        With several people in view, the callback is invoked once for each of
        them, sequentially, all detected in the same frame.

        Args:
            callback (Callable[[Person], None]): Function to call with one detected
                `Person` (keypoints dict and bounding box). None to unregister.
        """
        self._register_callback("keypoints", callback)

    def on_pose(self, pose: str, callback: Callable[[Pose], None] | None):
        """Register a callback for a named pose (e.g. "arms_up").

        Not implemented yet: pose classification with built-in pose names will
        be added in a future version of this brick. The callback will receive a
        `Pose` event when a person assumes the named pose (event="enter") and
        when they leave it (event="exit").

        Args:
            pose (str): Name of the pose to detect.
            callback (Callable[[Pose], None]): Function to call with the pose event.
                None to unregister.

        Raises:
            NotImplementedError: Always, in this version of the brick.
        """
        raise NotImplementedError("Pose classification will be added in a future version of this brick.")

    def on_enter(self, callback: Callable[[], None] | None):
        """Register a callback for when the first person enters the scene.

        Args:
            callback (Callable[[], None]): Function to call when at least one person
                is detected after nobody was in view. None to unregister.
        """
        self._register_callback("enter", callback)

    def on_exit(self, callback: Callable[[], None] | None):
        """Register a callback for when the last person leaves the scene.

        Args:
            callback (Callable[[], None]): Function to call when no people are
                detected anymore. None to unregister.
        """
        self._register_callback("exit", callback)

    def on_count_change(self, callback: Callable[[int], None] | None):
        """Register a callback for when the number of detected people changes.

        Args:
            callback (Callable[[int], None]): Function to call with the new people count.
                None to unregister.
        """
        self._register_callback("count", callback)

    def on_frame(self, callback: Callable[[np.ndarray], None] | None):
        """Register a callback that receives each raw camera frame.

        Args:
            callback (Callable[[np.ndarray], None]): Function to call with camera frame data.
                None to unregister.
        """
        self._register_callback("frame", callback)

    def on_error(self, callback: Callable[[Exception], None] | None):
        """Register a callback invoked when an error occurs while processing detections.

        Args:
            callback (Callable[[Exception], None]): Function to call with the raised exception.
                None to unregister.
        """
        self._register_callback("error", callback)

    def _register_callback(self, key: str, callback: Callable | None):
        with self._callbacks_lock:
            if callback is None:
                self._callbacks.pop(key, None)
                self._callback_locks.pop(key, None)
            else:
                self._callbacks[key] = callback
                if key not in self._callback_locks:
                    self._callback_locks[key] = threading.Lock()

    def _get_callback(self, key: str) -> Callable | None:
        with self._callbacks_lock:
            return self._callbacks.get(key)

    @brick.loop
    def _capture_loop(self):
        """Continuously capture frames from camera (runs in dedicated thread)."""
        try:
            frame = self._camera.capture()
            if frame is None:
                time.sleep(0.01)
                return

            frame_cb = self._get_callback("frame")
            if frame_cb:
                try:
                    frame_cb(frame)
                except Exception as e:
                    logger.error(f"Error in frame callback: {e}")

            jpeg_frame = compress_to_jpeg(frame)
            if jpeg_frame is None:
                time.sleep(0.01)
                return

            try:
                self._camera_frame_queue.put(jpeg_frame, block=False)
            except queue.Full:
                # Drop oldest frame and add new one
                try:
                    self._camera_frame_queue.get_nowait()
                    self._camera_frame_queue.put(jpeg_frame, block=False)
                except (queue.Empty, queue.Full):
                    pass

        except Exception as e:
            if self._is_running:
                logger.error(f"Error capturing frame: {e}")

    @brick.execute
    def _send_receive_loop(self):
        """Run the asyncio event loop in a dedicated thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            tasks = asyncio.gather(self._send_frames_task(), self._receive_detections_task(), return_exceptions=True)
            loop.run_until_complete(tasks)

        except Exception as e:
            logger.error(f"Error in asyncio loop: {e}")
        finally:
            loop.close()

    async def _send_frames_task(self):
        """Send frames to the processing container via WebSocket."""
        while self._is_running:
            try:
                async with websockets.connect(self._ws_send_url) as ws:
                    while self._is_running:
                        try:
                            frame = await asyncio.get_event_loop().run_in_executor(None, self._camera_frame_queue.get, True, 0.1)
                        except queue.Empty:
                            continue

                        b64_frame = base64.b64encode(frame.tobytes()).decode("utf-8")
                        payload = {"frame": b64_frame}

                        await ws.send(json.dumps(payload))

            except Exception as e:
                if self._is_running:
                    logger.error(f"Error in send frames task: {e}. Reconnecting...")
                    await asyncio.sleep(3)

    async def _receive_detections_task(self):
        """Receive detection results and dispatch events."""
        while self._is_running:
            try:
                async with websockets.connect(self._ws_recv_url) as ws:
                    while self._is_running:
                        data = await ws.recv()
                        detection = json.loads(data)

                        self._process_detection(detection.get("metadata", {}))

            except json.JSONDecodeError as e:
                logger.error(f"Received invalid JSON data: {e}")
            except Exception as e:
                if self._is_running:
                    logger.error(f"Error in receive detections task: {e}. Reconnecting...")
                    await asyncio.sleep(3)

    def _process_detection(self, metadata: dict):
        """Process detection data and dispatch appropriate events."""
        try:
            people: list[Person] = []
            for entry in metadata.get("poses", []):
                if float(entry.get("score", 0.0)) < self._confidence:
                    continue
                keypoints = {
                    kp.get("name", ""): Keypoint(
                        name=kp.get("name", ""),
                        x=int(kp.get("x", 0)),
                        y=int(kp.get("y", 0)),
                        score=float(kp.get("score", 0.0)),
                    )
                    for kp in entry.get("keypoints", [])
                }
                bbox = entry.get("bounding_box_xyxy", [0, 0, 0, 0])
                people.append(Person(keypoints=keypoints, bounding_box_xyxy=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        except Exception as e:
            logger.error(f"Error parsing detection metadata: {e}")
            self._dispatch_error(e)
            return

        count = len(people)
        now = time.monotonic()

        # Dispatch person enter/exit events, debounced to filter out detection flicker
        present = count > 0
        if present != self._person_present and (now - self._presence_change_ts) >= self._debounce_sec:
            self._person_present = present
            self._presence_change_ts = now
            self._submit_callback("enter" if present else "exit")

        # Dispatch people count change events, debounced as well
        if count != self._person_count and (now - self._count_change_ts) >= self._debounce_sec:
            self._person_count = count
            self._count_change_ts = now
            self._submit_callback("count", count)

        # Dispatch keypoint events (not debounced: they are the raw detection stream)
        if people:
            self._submit_callback("keypoints", people, unroll=True)

    def _dispatch_error(self, error: Exception):
        callback = self._get_callback("error")
        if callback:
            self._submit_callback("error", error)

    def _submit_callback(self, key: str, *args, unroll: bool = False):
        """Acquire the per-callback lock and submit the callback to the executor.

        If the lock is already held (callback still running), the event is discarded.
        """
        callback = self._get_callback(key)
        if callback is None or self._executor is None:
            return
        with self._callbacks_lock:
            lock = self._callback_locks.get(key)
        if lock is None or not lock.acquire(blocking=False):
            return
        try:
            self._executor.submit(self._run_callback, lock, callback, *args, unroll=unroll)
        except RuntimeError:
            # Executor was shut down before the task could be submitted
            lock.release()

    def _run_callback(self, lock: threading.Lock, callback: Callable, *args, unroll: bool = False):
        """Run a callback and release its lock when done.

        With `unroll=True` the first argument is a list and the callback is invoked once per item.
        """
        try:
            payloads = args[0] if unroll and args else [args]
            for payload in payloads:
                call_args = (payload,) if unroll else args
                try:
                    callback(*call_args)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
                    error_cb = self._get_callback("error")
                    if error_cb and callback is not error_cb:
                        try:
                            error_cb(e)
                        except Exception as nested:
                            logger.error(f"Error in error callback: {nested}")
        finally:
            lock.release()
