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
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import websockets

from arduino.app_peripherals.camera import BaseCamera, Camera
from arduino.app_utils import brick, Logger
from arduino.app_utils.image.adjustments import compress_to_jpeg
from arduino.app_internal.core.module import load_brick_compose_file, resolve_address

from .pose_classifier import (
    ANCHOR_JOINTS,
    EMBEDDING_JOINTS,
    MIN_OBSERVED_SCORE,
    OUT_OF_FRAME_TOLERANCE,
    EmaHysteresis,
    KEYPOINT_NAMES,
    embed,
    load_pose_classifier,
    normalize_pose,
)

logger = Logger("PoseEstimation")

_POSE_CLASSIFIER_PATH = Path(__file__).resolve().parent / "assets" / "pose_classifier.npz"


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

    Delivered by `on_pose` callbacks when the tracked person assumes or leaves
    a built-in pose.

    Attributes:
        name (str): Built-in pose name, e.g. "sitting".
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
            confidence (float): Minimum detection score for a person to be reported. The score is
                the mean of the person's 17 keypoint scores, so partly visible people score lower.
                Applied by the model runner, so detections below it are neither emitted nor drawn
                on the overlay. Changeable at runtime with `set_confidence()`.
            debounce_sec (float): Minimum seconds a presence or people-count change must be stable
                before `on_enter`/`on_exit`/`on_count_change` fire again. Filters out detection
                flicker. Default is 0 (no debounce).

        Raises:
            RuntimeError: If the model runner host address could not be resolved.
        """
        self._camera = camera if camera else Camera(fps=30)
        self._confidence = confidence
        self._debounce_sec = debounce_sec

        # Callbacks
        self._callbacks: dict[str, Callable] = {}
        self._callbacks_lock = threading.Lock()

        self._frame_hw: tuple[int, int] | None = None

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

        # Built-in pose classification: the shipped reference database and the
        # dials it was tuned with travel together inside the asset.
        load_start = time.monotonic()
        self._pose_knn, self._pose_label_weights, self._pose_names = load_pose_classifier(_POSE_CLASSIFIER_PATH)
        logger.info(f"pose classifier ready in {time.monotonic() - load_start:.2f}s (poses: {', '.join(self._pose_names)})")
        self._pose_ema = EmaHysteresis(classes=self._pose_names)
        self._pose_last_ts: float | None = None
        self._pose_last_person: Person | None = None

        # Frame-gate accounting: which rule refuses frames, and how often.
        # Written only by the receive thread; readers get GIL-atomic snapshots.
        self._gate_counts: dict[str, int] = dict.fromkeys(("classified", "anchors", "missing", "out_of_frame", "torso"), 0)
        self._gate_last: tuple[str, float] = ("none", 0.0)
        self._pose_last_probs: dict[str, float] | None = None
        self._gate_log_ts = time.monotonic()
        self._gate_log_counts = dict(self._gate_counts)

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
        # Reset the temporal state so a restart begins from a clean slate
        self._pose_ema = EmaHysteresis(classes=self._pose_names)
        self._pose_last_ts = None
        self._pose_last_person = None

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
        """Register a callback for a built-in pose (e.g. "sitting").

        The classifier follows ONE person: the largest bounding box in the
        frame, normally the closest to the camera. Other people stay visible
        through `on_keypoints` but do not fire pose events. Per-frame
        classifications are smoothed over time with hysteresis, so the
        callback receives stable edges: a `Pose` with event="enter" when the
        tracked person assumes the pose, event="exit" when they leave it.

        Args:
            pose (str): One of the built-in pose names: "left_arm_raised",
                "right_arm_raised", "sitting", "standing".
            callback (Callable[[Pose], None]): Function to call with the pose
                event. None to unregister.

        Raises:
            ValueError: If `pose` is not one of the built-in pose names.
        """
        if pose not in self._pose_names:
            raise ValueError(f"unknown pose {pose!r} (available: {', '.join(self._pose_names)})")
        self._register_callback(f"pose:{pose}", callback)

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

    def set_confidence(self, confidence: float):
        """Change the minimum detection score for a person, effective immediately.

        Args:
            confidence (float): New threshold in [0.0, 1.0], forwarded to the model runner.

        Raises:
            ValueError: If confidence is not a number in [0.0, 1.0].
        """
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"confidence must be a number in [0.0, 1.0], got {confidence!r}")
        self._confidence = float(confidence)
        logger.info(f"detection confidence set to {self._confidence}")

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
            self._frame_hw = frame.shape[:2]

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
                    sent_confidence: float | None = None
                    while self._is_running:
                        if self._confidence != sent_confidence:
                            await ws.send(json.dumps({"config": {"min_person_score": self._confidence}}))
                            sent_confidence = self._confidence
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
            for entry in metadata.get("persons", []):
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

        self._update_pose_classification(people, now)

    def _update_pose_classification(self, people: list[Person], now: float):
        """Classify the tracked person (largest box) and dispatch pose edges."""
        dt = 0.0 if self._pose_last_ts is None else now - self._pose_last_ts
        self._pose_last_ts = now

        tracked: Person | None = None
        probs: dict[str, float] | None = None
        if people:
            tracked = max(people, key=lambda person: self._box_area(person.bounding_box_xyxy))
            self._pose_last_person = tracked
            probs = self._classify_person(tracked)
            self._pose_last_probs = probs

        events = self._pose_ema.update(probs, dt, person_present=bool(people))
        if not events:
            return
        # An exit can fire after the person left the frame (grace decay): the
        # event then carries the last skeleton the pose was read from.
        subject = tracked or self._pose_last_person
        keypoints = subject.keypoints if subject else {}
        bbox = subject.bounding_box_xyxy if subject else (0, 0, 0, 0)
        for event, name in events:
            self._submit_callback(
                f"pose:{name}",
                Pose(
                    name=name,
                    event=event,
                    confidence=self._pose_ema.smoothed[name],
                    keypoints=keypoints,
                    bounding_box_xyxy=bbox,
                ),
            )

    @staticmethod
    def _box_area(box: tuple[int, int, int, int]) -> int:
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def _gate(self, outcome: str) -> None:
        """Record one frame-gate outcome and, at most every 15 s, log the tally."""
        self._gate_counts[outcome] += 1
        self._gate_last = (outcome, time.monotonic())
        now = time.monotonic()
        if now - self._gate_log_ts >= 15.0:
            delta = {key: self._gate_counts[key] - self._gate_log_counts[key] for key in self._gate_counts}
            if any(count for key, count in delta.items() if key != "classified"):
                logger.debug("frame gates (15s): " + " ".join(f"{key}={count}" for key, count in delta.items()))
            self._gate_log_ts = now
            self._gate_log_counts = dict(self._gate_counts)

    def _classify_person(self, person: Person) -> dict[str, float] | None:
        """Per-frame pose probabilities for one person, or None when unreadable.

        None means "no evidence" (every anchor guessed, missing joints,
        joints far outside the frame, collapsed torso) and freezes the
        temporal layer.
        an all-zeros dict from the classifier means "read fine, looks like
        nothing we know" and makes any active pose decay.
        """
        anchors_observed = any(
            (keypoint := person.keypoints.get(name)) is not None and keypoint.score >= MIN_OBSERVED_SCORE for name in ANCHOR_JOINTS
        )
        if not anchors_observed:
            self._gate("anchors")
            return None
        if any(name not in person.keypoints for name in KEYPOINT_NAMES):
            self._gate("missing")
            return None
        if self._frame_hw is not None:
            frame_h, frame_w = self._frame_hw
            margin_x = OUT_OF_FRAME_TOLERANCE * frame_w
            margin_y = OUT_OF_FRAME_TOLERANCE * frame_h
            for name in EMBEDDING_JOINTS:
                keypoint = person.keypoints[name]
                if not (-margin_x <= keypoint.x <= frame_w + margin_x and -margin_y <= keypoint.y <= frame_h + margin_y):
                    self._gate("out_of_frame")
                    return None
        xy = np.asarray(
            [[person.keypoints[name].x, person.keypoints[name].y] for name in KEYPOINT_NAMES],
            dtype=np.float32,
        )
        norm = normalize_pose(xy)
        if norm is None:
            self._gate("torso")
            return None
        self._gate("classified")
        return self._pose_knn.classify(embed(norm), label_weights=self._pose_label_weights)

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
