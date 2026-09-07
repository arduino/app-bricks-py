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
from dataclasses import asdict
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
import websockets

from arduino.app_peripherals.camera import BaseCamera, Camera
from arduino.app_utils import brick, Logger
from arduino.app_utils.image.adjustments import compress_to_jpeg
from arduino.app_internal.core.module import load_brick_compose_file, resolve_address

from .pose_classifier import (
    ACTION_SMOOTHING_SECONDS,
    DEFAULT_ACTION_DURATION,
    DEFAULT_SMOOTHING_SECONDS,
    OUT_OF_FRAME_TOLERANCE,
    EmaHysteresis,
    embed_person,
    load_pose_classifier,
)
from .pose_enrollment import OTHER, Enrollment, enroll
from .pose_enrollment.cache import CACHE_FILE, EnrollmentCache
from .pose_enrollment.photos import build_buckets, custom_folders, embed_photos, photos
from .pose_types import Person, Pose, parse_people
from .pose_vocabulary import PoseSpec, parse_poses

logger = Logger("PoseEstimation")

_POSE_CLASSIFIER_PATH = Path(__file__).resolve().parent / "assets" / "pose_classifier.npz"

_UNREADABLE_HOLD_SEC = 0.5

_CUSTOM_POSES_DIR = "/app/poses"

"""Names of the built-in poses, the ones an app can listen to without teaching anything."""
BUILTIN_POSE_NAMES: tuple[str, ...] = load_pose_classifier(_POSE_CLASSIFIER_PATH)[2]


@brick
class PoseEstimation:
    def __init__(
        self,
        camera: BaseCamera | None = None,
        confidence: float = 0.25,
        count_debounce_sec: float = 0.0,
        out_of_frame_tolerance: float = OUT_OF_FRAME_TOLERANCE,
        draw_bboxes: bool = False,
        draw_low_confidence_points: bool = True,
        bbox_padding: float | tuple[float, float, float, float] = 0.0,
        poses: list[str | dict[str, Any]] | None = None,
        custom_poses_dir: str = _CUSTOM_POSES_DIR,
    ) -> None:
        """Initialize the PoseEstimation brick.

        Args:
            camera (BaseCamera): The camera instance to use for capturing video. If None, a default
                camera will be initialized. Pass the same instance shared with other bricks to reuse
                a single camera.
            poses (list[str | dict] | None): The poses this instance listens to. None (default)
                selects the built-in poses with their shipped settings. Otherwise a list whose items
                are pose names, or dicts with the key `name` plus any of `thresholds` ({"enter":
                float, "exit": float}, with votes in [0, 1], replacing the shipped values) and
                `smoothing` (seconds, the time constant of the moving average). A name is either
                one of `BUILTIN_POSE_NAMES` or a sub-folder of `custom_poses_dir` holding the photos
                of a pose of yours, taught at `start()`; a custom pose also takes `type` ("state",
                the default, for a held pose, or "action" for a movement) and `duration` (seconds,
                actions only: the typical length of one occurrence, 0.7 by default). The built-in
                poses left out stay in the classifier as negatives and never fire.
            custom_poses_dir (str): Where the photo folders of the custom poses live, one folder per
                pose named like the pose, plus an optional `other` folder with photos of what is not
                any of your poses. Default "/app/poses".
            confidence (float): Minimum detection score for a person to be reported. The score is
                the mean of the person's 17 keypoint scores, so partly visible people score lower.
                Applied by the model runner, so detections below it are neither emitted nor drawn
                on the overlay. Changeable at runtime with `set_confidence()`.
            count_debounce_sec (float): Minimum seconds a person leaving, or the people count
                dropping, must hold before `on_exit`/`on_count_change` report it, so that a
                dropped detection frame cannot fake it. People appearing are always reported at
                once. Default is 0 (no debounce). Pose events are not affected: they have their
                own temporal smoothing.
            out_of_frame_tolerance (float): How far past the frame edges a joint may be
                extrapolated before the skeleton counts as unreadable, as a fraction of the
                frame size: no pose is classified and `on_readable_change` reports False.
                Default is 0.25; 0 demands every joint inside the picture.
            draw_bboxes (bool): Draw each detected person's bounding box on the skeleton overlay
                served by the model runner. Off by default. Changeable at runtime with
                `set_draw_bboxes()`.
            draw_low_confidence_points (bool): Mark low-confidence keypoints on the overlay too, as small
                hollow dots joined by darker lines. On by default; set to False for an overlay
                that shows only the confident keypoints. Changeable at runtime with
                `set_draw_low_confidence_points()`.
            bbox_padding (float | tuple[float, float, float, float]): Expand every reported
                and drawn bounding box, CSS style: a single number applies to all sides, a
                4-tuple is (top, right, bottom, left). Top/bottom are fractions of the box
                height, left/right of its width, each in [0.0, 1.0]. Default is 0 (no
                expansion). Changeable at runtime with `set_bbox_padding()`.

        Raises:
            ValueError: If `poses` is malformed or names a pose that does not exist, or if a folder
                in `custom_poses_dir` carries the name of a built-in pose.
            RuntimeError: If the model runner host address could not be resolved.
        """
        self._custom_poses_dir = Path(custom_poses_dir)
        self._pose_specs = parse_poses(poses, BUILTIN_POSE_NAMES, custom_folders(self._custom_poses_dir, BUILTIN_POSE_NAMES))
        self._camera = camera if camera else Camera(fps=30)
        self._confidence = confidence
        self._count_debounce_sec = count_debounce_sec
        self._out_of_frame_tolerance = out_of_frame_tolerance
        self._draw_bboxes = draw_bboxes
        self._draw_low_confidence_points = draw_low_confidence_points
        self._bbox_padding = self._validate_bbox_padding(bbox_padding)

        # Callbacks
        self._callbacks: dict[str, Callable] = {}
        self._callbacks_lock = threading.Lock()

        self._frame_hw: tuple[int, int] | None = None

        # State tracking
        self._person_present = False
        self._presence_since: float | None = None
        self._person_count = 0
        self._count_candidate: int | None = None
        self._count_since = 0.0
        self._readable = False
        self._readable_since: float | None = None
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

        # Built-in pose classification: the shipped reference database, the
        # dials and the operating point it was tuned with travel together
        # inside the asset.
        load_start = time.monotonic()
        self._pose_knn, self._pose_label_weights, _, shipped_thresholds = load_pose_classifier(_POSE_CLASSIFIER_PATH)
        self._pose_names = BUILTIN_POSE_NAMES if poses is None else tuple(spec.name for spec in self._pose_specs)
        if self._pose_label_weights is not None and set(self._pose_names) != set(BUILTIN_POSE_NAMES):
            raise RuntimeError("a reduced pose vocabulary needs other_weight == 1.0 in the classifier asset")
        self._pose_thresholds = {
            "enter": {spec.name: self._declared_threshold(spec, "enter", shipped_thresholds) for spec in self._pose_specs},
            "exit": {spec.name: self._declared_threshold(spec, "exit", shipped_thresholds) for spec in self._pose_specs},
        }
        self._pose_smoothing = {
            spec.name: (ACTION_SMOOTHING_SECONDS if spec.type == "action" else DEFAULT_SMOOTHING_SECONDS)
            if spec.smoothing is None
            else spec.smoothing
            for spec in self._pose_specs
        }
        self._pose_action_duration = {
            spec.name: DEFAULT_ACTION_DURATION if spec.duration is None else spec.duration for spec in self._pose_specs if spec.type == "action"
        }
        logger.info(f"pose classifier ready in {time.monotonic() - load_start:.2f}s (poses: {', '.join(self._pose_names)})")
        self._pose_ema = EmaHysteresis(
            classes=self._pose_names,
            smoothing_tau=self._pose_smoothing,
            enter_threshold=self._pose_thresholds["enter"],
            exit_threshold=self._pose_thresholds["exit"],
            action_duration=self._pose_action_duration,
        )
        self._pose_last_ts: float | None = None
        self._pose_last_person: Person | None = None

        # Frame-gate accounting: which rule refuses frames, and how often.
        # Written only by the receive thread; readers get GIL-atomic snapshots.
        self._gate_counts: dict[str, int] = dict.fromkeys(("classified", "anchors", "missing", "out_of_frame", "torso"), 0)
        self._gate_last: tuple[str, float] = ("none", 0.0)
        self._pose_last_probs: dict[str, float] | None = None
        self._gate_log_ts = time.monotonic()
        self._gate_log_counts = dict(self._gate_counts)

    def start(self) -> None:
        """Teach the custom poses, if any, then start the capture thread and the asyncio event loop.

        Raises:
            ValueError: If a custom pose is not accepted; the message carries its report.
            RuntimeError: If the model runner cannot be reached to read the photos.
        """
        if any(not spec.builtin for spec in self._pose_specs):
            self._install_enrollment(self._enroll_custom_poses())
        self._executor = ThreadPoolExecutor()
        self._camera.start()
        self._is_running = True

    def stop(self) -> None:
        """Stop all tracking and close connections."""
        self._is_running = False
        self._camera.stop()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        # Reset the temporal state so a restart begins from a clean slate
        self._pose_ema = EmaHysteresis(
            classes=self._pose_names,
            smoothing_tau=self._pose_smoothing,
            enter_threshold=self._pose_thresholds["enter"],
            exit_threshold=self._pose_thresholds["exit"],
            action_duration=self._pose_action_duration,
        )
        self._pose_last_ts = None
        self._pose_last_person = None
        self._readable = False
        self._readable_since = None

    def on_keypoints(self, callback: Callable[[Person], None] | None) -> None:
        """Register a callback invoked once per detected person, for every processed frame.

        With several people in view, the callback is invoked once for each of
        them, sequentially, all detected in the same frame.

        Args:
            callback (Callable[[Person], None]): Function to call with one detected
                `Person` (keypoints dict and bounding box). None to unregister.
        """
        self._register_callback("keypoints", callback)

    def on_pose(self, pose: str, callback: Callable[[Pose], None] | None) -> None:
        """Register a callback for one of the poses this instance listens to (e.g. "sitting").

        The classifier follows ONE person: the largest bounding box in the
        frame, normally the closest to the camera. Other people stay visible
        through `on_keypoints` but do not fire pose events. Per-frame
        classifications are smoothed over time with hysteresis, so the
        callback receives stable edges: a `Pose` with event="enter" when the
        tracked person assumes the pose, event="exit" when they leave it.

        Args:
            pose (str): One of `pose_names`: the built-in poses ("left_arm_raised",
                "right_arm_raised", "sitting", "standing") unless the constructor's
                `poses` argument narrowed them down.
            callback (Callable[[Pose], None]): Function to call with the pose
                event. None to unregister.

        Raises:
            ValueError: If `pose` is not one of `pose_names`.
        """
        if pose not in self._pose_names:
            raise ValueError(f"unknown pose {pose!r} (available: {', '.join(self._pose_names)})")
        self._register_callback(f"pose:{pose}", callback)

    def on_enter(self, callback: Callable[[], None] | None) -> None:
        """Register a callback for when the first person enters the scene.

        Args:
            callback (Callable[[], None]): Function to call when at least one person
                is detected after nobody was in view. None to unregister.
        """
        self._register_callback("enter", callback)

    def on_exit(self, callback: Callable[[], None] | None) -> None:
        """Register a callback for when the last person leaves the scene.

        Args:
            callback (Callable[[], None]): Function to call when no people are
                detected anymore. None to unregister.
        """
        self._register_callback("exit", callback)

    def on_count_change(self, callback: Callable[[int], None] | None) -> None:
        """Register a callback for when the number of detected people changes.

        Args:
            callback (Callable[[int], None]): Function to call with the new people count.
                None to unregister.
        """
        self._register_callback("count", callback)

    def on_readable_change(self, callback: Callable[[bool], None] | None) -> None:
        """Register a callback for when the tracked person becomes readable, or stops being.

        The pose classifier needs the skeleton to be complete enough to judge: joints
        wildly outside the frame, guessed anchors or a collapsed torso make the frame
        unreadable, and no pose event is emitted while it stays that way. Becoming
        readable is reported at once, losing it only when it holds.

        Args:
            callback (Callable[[bool], None]): Function to call with True when the tracked
                person's pose can be read, False when it cannot. None to unregister.
        """
        self._register_callback("readable", callback)

    @property
    def readable(self) -> bool:
        """Whether the tracked person's pose can be read right now."""
        return self._readable

    @property
    def people_count(self) -> int:
        """How many people are in view right now, the value `on_count_change` last reported."""
        return self._person_count

    @property
    def pose_names(self) -> tuple[str, ...]:
        """The poses this instance listens to, the names `on_pose` accepts."""
        return self._pose_names

    def set_confidence(self, confidence: float) -> None:
        """Change the minimum detection score for a person, effective immediately.

        Args:
            confidence (float): New threshold in [0.0, 1.0], forwarded to the model runner.

        Raises:
            ValueError: If confidence is not a number in [0.0, 1.0].
        """
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"confidence must be a number in [0.0, 1.0], got {confidence!r}")
        self._confidence = float(confidence)
        logger.debug(f"detection confidence set to {self._confidence}")

    def set_draw_bboxes(self, draw_bboxes: bool) -> None:
        """Show or hide each detected person's bounding box on the overlay, effective immediately.

        Args:
            draw_bboxes (bool): True to draw every detected person's bounding box on the
                skeleton overlay served by the model runner, False to hide the boxes.

        Raises:
            ValueError: If draw_bboxes is not a boolean.
        """
        if not isinstance(draw_bboxes, bool):
            raise ValueError(f"draw_bboxes must be a boolean, got {draw_bboxes!r}")
        self._draw_bboxes = draw_bboxes
        logger.debug(f"bbox overlay {'enabled' if draw_bboxes else 'disabled'}")

    def set_draw_low_confidence_points(self, draw_low_confidence_points: bool) -> None:
        """Show or hide low-confidence keypoints on the overlay, effective immediately.

        Args:
            draw_low_confidence_points (bool): True to mark low-confidence keypoints too, False for an
                overlay that shows only the confident keypoints and connections.

        Raises:
            ValueError: If draw_low_confidence_points is not a boolean.
        """
        if not isinstance(draw_low_confidence_points, bool):
            raise ValueError(f"draw_low_confidence_points must be a boolean, got {draw_low_confidence_points!r}")
        self._draw_low_confidence_points = draw_low_confidence_points
        logger.debug(f"uncertain keypoints overlay {'enabled' if draw_low_confidence_points else 'disabled'}")

    def set_bbox_padding(self, padding: float | tuple[float, float, float, float]) -> None:
        """Expand every reported and drawn bounding box, effective immediately.

        The value passed replaces the current padding entirely.

        Args:
            padding (float | tuple[float, float, float, float]): CSS style: a single number
                applies to all sides, a 4-tuple is (top, right, bottom, left). Top/bottom are
                fractions of the box height, left/right of its width, each in [0.0, 1.0].

        Raises:
            ValueError: If padding is not a number in [0.0, 1.0] or a 4-tuple of them.
        """
        self._bbox_padding = self._validate_bbox_padding(padding)
        top, right, bottom, left = self._bbox_padding
        logger.debug(f"bbox padding set to top={top}, right={right}, bottom={bottom}, left={left}")

    @staticmethod
    def _validate_bbox_padding(padding: float | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Normalize a CSS-style padding value to a (top, right, bottom, left) tuple."""
        values = padding if isinstance(padding, (tuple, list)) else (padding, padding, padding, padding)
        if len(values) != 4:
            raise ValueError(f"padding must be a number or a (top, right, bottom, left) tuple, got {padding!r}")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"padding values must be numbers in [0.0, 1.0], got {value!r}")
        return tuple(float(value) for value in values)

    @staticmethod
    def _declared_threshold(spec: PoseSpec, edge: str, shipped: dict[str, dict[str, float]]) -> float:
        """The threshold a pose starts with: declared, shipped, or the reference point until enrollment derives it."""
        declared = spec.enter if edge == "enter" else spec.exit
        if declared is not None:
            return declared
        if spec.builtin:
            return shipped[edge][spec.name]
        return 0.55 if edge == "enter" else 0.35

    def _embed_photos(self, paths: list[Path]) -> dict[Path, tuple[np.ndarray | None, str | None]]:
        config = {"min_person_score": self._confidence, "draw_bboxes": False, "draw_low_confidence_points": False}
        return embed_photos(paths, self._ws_send_url, self._ws_recv_url, config, self._confidence, self._out_of_frame_tolerance)

    def _enroll_custom_poses(self) -> Enrollment:
        """Turn the photo folders into buckets, reusing the cached embeddings, and run the enrollment."""
        started = time.monotonic()
        root = self._custom_poses_dir
        cache = EnrollmentCache(root / CACHE_FILE, _POSE_CLASSIFIER_PATH, self._confidence, self._out_of_frame_tolerance)
        names = [spec.name for spec in self._pose_specs if not spec.builtin] + ([OTHER] if (root / OTHER).is_dir() else [])
        folders = {name: photos(root / name) for name in names}
        keys = {path: cache.key(root, path) for paths in folders.values() for path in paths}
        pending = [path for path, key in keys.items() if key not in cache.embeddings]
        declaration = json.dumps([asdict(spec) for spec in self._pose_specs])
        changed = bool(pending) or set(keys.values()) != set(cache.embeddings) or declaration != cache.declaration
        if pending:
            logger.info(f"reading {len(pending)} new photo(s) of the custom poses through the model runner")
            for path, (embedding, reason) in self._embed_photos(pending).items():
                cache.store(keys[path], embedding, reason)
            cache.save(list(keys.values()))
        buckets = build_buckets(root, folders, cache)
        other = buckets.pop(OTHER, None)

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        enrollment = enroll(_POSE_CLASSIFIER_PATH, self._pose_specs, buckets, other, now)
        for name, outcome in enrollment.outcomes.items():
            if changed:
                stem = f"report_{now.replace(' ', '_').replace(':', '-')}"
                report_path = root / name / f"{stem}.txt"
                copy = 1
                while report_path.exists():
                    copy += 1
                    report_path = root / name / f"{stem}-{copy}.txt"
                report_path.write_text(outcome.report + "\n")
                logger.info(f"enrollment report for {name!r} written to {report_path}:\n{outcome.report}")
            else:
                verdict = "accepted" if outcome.accepted else "not accepted"
                logger.info(f"pose {name!r}: {verdict} (same photos and declaration as last time, report unchanged)")
        if changed:
            cache.declaration = declaration
            cache.save(list(keys.values()))
        logger.info(f"custom poses enrolled in {time.monotonic() - started:.1f}s")
        return enrollment

    def _install_enrollment(self, enrollment: Enrollment) -> None:
        """Make the composed classifier and the derived thresholds the ones the frames are judged with."""
        refused = [outcome.report for outcome in enrollment.outcomes.values() if not outcome.accepted]
        if refused:
            raise ValueError("custom pose(s) not accepted, see the report(s):\n\n" + "\n\n".join(refused))
        self._pose_knn = enrollment.knn
        self._pose_label_weights = None
        for name, outcome in enrollment.outcomes.items():
            self._pose_thresholds["enter"][name] = outcome.enter
            self._pose_thresholds["exit"][name] = outcome.exit
        self._pose_ema = EmaHysteresis(
            classes=self._pose_names,
            smoothing_tau=self._pose_smoothing,
            enter_threshold=self._pose_thresholds["enter"],
            exit_threshold=self._pose_thresholds["exit"],
            action_duration=self._pose_action_duration,
        )

    def on_frame(self, callback: Callable[[np.ndarray], None] | None) -> None:
        """Register a callback that receives each raw camera frame.

        Args:
            callback (Callable[[np.ndarray], None]): Function to call with camera frame data.
                None to unregister.
        """
        self._register_callback("frame", callback)

    def on_error(self, callback: Callable[[Exception], None] | None) -> None:
        """Register a callback invoked when an error occurs while processing detections.

        Args:
            callback (Callable[[Exception], None]): Function to call with the raised exception.
                None to unregister.
        """
        self._register_callback("error", callback)

    def _register_callback(self, key: str, callback: Callable | None) -> None:
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
    def _capture_loop(self) -> None:
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
    def _send_receive_loop(self) -> None:
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

    async def _send_frames_task(self) -> None:
        """Send frames to the processing container via WebSocket."""
        while self._is_running:
            try:
                async with websockets.connect(self._ws_send_url) as ws:
                    sent_config: dict | None = None
                    while self._is_running:
                        top, right, bottom, left = self._bbox_padding
                        config = {
                            "min_person_score": self._confidence,
                            "draw_bboxes": self._draw_bboxes,
                            "draw_low_confidence_points": self._draw_low_confidence_points,
                            "bbox_padding_top": top,
                            "bbox_padding_right": right,
                            "bbox_padding_bottom": bottom,
                            "bbox_padding_left": left,
                        }
                        if config != sent_config:
                            await ws.send(json.dumps({"config": config}))
                            sent_config = config
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

    async def _receive_detections_task(self) -> None:
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

    def _process_detection(self, metadata: dict) -> None:
        """Process detection data and dispatch appropriate events."""
        try:
            people = parse_people(metadata, self._confidence)
        except Exception as e:
            logger.error(f"Error parsing detection metadata: {e}")
            self._dispatch_error(e)
            return

        count = len(people)
        now = time.monotonic()

        # Dispatch person enter/exit events, debounced to filter out detection flicker
        present = count > 0
        if present == self._person_present:
            self._presence_since = None
        else:
            if self._presence_since is None:
                self._presence_since = now
            if present or (now - self._presence_since) >= self._count_debounce_sec:
                self._person_present = present
                self._presence_since = None
                self._submit_callback("enter" if present else "exit")

        # Dispatch people count change events, debounced as well
        if count == self._person_count:
            self._count_candidate = None
        else:
            if self._count_candidate != count:
                self._count_candidate = count
                self._count_since = now
            if count > self._person_count or (now - self._count_since) >= self._count_debounce_sec:
                self._person_count = count
                self._count_candidate = None
                self._submit_callback("count", count)

        # Dispatch keypoint events (not debounced: they are the raw detection stream)
        if people:
            self._submit_callback("keypoints", people, unroll=True)

        self._update_pose_classification(people, now)

    def _update_pose_classification(self, people: list[Person], now: float) -> None:
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

        self._update_readable(probs is not None, now)

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

    def _update_readable(self, readable: bool, now: float) -> None:
        """Dispatch readability changes: gained at once, lost only when it holds."""
        if readable == self._readable:
            self._readable_since = None
            return
        if self._readable_since is None:
            self._readable_since = now
        if readable or (now - self._readable_since) >= _UNREADABLE_HOLD_SEC:
            self._readable = readable
            self._readable_since = None
            self._submit_callback("readable", readable)

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
        embedding, gate = embed_person(person, self._frame_hw, self._out_of_frame_tolerance)
        self._gate(gate)
        if embedding is None:
            return None
        return self._pose_knn.classify(embedding, label_weights=self._pose_label_weights)

    def _dispatch_error(self, error: Exception) -> None:
        callback = self._get_callback("error")
        if callback:
            self._submit_callback("error", error)

    def _submit_callback(self, key: str, *args: Any, unroll: bool = False) -> None:
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

    def _run_callback(self, lock: threading.Lock, callback: Callable, *args: Any, unroll: bool = False) -> None:
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
