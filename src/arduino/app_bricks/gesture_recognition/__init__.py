# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import base64
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal

import numpy as np
import websockets

from arduino.app_peripherals.camera import BaseCamera, Camera
from arduino.app_utils import brick, Logger
from arduino.app_utils.image.adjustments import compress_to_jpeg
from arduino.app_internal.core.module import load_brick_compose_file, resolve_address

logger = Logger("GestureRecognition")


@brick
class GestureRecognition:
    def __init__(self, camera: BaseCamera | None = None):
        if camera is None:
            camera = Camera(fps=30)
        self._camera = camera

        # Callbacks
        self._gesture_callbacks = {}  # {(gesture, hand): callback}
        self._enter_callback = None
        self._exit_callback = None
        self._frame_callback = None
        self._callbacks_lock = threading.Lock()

        # State tracking
        self._had_hands = False
        self._is_running = False

        self._camera_frame_queue = queue.Queue(maxsize=2)

        # Callback executor and per-callback in-progress locks
        self._executor: ThreadPoolExecutor | None = None
        self._callback_locks: dict[str | tuple, threading.Lock] = {}  # keyed by "enter", "exit", or (gesture, hand)

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

    def on_gesture(self, gesture: str, callback: Callable[[dict], None], hand: Literal["left", "right", "both"] = "both"):
        """
        Register or unregister a gesture callback.

        Args:
            gesture (str): The gesture name to detect
            callback (Callable[[dict], None]): Function to call when gesture is detected. None to unregister.
                The callback receives a metadata dictionary with details about the detection, including:
                - "hand": Which hand performed the gesture ("left" or "right")
                - "gesture": Name of the detected gesture
                - "confidence": Confidence score of the detection (0.0 to 1.0)
                - "landmarks": List of key points of the detected hand (in (x, y, z) format where
                    x and y are pixel coordinates and z is normalized depth)
                - "bounding_box_xyxy": [x_min, y_min, x_max, y_max] of the detected hand bounding box
            hand (Literal["left", "right", "both"]): Which hand(s) to track

        Raises:
            ValueError: If 'hand' argument is not valid
        """
        if hand not in ("left", "right", "both"):
            raise ValueError("hand must be 'left', 'right', or 'both'")

        with self._callbacks_lock:
            key = (gesture, hand)
            if callback is None:
                if key in self._gesture_callbacks:
                    del self._gesture_callbacks[key]
                    self._callback_locks.pop(key, None)
            else:
                self._gesture_callbacks[key] = callback
                if key not in self._callback_locks:
                    self._callback_locks[key] = threading.Lock()

    def on_enter(self, callback: Callable[[], None]):
        """
        Register a callback for when hands become visible.

        Args:
            callback (Callable[[], None]): Function to call when at least one hand is detected
        """
        with self._callbacks_lock:
            self._enter_callback = callback
            if callback is not None:
                self._callback_locks["enter"] = threading.Lock()
            else:
                self._callback_locks.pop("enter", None)

    def on_exit(self, callback: Callable[[], None]):
        """
        Register a callback for when hands are no longer visible.

        Args:
            callback (Callable[[], None]): Function to call when no hands are detected anymore
        """
        with self._callbacks_lock:
            self._exit_callback = callback
            if callback is not None:
                self._callback_locks["exit"] = threading.Lock()
            else:
                self._callback_locks.pop("exit", None)

    def on_frame(self, callback: Callable[[np.ndarray], None]):
        """
        Register a callback that receives each camera frame.

        Args:
            callback (Callable[[np.ndarray], None]): Function to call with camera frame data. None to unregister.
        """
        with self._callbacks_lock:
            self._frame_callback = callback

    @brick.loop
    def _capture_loop(self):
        """Continuously capture frames from camera (runs in dedicated thread)."""
        try:
            frame = self._camera.capture()
            if frame is None:
                time.sleep(0.01)
                return

            with self._callbacks_lock:
                frame_cb = self._frame_callback
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
                except:
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
                pass
            except Exception as e:
                if self._is_running:
                    logger.error(f"Error in receive detections task: {e}. Reconnecting...")
                    await asyncio.sleep(3)

    def _process_detection(self, metadata: dict):
        """Process detection data and dispatch appropriate events."""
        hands_data = metadata.get("hands", [])
        has_hands = bool(hands_data)

        # Dispatch hand enter/exit events
        if has_hands and not self._had_hands:
            self._dispatch_enter()
        elif self._had_hands and not has_hands:
            self._dispatch_exit()

        self._had_hands = has_hands

        # Dispatch hand gesture events
        for hand_data in hands_data:
            hand = hand_data.get("hand", "")
            gesture = hand_data.get("gesture", "")
            if hand in ("left", "right") and gesture:
                self._dispatch_gesture(gesture, hand, metadata)

    def _dispatch_enter(self):
        """Dispatch hand enter event."""
        with self._callbacks_lock:
            callback = self._enter_callback

        if callback:
            self._submit_callback("enter", callback)

    def _dispatch_exit(self):
        """Dispatch hand exit event."""
        with self._callbacks_lock:
            callback = self._exit_callback

        if callback:
            self._submit_callback("exit", callback)

    def _dispatch_gesture(self, gesture: str, hand: Literal["left", "right"], metadata: dict):
        """Dispatch gesture event to registered callbacks."""
        with self._callbacks_lock:
            exact_key = (gesture, hand)
            both_key = (gesture, "both")
            keys_and_callbacks = [(k, self._gesture_callbacks[k]) for k in (exact_key, both_key) if k in self._gesture_callbacks]

        for key, callback in keys_and_callbacks:
            self._submit_callback(key, callback, metadata)

    def _submit_callback(self, key: str | tuple, callback: Callable, *args):
        """Acquire the per-callback lock and submit callback to the executor.

        If the lock is already held (callback still running), the event is discarded.
        """
        if self._executor is None:
            return
        lock = self._callback_locks.get(key)
        if lock is None or not lock.acquire(blocking=False):
            return
        self._executor.submit(self._run_callback, lock, callback, *args)

    def _run_callback(self, lock: threading.Lock, callback: Callable, *args):
        """Run a callback and release its lock when done."""
        try:
            callback(*args)
        except Exception as e:
            logger.error(f"Error in callback: {e}")
        finally:
            lock.release()
