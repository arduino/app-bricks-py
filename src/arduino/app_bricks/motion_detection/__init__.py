# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import collections
import threading
import queue
import inspect
import numpy as np
import time
from collections.abc import Callable, Iterable
from typing import Any
from arduino.app_internal.core import EdgeImpulseRunnerFacade
from arduino.app_utils import brick, Logger, SlidingWindowBuffer

logger = Logger("MotionDetection")

# Bounded wait on the sensor window: pull() returns an empty window on timeout, the loop returns and
# the worker checks whether the brick is still running before pulling again. Without it stop() leaves
# the loop waiting for a window that never comes, and the worker thread never terminates.
_WINDOW_PULL_TIMEOUT_S = 0.1


@brick
class MotionDetection(EdgeImpulseRunnerFacade):
    """This Motion Detection module classifies motion patterns using accelerometer data."""

    def __init__(self, confidence: float = 0.4) -> None:
        """Initialize the MotionDetection module.

        Args:
            confidence (float): Confidence level for detection. Default is 0.4 (40%).
        """
        self._confidence = confidence
        super().__init__()
        model_info = self.get_model_info()
        if not model_info:
            raise ValueError("Failed to retrieve model information. Ensure the EI model runner service is running.")
        if model_info.frequency <= 0 or model_info.input_features_count <= 0:
            raise ValueError("Model parameters are missing or incomplete in the retrieved model information.")
        self._model_info = model_info

        self._handlers: dict[str, Callable[..., None]] = {}  # Dictionary to hold handlers for different keywords
        self._handlers_lock = threading.Lock()

        # The latest samples, kept for get_sensor_samples() (e.g. to chart them): a full deque drops the
        # oldest one in the same append, so the producer never races the reader on the drop.
        self._sensor_samples: collections.deque[tuple[float, float, float]] = collections.deque(maxlen=100)
        self._sensor_samples_available = threading.Condition()

        self._buffer = SlidingWindowBuffer(window_size=model_info.input_features_count, slide_amount=int(model_info.input_features_count))

    def start(self) -> None:
        self._buffer.flush()

    def stop(self) -> None:
        self._buffer.flush()

    def on_movement_detection(self, movement: str, callback: Callable[..., None]) -> None:
        """Register a callback function to be invoked when a specific motion pattern is detected.

        Args:
            movement (str): The motion pattern name to check for in the classification results.
            callback (Callable[..., None]): Function to call when the specified motion pattern is detected. It takes
                no argument, or one receiving the classification results, or a ``classification`` keyword argument.
        """
        with self._handlers_lock:
            if movement in self._handlers:
                logger.warning(f"Handler for movement '{movement}' already exists. Overwriting.")
            self._handlers[movement] = callback

    def accumulate_samples(self, accelerometer_samples: tuple[float, float, float]) -> None:
        """Accumulate accelerometer samples for motion detection.

        Args:
            accelerometer_samples (tuple): A tuple containing x, y, z acceleration values. Typically, these values are
                in m/s^2, but depends on the model configuration.
        """
        if not accelerometer_samples or len(accelerometer_samples) != 3:
            raise ValueError("Invalid accelerometer samples. Expected a tuple of three float values (x, y, z).")

        # Fill time window buffer with the latest accelerometer samples.
        chunk = np.array([accelerometer_samples[0], accelerometer_samples[1], accelerometer_samples[2]])
        if not self._buffer.push(chunk):
            logger.debug(f"Samples not pushed to the buffer. Buffer is full or has insufficient capacity.")

        # Keep the sample for get_sensor_samples().
        with self._sensor_samples_available:
            self._sensor_samples.append(accelerometer_samples)
            self._sensor_samples_available.notify()

    def get_sensor_samples(self) -> Iterable[tuple[float, float, float]]:
        """Get the current sensor samples.

        Returns:
            iterable: An iterable containing the accumulated sensor data (x, y, z acceleration values).
        """
        while True:
            with self._sensor_samples_available:
                self._sensor_samples_available.wait_for(lambda: len(self._sensor_samples) > 0)
                sample = self._sensor_samples.popleft()
            yield sample

    def _movement_spotted(self, item: dict[str, Any] | None) -> tuple[str, float, dict[str, float]] | None:
        """Verify if a movement has been spotted.

        Args:
            item (dict | None): The item containing classification results, None when the inference failed.

        Returns:
            tuple[str, float, dict] | None: A tuple containing the detected class name, confidence level, and complete
                classification results; otherwise, None.
        """
        classification = self._extract_classification(item, confidence=0.0)  # Use 0.0 to get all results without filtering by confidence
        if not classification:
            return None

        detected_class = None
        detected_class_confidence = 0.0
        classification_dict: dict[str, float] = {}

        class_results = classification["classification"]
        for class_detected in class_results:
            class_name = class_detected["class_name"]
            class_confidence = float(class_detected["confidence"])
            classification_dict[class_name] = class_confidence
            if class_confidence < self._confidence:
                continue

            if class_confidence >= self._confidence and class_confidence > detected_class_confidence:
                detected_class = class_name
                detected_class_confidence = class_confidence

        if detected_class is None:
            return None
        return detected_class, detected_class_confidence, classification_dict

    @brick.loop
    def _detection_loop(self) -> None:
        """Main loop for motion detection, processing sensor data and invoking callbacks when movements are detected."""
        features = self._buffer.pull(timeout=_WINDOW_PULL_TIMEOUT_S)
        if len(features) == 0:
            return

        try:
            ret = self.infer_from_features(features.flatten().tolist())
            spotted_movement = self._movement_spotted(ret)
            if spotted_movement is not None:
                keyword, confidence, complete_detection = spotted_movement
                logger.debug(f"Movement '{keyword}' detected in sensor data with confidence {confidence:2f}%.")
                if keyword in self._handlers:
                    callback = self._handlers[keyword]
                    if callback is not None and inspect.isfunction(callback):
                        logger.debug(f"Invoking callback for keyword '{keyword}'.")
                        callback_signature = inspect.signature(callback)  # Validate callback signature
                        if len(callback_signature.parameters) == 0:
                            callback()
                        elif len(callback_signature.parameters) >= 1:
                            if len(callback_signature.parameters) == 1:
                                # If the callback expects one parameter, pass the complete detection result
                                callback(complete_detection)
                            else:
                                # If the callback expects more than one parameter, try to find a parameter that matches
                                # the expected name: "classification"
                                if "classification" in callback_signature.parameters:
                                    callback(classification=complete_detection)
                        else:
                            logger.error(
                                f"Callback for keyword '{keyword}' has an unsupported signature. "
                                f"Expected 0 or 1 parameters, got {len(callback_signature.parameters)}."
                            )

        except queue.Empty:
            return
        except queue.ShutDown:
            raise StopIteration()
        except Exception as e:
            logger.error(f"Error {e}")
            time.sleep(1)  # Sleep briefly to avoid tight loop in case of errors
