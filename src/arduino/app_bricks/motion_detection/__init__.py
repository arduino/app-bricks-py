# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import threading
import queue
import inspect
import numpy as np
import time
from typing import Iterable, Tuple
from arduino.app_internal.core import EdgeImpulseRunnerFacade
from arduino.app_utils import brick, Logger, SlidingWindowBuffer

logger = Logger("MotionDetection")


@brick
class MotionDetection(EdgeImpulseRunnerFacade):
    """This Motion Detection module classifies motion patterns using accelerometer data."""

    def __init__(self, confidence: float = 0.4):
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

        self._handlers = {}  # Dictionary to hold handlers for different keywords
        self._handlers_lock = threading.Lock()

        # TODO: remove this queue and its handling
        self._external_notification_queue = queue.Queue(
            maxsize=100
        )  # Queue to hold chunks of sensor data for external notifications (like processing them in a chart)

        self._buffer = SlidingWindowBuffer(window_size=model_info.input_features_count, slide_amount=int(model_info.input_features_count))

    def start(self):
        self._buffer.flush()

    def stop(self):
        self._buffer.flush()

    def on_movement_detection(self, movement: str, callback: callable):
        """Register a callback function to be invoked when a specific motion pattern is detected.

        Args:
            movement (str): The motion pattern name to check for in the classification results.
            callback (callable): Function to call when the specified motion pattern is detected.
        """
        with self._handlers_lock:
            if movement in self._handlers:
                logger.warning(f"Handler for movement '{movement}' already exists. Overwriting.")
            self._handlers[movement] = callback

    def accumulate_samples(self, accelerometer_samples: Tuple[float, float, float]):
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

        # Fill buffer of external notification queue with the latest accelerometer samples.
        if self._external_notification_queue.full():
            self._external_notification_queue.get_nowait()
        self._external_notification_queue.put_nowait(accelerometer_samples)

    def get_sensor_samples(self) -> Iterable[Tuple[float, float, float]]:
        """Get the current sensor samples.

        Returns:
            iterable: An iterable containing the accumulated sensor data (x, y, z acceleration values).
        """
        while True:
            acquired_samples = self._external_notification_queue.get()
            if acquired_samples is None:
                continue
            yield acquired_samples

    def _movement_spotted(self, item: dict) -> Tuple[str, float, dict] | None:
        """Verify if a movement has been spotted.

        Args:
            item (dict): The item containing classification results.

        Returns:
            tuple[str, float, dict] | None: A tuple containing the detected class name, confidence level, and complete
                classification results; otherwise, None.
        """
        classification = self._extract_classification(item, confidence=0.0)  # Use 0.0 to get all results without filtering by confidence
        if not classification:
            return None

        detected_class = None
        detected_class_confidence = 0.0
        classification_dict = {}

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

        return detected_class, detected_class_confidence, classification_dict

    @brick.loop
    def _detection_loop(self):
        """Main loop for motion detection, processing sensor data and invoking callbacks when movements are detected."""
        features = self._buffer.pull()
        if features is None or len(features) == 0:
            return

        try:
            ret = super().infer_from_features(features[: int(self._model_info.input_features_count)].flatten().tolist())
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
