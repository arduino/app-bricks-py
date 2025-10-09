# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import threading
import queue
import inspect
import numpy as np
import time
from typing import Iterable
from arduino.app_internal.core import EdgeImpulseRunnerFacade
from arduino.app_utils import Logger, SlidingWindowBuffer, brick

logger = Logger("AnomalyDetection")


@brick
class VibrationAnomalyDetection(EdgeImpulseRunnerFacade):
    """Detect vibration anomalies from accelerometer time-series using a pre-trained
    Edge Impulse model.

    This Brick buffers incoming samples into a sliding window sized to the model’s
    `input_features_count`, runs inference when a full window is available, extracts
    the **anomaly score**, and (optionally) invokes a user-registered callback when
    the score crosses a configurable threshold.

    Notes:
        - Requires an active Edge Impulse runner; model info is fetched at init.
        - The window size equals the model’s `input_features_count`; samples pushed
          via `accumulate_samples()` are flattened before inference.
        - The expected **units, axis order, and sampling rate** must match those
          used during model training (e.g., m/s² vs g, [ax, ay, az], 100 Hz).
        - A single callback is supported at a time (thread-safe registration).
    """

    def __init__(self, anomaly_detection_threshold: float = 1.0):
        """Initialize the vibration anomaly detector.

        Args:
            anomaly_detection_threshold (float): Threshold applied to the model’s
                anomaly score to decide whether to trigger the registered callback.
                Typical starting point is 1.0; tune based on your dataset.

        Raises:
            ValueError: If the Edge Impulse runner is unreachable, or if the model
                info is missing/invalid (e.g., non-positive `frequency` or
                `input_features_count`).
        """
        self._anomaly_detection_threshold = anomaly_detection_threshold
        super().__init__()
        model_info = self.get_model_info()
        if not model_info:
            raise ValueError("Failed to retrieve model information. Ensure the EI model runner service is running.")
        if model_info.frequency <= 0 or model_info.input_features_count <= 0:
            raise ValueError("Model parameters are missing or incomplete in the retrieved model information.")
        self._model_info = model_info

        self._handler = None  # Single handler for anomaly detection
        self._handler_lock = threading.Lock()

        self._buffer = SlidingWindowBuffer(window_size=model_info.input_features_count, slide_amount=int(model_info.input_features_count))

    def accumulate_samples(self, sensor_samples: Iterable[float]) -> None:
        """Append one or more accelerometer samples to the sliding window buffer.

        Args:
            sensor_samples (Iterable[float]): A sequence of numeric values. This can
                be a single 3-axis sample `(ax, ay, az)`, multiple concatenated
                triples, or any iterable whose flattened length contributes toward
                the model’s `input_features_count`.

        Raises:
            ValueError: If `sensor_samples` is empty or None.

        Notes:
            - Ensure **units** match model training (convert g → m/s² if required).
            - Provide samples at approximately the model’s `frequency` to avoid
              under- or oversampling.
            - When the buffer reaches a full window, `loop()` will consume it.
        """
        if not sensor_samples or len(sensor_samples) == 0:
            raise ValueError("Invalid sensor samples. Expected an array with len > 0.")

        # Fill time window buffer with the latest accelerometer samples.
        chunk = np.array(sensor_samples)
        if not self._buffer.push(chunk):
            logger.debug(f"Samples not pushed to the buffer. Buffer is full or has insufficient capacity.")

    def on_anomaly(self, callback: callable):
        """Register a handler to be invoked when an anomaly is detected.

        The callback signature can be one of:
            - `callback()`
            - `callback(anomaly_score: float)`
            - `callback(anomaly_score: float, classification: dict)`

        Args:
            callback (callable): Function to invoke when `anomaly_score >= threshold`.
                If a signature with `classification` is used and the model returns
                an auxiliary classification head, a dict with label scores is passed.

        Notes:
            - Registration is thread-safe and **replaces** any previously set handler.
            - No callback is invoked if no handler is registered or if the score is
              below the threshold.
        """
        self._handler_lock.acquire()
        try:
            self._handler = callback
        finally:
            self._handler_lock.release()

    def loop(self):
        """Non-blocking processing step; run this periodically.

        Behavior:
            - Pulls a full window from the buffer (if available).
            - Runs inference via `infer_from_features(...)`.
            - Extracts the anomaly score and, if `>= threshold`, invokes the
              registered callback (respecting its signature).

        Raises:
            StopIteration: Propagated if an internal shutdown condition is signaled.

        Notes:
            - Call at (or faster than) `1 / model_info.frequency` seconds.
            - Consumes **at most one** full window per call.
            - Handles transient queue conditions internally and throttles on errors
              to avoid tight loops.
        """
        try:
            features = self._buffer.pull()
            if features is None or len(features) == 0:
                return

            ret = super().infer_from_features(features[: int(self._model_info.input_features_count)].flatten().tolist())
            logger.debug(f"Inference result: {ret}")
            spotted_anomaly = self._extract_anomaly_score(ret)
            if spotted_anomaly is not None:
                if spotted_anomaly >= self._anomaly_detection_threshold and self._handler is not None:
                    callback = self._handler
                    if callback is not None and inspect.isfunction(callback):
                        logger.debug(f"Invoking callback handler for anomaly.")
                        callback_signature = inspect.signature(callback)  # Validate callback signature
                        if len(callback_signature.parameters) == 0:
                            callback()
                        elif len(callback_signature.parameters) >= 1:
                            if len(callback_signature.parameters) == 1:
                                # If the callback expects one parameter, pass the anomaly score
                                callback(spotted_anomaly)
                            else:
                                # If the callback expects more than one parameter, try to find a parameter that matches
                                # the expected name: "classification"
                                if "classification" in callback_signature.parameters:
                                    complete_detection = self._extract_classification(ret)
                                    callback(spotted_anomaly, classification=complete_detection)
                        else:
                            logger.error(
                                f"Callback has an unsupported signature. Expected 0 or 1 parameters, got {len(callback_signature.parameters)}."
                            )

        except queue.Empty:
            return
        except queue.ShutDown:
            raise StopIteration()
        except Exception as e:
            logger.error(f"Error {e}")
            time.sleep(1)  # Sleep briefly to avoid tight loop in case of errors

    def start(self):
        """Prepare the detector for a new session.

        Notes:
            - Flushes the internal buffer so the next window starts clean.
            - Call before beginning to stream new samples.
        """
        self._buffer.flush()

    def stop(self):
        """Stop the detector and release transient resources.

        Notes:
            - Clears the internal buffer; does not alter the registered callback.
        """
        self._clear()

    def _clear(self):
        """Internal helper: flush the sensor data buffer and log the action.

        Notes:
            - Intended for internal lifecycle management; use `stop()` instead of
              calling this directly from user code.
        """
        self._buffer.flush()
        logger.info(f"Sensor data buffer cleared")
