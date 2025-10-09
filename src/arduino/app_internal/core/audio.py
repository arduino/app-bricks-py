# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import time
import inspect
import math
import threading
from typing import Callable
from arduino.app_internal.core import EdgeImpulseRunnerFacade
from arduino.app_peripherals.microphone import Microphone
from arduino.app_utils import Logger, SlidingWindowBuffer, brick

logger = Logger(__name__)


class AudioDetector(EdgeImpulseRunnerFacade):
    """AudioDetector module for detecting sounds and classifying audio using a specified model."""

    def __init__(self, mic: Microphone = None, confidence: float = 0.8, debounce_sec: float = 2.0):
        """Initialize the AudioDetector class.

        Args:
            mic (Microphone): Microphone instance for audio input. If None, a default Microphone will be initialized.
            confidence (float): Confidence level for detection. Default is 0.8 (80%).
            debounce_sec (float): Minimum seconds between repeated detections of the same keyword. Default is 2.0 seconds.

        Raises:
            ValueError: If the model information cannot be retrieved or if the model parameters are incomplete.
        """
        super().__init__()

        self.confidence = confidence

        self._debounce_sec = debounce_sec
        self._last_detected = {}

        model_info = self.get_model_info()
        if not model_info:
            raise ValueError("Failed to retrieve model information. Ensure the Edge Impulse service is running.")
        if model_info.frequency <= 0 or model_info.input_features_count <= 0:
            raise ValueError("Model parameters are missing or incomplete in the retrieved model information.")
        self.model_info = model_info

        self._mic = mic if mic else Microphone(sample_rate=model_info.frequency, channels=model_info.axis_count)
        self._mic_lock = threading.Lock()

        self._window_size = int(model_info.input_features_count / model_info.axis_count)
        self._duration = model_info.input_features_count / model_info.axis_count * model_info.interval_ms
        self._buffer = SlidingWindowBuffer(self._window_size, slide_amount=math.floor(self._window_size * 0.4))

        self.handlers = {}  # Dictionary to hold handlers for different keywords
        self.handlers_lock = threading.Lock()

    def on_detect(self, keyword: str, callback: Callable[[], None]):
        """Register a callback function to be invoked when a specific keyword is detected.

        Args:
            keyword (str): The keyword to check for in the classification results.
            callback (callable): a callback function to handle the keyword spotted.

        Raises:
            TypeError: If callback is not callable.
            ValueError: If callback accepts any argument.
        """
        if not inspect.isfunction(callback):
            raise TypeError("Callback must be a callable function.")
        sig_args = inspect.signature(callback).parameters
        if len(sig_args) > 0:
            raise ValueError("Callback must not accept any arguments.")

        keyword = keyword.lower()
        with self.handlers_lock:
            if keyword in self.handlers:
                logger.warning(f"Handler for keyword '{keyword}' already exists. Overwriting.")
            self.handlers[keyword] = callback

    def start(self):
        self._buffer.flush()
        with self._mic_lock:
            self._mic.start()

    def stop(self):
        with self._mic_lock:
            self._mic.stop()
        self._buffer.flush()

    def get_best_match(self, item: dict, confidence: int = None) -> tuple[str, float] | None:
        """Extract the best matched keyword from the classification results.

        Args:
        item (dict): The classification result from the inference.
        confidence (int): The confidence threshold for classification. If None, uses the instance's confidence level.

        Returns:
        tuple[str, float] | None: The best matched keyword and its confidence, or None if no match is found.
        """
        classification = _extract_classification(item, confidence or self.confidence)
        if not classification:
            return None

        best_matched_keyword = None
        best_matched_keyword_confidence = 0.0

        for possible_keyword in classification:
            keyword_name = possible_keyword["class_name"]
            keyword_confidence = float(possible_keyword["confidence"])

            if keyword_confidence > best_matched_keyword_confidence:
                best_matched_keyword = keyword_name
                best_matched_keyword_confidence = keyword_confidence

        return best_matched_keyword, best_matched_keyword_confidence

    @brick.loop
    def _read_mic_loop(self):
        try:
            with self._mic_lock:
                stream = self._mic.stream()
                for chunk in stream:
                    if chunk is None:
                        continue
                    self._buffer.push(chunk)
        except StopIteration:
            raise
        except Exception:
            logger.error("Error reading microphone")
            raise

    @brick.loop
    def _inference_loop(self):
        now = time.time()
        # If in debounce period, skip the inference
        if hasattr(self, "_debounce_until") and now < self._debounce_until:
            time.sleep(0.05)
            return

        features = self._buffer.pull()
        if len(features) == 0:
            return

        logger.debug(f"Processing sensor data with {len(features)} features.")
        try:
            ret = super().infer_from_features(features[: int(self.model_info.input_features_count)].tolist())
            spotted_keyword = self.get_best_match(ret)
            if spotted_keyword:
                keyword, confidence = spotted_keyword
                keyword = keyword.lower()
                logger.debug(f"Keyword '{keyword}' detected with confidence {confidence:2f}%.")
                callback = None
                with self.handlers_lock:
                    if keyword in self.handlers:
                        callback = self.handlers[keyword]

                if callback:
                    last_time = self._last_detected.get(keyword, 0)
                    if now - last_time >= self._debounce_sec:
                        self._last_detected[keyword] = now
                        logger.debug(f"Invoking callback for keyword '{keyword}'.")
                        callback()
                    else:
                        self._debounce_until = now + self._debounce_sec
        except Exception as e:
            logger.exception(f"Error running inference: {e}")
            time.sleep(1)  # Sleep briefly to avoid tight loop in case of errors


def _extract_classification(item, confidence: float) -> list | None:
    if not item:
        return None

    if "result" in item:
        class_results = item["result"]
        if class_results and "classification" in class_results:
            class_results = class_results["classification"]

            classification = []
            for class_name in class_results:
                class_confidence = float(class_results[class_name])

                if class_confidence < confidence:
                    continue

                class_confidence *= 100.0  # Convert to percentage
                obj = {
                    "class_name": class_name,
                    "confidence": f"{class_confidence:.2f}",
                }
                classification.append(obj)

            return classification

    return None
