# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import brick, Logger
from arduino.app_internal.core import EdgeImpulseRunnerFacade

logger = Logger("ObjectDetection")


@brick
class ObjectDetection(EdgeImpulseRunnerFacade):
    """Module for object detection in images using a specified machine learning model.

    This module processes an input image and returns:
    - Bounding boxes for detected objects
    - Corresponding class labels
    - Confidence scores for each detection
    """

    def __init__(self, confidence: float = 0.3):
        self.confidence = confidence
        super().__init__()

    def detect_from_file(self, image_path: str, confidence: float = None) -> dict | None:
        """Process a local image file to detect and identify objects.

        Args:
            image_path: Path to the image file on the local file system.
            confidence: Minimum confidence threshold for detections. Default is None (use module defaults).

        Returns:
            dict: Detection results containing class names, confidence, and bounding boxes.
        """
        if not image_path:
            return None
        ret = super().infer_from_file(image_path)
        return self._extract_detection(ret, confidence)

    def detect(self, image_bytes, image_type: str = "jpg", confidence: float = None) -> dict:
        """Process an in-memory image to detect and identify objects.

        Args:
            image_bytes: Can be raw bytes (e.g., from a file or stream) or a preloaded PIL image.
            image_type: The image format ('jpg', 'jpeg', or 'png'). Required if using raw bytes. Defaults to 'jpg'.
            confidence: Minimum confidence threshold for detections. Default is None (use module defaults).

        Returns:
            dict: Detection results containing class names, confidence, and bounding boxes.
        """
        if not image_bytes or not image_type:
            return None
        ret = super().infer_from_image(image_bytes, image_type)
        return self._extract_detection(ret, confidence)

    def _extract_detection(self, item, confidence: float = None):
        if not item:
            return None

        if "result" in item:
            results = item["result"]
            if results and "bounding_boxes" in results:
                results = results["bounding_boxes"]
            else:
                return None

            detection = []
            for result in results:
                if "label" in result and "value" in result:
                    class_name = result["label"]
                    class_confidence = result["value"]

                    if class_confidence < (confidence or self.confidence):
                        continue

                    class_confidence = class_confidence * 100.0
                    obj = {
                        "class_name": class_name,
                        "confidence": f"{class_confidence:.2f}",
                        "bounding_box_xyxy": [
                            float(result["x"]),
                            float(result["y"]),
                            float(result["x"] + result["width"]),
                            float(result["y"] + result["height"]),
                        ],
                    }
                    detection.append(obj)

            return {"detection": detection}

        return None

    def process(self, item):
        """Process an item to detect objects in an image.

        This method supports two input formats:
        - A string path to a local image file.
        - A dictionary containing raw image bytes under the 'image' key, and optionally an 'image_type' key (e.g., 'jpg', 'png').

        Args:
            item: A file path (str) or a dictionary with the 'image' and 'image_type' keys (dict).
                'image_type' is optional while 'image' contains image as bytes.
        """
        return self._extract_detection(super().process(item))
