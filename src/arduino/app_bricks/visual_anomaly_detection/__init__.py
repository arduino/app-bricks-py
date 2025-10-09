# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_internal.core import EdgeImpulseRunnerFacade
from arduino.app_utils import brick, Logger

logger = Logger("VisualAnomalyDetection")


@brick
class VisualAnomalyDetection(EdgeImpulseRunnerFacade):
    """Module for detecting **visual anomalies** in images using a specified model.

    This module processes an input image and returns:
    - Global anomaly metrics (`anomaly_max_score`, `anomaly_mean_score`), when available.
    - A list of localized anomaly detections with label, score, and bounding boxes.

    Notes:
        - Bounding boxes are returned as `[x_min, y_min, x_max, y_max]` (float).
        - Methods return `None` when input is invalid or when the model output
          does not contain expected anomaly fields.
    """

    def __init__(self):
        super().__init__()

    def detect_from_file(self, image_path: str) -> dict:
        """Process a local image file to detect anomalies.

        Args:
            image_path (str): Path to the image file on the local file system.

        Returns:
            dict | None: A dictionary with anomaly information, or `None` on error.
                Example successful payload:
                {
                    "anomaly_max_score": <float>,              # optional, if provided by model
                    "anomaly_mean_score": <float>,             # optional, if provided by model
                    "detection": [
                        {
                            "class_name": <str>,
                            "score": <float>,                   # anomaly score for this region
                            "bounding_box_xyxy": [x1, y1, x2, y2]
                        },
                        ...
                    ]
                }

            - Returns `None` if `image_path` is falsy or if the inference result
              does not include anomaly data.
        """
        if not image_path:
            return None
        ret = super().infer_from_file(image_path)
        return self._extract_anomalies(ret)

    def detect(self, image_bytes, image_type: str = "jpg") -> dict:
        """Process an in-memory image to detect anomalies.

        Args:
            image_bytes: Raw image bytes (e.g., from a file or camera) or a PIL Image.
            image_type (str): Image format ('jpg', 'jpeg', 'png'). Required when passing raw bytes.
                              Defaults to 'jpg'.

        Returns:
            dict | None: A dictionary with anomaly information, or `None` on error.
                See `detect_from_file` for the response schema.

            - Returns `None` if `image_bytes` or `image_type` is missing/invalid.
        """
        if not image_bytes or not image_type:
            return None
        ret = super().infer_from_image(image_bytes, image_type)
        return self._extract_anomalies(ret)

    def _extract_anomalies(self, item):
        if not item:
            return None
        out_result = {}

        if "result" in item:
            results = item["result"]
            if "visual_anomaly_max" in results and "visual_anomaly_mean" in results:
                out_result["anomaly_max_score"] = results["visual_anomaly_max"]
                out_result["anomaly_mean_score"] = results["visual_anomaly_mean"]

            if results and "visual_anomaly_grid" in results:
                results = results["visual_anomaly_grid"]
            else:
                return None

            anomalies = []
            for result in results:
                if "label" in result and "value" in result:
                    class_name = result["label"]
                    score = result["value"]
                    obj = {
                        "class_name": class_name,
                        "score": score,
                        "bounding_box_xyxy": [
                            float(result["x"]),
                            float(result["y"]),
                            float(result["x"] + result["width"]),
                            float(result["y"] + result["height"]),
                        ],
                    }
                    anomalies.append(obj)

            out_result["detection"] = anomalies

            return out_result

        return None

    def process(self, item):
        """Process an item to detect anomalies (file path or in-memory image).

        This method supports two input formats:
        - A string path to a local image file.
        - A dictionary containing raw image bytes under the `'image'` key, and
          optionally an `'image_type'` key (e.g., `'jpg'`, `'png'`).

        Args:
            item (str | dict): File path or a dict with `'image'` (bytes/PIL) and
                optional `'image_type'` (str).

        Returns:
            dict | None: Normalized anomaly payload or `None` if an error occurs or
            the result lacks anomaly data.

        Example:
            process("path/to/image.jpg")
            # or
            process({"image": image_bytes, "image_type": "png"})
        """
        return self._extract_anomalies(super().process(item))
