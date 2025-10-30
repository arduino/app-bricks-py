# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import requests
import io
from arduino.app_internal.core import load_brick_compose_file, resolve_address
from arduino.app_utils.image import get_image_bytes, get_image_type
from arduino.app_utils import Logger, HttpClient

logger = Logger(__name__)


class EdgeImpulseModelInfo:
    """Class to hold Edge Impulse model information."""

    def __init__(self, model_info: dict):
        """Initialize the EdgeImpulseModelInfo with model information."""
        if not model_info:
            raise ValueError("Model information cannot be empty.")

        if "project" in model_info:
            project = model_info["project"]
            self.name = project.get("name", "Unknown Model")
            self.project_id = project.get("id", -1)

        if "modelParameters" in model_info:
            model_params = model_info["modelParameters"]
            self.model_type = model_params.get("model_type", "Unknown Model Type")
            self.axis_count = int(model_params.get("axis_count", 1))
            self.frequency = int(model_params.get("frequency", -1))
            self.image_input_height = int(model_params.get("image_input_height", -1))
            self.image_input_width = int(model_params.get("image_input_width", -1))
            self.input_features_count = int(model_params.get("input_features_count", -1))
            self.label_count = int(model_params.get("label_count", -1))
            self.labels = model_params.get("labels", [])
            self.interval_ms = float(model_params.get("interval_ms", -1))
            self.thresholds = model_params["thresholds"]


class EdgeImpulseRunnerFacade:
    """Facade for Edge Impulse Object Detection and Classification."""

    def __init__(self):
        """Initialize the EdgeImpulseRunnerFacade with the API path.

        Raises:
            RuntimeError: If the Edge Impulse runner address cannot be resolved.
        """
        self.url = self._get_ei_url()
        logger.info(f"[{self.__class__.__name__}] URL: {self.url}")

    def infer_from_file(self, image_path: str) -> dict | None:
        if not image_path or image_path == "":
            return None
        with open(image_path, "rb") as f:
            try:
                return self.infer_from_image(image_bytes=f.read(), image_type=image_path.split(".")[-1])
            except Exception as e:
                logger.error(f"Error: {e}")
                return None

    def infer_from_image(self, image_bytes, image_type: str = "jpg") -> dict | None:
        image_bytes = get_image_bytes(image_bytes)
        if not image_bytes or not image_type:
            return None

        if image_type not in ["jpg", "jpeg", "png"]:
            logger.warning(f"[{self.__class__.__name__}] Invalid image type: {image_type}. Discarding image.")
            return None
        elif image_type == "jpg":
            image_type = "jpeg"

        try:
            logger.debug(f"[{self.__class__.__name__}] Detecting image of type: {image_type} -> {len(image_bytes)} bytes")

            files = {"file": (f"image.{image_type}", io.BytesIO(image_bytes), f"image/{image_type}")}
            response = requests.post(f"{self.url}/api/image", files=files)

        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Error: {e}")
            return None

        # Check the response
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"[{self.__class__}] error: {response.status_code}. Message: {response.text}")
            return None

    def process(self, item):
        """Process an item to detect objects in an image.

        Args:
            item: A file path (str) or a dictionary with the 'image' and 'image_type' keys (dict).
                'image_type' is optional while 'image' contains image as bytes.
        """
        try:
            if isinstance(item, str):
                # Use this like a file path
                with open(item, "rb") as f:
                    return self.infer_from_image(f.read(), item.split(".")[-1])
            elif isinstance(item, dict) and "image" in item and item["image"] != "":
                image = item["image"]
                if "image_type" in item and item["image_type"] != "":
                    image_type = item["image_type"]
                else:
                    image_type = get_image_type(image)

                if image_type is None:
                    logger.debug(f"[{self.__class__}] Discarding not supported file type")
                    return None

                return self.infer_from_image(image, image_type.lower())
            return item  # No processing needed
        except FileNotFoundError:
            logger.error(f"[{self.__class__}] File not found: {item}")
        except Exception as e:
            logger.error(f"[{self.__class__}] Error processing file {item}: {e}")
        return None

    @classmethod
    def infer_from_features(cls, features: list) -> dict | None:
        """
        Infer from features using the Edge Impulse API.

        Args:
            cls: The class method caller.
            features (list): A list of features to send to the Edge Impulse API.

        Returns:
            dict | None: The response from the Edge Impulse API as a dictionary, or None if an error occurs.
        """
        try:
            url = cls._get_ei_url()
            model_info = cls.get_model_info(url)
            features = features[: int(model_info.input_features_count)]

            response = requests.post(f"{url}/api/features", json={"features": features})
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"[{cls.__name__}] error: {response.status_code}. Message: {response.text}")
                return None
        except Exception as e:
            logger.error(f"[{cls.__name__}] Error: {e}")
            return None

    @classmethod
    def get_model_info(cls, url: str = None) -> EdgeImpulseModelInfo | None:
        """Get model information from the Edge Impulse API.

        Args:
            cls: The class method caller.
            url (str): The base URL of the Edge Impulse API. If None, it will be determined automatically.

        Returns:
            model_info (EdgeImpulseModelInfo | None): An instance of EdgeImpulseModelInfo containing model details, None if an error occurs.
        """
        if not url:
            url = cls._get_ei_url()

        http_client = HttpClient(total_retries=6)  # Initialize the HTTP client with retry logic
        try:
            response = http_client.request_with_retry(f"{url}/api/info")
            if response.status_code == 200:
                logger.debug(f"[{cls.__name__}] Fetching model info from {url}/api/info -> {response.status_code} {response.json}")
                return EdgeImpulseModelInfo(response.json())
            else:
                logger.warning(f"[{cls}] Error fetching model info: {response.status_code}. Message: {response.text}")
                return None
        except Exception as e:
            logger.error(f"[{cls}] Error fetching model info: {e}")
            return None
        finally:
            http_client.close()  # Close the HTTP client session

    @staticmethod
    def parse_model_info_message(model_info: dict) -> EdgeImpulseModelInfo | None:
        """Parse Edge Impulse model definition message.

        Args:
            model_info (dict): A dictionary containing model information.

        Returns:
            EdgeImpulseModelInfo | None: An instance of EdgeImpulseModelInfo if parsing is successful, None otherwise.
        """
        return EdgeImpulseModelInfo(model_info)

    def _extract_classification(self, item: dict, confidence: float = 0.0):
        """Extract classification results from the item.

        Args:
            item (dict): The item containing classification results.
            confidence (float): The minimum confidence level for classification results.

        Returns:
            dict: A dictionary containing classification results or None if no valid results are found.
        """
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

                return {"classification": classification}
            else:
                return None

        return None

    def _extract_anomaly_score(self, item: dict):
        """Extract anomaly score for anomaly detection use case.

        Args:
            item (dict): The item containing anomaly score.

        Returns:
            float | None: The anomaly score or None if not found.
        """
        if not item:
            return None

        if "result" in item:
            class_results = item["result"]
            if class_results and "anomaly" in class_results:
                return class_results["anomaly"]

        return None

    @classmethod
    def _get_ei_url(cls):
        infra = load_brick_compose_file(cls)
        if not infra or "services" not in infra:
            raise RuntimeError("Cannot load Brick Compose file to resolve Edge Impulse runner address.")
        host = None
        for k, v in infra["services"].items():
            host = k
            break
        if not host:
            raise RuntimeError("Cannot resolve Edge Impulse runner address from Brick Compose file.")
        addr = resolve_address(host)
        if not addr:
            raise RuntimeError("Host address resolution failed for Edge Impulse runner.")
        return f"http://{addr}:1337"
