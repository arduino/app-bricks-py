# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from PIL import Image

from arduino.app_internal.core import EdgeImpulseRunnerFacade
from arduino.app_internal.core.ei import brick_model_requires_softmax, compute_softmax_over_ei_classification
from arduino.app_utils import brick, Logger

logger = Logger("ImageClassification")


@brick
class ImageClassification(EdgeImpulseRunnerFacade):
    """Module for image analysis and content classification using machine learning.

    This module processes an input image and returns:
    - Corresponding class labels
    - Confidence scores for each classification

    """

    def __init__(self, confidence: float = 0.3) -> None:
        """Initialize the ImageClassification module.

        Args:
            confidence (float, optional): Minimum confidence threshold for
                classification results. Defaults to 0.3.
        """
        self._confidence = confidence
        super().__init__()
        self.confidence = confidence
        # Some models (e.g. EfficientNet-B4) return raw logits: apply a softmax only when the
        # configured model is flagged with `requires_softmax` in the models list.
        self.apply_softmax = brick_model_requires_softmax(self.__class__)

    def classify_from_file(self, image_path: str, confidence: float = None) -> dict | None:
        """Process a local image file to be classified.

        Args:
            image_path (str): Path to the image file on the local file system.
            confidence (float): Minimum confidence threshold for classification results. Default is None (use module defaults).

        Returns:
            dict: Classification results containing class names and confidence, or None if an error occurs.
        """
        if not image_path or image_path == "":
            return None
        ret = self._apply_softmax_if_required(super().infer_from_file(image_path))
        return self._extract_classification(ret, confidence or self._confidence)

    def classify(self, image_bytes: bytes | Image.Image, image_type: str = "jpg", confidence: float = None) -> dict | None:
        """Process an in-memory image to be classified.

        Args:
            image_bytes: Can be raw bytes (e.g., from a file or stream) or a preloaded PIL image.
            image_type (str): The image format ('jpg', 'jpeg', or 'png'). Required if using raw bytes. Defaults to 'jpg'.
            confidence (float): Minimum confidence threshold for classification results. Default is None (use module defaults).

        Returns:
            dict: Classification results containing class names and confidence, or None if an error occurs.
        """
        if not image_bytes or not image_type:
            return None
        ret = self._apply_softmax_if_required(super().infer_from_image(image_bytes, image_type))
        return self._extract_classification(ret, confidence or self._confidence)

    def process(self, item: str | dict) -> dict | None:
        """Process an item to classify objects in an image.

        This method supports two input formats:
        - A string path to a local image file.
        - A dictionary containing raw image bytes under the 'image' key, and optionally an 'image_type' key (e.g., 'jpg', 'png').

        Args:
            item: A file path (str) or a dictionary with the 'image' and 'image_type' keys (dict).
                'image_type' is optional while 'image' contains image as bytes.

        Returns:
            dict: Classification results or None if an error occurs.
        """
        return self._extract_classification(self._apply_softmax_if_required(super().process(item)))

    def _apply_softmax_if_required(self, item: dict | None) -> dict | None:
        """Normalize raw runner logits to probabilities when the configured model requires it.

        Args:
            item: The raw Edge Impulse runner response.

        Returns:
            dict | None: The same response, with the classification scores replaced by their softmax
            when `apply_softmax` is enabled; the response untouched otherwise.
        """
        if not self.apply_softmax or not item or not isinstance(item, dict):
            return item
        result = item.get("result")
        if not isinstance(result, dict) or not result.get("classification"):
            return item
        # Softmax over the full logit vector, so probabilities keep the network calibration.
        result["classification"] = compute_softmax_over_ei_classification(result["classification"])
        return item
