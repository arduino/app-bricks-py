# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The model a brick runs on the Edge Impulse inference service.

`EdgeImpulseModel` is the base of every brick running an Edge Impulse model, whatever its input: it resolves
the model from the app configuration, opens it on the service with the confidence of the brick, keeps the
connection and reconnects when the service goes away. The bricks working on a camera build on
`VideoInference`, the ones working on single images call `infer` directly.
"""

import os
import threading

import numpy as np

from arduino.app_internal import ei_inference
from arduino.app_internal.core.module import get_brick_config, get_brick_configured_model, load_model_list
from arduino.app_internal.ei_inference import InferenceClient, Result, ServerError, model_name_from_path
from arduino.app_utils import Logger

logger = Logger("EdgeImpulseModel")


class EdgeImpulseModel:
    """A model of the Edge Impulse inference service, `arduino:edge_impulse`, and the connection to it.

    The model is the one configured for the brick in the app, through the class attribute `MODEL_VARIABLE`, the
    app variable the CLI sets to the `.eim` path, with the models list as the fallback; a subclass names it. The
    confidence is the score the results must reach: the service applies it and returns nothing below it.
    Subclasses set their other model parameters in `_configure`, called every time the model is opened.
    """

    MODEL_VARIABLE = ""  # the app variable naming the .eim file of the brick, set by each brick class

    _RETRY_SEC = 2.0  # seconds between attempts to reach the inference service

    def __init__(self, confidence: float | None = None) -> None:
        """Resolve the model configured for the brick, without opening it yet.

        Args:
            confidence (float | None): Score the results must reach, between 0 and 1. None takes everything the
                model reports.

        Raises:
            RuntimeError: If no model is configured.
        """
        self._model = self._configured_model()
        self._confidence = confidence
        self._client: InferenceClient | None = None
        self._client_lock = threading.Lock()

    @property
    def model(self) -> str:
        """The name of the model on the inference service."""
        return self._model

    @property
    def confidence(self) -> float | None:
        """The score the results must reach, None when everything the model reports is taken; assign to change it."""
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        self.override_threshold(value)

    @classmethod
    def _configured_model(cls) -> str:
        """The model configured for the brick, as the inference service names it.

        The app CLI sets the model variable when the app selects a model; for the default model of the board
        the path comes from the models list, through the model the brick configuration selects.
        """
        if not cls.MODEL_VARIABLE:
            raise RuntimeError(f"{cls.__name__} names no model variable.")
        model_path = os.getenv(cls.MODEL_VARIABLE) or cls._default_model_path()
        if not model_path:
            raise RuntimeError(f"No model configured: select a model for the brick in the app or set the {cls.MODEL_VARIABLE} variable.")
        try:
            return model_name_from_path(model_path)
        except ValueError as e:
            raise RuntimeError(str(e)) from e

    @classmethod
    def _default_model_path(cls) -> str | None:
        """The model path the models list configures for this brick and its default model, None when unknown."""
        brick_config = get_brick_config(cls)
        if not brick_config:
            return None
        brick_id = brick_config.get("id")
        model_id = get_brick_configured_model(brick_id, brick_config) if brick_id else None
        models = load_model_list() or {}
        entry = models.get(model_id) if model_id else None
        if entry is None:
            return None
        for brick in entry.bricks:
            if brick.id == brick_id and cls.MODEL_VARIABLE in brick.model_configuration:
                logger.info(f"Model '{model_id}' selected for the brick")
                return brick.model_configuration[cls.MODEL_VARIABLE]
        return None

    def override_threshold(self, value: object) -> None:
        """Override the confidence threshold of the detections.

        Args:
            value (float): The new value for the threshold in the range [0.0, 1.0].

        Raises:
            TypeError: If the value is not a number.
        """
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("Invalid types for value.")
        logger.info(f"Overriding detection threshold. New confidence: {value}")
        self._confidence = float(value)
        client = self._connected()
        if client is None:
            return  # set when the model is opened
        try:
            client.set_confidence(self._confidence)
        except (ServerError, TimeoutError, ConnectionError) as e:
            logger.warning(f"The confidence could not be set on the model, it will be on the next connection: {e}")

    def infer(
        self, image: np.ndarray, confidence: float | None = None, color: str = "bgr", timeout: float | None = 30.0, keep_frame: bool = False
    ) -> Result:
        """Run the model on one image, opening it on the service if needed.

        Args:
            image (np.ndarray): HxWx3 uint8 image.
            confidence (float | None): Score the results of this call must reach, the confidence of the brick when None.
            color (str): Channel order of the image, "bgr" (OpenCV) or "rgb" (PIL).
            timeout (float | None): Seconds to wait for the model and its result, None waits indefinitely.
            keep_frame (bool): Keep a copy of the image in the result.

        Returns:
            Result: The boxes, tracks and class scores reaching the confidence, in the coordinates of the image.

        Raises:
            ServerError: If the service refuses the model or the confidence.
            OSError: If the service is not reachable.
            TimeoutError: If the model or its result do not come in time.
            ConnectionError: If the connection is lost, the next call opens the model again.
        """
        client = self._connected() or self._open(timeout)
        try:
            if confidence is None or confidence == client.confidence:
                return client.infer(image, color=color, keep_frame=keep_frame, timeout=timeout)
            client.set_confidence(confidence)  # for this call only
            try:
                return client.infer(image, color=color, keep_frame=keep_frame, timeout=timeout)
            finally:
                if not client.closed:
                    client.set_confidence(self._confidence)
        except ConnectionError:
            self._close()
            raise

    def _open(self, timeout: float | None = None) -> InferenceClient:
        """Open the model on the inference service and configure it, waiting up to `timeout` seconds for it to load.

        Raises:
            ServerError: If the service refuses the model.
            OSError: If the service is not reachable, or the model is not ready in time (TimeoutError).
        """
        client = InferenceClient(self._model, ei_inference.DEFAULT_SOCKET_PATH, open_timeout=timeout, confidence=self._confidence)
        try:
            self._configure(client)
        except BaseException:
            client.close()
            raise
        with self._client_lock:
            self._client = client
        logger.info(f"Connected to the inference service, model '{client.model}' input {client.input_size[0]}x{client.input_size[1]}")
        return client

    def _configure(self, client: InferenceClient) -> None:
        """Set the parameters of the model every time it is opened; the base class has none beyond the confidence."""

    def _connect(self, running: threading.Event) -> InferenceClient | None:
        """The open connection to the model, or a new one once the service is ready, retrying while `running` is set."""
        client = self._connected()
        if client is not None:
            return client
        while running.is_set():
            try:
                return self._open()
            except ServerError as e:
                logger.error(f"The inference service refused model '{self._model}' ({e.code}): {e}. Retrying...")
            except OSError:
                logger.debug(f"Waiting for the inference service at {ei_inference.DEFAULT_SOCKET_PATH}. Retrying...")
            running.wait(self._RETRY_SEC)
        return None

    def _connected(self) -> InferenceClient | None:
        """The open connection to the model, None while there is none."""
        with self._client_lock:
            client = self._client
        return client if client is not None and not client.closed else None

    def _close(self) -> None:
        """Close the connection, the service releases the model."""
        with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            client.close()
