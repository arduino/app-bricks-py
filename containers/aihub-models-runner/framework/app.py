# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import time
from typing import Callable, List

import numpy as np

from aihub.logging import setup_logging, logger


class AIHubApp:
    """
    Main application class that orchestrates input sources and output sinks.

    Example usage:
        def my_inference(frame: np.ndarray) -> tuple[np.ndarray, dict]:
            # ... your inference code here ...
            return annotated_frame, {}

        app = AIHubApp(
            inference_cb=my_inference,
            input_type="gstreamer",
            output_types=["mjpeg", "websocket"],
            gst_source="v4l2src device=/dev/video0",
            gst_width=1024,
            gst_height=768,
            ws_output_port=5001,
            mjpeg_port=5002,
        )
        app.run()
    """

    def __init__(
        self,
        inference_cb: Callable[[np.ndarray], tuple[np.ndarray, dict]],
        input_type: str = "gstreamer",
        output_types: List[str] = ["mjpeg"],
        **kwargs,
    ):
        """
        Initialize AIHubApp.

        Args:
            inference_cb: User callback for frame processing. Receives RGB frame,
                returns processed RGB frame and metadata.
            input_type: Input source type ("gstreamer" or "websocket").
            output_types: List of output sink types (accepted values are "mjpeg",
                "websocket"). Defaults to ["mjpeg"].
            **kwargs: Forwarded to input/output constructors based on prefix:
                - gst_* -> GStreamer input
                - ws_input_* -> WebSocket input
                - mjpeg_* -> MJPEG output
                - ws_output_* -> WebSocket output
        """
        self._inference_cb = inference_cb
        self._input_type = input_type
        self._output_types = output_types
        self._kwargs = kwargs

        self._input_source = None
        self._output_sinks: List = []
        self._is_running = False

        # Setup logging
        self._verbose = kwargs.get("verbose", False)
        setup_logging(self._verbose)

        # FPS tracking state
        self._fps_start_time = time.perf_counter()
        self._fps_frame_count = 0

    def run(self) -> None:
        """
        Start the application and block until stopped by Ctrl+C.
        """
        try:
            self._setup()
            self._is_running = True

            for sink in self._output_sinks:
                sink.start()

            if self._input_source:
                self._input_source.start()  # This call blocks until stop() is called

        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(e)
        finally:
            logger.info("Shutting down...")
            self._cleanup()

    def stop(self) -> None:
        """Signal the application to stop."""
        self._is_running = False
        if self._input_source:
            self._input_source.stop()

    def _setup(self) -> None:
        """Initialize input source and output sinks based on configuration."""
        # Setup input source
        if self._input_type == "gstreamer":
            input_kwargs = self._extract_kwargs("gst_")
        elif self._input_type == "websocket":
            input_kwargs = self._extract_kwargs("ws_input_")
        else:
            input_kwargs = {}

        InputClass = self._get_input_class(self._input_type)
        self._input_source = InputClass(
            on_frame_cb=self._frame_callback,
            **input_kwargs,
        )

        # Setup output sinks
        for output_type in self._output_types:
            if output_type == "mjpeg":
                output_kwargs = self._extract_kwargs("mjpeg_")
            elif output_type == "websocket":
                output_kwargs = self._extract_kwargs("ws_output_")
            else:
                output_kwargs = {}

            OutputClass = self._get_output_class(output_type)
            sink = OutputClass(**output_kwargs)
            self._output_sinks.append(sink)

    def _extract_kwargs(self, prefix: str) -> dict:
        """Extract kwargs with given prefix, stripping the prefix from keys."""
        result = {}
        for key, value in self._kwargs.items():
            if key.startswith(prefix):
                result[key.removeprefix(prefix)] = value
        return result

    def _get_input_class(self, input_type: str):
        """Lazily load and return the input class for the given type."""
        if input_type == "gstreamer":
            from aihub.gstreamer.input import GStreamerInput
            return GStreamerInput
        elif input_type == "websocket":
            from aihub.websocket.input import WebSocketInput
            return WebSocketInput
        else:
            raise ValueError(f"Unknown input type: {input_type}")

    def _get_output_class(self, output_type: str):
        """Lazily load and return the output class for the given type."""
        if output_type == "mjpeg":
            from aihub.mjpeg.output import MJPEGOutput
            return MJPEGOutput
        elif output_type == "websocket":
            from aihub.websocket.output import WebSocketOutput
            return WebSocketOutput
        else:
            raise ValueError(f"Unknown output type: {output_type}")

    def _frame_callback(self, frame: np.ndarray):
        """
        Internal callback that wraps user callback and distributes to outputs.
        """
        processed_frame, metadata = self._inference_cb(frame)

        for sink in self._output_sinks:
            try:
                sink.send_frame(processed_frame, metadata)
            except Exception as e:
                logger.warning(f"Output sink error: {e}")

        # FPS tracking
        self._fps_frame_count += 1
        cur_time = time.perf_counter()
        elapsed = cur_time - self._fps_start_time
        if elapsed >= 1.0:
            fps = self._fps_frame_count / elapsed
            logger.debug(f"FPS: {fps:.1f}")
            self._fps_start_time = cur_time
            self._fps_frame_count = 0

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._is_running = False

        if self._input_source:
            try:
                self._input_source.stop()
            except Exception as e:
                logger.warning(f"Error stopping input: {e}")

        for sink in self._output_sinks:
            try:
                sink.stop()
            except Exception as e:
                logger.warning(f"Error stopping output: {e}")
