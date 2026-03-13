# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""GStreamer input source."""

import queue
from typing import Callable

import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from aihub.base import InputSource
from aihub.logging import logger


# Default configuration
DEFAULT_SOURCE = "v4l2src device=/dev/video0"
DEFAULT_VIDEO_WIDTH = 1024
DEFAULT_VIDEO_HEIGHT = 768


class GStreamerInput(InputSource):
    """
    GStreamer-based frame input source.

    Captures frames from a GStreamer pipeline and delivers them to the
    provided callback.
    """

    PIPELINE_TEMPLATE = (
        "{source} ! "
        "videoconvert ! "
        "videoscale ! "
        "video/x-raw,format=RGB,width={width},height={height} ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "appsink name=appsink drop=true sync=false max-buffers=1 emit-signals=true"
    )

    def __init__(
        self,
        on_frame_cb: Callable[[np.ndarray], None],
        source: str = DEFAULT_SOURCE,
        width: int = DEFAULT_VIDEO_WIDTH,
        height: int = DEFAULT_VIDEO_HEIGHT,
        **kwargs,
    ):
        """
        Initialize GStreamer input.

        Args:
            on_frame_cb: Callback for each frame.
            source: GStreamer source element string (e.g., "v4l2src device=/dev/video0").
            width: Video width.
            height: Video height.
        """
        self._on_frame_cb = on_frame_cb

        self._pipeline_str = self.PIPELINE_TEMPLATE.format(
            source=source,
            width=width,
            height=height,
        )
        self._pipeline = None
        self._appsink = None

        self._running = False
        self._frame_queue = queue.Queue(maxsize=4)

        # Cached frame metadata (set on first frame for performance)
        self._frame_width = None
        self._frame_height = None
        self._rowstride = None
        self._has_padding = None

    def start(self) -> None:
        """Start the GStreamer pipeline and process frames."""
        Gst.init(None)

        self._pipeline = Gst.parse_launch(self._pipeline_str)

        self._appsink = self._pipeline.get_by_name("appsink")
        if not self._appsink:
            raise RuntimeError(
                "Could not find appsink element named 'appsink'. "
                "Ensure your pipeline includes 'appsink name=appsink'."
            )
        self._appsink.connect("new-sample", self._on_new_sample)

        logger.info("Starting GStreamer pipeline...")
        self._pipeline.set_state(Gst.State.PLAYING)
        self._running = True

        try:
            while self._running:
                try:
                    frame = self._frame_queue.get(timeout=1.0)
                    self._on_frame_cb(frame)
                except queue.Empty:
                    continue
        finally:
            self._pipeline.set_state(Gst.State.NULL)
            self._running = False

    def stop(self) -> None:
        """Stop the GStreamer pipeline."""
        self._running = False

    def _on_new_sample(self, sink) -> int:
        """GStreamer callback for new samples."""
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()

        # Cache dimensions on first frame (caps don't change during stream)
        if self._frame_width is None:
            caps = sample.get_caps().get_structure(0)
            self._frame_width = caps.get_value("width")
            self._frame_height = caps.get_value("height")

        # Map buffer memory as read-only
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK

        try:
            h, w = self._frame_height, self._frame_width

            # Cache rowstride on first frame
            if self._rowstride is None:
                self._rowstride = mapinfo.size // h
                self._has_padding = self._rowstride != w * 3

            if self._has_padding:
                # The buffer has padding bytes at the end of each row
                arr = np.frombuffer(
                    mapinfo.data, dtype=np.uint8, count=h * self._rowstride
                )
                arr = np.lib.stride_tricks.as_strided(
                    arr, shape=(h, w, 3), strides=(self._rowstride, 3, 1)
                )
            else:
                # No padding, reshape directly
                arr = np.frombuffer(
                    mapinfo.data, dtype=np.uint8, count=h * w * 3
                )
                arr = arr.reshape((h, w, 3))
            
            arr = arr.copy()  # Ensure data is owned by numpy as it will be unmapped
        finally:
            buf.unmap(mapinfo)

        try:
            self._frame_queue.put_nowait(arr)
        except queue.Full:
            pass

        return Gst.FlowReturn.OK

    @property
    def is_running(self) -> bool:
        return self._running
