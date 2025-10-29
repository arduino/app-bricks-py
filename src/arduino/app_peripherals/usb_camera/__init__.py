# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import io
import warnings
from PIL import Image
from arduino.app_peripherals.camera import Camera, CameraReadError as CRE, CameraOpenError as COE
from arduino.app_peripherals.camera.v4l_camera import V4LCamera
from arduino.app_utils.image import letterboxed, compressed_to_png
from arduino.app_utils import Logger

logger = Logger("USB Camera")

CameraReadError = CRE

CameraOpenError = COE


@warnings.deprecated("Use the Camera peripheral instead of this one")
class USBCamera:
    """Represents an input peripheral for capturing images from a USB camera device.
    This class uses OpenCV to interface with the camera and capture images.
    """

    def __init__(
        self,
        camera: int = 0,
        resolution: tuple[int, int] = None,
        fps: int = 10,
        compression: bool = False,
        letterbox: bool = False,
    ):
        """Initialize the USB camera.

        Args:
            camera (int): Camera index (default is 0 - index is related to the first camera available from /dev/v4l/by-id devices).
            resolution (tuple[int, int]): Resolution as (width, height). If None, uses default resolution.
            fps (int): Frames per second for the camera. If None, uses default FPS.
            compression (bool): Whether to compress the captured images. If True, images are compressed to PNG format.
            letterbox (bool): Whether to apply letterboxing to the captured images.
        """
        self.compression = compression

        pipe = None
        if compression:
            pipe = compressed_to_png()
        if letterbox:
            pipe = pipe | letterboxed() if pipe else letterboxed()

        self._wrapped_camera = V4LCamera(camera, resolution, fps, pipe)

    def capture(self) -> Image.Image | None:
        """Captures a frame from the camera, blocking to respect the configured FPS.

        Returns:
            PIL.Image.Image | None: The captured frame as a PIL Image, or None if no frame is available.
        """
        image_bytes = self._wrapped_camera.capture()
        if image_bytes is None:
            return None
        try:
            if self.compression:
                # If compression is enabled, we expect image_bytes to be in PNG format
                return Image.open(io.BytesIO(image_bytes))
            else:
                return Image.fromarray(image_bytes)
        except Exception as e:
            logger.exception(f"Error converting captured bytes to PIL Image: {e}")
            return None

    def capture_bytes(self) -> bytes | None:
        """Captures a frame from the camera and returns its raw bytes, blocking to respect the configured FPS.

        Returns:
            bytes | None: The captured frame as a bytes array, or None if no frame is available.
        """
        frame = self._wrapped_camera.capture()
        if frame is None:
            return None
        return frame.tobytes()

    def start(self):
        """Starts the camera capture."""
        self._wrapped_camera.start()

    def stop(self):
        """Stops the camera and releases its resources."""
        self._wrapped_camera.stop()

    def produce(self):
        """Alias for capture method."""
        return self.capture()
