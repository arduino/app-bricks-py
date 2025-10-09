# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time
import cv2
import io
import os
import re
from PIL import Image
from arduino.app_utils import Logger

logger = Logger("USB Camera")


class CameraReadError(Exception):
    """Exception raised when the specified camera cannot be found."""

    pass


class CameraOpenError(Exception):
    """Exception raised when the camera cannot be opened."""

    pass


class USBCamera:
    """Represents an input peripheral for capturing images from a USB camera device.
    This class uses OpenCV to interface with the camera and capture images.
    """

    def __init__(
        self,
        camera: int = 0,
        resolution: tuple[int, int] = (None, None),
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
        video_devices = self._get_video_devices_by_index()
        if camera in video_devices:
            self.camera = int(video_devices[camera])
        else:
            raise CameraOpenError(
                f"Not available camera at index 0 {camera}. Verify the connected cameras and fi cameras are listed "
                f"inside devices listed here: /dev/v4l/by-id"
            )

        self.resolution = resolution
        self.fps = fps
        self.compression = compression
        self.letterbox = letterbox
        self._cap = None
        self._cap_lock = threading.Lock()
        self._last_capture_time_monotonic = time.monotonic()
        if self.fps > 0:
            self.desired_interval = 1.0 / self.fps
        else:
            # Capture as fast as possible
            self.desired_interval = 0

    def capture(self) -> Image.Image | None:
        """Captures a frame from the camera, blocking to respect the configured FPS.

        Returns:
            PIL.Image.Image | None: The captured frame as a PIL Image, or None if no frame is available.
        """
        image_bytes = self._extract_frame()
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
        frame = self._extract_frame()
        if frame is None:
            return None
        return frame.tobytes()

    def _extract_frame(self) -> cv2.typing.MatLike | None:
        # Without locking, 'elapsed_time' could be a stale value but this scenario is unlikely to be noticeable in
        # practice, also its effects would disappear in the next capture. This optimization prevents us from calling
        # time.sleep while holding a lock.
        current_time_monotonic = time.monotonic()
        elapsed_time = current_time_monotonic - self._last_capture_time_monotonic
        if elapsed_time < self.desired_interval:
            sleep_duration = self.desired_interval - elapsed_time
            time.sleep(sleep_duration)  # Keep time.sleep out of the locked section!

        with self._cap_lock:
            if self._cap is None:
                return None

            ret, bgr_frame = self._cap.read()
            if not ret:
                raise CameraReadError(f"Failed to read from camera {self.camera}.")
            self._last_capture_time_monotonic = time.monotonic()
            if bgr_frame is None:
                # No frame available, skip this iteration
                return None

        try:
            if self.letterbox:
                bgr_frame = self._letterbox(bgr_frame)
            if self.compression:
                success, rgb_frame = cv2.imencode(".png", bgr_frame)
                if success:
                    return rgb_frame
                else:
                    return None
            else:
                return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            logger.exception(f"Error converting frame: {e}")
            return None

    def _letterbox(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Applies letterboxing to the frame to make it square.

        Args:
            frame (cv2.typing.MatLike): The input frame to be letterboxed (as cv2 supported format - numpy like).

        Returns:
            cv2.typing.MatLike: The letterboxed frame (as cv2 supported format - numpy like).
        """
        h, w = frame.shape[:2]
        if w != h:
            # Letterbox: add padding to make it square (yolo colors)
            size = max(h, w)
            return cv2.copyMakeBorder(
                frame,
                top=(size - h) // 2,
                bottom=(size - h + 1) // 2,
                left=(size - w) // 2,
                right=(size - w + 1) // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
        else:
            return frame

    def _get_video_devices_by_index(self):
        """Reads symbolic links in /dev/v4l/by-id/, resolves them, and returns a
        dictionary mapping the numeric index to the system /dev/videoX device.

        Returns:
            dict[int, str]: a dict where keys are ordinal integer indices (e.g., 0, 1) and values are the
            /dev/videoX device names (e.g., "0", "1").
        """
        devices_by_index = {}
        directory_path = "/dev/v4l/by-id/"

        # Check if the directory exists
        if not os.path.exists(directory_path):
            logger.error(f"Error: Directory '{directory_path}' not found.")
            return devices_by_index

        try:
            # List all entries in the directory
            entries = os.listdir(directory_path)

            for entry in entries:
                full_path = os.path.join(directory_path, entry)

                # Check if the entry is a symbolic link
                if os.path.islink(full_path):
                    # Use a regular expression to find the numeric index at the end of the filename
                    match = re.search(r"index(\d+)$", entry)
                    if match:
                        index_str = match.group(1)
                        try:
                            index = int(index_str)

                            # Resolve the symbolic link to its absolute path
                            resolved_path = os.path.realpath(full_path)

                            # Get just the filename (e.g., "video0") from the resolved path
                            device_name = os.path.basename(resolved_path)

                            # Remove the "video" prefix to get just the number
                            device_number = device_name.replace("video", "")

                            # Add the index and device number to the dictionary
                            devices_by_index[index] = device_number

                        except ValueError:
                            logger.warning(f"Warning: Could not convert index '{index_str}' to an integer for '{entry}'. Skipping.")
                            continue
        except OSError as e:
            logger.error(f"Error accessing directory '{directory_path}': {e}")
            return devices_by_index

        return devices_by_index

    def start(self):
        """Starts the camera capture."""
        with self._cap_lock:
            if self._cap is not None:
                return

            temp_cap = cv2.VideoCapture(self.camera)
            if not temp_cap.isOpened():
                raise CameraOpenError(f"Failed to open camera {self.camera}.")

            self._cap = temp_cap  # Assign only after successful initialization
            self._last_capture_time_monotonic = time.monotonic()

            if self.resolution[0] is not None and self.resolution[1] is not None:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                # Verify if setting resolution was successful
                actual_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if actual_width != self.resolution[0] or actual_height != self.resolution[1]:
                    logger.warning(
                        f"Camera {self.camera} could not be set to {self.resolution[0]}x{self.resolution[1]}, "
                        f"actual resolution: {int(actual_width)}x{int(actual_height)}",
                    )

    def stop(self):
        """Stops the camera and releases its resources."""
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def produce(self):
        """Alias for capture method."""
        return self.capture()
