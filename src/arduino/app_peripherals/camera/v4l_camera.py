# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import os
import re
import cv2
import numpy as np
from typing import Optional, Union, Dict

from arduino.app_utils import Logger

from .camera import BaseCamera
from .errors import CameraOpenError, CameraReadError

logger = Logger("V4LCamera")


class V4LCamera(BaseCamera):
    """
    V4L (Video4Linux) camera implementation for USB and local cameras.
    
    This class handles USB cameras and other V4L-compatible devices on Linux systems.
    It supports both device indices and device paths.
    """

    def __init__(self, camera: Union[str, int] = 0, **kwargs):
        """
        Initialize V4L camera.

        Args:
            camera: Camera identifier - can be:
                   - int: Camera index (e.g., 0, 1)  
                   - str: Camera index as string or device path
            **kwargs: Additional camera parameters
        """
        super().__init__(**kwargs)
        self.camera_id = self._resolve_camera_id(camera)
        self._cap = None

    def _resolve_camera_id(self, camera: Union[str, int]) -> int:
        """
        Resolve camera identifier to a numeric device ID.
        
        Args:
            camera: Camera identifier
            
        Returns:
            Numeric camera device ID
            
        Raises:
            CameraOpenError: If camera cannot be resolved
        """
        if isinstance(camera, int):
            return camera
        
        if isinstance(camera, str):
            # If it's a numeric string, convert directly
            if camera.isdigit():
                device_id = int(camera)
                # Validate using device index mapping
                video_devices = self._get_video_devices_by_index()
                if device_id in video_devices:
                    return int(video_devices[device_id])
                else:
                    # Fallback to direct device ID if mapping not available
                    return device_id
            
            # If it's a device path like "/dev/video0"
            if camera.startswith('/dev/video'):
                return int(camera.replace('/dev/video', ''))
        
        raise CameraOpenError(f"Cannot resolve camera identifier: {camera}")

    def _get_video_devices_by_index(self) -> Dict[int, str]:
        """
        Map camera indices to device numbers by reading /dev/v4l/by-id/.
        
        Returns:
            Dict mapping index to device number
        """
        devices_by_index = {}
        directory_path = "/dev/v4l/by-id/"

        # Check if the directory exists
        if not os.path.exists(directory_path):
            logger.warning(f"Directory '{directory_path}' not found.")
            return devices_by_index

        try:
            entries = os.listdir(directory_path)
            for entry in entries:
                full_path = os.path.join(directory_path, entry)

                if os.path.islink(full_path):
                    # Find numeric index at end of filename
                    match = re.search(r"index(\d+)$", entry)
                    if match:
                        try:
                            index = int(match.group(1))
                            resolved_path = os.path.realpath(full_path)
                            device_name = os.path.basename(resolved_path)
                            device_number = device_name.replace("video", "")
                            devices_by_index[index] = device_number
                        except ValueError:
                            logger.warning(f"Could not parse index from '{entry}'")
                            continue
        except OSError as e:
            logger.error(f"Error accessing directory '{directory_path}': {e}")

        return devices_by_index

    def _open_camera(self) -> None:
        """Open the V4L camera connection."""
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            raise CameraOpenError(f"Failed to open V4L camera {self.camera_id}")

        # Set resolution if specified
        if self.resolution and self.resolution[0] and self.resolution[1]:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Verify resolution setting
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width != self.resolution[0] or actual_height != self.resolution[1]:
                logger.warning(
                    f"Camera {self.camera_id} resolution set to {actual_width}x{actual_height} "
                    f"instead of requested {self.resolution[0]}x{self.resolution[1]}"
                )

        logger.info(f"Opened V4L camera {self.camera_id}")

    def _close_camera(self) -> None:
        """Close the V4L camera connection."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the V4L camera."""
        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CameraReadError(f"Failed to read from V4L camera {self.camera_id}")

        return frame
