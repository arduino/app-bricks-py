# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class InputSource(ABC):
    """Abstract base class for frame input sources."""

    @abstractmethod
    def __init__(self, on_frame_cb: Callable[[np.ndarray], None], **kwargs):
        """
        Initialize the input source.

        Args:
            on_frame_cb: Callback invoked for each frame. Receives RGB np.ndarray,
                         returns a tuple of processed RGB np.ndarray and metadata dict.
            **kwargs: Implementation-specific configuration.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """Start receiving frames. Blocks until stop() is called or error occurs."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Signal the input source to stop. May be called from another thread."""
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Return True if the input source is currently running."""
        pass


class OutputSink(ABC):
    """Abstract base class for frame output sinks."""

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the output sink.

        Args:
            **kwargs: Implementation-specific configuration.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """Start the output sink (e.g., start server thread)."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the output sink gracefully."""
        pass

    @abstractmethod
    def send_frame(self, frame: np.ndarray, metadata: dict) -> None:
        """
        Send a processed frame to this output.

        Args:
            frame: RGB np.ndarray to output.
            metadata: Dictionary containing additional information about the frame.
        """
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Return True if the output sink is currently running."""
        pass
