# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import struct
import wave
from typing import Callable

from arduino.app_internal.core.audio import AudioDetector
from arduino.app_peripherals.microphone import Microphone
from arduino.app_utils import brick, Logger

logger = Logger("AudioClassification")


class AudioClassificationException(Exception):
    """Custom exception for AudioClassification errors."""

    pass


@brick
class AudioClassification(AudioDetector):
    """AudioClassification module for detecting sounds and classifying audio using a specified model."""

    def __init__(self, mic: Microphone = None, confidence: float = 0.8):
        """Initialize the AudioClassification class.

        Args:
            mic (Microphone, optional): Microphone instance used as the audio source. If None, a default Microphone will be initialized.
            confidence (float, optional): Minimum confidence threshold (0.0–1.0) required
                for a detection to be considered valid. Defaults to 0.8 (80%).

        Raises:
            ValueError: If the model information cannot be retrieved, or if model parameters are missing or incomplete.
        """
        super().__init__(mic=mic, confidence=confidence)

    def on_detect(self, class_name: str, callback: Callable[[], None]):
        """Register a callback function to be invoked when a specific class is detected.

        Args:
            class_name (str): The class to check for in the classification results.
                Must match one of the classes defined in the loaded model.
            callback (callable): Function to execute when the class is detected.
                The callback must take no arguments and return None.

        Raises:
            TypeError: If `callback` is not callable.
            ValueError: If `callback` accepts any argument.
        """
        super().on_detect(class_name, callback)

    def start(self):
        """Start real-time audio classification.

        Begins capturing audio from the configured microphone and
        continuously classifies the incoming audio stream until stopped.
        """
        super().start()

    def stop(self):
        """Stop real-time audio classification.

        Terminates audio capture and releases any associated resources.
        """
        super().stop()

    def classify_from_file(self, audio_path: str, confidence: int = None) -> dict | None:
        """Classify audio content from a WAV file.

        Supported sample widths:
            - 8-bit unsigned
            - 16-bit signed
            - 24-bit signed
            - 32-bit signed

        Args:
            audio_path (str): Path to the `.wav` audio file to classify.
            confidence (int, optional): Confidence threshold (0–1). If None,
                the default confidence level specified during initialization
                will be applied.

        Returns:
            dict | None: A dictionary with keys:
            - ``class_name`` (str): The detected sound class.
            - ``confidence`` (float): Confidence score of the detection.
            Returns None if no valid classification is found.

        Raises:
            AudioClassificationException: If the file cannot be found, read, or processed.
            ValueError: If the file uses an unsupported sample width.
        """
        try:
            with wave.open(audio_path, "rb") as wf:
                # Get WAV file properties
                n_channels = wf.getnchannels()
                samp_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)

                # Unpack audio data
                features = []
                if samp_width == 1:
                    # 8-bit audio (unsigned char)
                    fmt = f"{n_frames * n_channels}B"
                    features = list(struct.unpack(fmt, frames))
                elif samp_width == 2:
                    # 16-bit audio (signed short)
                    fmt = f"{n_frames * n_channels}h"
                    features = list(struct.unpack(fmt, frames))
                elif samp_width == 3:
                    # 24-bit audio (custom unpacking as struct doesn't have a direct 3-byte type)
                    for i in range(0, len(frames), 3):
                        # Interpret 3 bytes as a signed 24-bit integer
                        value = int.from_bytes(frames[i : i + 3], byteorder="little", signed=True)
                        features.append(value)
                elif samp_width == 4:
                    # 32-bit audio (signed int)
                    fmt = f"{n_frames * n_channels}i"
                    features = list(struct.unpack(fmt, frames))
                else:
                    raise ValueError(f"Unsupported sample width: {samp_width} bytes. Cannot process this WAV file.")

                classification = super().infer_from_features(features[: int(self.model_info.input_features_count)])
                best_match = super().get_best_match(classification, confidence)
                if not best_match:
                    return None
                keyword, confidence = best_match
                if keyword and confidence:
                    return {"class_name": keyword, "confidence": confidence}
        except FileNotFoundError:
            raise AudioClassificationException(f"File not found: {audio_path}")
        except wave.Error as e:
            raise AudioClassificationException(f"Error reading WAV file '{audio_path}': {e}")
        except Exception as e:
            raise AudioClassificationException(f"An unexpected error occurred while processing '{audio_path}': {e}")

        return None
