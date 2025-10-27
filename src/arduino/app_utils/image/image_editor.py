# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image

from arduino.app_utils.image.pipeable import PipeableFunction


class ImageEditor:
    """
    Image processing utilities for camera frames.
    
    Handles common image operations like compression, letterboxing, resizing, and format conversions.
    
    This class provides traditional static methods for image processing operations.
    For functional composition with pipe operators, use the standalone functions below the class.
    
    Examples:
        Traditional API:
        result = ImageEditor.letterbox(frame, target_size=(640, 640))
        
        Functional API:
        result = frame | letterboxed(target_size=(640, 640))
        
        Chained operations:
        result = frame | letterboxed(target_size=(640, 640)) | greyscaled()
    """

    @staticmethod
    def letterbox(frame: np.ndarray, 
                  target_size: Optional[Tuple[int, int]] = None, 
                  color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
        """
        Add letterboxing to frame to achieve target size while maintaining aspect ratio.
        
        Args:
            frame (np.ndarray): Input frame
            target_size (tuple, optional): Target size as (width, height). If None, makes frame square.
            color (tuple): RGB color for padding borders. Default: (114, 114, 114)
            
        Returns:
            np.ndarray: Letterboxed frame
        """
        if target_size is None:
            # Make square based on the larger dimension
            max_dim = max(frame.shape[0], frame.shape[1])
            target_size = (max_dim, max_dim)
        
        target_w, target_h = target_size
        h, w = frame.shape[:2]
        
        # Handle empty frames
        if w == 0 or h == 0:
            raise ValueError("Cannot letterbox empty frame")
        
        # Calculate scaling factor to fit image inside target size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        # Add padding
        return cv2.copyMakeBorder(
            resized,
            top=pad_h // 2,
            bottom=(pad_h + 1) // 2,
            left=pad_w // 2,
            right=(pad_w + 1) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=color
        )

    @staticmethod
    def resize(frame: np.ndarray, 
               target_size: Tuple[int, int], 
               maintain_ratio: bool = False, 
               interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize frame to target size.
        
        Args:
            frame (np.ndarray): Input frame
            target_size (tuple): Target size as (width, height)
            maintain_ratio (bool): If True, use letterboxing to maintain aspect ratio
            interpolation (int): OpenCV interpolation method
            
        Returns:
            np.ndarray: Resized frame
        """
        if maintain_ratio:
            return ImageEditor.letterbox(frame, target_size)
        else:
            return cv2.resize(frame, (target_size[1], target_size[0]), interpolation=interpolation)

    @staticmethod
    def greyscale(frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to greyscale and maintain 3 channels for consistency.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            
        Returns:
            np.ndarray: Greyscale frame (3 channels, all identical)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert back to 3 channels for consistency
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def compress_to_jpeg(frame: np.ndarray, quality: int = 80) -> Optional[np.ndarray]:
        """
        Compress frame to JPEG format.
        
        Args:
            frame (np.ndarray): Input frame as numpy array
            quality (int): JPEG quality (0-100, higher = better quality)
            
        Returns:
            bytes: Compressed JPEG data, or None if compression failed
        """
        quality = int(quality)  # Gstreamer doesn't like quality to be float
        try:
            success, encoded = cv2.imencode(
                '.jpg', 
                frame, 
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            return encoded if success else None
        except Exception:
            return None

    @staticmethod
    def compress_to_png(frame: np.ndarray, compression_level: int = 6) -> Optional[np.ndarray]:
        """
        Compress frame to PNG format.
        
        Args:
            frame (np.ndarray): Input frame as numpy array
            compression_level (int): PNG compression level (0-9, higher = better compression)
            
        Returns:
            bytes: Compressed PNG data, or None if compression failed
        """
        compression_level = int(compression_level)  # Gstreamer doesn't like compression_level to be float
        try:
            success, encoded = cv2.imencode(
                '.png', 
                frame, 
                [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
            )
            return encoded if success else None
        except Exception:
            return None

    @staticmethod
    def numpy_to_pil(frame: np.ndarray) -> Image.Image:
        """
        Convert numpy array to PIL Image.
        
        Args:
            frame (np.ndarray): Input frame in BGR format (OpenCV default)
            
        Returns:
            PIL.Image.Image: PIL Image in RGB format
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    @staticmethod
    def pil_to_numpy(image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to numpy array.
        
        Args:
            image (PIL.Image.Image): PIL Image
            
        Returns:
            np.ndarray: Numpy array in BGR format (OpenCV default)
        """
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy and then BGR
        rgb_array = np.array(image)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


# =============================================================================
# Functional API - Standalone pipeable functions
# =============================================================================

def letterboxed(target_size: Optional[Tuple[int, int]] = None, 
                color: Tuple[int, int, int] = (114, 114, 114)):
    """
    Pipeable letterbox function - apply letterboxing with pipe operator support.
    
    Args:
        target_size (tuple, optional): Target size as (width, height). If None, makes frame square.
        color (tuple): RGB color for padding borders. Default: (114, 114, 114)
        
    Returns:
        Partial function that takes a frame and returns letterboxed frame
        
    Examples:
        pipe = letterboxed(target_size=(640, 640))
        pipe = letterboxed() | greyscaled()
    """
    return PipeableFunction(ImageEditor.letterbox, target_size=target_size, color=color)


def resized(target_size: Tuple[int, int], 
            maintain_ratio: bool = False, 
            interpolation: int = cv2.INTER_LINEAR):
    """
    Pipeable resize function - resize frame with pipe operator support.
    
    Args:
        target_size (tuple): Target size as (width, height)
        maintain_ratio (bool): If True, use letterboxing to maintain aspect ratio
        interpolation (int): OpenCV interpolation method
        
    Returns:
        Partial function that takes a frame and returns resized frame
        
    Examples:
        pipe = resized(target_size=(640, 480))
        pipe = letterboxed() | resized(target_size=(320, 240))
    """
    return PipeableFunction(ImageEditor.resize, target_size=target_size, maintain_ratio=maintain_ratio, interpolation=interpolation)


def greyscaled():
    """
    Pipeable greyscale function - convert frame to greyscale with pipe operator support.
    
    Returns:
        Function that takes a frame and returns greyscale frame
        
    Examples:
        pipe = greyscaled()
        pipe = letterboxed() | greyscaled() | greyscaled()
    """
    return PipeableFunction(ImageEditor.greyscale)


def compressed_to_jpeg(quality: int = 80):
    """
    Pipeable JPEG compression function - compress frame to JPEG with pipe operator support.
    
    Args:
        quality (int): JPEG quality (0-100, higher = better quality)
        
    Returns:
        Partial function that takes a frame and returns compressed JPEG bytes as Numpy array or None
        
    Examples:
        pipe = compressed_to_jpeg(quality=95)
        pipe = resized(target_size=(640, 480)) | compressed_to_jpeg()
    """
    return PipeableFunction(ImageEditor.compress_to_jpeg, quality=quality)


def compressed_to_png(compression_level: int = 6):
    """
    Pipeable PNG compression function - compress frame to PNG with pipe operator support.
    
    Args:
        compression_level (int): PNG compression level (0-9, higher = better compression)
        
    Returns:
        Partial function that takes a frame and returns compressed PNG bytes as Numpy array or None
        
    Examples:
        pipe = compressed_to_png(compression_level=9)
        pipe = letterboxed() | compressed_to_png()
    """
    return PipeableFunction(ImageEditor.compress_to_png, compression_level=compression_level)
