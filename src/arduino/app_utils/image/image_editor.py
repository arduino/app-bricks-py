# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image

from .pipeable import pipeable


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
        result = frame | letterboxed(target_size=(640, 640)) | adjusted(brightness=10)
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
               maintain_aspect: bool = False, 
               interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize frame to target size.
        
        Args:
            frame (np.ndarray): Input frame
            target_size (tuple): Target size as (width, height)
            maintain_aspect (bool): If True, use letterboxing to maintain aspect ratio
            interpolation (int): OpenCV interpolation method
            
        Returns:
            np.ndarray: Resized frame
        """
        if maintain_aspect:
            return ImageEditor.letterbox(frame, target_size)
        else:
            return cv2.resize(frame, target_size, interpolation=interpolation)

    @staticmethod
    def adjust(frame: np.ndarray, 
               brightness: float = 0.0, 
               contrast: float = 1.0,
               saturation: float = 1.0) -> np.ndarray:
        """
        Apply basic image filters.
        
        Args:
            frame (np.ndarray): Input frame
            brightness (float): Brightness adjustment (-100 to 100)
            contrast (float): Contrast multiplier (0.0 to 3.0)
            saturation (float): Saturation multiplier (0.0 to 3.0)
            
        Returns:
            np.ndarray: adjusted frame
        """
        # Apply brightness and contrast
        result = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        
        # Apply saturation if needed
        if saturation != 1.0:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result

    @staticmethod
    def greyscale(frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to greyscale.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            
        Returns:
            np.ndarray: Greyscale frame (still 3 channels for consistency)
        """
        # Convert to greyscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert back to 3 channels for consistency with other operations
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def compress_to_jpeg(frame: np.ndarray, quality: int = 90) -> Optional[bytes]:
        """
        Compress frame to JPEG format.
        
        Args:
            frame (np.ndarray): Input frame as numpy array
            quality (int): JPEG quality (0-100, higher = better quality)
            
        Returns:
            bytes: Compressed JPEG data, or None if compression failed
        """
        try:
            success, encoded = cv2.imencode(
                '.jpg', 
                frame, 
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            return encoded.tobytes() if success else None
        except Exception:
            return None

    @staticmethod
    def compress_to_png(frame: np.ndarray, compression_level: int = 6) -> Optional[bytes]:
        """
        Compress frame to PNG format.
        
        Args:
            frame (np.ndarray): Input frame as numpy array
            compression_level (int): PNG compression level (0-9, higher = better compression)
            
        Returns:
            bytes: Compressed PNG data, or None if compression failed
        """
        try:
            success, encoded = cv2.imencode(
                '.png', 
                frame, 
                [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
            )
            return encoded.tobytes() if success else None
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

    @staticmethod
    def get_frame_info(frame: np.ndarray) -> dict:
        """
        Get information about a frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            dict: Frame information including dimensions, channels, dtype, size
        """
        return {
            'height': frame.shape[0],
            'width': frame.shape[1],
            'channels': frame.shape[2] if len(frame.shape) > 2 else 1,
            'dtype': str(frame.dtype),
            'size_bytes': frame.nbytes,
            'shape': frame.shape
        }


# =============================================================================
# Functional API - Standalone pipeable functions
# =============================================================================

@pipeable
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
        result = frame | letterboxed(target_size=(640, 640))
        result = frame | letterboxed() | adjusted(brightness=10)
    """
    from functools import partial
    return partial(ImageEditor.letterbox, target_size=target_size, color=color)


@pipeable
def resized(target_size: Tuple[int, int], 
            maintain_aspect: bool = False, 
            interpolation: int = cv2.INTER_LINEAR):
    """
    Pipeable resize function - resize frame with pipe operator support.
    
    Args:
        target_size (tuple): Target size as (width, height)
        maintain_aspect (bool): If True, use letterboxing to maintain aspect ratio
        interpolation (int): OpenCV interpolation method
        
    Returns:
        Partial function that takes a frame and returns resized frame
        
    Examples:
        result = frame | resized(target_size=(640, 480))
        result = frame | letterboxed() | resized(target_size=(320, 240))
    """
    from functools import partial
    return partial(ImageEditor.resize, target_size=target_size, maintain_aspect=maintain_aspect, interpolation=interpolation)


@pipeable
def adjusted(brightness: float = 0.0, 
             contrast: float = 1.0,
             saturation: float = 1.0):
    """
    Pipeable filter function - apply filters with pipe operator support.
    
    Args:
        brightness (float): Brightness adjustment (-100 to 100)
        contrast (float): Contrast multiplier (0.0 to 3.0)
        saturation (float): Saturation multiplier (0.0 to 3.0)
        
    Returns:
        Partial function that takes a frame and returns the adjusted frame
        
    Examples:
        result = frame | adjusted(brightness=10, contrast=1.2)
        result = frame | letterboxed() | adjusted(brightness=5) | resized(target_size=(320, 240))
    """
    from functools import partial
    return partial(ImageEditor.adjust, brightness=brightness, contrast=contrast, saturation=saturation)


@pipeable
def greyscaled():
    """
    Pipeable greyscale function - convert frame to greyscale with pipe operator support.
    
    Returns:
        Function that takes a frame and returns greyscale frame
        
    Examples:
        result = frame | greyscaled()
        result = frame | letterboxed() | greyscaled() | adjusted(contrast=1.2)
    """
    return ImageEditor.greyscale


@pipeable
def compressed_to_jpeg(quality: int = 90):
    """
    Pipeable JPEG compression function - compress frame to JPEG with pipe operator support.
    
    Args:
        quality (int): JPEG quality (0-100, higher = better quality)
        
    Returns:
        Partial function that takes a frame and returns compressed JPEG bytes
        
    Examples:
        jpeg_bytes = frame | compressed_to_jpeg(quality=95)
        jpeg_bytes = frame | resized(target_size=(640, 480)) | compressed_to_jpeg()
    """
    from functools import partial
    return partial(ImageEditor.compress_to_jpeg, quality=quality)


@pipeable
def compressed_to_png(compression_level: int = 6):
    """
    Pipeable PNG compression function - compress frame to PNG with pipe operator support.
    
    Args:
        compression_level (int): PNG compression level (0-9, higher = better compression)
        
    Returns:
        Partial function that takes a frame and returns compressed PNG bytes
        
    Examples:
        png_bytes = frame | compressed_to_png(compression_level=9)
        png_bytes = frame | letterboxed() | compressed_to_png()
    """
    from functools import partial
    return partial(ImageEditor.compress_to_png, compression_level=compression_level)
