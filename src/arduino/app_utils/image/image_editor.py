# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image

from arduino.app_utils.image.pipeable import PipeableFunction

# NOTE: we use the following formats for image shapes (H = height, W = width, C = channels):
# - When receiving a resolution as argument we expect (W, H) format which is more user-friendly
# - When receiving images we expect (H, W, C) format with C = BGR, BGRA or greyscale
# - When returning images we use (H, W, C) format with C = BGR, BGRA or greyscale (depending on input)
# Keep in mind OpenCV uses (W, H, C) format with C = BGR whereas numpy uses (H, W, C) format with any C.
# The below functions all support unsigned integer types used by OpenCV (uint8, uint16 and uint32).


class ImageEditor:
    """
    Image processing utilities handling common image operations like letterboxing, resizing,
    adjusting, compressing and format conversions.
    Frames are expected to be in BGR, BGRA or greyscale format.
    """

    @staticmethod
    def letterbox(frame: np.ndarray, 
                  target_size: Optional[Tuple[int, int]] = None, 
                  color: int | Tuple[int, int, int] = (114, 114, 114),
                  interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Add letterboxing to frame to achieve target size while maintaining aspect ratio.
        
        Args:
            frame (np.ndarray): Input frame
            target_size (tuple, optional): Target size as (width, height). If None, makes frame square.
            color (int or tuple, optional): BGR color for padding borders, can be a scalar or a tuple
            matching the frame's channel count. Default: (114, 114, 114)
            interpolation (int, optional): OpenCV interpolation method. Default: cv2.INTER_LINEAR

        Returns:
            np.ndarray: Letterboxed frame
        """
        original_dtype = frame.dtype
        orig_h, orig_w = frame.shape[:2]    

        if target_size is None:
            # Default to a square canvas based on the longest side
            max_dim = max(orig_h, orig_w)
            target_w, target_h = int(max_dim), int(max_dim)
        else:
            target_w, target_h = int(target_size[0]), int(target_size[1])

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

        if frame.ndim == 2:
            # Greyscale
            if hasattr(color, '__len__'):
                color = color[0]
            canvas = np.full((target_h, target_w), color, dtype=original_dtype)
        else:
            # Colored (BGR/BGRA)
            channels = frame.shape[2]
            if not hasattr(color, '__len__'):
                color = (color,) * channels
            elif len(color) != channels:
                raise ValueError(
                    f"color length ({len(color)}) must match frame channels ({channels})."
                )
            canvas = np.full((target_h, target_w, channels), color, dtype=original_dtype)

        # Calculate offsets to center the image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        # Paste the resized image onto the canvas
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame

        return canvas

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
            return cv2.resize(frame, (target_size[0], target_size[1]), interpolation=interpolation)

    @staticmethod
    def adjust(frame: np.ndarray,
               brightness: float = 0.0,
               contrast: float = 1.0,
               saturation: float = 1.0,
               gamma: float = 1.0) -> np.ndarray:
        """
        Apply image adjustments to a BGR or BGRA frame, preserving channel count
        and data type.
        
        Args:
            frame (np.ndarray): Input frame (uint8, uint16, uint32).
            brightness (float): -1.0 to 1.0 (default: 0.0).
            contrast (float): 0.0 to N (default: 1.0).
            saturation (float): 0.0 to N (default: 1.0).
            gamma (float): > 0 (default: 1.0).
            
        Returns:
            np.ndarray: The adjusted input with same dtype as frame.
        """
        original_dtype = frame.dtype
        dtype_info = np.iinfo(original_dtype)
        max_val = dtype_info.max
        
        # Use float64 for int types with > 24 bits of precision (e.g., uint32)
        processing_dtype = np.float64 if dtype_info.bits > 24 else np.float32
        
        # Apply the adjustments in float space to reduce clipping and data loss
        frame_float = frame.astype(processing_dtype) / max_val

        # If present, separate alpha channel
        alpha_channel = None
        if frame.ndim == 3 and frame.shape[2] == 4:
            alpha_channel = frame_float[:, :, 3]
            frame_float = frame_float[:, :, :3]
        
        # Saturation
        if saturation != 1.0 and frame.ndim == 3:  # Ensure frame has color channels
            # This must be done with float32 so it's lossy only for uint32
            frame_float_32 = frame_float.astype(np.float32)
            hsv = cv2.cvtColor(frame_float_32, cv2.COLOR_BGR2HSV)
            h, s, v = ImageEditor.split_channels(hsv)
            s = np.clip(s * saturation, 0.0, 1.0)
            frame_float_32 = cv2.cvtColor(np.stack([h, s, v], axis=2), cv2.COLOR_HSV2BGR)
            frame_float = frame_float_32.astype(processing_dtype)

        # Brightness
        if brightness != 0.0:
            frame_float = frame_float + brightness
        
        # Contrast
        if contrast != 1.0:
            frame_float = (frame_float - 0.5) * contrast + 0.5
            
        # We need to clip before reaching gamma correction
        # Clipping to 0 is mandatory to avoid handling complex numbers
        # Clipping to 1 is handy to avoid clipping again after gamma correction
        frame_float = np.clip(frame_float, 0.0, 1.0)
        
        # Gamma
        if gamma != 1.0:
            if gamma <= 0:
                # This check is critical to prevent math errors (NaN/Inf)
                raise ValueError("Gamma value must be greater than 0.")
            frame_float = np.power(frame_float, gamma)
        
        # Convert back to original dtype
        final_frame_bgr = (frame_float * max_val).astype(original_dtype)

        # If present, reattach alpha channel
        if alpha_channel is not None:
            final_alpha = (alpha_channel * max_val).astype(original_dtype)
            b, g, r = ImageEditor.split_channels(final_frame_bgr)
            final_frame = np.stack([b, g, r, final_alpha], axis=2)
        else:
            final_frame = final_frame_bgr
        
        return final_frame

    @staticmethod
    def split_channels(frame: np.ndarray) -> tuple:
        """
        Split a multi-channel frame into individual channels using numpy indexing.
        This function provides better data type compatibility than cv2.split,
        especially for uint32 data which OpenCV doesn't fully support.
        
        Args:
            frame (np.ndarray): Input frame with 3 or 4 channels
            
        Returns:
            tuple: Individual channel arrays. For BGR: (b, g, r). For BGRA: (b, g, r, a).
                   For HSV: (h, s, v). For other 3-channel: (ch0, ch1, ch2).
        """
        if frame.ndim != 3:
            raise ValueError("Frame must be 3-dimensional (H, W, C)")
        
        channels = frame.shape[2]
        if channels == 3:
            return frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        elif channels == 4:
            return frame[:, :, 0], frame[:, :, 1], frame[:, :, 2], frame[:, :, 3]
        else:
            raise ValueError(f"Unsupported number of channels: {channels}. Expected 3 or 4.")

    @staticmethod
    def greyscale(frame: np.ndarray) -> np.ndarray:
        """
        Converts a BGR or BGRA frame to greyscale, preserving channel count and
        data type. A greyscale frame is returned unmodified.
        
        Args:
            frame (np.ndarray): Input frame (uint8, uint16, uint32).
            
        Returns:
            np.ndarray: The greyscaled frame with same dtype and channel count as frame.
        """
        # If already greyscale or unknown format, return the original frame
        if frame.ndim != 3:
            return frame
            
        original_dtype = frame.dtype
        dtype_info = np.iinfo(original_dtype)
        max_val = dtype_info.max
        
        # Use float64 for int types with > 24 bits of precision (e.g., uint32)
        processing_dtype = np.float64 if dtype_info.bits > 24 else np.float32
        
        # Apply the adjustments in float space to reduce clipping and data loss
        frame_float = frame.astype(processing_dtype) / max_val
        
        # If present, separate alpha channel
        alpha_channel = None
        if frame.shape[2] == 4:
            alpha_channel = frame_float[:, :, 3]
            frame_float = frame_float[:, :, :3]

        # Convert to greyscale using standard BT.709 weights
        # GREY = 0.0722 * B + 0.7152 * G + 0.2126 * R
        grey_float = (0.0722 * frame_float[:, :, 0] +
                      0.7152 * frame_float[:, :, 1] +
                      0.2126 * frame_float[:, :, 2])
        
        # Convert back to original dtype
        final_grey = (grey_float * max_val).astype(original_dtype)

        # If present, reattach alpha channel
        if alpha_channel is not None:
            final_alpha = (alpha_channel * max_val).astype(original_dtype)
            final_frame = np.stack([final_grey, final_grey, final_grey, final_alpha], axis=2)
        else:
            final_frame = np.stack([final_grey, final_grey, final_grey], axis=2)

        return final_frame
        
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
            frame (np.ndarray): Input frame in BGR format
            
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
            np.ndarray: Numpy array in BGR format
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy and then BGR
        rgb_array = np.array(image)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


# =============================================================================
# Functional API - Standalone pipeable functions
# =============================================================================

def letterboxed(target_size: Optional[Tuple[int, int]] = None, 
                color: Tuple[int, int, int] = (114, 114, 114),
                interpolation: int = cv2.INTER_LINEAR):
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
    return PipeableFunction(ImageEditor.letterbox, target_size=target_size, color=color, interpolation=interpolation)


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


def adjusted(brightness: float = 0.0,
             contrast: float = 1.0,
             saturation: float = 1.0,
             gamma: float = 1.0):
    """
    Pipeable adjust function - apply image adjustments with pipe operator support.
    
    Args:
        brightness (float): -1.0 to 1.0 (default: 0.0).
        contrast (float): 0.0 to N (default: 1.0).
        saturation (float): 0.0 to N (default: 1.0).
        gamma (float): > 0 (default: 1.0).

    Returns:
        Partial function that takes a frame and returns adjusted frame

    Examples:
        pipe = adjusted(brightness=0.1, contrast=1.2)
        pipe = letterboxed() | adjusted(saturation=0.8)
    """
    return PipeableFunction(ImageEditor.adjust, brightness=brightness, contrast=contrast, saturation=saturation, gamma=gamma)


def greyscaled():
    """
    Pipeable greyscale function - convert frame to greyscale with pipe operator support.
    
    Returns:
        Function that takes a frame and returns greyscale frame
        
    Examples:
        pipe = greyscaled()
        pipe = letterboxed() | greyscaled()
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
