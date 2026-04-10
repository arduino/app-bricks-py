# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image

from arduino.app_utils.image.pipeable import PipeableFunction

# NOTE: we adopt the following conventions for color and image shape arguments:
# (H = height, W = width, C = channels)
# - When receiving colors we expect BGR or BGRA formats as a tuple
# - When receiving a resolution we expect (W, H) format which is more user-friendly
# - When receiving images we expect (H, W, C) format with C = BGR, BGRA or greyscale
# - When returning images we use (H, W, C) format with C = BGR, BGRA or greyscale (depending on input)
# Keep in mind OpenCV uses (W, H, C) format with C = BGR whereas numpy uses (H, W, C) format with any C.
# The below functions all support unsigned integer types used by OpenCV (uint8, uint16 and uint32).


"""
Image processing utilities handling common image operations like letterboxing, resizing,
adjusting, compressing and format conversions.
Frames are expected to be in BGR, BGRA or greyscale format.
"""


def letterbox(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    color: Tuple[int, int, int] | Tuple[int, int, int, int] = (114, 114, 114),
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Add letterboxing to frame to achieve target size while maintaining aspect ratio.

    Args:
        frame (np.ndarray): Input frame
        target_size (tuple, optional): Target size as (width, height). If None, makes frame square.
        color (tuple): BGR or BGRA color for padding borders. Default: (114, 114, 114)
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

    if new_w == orig_w and new_h == orig_h:
        resized_frame = frame
    else:
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    if frame.ndim == 2:
        # Greyscale
        canvas = np.full((target_h, target_w), color[0], dtype=original_dtype)
    else:
        # Colored (BGR/BGRA)
        channels = frame.shape[2]
        if len(color) != channels:
            raise ValueError(f"color length ({len(color)}) must match frame channels ({channels}).")
        canvas = np.full((target_h, target_w, channels), color, dtype=original_dtype)

    # Calculate offsets to center the image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    # Paste the resized image onto the canvas
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_frame

    return canvas


def resize(frame: np.ndarray, target_size: Tuple[int, int], maintain_ratio: bool = False, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize frame to target size.

    Args:
        frame (np.ndarray): Input frame
        target_size (tuple): Target size as (width, height)
        maintain_ratio (bool): If True, use letterboxing to maintain aspect ratio. Default: False.
        interpolation (int): OpenCV interpolation method. Default: cv2.INTER_LINEAR.

    Returns:
        np.ndarray: Resized frame
    """
    if frame.shape[1] == target_size[0] and frame.shape[0] == target_size[1]:
        return frame

    if maintain_ratio:
        return letterbox(frame, target_size)
    else:
        return cv2.resize(frame, (target_size[0], target_size[1]), interpolation=interpolation)


def flip_h(frame: np.ndarray) -> np.ndarray:
    """
    Flip frame horizontally.

    Args:
        frame (np.ndarray): Input frame

    Returns:
        np.ndarray: Horizontally flipped frame
    """
    return frame[:, ::-1, ...]


def flip_v(frame: np.ndarray) -> np.ndarray:
    """
    Flip frame vertically.

    Args:
        frame (np.ndarray): Input frame

    Returns:
        np.ndarray: Vertically flipped frame
    """
    return frame[::-1, :, ...]


def crop(frame: np.ndarray, width: int, height: int, x: Optional[int] = None, y: Optional[int] = None) -> np.ndarray:
    """
    Crop frame to specified region. If x and y are not provided, the crop is centered.

    Args:
        frame (np.ndarray): Input frame
        width (int): Width of crop region
        height (int): Height of crop region
        x (int, optional): Left coordinate of crop region. If None, centers horizontally.
            Default: None.
        y (int, optional): Top coordinate of crop region. If None, centers vertically.
            Default: None.

    Returns:
        np.ndarray: Cropped frame
    """
    orig_h, orig_w = frame.shape[:2]

    if x is None:
        x = (orig_w - width) // 2
    if y is None:
        y = (orig_h - height) // 2

    x = max(0, min(x, orig_w))
    y = max(0, min(y, orig_h))
    x2 = max(0, min(x + width, orig_w))
    y2 = max(0, min(y + height, orig_h))

    return frame[y:y2, x:x2, ...]


def crop_to_aspect_ratio(
    frame: np.ndarray,
    aspect_ratio: Tuple[int, int],
    x: Optional[int] = None,
    y: Optional[int] = None,
) -> np.ndarray:
    """
    Crop frame to specified aspect ratio. If x and y are not provided, the crop is
    centered. The function will crop the minimum amount necessary to achieve the
    target aspect ratio.

    Args:
        frame (np.ndarray): Input frame
        aspect_ratio (tuple): Target aspect ratio as tuple (e.g., (16, 9), (1, 1))
        x (int, optional): Left coordinate of crop region. If None, centers horizontally.
            Default: None.
        y (int, optional): Top coordinate of crop region. If None, centers vertically.
            Default: None.

    Returns:
        np.ndarray: Cropped frame with target aspect ratio

    Examples:
        crop_to_aspect_ratio(frame, (16, 9))  # Crop to 16:9 aspect ratio
        crop_to_aspect_ratio(frame, (4, 3))  # Crop to 4:3 aspect ratio
        crop_to_aspect_ratio(frame, (1, 1))  # Crop to square
    """
    aspect_ratio_float = aspect_ratio[0] / aspect_ratio[1]
    orig_h, orig_w = frame.shape[:2]
    current_aspect = orig_w / orig_h
    # Determine which dimension to crop
    if current_aspect > aspect_ratio_float:
        # Crop width
        new_width = int(orig_h * aspect_ratio_float)
        new_height = orig_h
    else:
        # Crop height
        new_width = orig_w
        new_height = int(orig_w / aspect_ratio_float)

    return crop(frame, new_width, new_height, x, y)


def rotate(
    frame: np.ndarray,
    angle: float,
    center: Optional[Tuple[int, int]] = None,
    expand: bool = False,
    color: Tuple[int, int, int] | Tuple[int, int, int, int] = (0, 0, 0),
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Rotate frame by specified angle.

    Args:
        frame (np.ndarray): Input frame
        angle (float): Rotation angle in degrees (positive values rotate counter-clockwise)
        center (tuple, optional): Center of rotation as (x, y). If None, uses image center.
            Default: None.
        expand (bool): If True, expands the output frame to fit the entire rotated image.
            If False, output frame size matches input. Default: False.
        color (tuple): BGR or BGRA color for border pixels. Default: (0, 0, 0).
        interpolation (int): OpenCV interpolation method. Default: cv2.INTER_LINEAR.

    Returns:
        np.ndarray: Rotated frame

    Examples:
        rotate(frame, 45)  # Rotate 45 degrees counter-clockwise
        rotate(frame, -90)  # Rotate 90 degrees clockwise
        rotate(frame, 30, expand=True)  # Rotate 30 degrees and expand to fit
    """
    if angle == 0.0:
        return frame

    if center is None and not expand:
        normalized_angle = angle % 360

        # Check for 90-degree increments
        if normalized_angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif normalized_angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif normalized_angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    orig_h, orig_w = frame.shape[:2]

    if center is None:
        center = (orig_w // 2, orig_h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    if expand:
        # Calculate new dimension after rotation
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int((orig_h * sin) + (orig_w * cos))
        new_h = int((orig_h * cos) + (orig_w * sin))

        # Center the rotated image
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        output_size = (new_w, new_h)
    else:
        output_size = (orig_w, orig_h)

    rotated = cv2.warpAffine(
        frame,
        rotation_matrix,
        output_size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=color,
    )

    return rotated


def adjust(frame: np.ndarray, brightness: float = 0.0, contrast: float = 1.0, saturation: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """
    Apply image adjustments to a BGR or BGRA frame, preserving channel count
    and data type.

    Args:
        frame (np.ndarray): Input frame (uint8, uint16, uint32).
        brightness (float): -1.0 to 1.0. Default: 0.0.
        contrast (float): 0.0 to N. Default: 1.0.
        saturation (float): 0.0 to N. Default: 1.0.
        gamma (float): > 0. Default: 1.0.

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
        h, s, v = split_channels(hsv)
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
        b, g, r = split_channels(final_frame_bgr)
        final_frame = np.stack([b, g, r, final_alpha], axis=2)
    else:
        final_frame = final_frame_bgr

    return final_frame


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
    grey_float = 0.0722 * frame_float[:, :, 0] + 0.7152 * frame_float[:, :, 1] + 0.2126 * frame_float[:, :, 2]

    # Convert back to original dtype
    final_grey = (grey_float * max_val).astype(original_dtype)

    # If present, reattach alpha channel
    if alpha_channel is not None:
        final_alpha = (alpha_channel * max_val).astype(original_dtype)
        final_frame = np.stack([final_grey, final_grey, final_grey, final_alpha], axis=2)
    else:
        final_frame = np.stack([final_grey, final_grey, final_grey], axis=2)

    return final_frame


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
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return encoded if success else None
    except Exception:
        return None


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
        success, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        return encoded if success else None
    except Exception:
        return None


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


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.

    Args:
        image (PIL.Image.Image): PIL Image

    Returns:
        np.ndarray: Numpy array in BGR format
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to numpy and then BGR
    rgb_array = np.array(image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


# =============================================================================
# Functional API - Standalone pipeable functions
# =============================================================================


def letterboxed(
    target_size: Optional[Tuple[int, int]] = None,
    color: Tuple[int, int, int] | Tuple[int, int, int, int] = (114, 114, 114),
    interpolation: int = cv2.INTER_LINEAR,
):
    """
    Pipeable letterbox function - apply letterboxing with pipe operator support.

    Args:
        target_size (tuple, optional): Target size as (width, height). If None, makes frame square.
        color (tuple): BGR or BGRA color for padding borders. Default: (114, 114, 114)
        interpolation (int): OpenCV interpolation method. Default: cv2.INTER_LINEAR

    Returns:
        Function that takes a frame and returns letterboxed frame

    Examples:
        pipe = letterboxed(target_size=(640, 640))
        pipe = letterboxed() | greyscaled()
    """
    return PipeableFunction(letterbox, target_size=target_size, color=color, interpolation=interpolation)


def resized(target_size: Tuple[int, int], maintain_ratio: bool = False, interpolation: int = cv2.INTER_LINEAR):
    """
    Pipeable resize function - resize frame with pipe operator support.

    Args:
        target_size (tuple): Target size as (width, height)
        maintain_ratio (bool): If True, use letterboxing to maintain aspect ratio
        interpolation (int): OpenCV interpolation method. Default: cv2.INTER_LINEAR

    Returns:
        Function that takes a frame and returns resized frame

    Examples:
        pipe = resized(target_size=(640, 480))
        pipe = letterboxed() | resized(target_size=(320, 240))
    """
    return PipeableFunction(resize, target_size=target_size, maintain_ratio=maintain_ratio, interpolation=interpolation)


def flipped_h():
    """
    Pipeable horizontal flip function - flip frame horizontally with pipe operator support.

    Returns:
        Function that takes a frame and returns horizontally flipped frame
    """
    return PipeableFunction(flip_h)


def flipped_v():
    """
    Pipeable vertical flip function - flip frame vertically with pipe operator support.

    Returns:
        Function that takes a frame and returns vertically flipped frame
    """
    return PipeableFunction(flip_v)


def cropped(width: int, height: int, x: Optional[int] = None, y: Optional[int] = None):
    """
    Pipeable crop function - crop frame with pipe operator support.
    If x and y are not provided, the crop is centered.

    Args:
        width (int): Width of crop region
        height (int): Height of crop region
        x (int, optional): Left coordinate of crop region. If None, centers
            horizontally. Default: None.
        y (int, optional): Top coordinate of crop region. If None, centers
            vertically. Default: None.

    Returns:
        Function that takes a frame and returns cropped frame

    Examples:
        pipe = cropped(width=400, height=300)  # Centered crop
        pipe = cropped(width=400, height=300, x=100, y=100)
        pipe = letterboxed() | cropped(width=640, height=480)
    """
    return PipeableFunction(crop, width=width, height=height, x=x, y=y)


def cropped_to_aspect_ratio(aspect_ratio: Tuple[int, int], x: Optional[int] = None, y: Optional[int] = None):
    """
    Pipeable crop to aspect ratio function - crop frame to aspect ratio with
    pipe operator support.
    If x and y are not provided, the crop is centered.

    Args:
        aspect_ratio (tuple): Target aspect ratio as tuple (e.g., (16, 9), (4, 3), (1, 1))
        x (int, optional): Left coordinate of crop region. If None, centers horizontally.
            Default: None.
        y (int, optional): Top coordinate of crop region. If None, centers vertically.
            Default: None.

    Returns:
        Function that takes a frame and returns cropped frame with target aspect ratio

    Examples:
        pipe = cropped_to_aspect_ratio((16, 9))  # Crop to 16:9 aspect ratio
        pipe = cropped_to_aspect_ratio((4, 3))  # Crop to 4:3 aspect ratio
        pipe = letterboxed() | cropped_to_aspect_ratio((1, 1))  # Square crop
    """
    return PipeableFunction(crop_to_aspect_ratio, aspect_ratio=aspect_ratio, x=x, y=y)


def rotated(
    angle: float,
    center: Optional[Tuple[int, int]] = None,
    expand: bool = False,
    color: Tuple[int, int, int] | Tuple[int, int, int, int] = (0, 0, 0),
    interpolation: int = cv2.INTER_LINEAR,
):
    """
    Pipeable rotate function - rotate frame with pipe operator support.
    If center is not provided, rotates around the image center.

    Args:
        angle (float): Rotation angle in degrees (positive values rotate counter-clockwise)
        center (tuple, optional): Center of rotation as (x, y). If None, uses image center.
            Default: None.
        expand (bool): If True, expands the output frame to fit the entire rotated image.
            If False, output frame size matches input. Default: False.
        color (tuple): BGR or BGRA color for border pixels. Default: (0, 0, 0).
        interpolation (int): OpenCV interpolation method. Default: cv2.INTER_LINEAR.

    Returns:
        Function that takes a frame and returns rotated frame

    Examples:
        pipe = rotated(45)  # Rotate 45 degrees counter-clockwise
        pipe = rotated(-90)  # Rotate 90 degrees clockwise
        pipe = letterboxed() | rotated(30, expand=True)
    """
    return PipeableFunction(rotate, angle=angle, center=center, expand=expand, color=color, interpolation=interpolation)


def adjusted(brightness: float = 0.0, contrast: float = 1.0, saturation: float = 1.0, gamma: float = 1.0):
    """
    Pipeable adjust function - apply image adjustments with pipe operator support.

    Args:
        brightness (float): -1.0 to 1.0. Default: 0.0.
        contrast (float): 0.0 to N. Default: 1.0.
        saturation (float): 0.0 to N. Default: 1.0.
        gamma (float): > 0. Default: 1.0.

    Returns:
        Function that takes a frame and returns adjusted frame

    Examples:
        pipe = adjusted(brightness=0.1, contrast=1.2)
        pipe = letterboxed() | adjusted(saturation=0.8)
    """
    return PipeableFunction(adjust, brightness=brightness, contrast=contrast, saturation=saturation, gamma=gamma)


def greyscaled():
    """
    Pipeable greyscale function - convert frame to greyscale with pipe operator support.

    Returns:
        Function that takes a frame and returns greyscale frame

    Examples:
        pipe = greyscaled()
        pipe = letterboxed() | greyscaled()
    """
    return PipeableFunction(greyscale)


def compressed_to_jpeg(quality: int = 80):
    """
    Pipeable JPEG compression function - compress frame to JPEG with pipe operator support.

    Args:
        quality (int): JPEG quality (0-100, higher = better quality)

    Returns:
        Function that takes a frame and returns compressed JPEG bytes as Numpy array or None

    Examples:
        pipe = compressed_to_jpeg(quality=95)
        pipe = resized(target_size=(640, 480)) | compressed_to_jpeg()
    """
    return PipeableFunction(compress_to_jpeg, quality=quality)


def compressed_to_png(compression_level: int = 6):
    """
    Pipeable PNG compression function - compress frame to PNG with pipe operator support.

    Args:
        compression_level (int): PNG compression level (0-9, higher = better compression)

    Returns:
        Function that takes a frame and returns compressed PNG bytes as Numpy array or None

    Examples:
        pipe = compressed_to_png(compression_level=9)
        pipe = letterboxed() | compressed_to_png()
    """
    return PipeableFunction(compress_to_png, compression_level=compression_level)
