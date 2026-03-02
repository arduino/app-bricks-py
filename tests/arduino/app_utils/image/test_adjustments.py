# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from arduino.app_utils.image.adjustments import (
    letterbox,
    resize,
    adjust,
    split_channels,
    greyscale,
    flip_h,
    flip_v,
    crop,
    crop_to_aspect_ratio,
    rotate,
)


# FIXTURES


def create_gradient_frame(dtype):
    """Helper: Creates a 100x100 3-channel (BGR) frame with gradients."""
    iinfo = np.iinfo(dtype)
    max_val = iinfo.max
    frame = np.zeros((100, 100, 3), dtype=dtype)
    frame[:, :, 0] = np.linspace(0, max_val // 2, 100, dtype=dtype)  # Blue
    frame[:, :, 1] = np.linspace(0, max_val, 100, dtype=dtype)  # Green
    frame[:, :, 2] = np.linspace(max_val // 2, max_val, 100, dtype=dtype)  # Red
    return frame


def create_greyscale_frame(dtype):
    """Helper: Creates a 100x100 1-channel (greyscale) frame."""
    iinfo = np.iinfo(dtype)
    max_val = iinfo.max
    frame = np.zeros((100, 100), dtype=dtype)
    frame[:, :] = np.linspace(0, max_val, 100, dtype=dtype)
    return frame


def create_bgra_frame(dtype):
    """Helper: Creates a 100x100 4-channel (BGRA) frame."""
    iinfo = np.iinfo(dtype)
    max_val = iinfo.max
    bgr = create_gradient_frame(dtype)
    alpha = np.zeros((100, 100), dtype=dtype)
    alpha[:, :] = np.linspace(max_val // 4, max_val, 100, dtype=dtype)
    frame = np.stack([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha], axis=2)
    return frame


# Fixture for a 100x100 uint8 BGR frame
@pytest.fixture
def frame_bgr_uint8():
    return create_gradient_frame(np.uint8)


# Fixture for a 100x100 uint8 BGRA frame
@pytest.fixture
def frame_bgra_uint8():
    return create_bgra_frame(np.uint8)


# Fixture for a 100x100 uint8 greyscale frame
@pytest.fixture
def frame_grey_uint8():
    return create_greyscale_frame(np.uint8)


# Fixtures for high bit-depth frames
@pytest.fixture
def frame_bgr_uint16():
    return create_gradient_frame(np.uint16)


@pytest.fixture
def frame_bgr_uint32():
    return create_gradient_frame(np.uint32)


@pytest.fixture
def frame_bgra_uint16():
    return create_bgra_frame(np.uint16)


@pytest.fixture
def frame_bgra_uint32():
    return create_bgra_frame(np.uint32)


# Fixture for a 200x100 (wide) uint8 BGR frame
@pytest.fixture
def frame_bgr_wide():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # Solid Red
    return frame


# Fixture for a 100x200 (tall) uint8 BGR frame
@pytest.fixture
def frame_bgr_tall():
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    frame[:, :, 1] = 255  # Solid Green
    return frame


# A parameterized fixture to test multiple data types
@pytest.fixture(params=[np.uint8, np.uint16, np.uint32])
def frame_any_dtype(request):
    """Provides a gradient frame for uint8, uint16, and uint32."""
    return create_gradient_frame(request.param)


# TESTS

# ADJUST TESTS


def test_adjust_dtype_preservation(frame_any_dtype):
    """Tests that the dtype of the frame is preserved."""
    dtype = frame_any_dtype.dtype
    adjusted = adjust(frame_any_dtype, brightness=0.1)
    assert adjusted.dtype == dtype


def test_adjust_no_op(frame_bgr_uint8):
    """Tests that default parameters do not change the frame."""
    adjusted = adjust(frame_bgr_uint8)
    assert np.array_equal(frame_bgr_uint8, adjusted)


def test_adjust_brightness(frame_bgr_uint8):
    """Tests brightness adjustment."""
    brighter = adjust(frame_bgr_uint8, brightness=0.1)
    darker = adjust(frame_bgr_uint8, brightness=-0.1)
    assert np.mean(brighter) > np.mean(frame_bgr_uint8)
    assert np.mean(darker) < np.mean(frame_bgr_uint8)


def test_adjust_contrast(frame_bgr_uint8):
    """Tests contrast adjustment."""
    higher_contrast = adjust(frame_bgr_uint8, contrast=1.5)
    lower_contrast = adjust(frame_bgr_uint8, contrast=0.5)
    assert np.std(higher_contrast) > np.std(frame_bgr_uint8)
    assert np.std(lower_contrast) < np.std(frame_bgr_uint8)


def test_adjust_gamma(frame_bgr_uint8):
    """Tests gamma correction."""
    # Gamma < 1.0 (e.g., 0.5) ==> brightens
    brighter = adjust(frame_bgr_uint8, gamma=0.5)
    # Gamma > 1.0 (e.g., 2.0) ==> darkens
    darker = adjust(frame_bgr_uint8, gamma=2.0)
    assert np.mean(brighter) > np.mean(frame_bgr_uint8)
    assert np.mean(darker) < np.mean(frame_bgr_uint8)


def test_adjust_saturation_to_greyscale(frame_bgr_uint8):
    """Tests that saturation=0.0 makes all color channels equal."""
    desaturated = adjust(frame_bgr_uint8, saturation=0.0)
    b, g, r = split_channels(desaturated)
    assert np.allclose(b, g, atol=1)
    assert np.allclose(g, r, atol=1)


def test_adjust_greyscale_input(frame_grey_uint8):
    """Tests that greyscale frames are handled safely."""
    adjusted = adjust(frame_grey_uint8, saturation=1.5, brightness=0.1)
    assert adjusted.ndim == 2
    assert adjusted.dtype == np.uint8
    assert np.mean(adjusted) > np.mean(frame_grey_uint8)


def test_adjust_bgra_input(frame_bgra_uint8):
    """Tests that BGRA frames are handled safely and alpha is preserved."""
    original_alpha = frame_bgra_uint8[:, :, 3]

    adjusted = adjust(frame_bgra_uint8, saturation=0.0, brightness=0.1)

    assert adjusted.ndim == 3
    assert adjusted.shape[2] == 4
    assert adjusted.dtype == np.uint8

    b, g, r, a = split_channels(adjusted)
    assert np.allclose(b, g, atol=1)  # Check desaturation
    assert np.allclose(g, r, atol=1)  # Check desaturation
    assert np.array_equal(original_alpha, a)  # Check alpha preservation


def test_adjust_gamma_zero_error(frame_bgr_uint8):
    """Tests that gamma <= 0 raises a ValueError."""
    with pytest.raises(ValueError, match="Gamma value must be greater than 0."):
        adjust(frame_bgr_uint8, gamma=0.0)

    with pytest.raises(ValueError, match="Gamma value must be greater than 0."):
        adjust(frame_bgr_uint8, gamma=-1.0)


def test_adjust_high_bit_depth_bgr(frame_bgr_uint16, frame_bgr_uint32):
    """
    Tests that brightness/contrast logic is correct on high bit-depth images.
    This validates that the float64 conversion is working.
    """
    # Test uint16
    brighter_16 = adjust(frame_bgr_uint16, brightness=0.1)
    darker_16 = adjust(frame_bgr_uint16, brightness=-0.1)
    assert np.mean(brighter_16) > np.mean(frame_bgr_uint16)
    assert np.mean(darker_16) < np.mean(frame_bgr_uint16)

    # Test uint32
    brighter_32 = adjust(frame_bgr_uint32, brightness=0.1)
    darker_32 = adjust(frame_bgr_uint32, brightness=-0.1)
    assert np.mean(brighter_32) > np.mean(frame_bgr_uint32)
    assert np.mean(darker_32) < np.mean(frame_bgr_uint32)


def test_adjust_high_bit_depth_bgra(frame_bgra_uint16, frame_bgra_uint32):
    """
    Tests that brightness/contrast logic is correct on high bit-depth
    BGRA images and that the alpha channel is preserved.
    """
    # Test uint16
    original_alpha_16 = frame_bgra_uint16[:, :, 3]
    brighter_16 = adjust(frame_bgra_uint16, brightness=0.1)
    assert brighter_16.dtype == np.uint16
    assert brighter_16.shape == frame_bgra_uint16.shape
    _, _, _, a16 = split_channels(brighter_16)
    assert np.array_equal(original_alpha_16, a16)
    assert np.mean(brighter_16) > np.mean(frame_bgra_uint16)

    # Test uint32
    original_alpha_32 = frame_bgra_uint32[:, :, 3]
    brighter_32 = adjust(frame_bgra_uint32, brightness=0.1)
    assert brighter_32.dtype == np.uint32
    assert brighter_32.shape == frame_bgra_uint32.shape
    _, _, _, a32 = split_channels(brighter_32)
    assert np.array_equal(original_alpha_32, a32)
    assert np.mean(original_alpha_32) > np.mean(frame_bgra_uint32)


# GREYSCALE TESTS


def test_greyscale(frame_bgr_uint8, frame_bgra_uint8, frame_grey_uint8):
    """Tests the standalone greyscale function."""
    # Test on BGR
    greyscaled_bgr = greyscale(frame_bgr_uint8)
    assert greyscaled_bgr.ndim == 3
    assert greyscaled_bgr.shape[2] == 3
    b, g, r = split_channels(greyscaled_bgr)
    assert np.allclose(b, g, atol=1)
    assert np.allclose(g, r, atol=1)

    # Test on BGRA
    original_alpha = frame_bgra_uint8[:, :, 3]
    greyscaled_bgra = greyscale(frame_bgra_uint8)
    assert greyscaled_bgra.ndim == 3
    assert greyscaled_bgra.shape[2] == 4
    b, g, r, a = split_channels(greyscaled_bgra)
    assert np.allclose(b, g, atol=1)
    assert np.allclose(g, r, atol=1)
    assert np.array_equal(original_alpha, a)

    # Test on 2D Greyscale (should be no-op)
    greyscaled_grey = greyscale(frame_grey_uint8)
    assert np.array_equal(frame_grey_uint8, greyscaled_grey)
    assert greyscaled_grey.ndim == 2


def test_greyscale_dtype_preservation(frame_any_dtype):
    """Tests that the dtype of the frame is preserved."""
    dtype = frame_any_dtype.dtype
    adjusted = adjust(frame_any_dtype, brightness=0.1)
    assert adjusted.dtype == dtype


def test_greyscale_high_bit_depth(frame_bgr_uint16, frame_bgr_uint32):
    """
    Tests that greyscale logic is correct on high bit-depth images.
    """
    # Test uint16
    greyscaled_16 = greyscale(frame_bgr_uint16)
    assert greyscaled_16.dtype == np.uint16
    assert greyscaled_16.shape == frame_bgr_uint16.shape
    b16, g16, r16 = split_channels(greyscaled_16)
    assert np.allclose(b16, g16, atol=1)
    assert np.allclose(g16, r16, atol=1)
    assert np.mean(b16) != np.mean(frame_bgr_uint16[:, :, 0])

    # Test uint32
    greyscaled_32 = greyscale(frame_bgr_uint32)
    assert greyscaled_32.dtype == np.uint32
    assert greyscaled_32.shape == frame_bgr_uint32.shape
    b32, g32, r32 = split_channels(greyscaled_32)
    assert np.allclose(b32, g32, atol=1)
    assert np.allclose(g32, r32, atol=1)
    assert np.mean(b32) != np.mean(frame_bgr_uint32[:, :, 0])


def test_high_bit_depth_greyscale_bgra_content(frame_bgra_uint16, frame_bgra_uint32):
    """
    Tests that greyscale logic is correct on high bit-depth
    BGRA images and that the alpha channel is preserved.
    """
    # Test uint16
    original_alpha_16 = frame_bgra_uint16[:, :, 3]
    greyscaled_16 = greyscale(frame_bgra_uint16)
    assert greyscaled_16.dtype == np.uint16
    assert greyscaled_16.shape == frame_bgra_uint16.shape
    b16, g16, r16, a16 = split_channels(greyscaled_16)
    assert np.allclose(b16, g16, atol=1)
    assert np.allclose(g16, r16, atol=1)
    assert np.array_equal(original_alpha_16, a16)

    # Test uint32
    original_alpha_32 = frame_bgra_uint32[:, :, 3]
    greyscaled_32 = greyscale(frame_bgra_uint32)
    assert greyscaled_32.dtype == np.uint32
    assert greyscaled_32.shape == frame_bgra_uint32.shape
    b32, g32, r32, a32 = split_channels(greyscaled_32)
    assert np.allclose(b32, g32, atol=1)
    assert np.allclose(g32, r32, atol=1)
    assert np.array_equal(original_alpha_32, a32)


# RESIZE TESTS


def test_resize_shape_and_dtype(frame_bgr_uint8, frame_bgra_uint8, frame_grey_uint8):
    """Tests that resize produces the correct shape and preserves dtype."""
    target_w, target_h = 50, 75

    # Test BGR
    resized_bgr = resize(frame_bgr_uint8, (target_w, target_h))
    assert resized_bgr.shape == (target_h, target_w, 3)
    assert resized_bgr.dtype == frame_bgr_uint8.dtype

    # Test BGRA
    resized_bgra = resize(frame_bgra_uint8, (target_w, target_h))
    assert resized_bgra.shape == (target_h, target_w, 4)
    assert resized_bgra.dtype == frame_bgra_uint8.dtype

    # Test Greyscale
    resized_grey = resize(frame_grey_uint8, (target_w, target_h))
    assert resized_grey.shape == (target_h, target_w)
    assert resized_grey.dtype == frame_grey_uint8.dtype


# LETTERBOX TESTS


def test_letterbox_wide_image(frame_bgr_wide):
    """Tests letterboxing a wide image (200x100) into a square (200x200)."""
    target_w, target_h = 200, 200
    # Frame is 200x100, solid red (255)
    # Scale = min(200/200, 200/100) = min(1, 2) = 1
    # new_w = 200 * 1 = 200
    # new_h = 100 * 1 = 100
    # y_offset = (200 - 100) // 2 = 50
    # x_offset = (200 - 200) // 2 = 0

    letterboxed = letterbox(frame_bgr_wide, (target_w, target_h), color=(0, 0, 0))

    assert letterboxed.shape == (target_h, target_w, 3)
    assert letterboxed.dtype == frame_bgr_wide.dtype

    # Check padding (top row, black)
    assert np.all(letterboxed[0, 0] == [0, 0, 0])
    # Check padding (bottom row, black)
    assert np.all(letterboxed[199, 199] == [0, 0, 0])
    # Check image data (center row, red)
    assert np.all(letterboxed[100, 100] == [0, 0, 255])
    # Check image edge (no left/right padding)
    assert np.all(letterboxed[100, 0] == [0, 0, 255])


def test_letterbox_tall_image(frame_bgr_tall):
    """Tests letterboxing a tall image (100x200) into a square (200x200)."""
    target_w, target_h = 200, 200
    # Frame is 100x200, solid green (255)
    # Scale = min(200/100, 200/200) = min(2, 1) = 1
    # new_w = 100 * 1 = 100
    # new_h = 200 * 1 = 200
    # y_offset = (200 - 200) // 2 = 0
    # x_offset = (200 - 100) // 2 = 50

    letterboxed = letterbox(frame_bgr_tall, (target_w, target_h), color=(0, 0, 0))

    assert letterboxed.shape == (target_h, target_w, 3)
    assert letterboxed.dtype == frame_bgr_tall.dtype

    # Check padding (left column, black)
    assert np.all(letterboxed[0, 0] == [0, 0, 0])
    # Check padding (right column, black)
    assert np.all(letterboxed[199, 199] == [0, 0, 0])
    # Check image data (center column, green)
    assert np.all(letterboxed[100, 100] == [0, 255, 0])
    # Check image edge (no top/bottom padding)
    assert np.all(letterboxed[0, 100] == [0, 255, 0])


def test_letterbox_color(frame_bgr_tall):
    """Tests letterboxing with a non-default color."""
    white = (255, 255, 255)
    letterboxed = letterbox(frame_bgr_tall, (200, 200), color=white)

    # Check padding (left column, white)
    assert np.all(letterboxed[0, 0] == white)
    # Check image data (center column, green)
    assert np.all(letterboxed[100, 100] == [0, 255, 0])


def test_letterbox_bgra(frame_bgra_uint8):
    """Tests letterboxing on a 4-channel BGRA image."""
    target_w, target_h = 200, 200
    # Opaque black padding
    padding = (0, 0, 0, 255)

    letterboxed = letterbox(frame_bgra_uint8, (target_w, target_h), color=padding)

    assert letterboxed.shape == (target_h, target_w, 4)
    # Check no padding (corner, original BGRA point)
    assert np.array_equal(letterboxed[0, 0], frame_bgra_uint8[0, 0])
    # Check image data (center, from fixture) - allow small tolerance for numerical precision differences
    assert np.allclose(letterboxed[100, 100], frame_bgra_uint8[50, 50], atol=1)


def test_letterbox_greyscale(frame_grey_uint8):
    """Tests letterboxing on a 2D greyscale image."""
    target_w, target_h = 200, 200
    letterboxed = letterbox(frame_grey_uint8, (target_w, target_h), color=(0, 0, 0))

    assert letterboxed.shape == (target_h, target_w)
    assert letterboxed.ndim == 2
    # Check padding (corner, black)
    assert letterboxed[0, 0] == 0
    # Check image data (center) - allow small tolerance for numerical precision differences
    assert np.allclose(letterboxed[100, 100], frame_grey_uint8[50, 50], atol=1)


def test_letterbox_none_target_size(frame_bgr_wide, frame_bgr_tall):
    """Tests that target_size=None creates a square based on the longest side."""
    # frame_bgr_wide is 200x100, longest side is 200
    letterboxed_wide = letterbox(frame_bgr_wide, target_size=None)
    assert letterboxed_wide.shape == (200, 200, 3)

    # frame_bgr_tall is 100x200, longest side is 200
    letterboxed_tall = letterbox(frame_bgr_tall, target_size=None)
    assert letterboxed_tall.shape == (200, 200, 3)


def test_letterbox_color_tuple_error(frame_bgr_uint8):
    """Tests that a mismatched padding tuple raises a ValueError."""
    with pytest.raises(ValueError, match="color length"):
        # BGR (3-ch) frame with 4-ch padding
        letterbox(frame_bgr_uint8, (200, 200), color=(0, 0, 0, 0))

    with pytest.raises(ValueError, match="color length"):
        # BGRA (4-ch) frame with 3-ch padding
        frame_bgra = create_bgra_frame(np.uint8)
        letterbox(frame_bgra, (200, 200), color=(0, 0, 0))


# FLIP TESTS


def test_flip_h_bgr(frame_bgr_uint8):
    """Test horizontal flip for BGR image."""
    flipped = flip_h(frame_bgr_uint8)
    # Flipping twice should return the original
    assert np.array_equal(flip_h(flipped), frame_bgr_uint8)
    # Check that first column becomes last
    assert np.array_equal(flipped[:, 0, :], frame_bgr_uint8[:, -1, :])
    assert np.array_equal(flipped[:, -1, :], frame_bgr_uint8[:, 0, :])


def test_flip_v_bgr(frame_bgr_uint8):
    """Test vertical flip for BGR image."""
    flipped = flip_v(frame_bgr_uint8)
    # Flipping twice should return the original
    assert np.array_equal(flip_v(flipped), frame_bgr_uint8)
    # Check that first row becomes last
    assert np.array_equal(flipped[0, :, :], frame_bgr_uint8[-1, :, :])
    assert np.array_equal(flipped[-1, :, :], frame_bgr_uint8[0, :, :])


def test_flip_h_greyscale(frame_grey_uint8):
    """Test horizontal flip for greyscale image."""
    flipped = flip_h(frame_grey_uint8)
    assert np.array_equal(flip_h(flipped), frame_grey_uint8)
    assert np.array_equal(flipped[:, 0], frame_grey_uint8[:, -1])
    assert np.array_equal(flipped[:, -1], frame_grey_uint8[:, 0])


def test_flip_v_greyscale(frame_grey_uint8):
    """Test vertical flip for greyscale image."""
    flipped = flip_v(frame_grey_uint8)
    assert np.array_equal(flip_v(flipped), frame_grey_uint8)
    assert np.array_equal(flipped[0, :], frame_grey_uint8[-1, :])
    assert np.array_equal(flipped[-1, :], frame_grey_uint8[0, :])


def test_flip_h_bgra(frame_bgra_uint8):
    """Test horizontal flip for BGRA image."""
    flipped = flip_h(frame_bgra_uint8)
    assert np.array_equal(flip_h(flipped), frame_bgra_uint8)
    assert np.array_equal(flipped[:, 0, :], frame_bgra_uint8[:, -1, :])
    assert np.array_equal(flipped[:, -1, :], frame_bgra_uint8[:, 0, :])


def test_flip_v_bgra(frame_bgra_uint8):
    """Test vertical flip for BGRA image."""
    flipped = flip_v(frame_bgra_uint8)
    assert np.array_equal(flip_v(flipped), frame_bgra_uint8)
    assert np.array_equal(flipped[0, :, :], frame_bgra_uint8[-1, :, :])
    assert np.array_equal(flipped[-1, :, :], frame_bgra_uint8[0, :, :])


# CROP TESTS


def test_crop_centered(frame_bgr_uint8):
    """Test crop with default centered behavior."""
    cropped = crop(frame_bgr_uint8, width=50, height=50)
    assert cropped.shape == (50, 50, 3)
    assert cropped.dtype == frame_bgr_uint8.dtype
    # Center of 100x100 frame with 50x50 crop: x=25, y=25
    assert np.array_equal(cropped[0, 0], frame_bgr_uint8[25, 25])


def test_crop_explicit_position(frame_bgr_uint8):
    """Test crop with explicit x, y position."""
    cropped = crop(frame_bgr_uint8, width=30, height=40, x=10, y=20)
    assert cropped.shape == (40, 30, 3)
    assert np.array_equal(cropped[0, 0], frame_bgr_uint8[20, 10])
    assert np.array_equal(cropped[-1, -1], frame_bgr_uint8[20 + 40 - 1, 10 + 30 - 1])


def test_crop_bounds_clipping(frame_bgr_uint8):
    """Test that crop clips to frame boundaries."""
    # Request crop that would go out of bounds
    cropped = crop(frame_bgr_uint8, width=80, height=80, x=50, y=50)
    # Should clip to 50x50 (from (50,50) to (100,100))
    assert cropped.shape == (50, 50, 3)
    assert np.array_equal(cropped[0, 0], frame_bgr_uint8[50, 50])


def test_crop_greyscale(frame_grey_uint8):
    """Test crop on greyscale image."""
    cropped = crop(frame_grey_uint8, width=40, height=60)
    assert cropped.shape == (60, 40)
    assert cropped.ndim == 2
    assert cropped.dtype == frame_grey_uint8.dtype


def test_crop_bgra(frame_bgra_uint8):
    """Test crop on BGRA image."""
    cropped = crop(frame_bgra_uint8, width=50, height=50, x=10, y=10)
    assert cropped.shape == (50, 50, 4)
    assert cropped.dtype == frame_bgra_uint8.dtype
    assert np.array_equal(cropped[0, 0], frame_bgra_uint8[10, 10])


def test_crop_high_bit_depth(frame_bgr_uint16, frame_bgr_uint32):
    """Test crop preserves high bit-depth dtypes."""
    cropped_16 = crop(frame_bgr_uint16, width=50, height=50)
    assert cropped_16.dtype == np.uint16
    assert cropped_16.shape == (50, 50, 3)

    cropped_32 = crop(frame_bgr_uint32, width=50, height=50)
    assert cropped_32.dtype == np.uint32
    assert cropped_32.shape == (50, 50, 3)


# CROP TO ASPECT RATIO TESTS


def test_crop_to_aspect_ratio_wider(frame_bgr_wide):
    """Test cropping a wide image (200x100) to 1:1 aspect ratio."""
    # frame_bgr_wide is 200x100, aspect 2:1
    # Target is 1:1, so we crop width to match height
    # new_width = 100 * (1/1) = 100
    # Centered: x = (200 - 100) // 2 = 50
    cropped = crop_to_aspect_ratio(frame_bgr_wide, aspect_ratio=(1, 1))
    assert cropped.shape == (100, 100, 3)
    assert cropped.dtype == frame_bgr_wide.dtype
    # Check that we got the center portion
    assert np.array_equal(cropped[0, 0], frame_bgr_wide[0, 50])


def test_crop_to_aspect_ratio_taller(frame_bgr_tall):
    """Test cropping a tall image (100x200) to 1:1 aspect ratio."""
    # frame_bgr_tall is 100x200, aspect 1:2
    # Target is 1:1, so we crop height to match width
    # new_height = 100 * (1/1) = 100
    # Centered: y = (200 - 100) // 2 = 50
    cropped = crop_to_aspect_ratio(frame_bgr_tall, aspect_ratio=(1, 1))
    assert cropped.shape == (100, 100, 3)
    assert cropped.dtype == frame_bgr_tall.dtype
    # Check that we got the center portion
    assert np.array_equal(cropped[0, 0], frame_bgr_tall[50, 0])


def test_crop_to_aspect_ratio_16_9(frame_bgr_uint8):
    """Test cropping to 16:9 aspect ratio."""
    # frame_bgr_uint8 is 100x100, aspect 1:1
    # Target is 16:9
    # current_aspect = 100/100 = 1.0
    # target_aspect = 16/9 ≈ 1.778
    # Since current < target, we need to crop height
    # new_height = 100 / (16/9) = 100 * 9/16 = 56.25 -> 56
    cropped = crop_to_aspect_ratio(frame_bgr_uint8, aspect_ratio=(16, 9))
    assert cropped.shape[1] == 100  # Width unchanged
    assert cropped.shape[0] == 56  # Height cropped
    assert cropped.shape[2] == 3


def test_crop_to_aspect_ratio_explicit_position(frame_bgr_uint8):
    """Test crop_to_aspect_ratio with explicit x, y position."""
    cropped = crop_to_aspect_ratio(frame_bgr_uint8, aspect_ratio=(1, 1), x=10, y=20)
    assert cropped.shape == (80, 90, 3)  # Clipped from requested position
    # Starting from (10, 20) with shape that gets clipped
    assert np.array_equal(cropped[0, 0], frame_bgr_uint8[20, 10])


def test_crop_to_aspect_ratio_greyscale(frame_grey_uint8):
    """Test crop_to_aspect_ratio on greyscale image."""
    cropped = crop_to_aspect_ratio(frame_grey_uint8, aspect_ratio=(1, 1))
    assert cropped.shape == (100, 100)
    assert cropped.ndim == 2
    assert cropped.dtype == frame_grey_uint8.dtype


def test_crop_to_aspect_ratio_bgra(frame_bgra_uint8):
    """Test crop_to_aspect_ratio on BGRA image."""
    cropped = crop_to_aspect_ratio(frame_bgra_uint8, aspect_ratio=(2, 1))
    assert cropped.shape[2] == 4
    assert cropped.dtype == frame_bgra_uint8.dtype
    # 100x100 to 2:1 -> crop height to 50
    assert cropped.shape == (50, 100, 4)


# ROTATE TESTS


def test_rotate_no_op(frame_bgr_uint8):
    """Test that 0-degree rotation returns unchanged frame."""
    rotated = rotate(frame_bgr_uint8, angle=0)
    assert np.array_equal(rotated, frame_bgr_uint8)


def test_rotate_90_degrees(frame_bgr_uint8):
    """Test 90-degree rotation (counter-clockwise)."""
    rotated = rotate(frame_bgr_uint8, angle=90)
    # 90° counter-clockwise: (100, 100, 3) -> (100, 100, 3)
    assert rotated.shape == frame_bgr_uint8.shape
    assert rotated.dtype == frame_bgr_uint8.dtype
    # Top-left corner becomes bottom-left corner
    assert np.array_equal(rotated[-1, 0], frame_bgr_uint8[0, 0])


def test_rotate_180_degrees(frame_bgr_uint8):
    """Test 180-degree rotation."""
    rotated = rotate(frame_bgr_uint8, angle=180)
    assert rotated.shape == frame_bgr_uint8.shape
    # 180° rotation: corners swap diagonally
    assert np.array_equal(rotated[0, 0], frame_bgr_uint8[-1, -1])
    assert np.array_equal(rotated[-1, -1], frame_bgr_uint8[0, 0])


def test_rotate_270_degrees(frame_bgr_uint8):
    """Test 270-degree rotation (counter-clockwise = 90° clockwise)."""
    rotated = rotate(frame_bgr_uint8, angle=270)
    assert rotated.shape == frame_bgr_uint8.shape
    assert rotated.dtype == frame_bgr_uint8.dtype
    # Top-left corner becomes top-right corner
    assert np.array_equal(rotated[0, -1], frame_bgr_uint8[0, 0])


def test_rotate_arbitrary_angle(frame_bgr_uint8):
    """Test rotation by arbitrary angle (45 degrees)."""
    rotated = rotate(frame_bgr_uint8, angle=45)
    # Shape remains 100x100 without expand
    assert rotated.shape == frame_bgr_uint8.shape
    assert rotated.dtype == frame_bgr_uint8.dtype


def test_rotate_expand(frame_bgr_uint8):
    """Test rotation with expand=True to fit entire rotated image."""
    rotated = rotate(frame_bgr_uint8, angle=45, expand=True)
    # Diagonal of 100x100 square is ~141, so expanded frame is larger
    assert rotated.shape[0] > 100
    assert rotated.shape[1] > 100
    assert rotated.shape[2] == 3
    assert rotated.dtype == frame_bgr_uint8.dtype


def test_rotate_border_color_bgr(frame_bgr_uint8):
    """Test rotation with custom border color for BGR image."""
    white = (255, 255, 255)
    rotated = rotate(frame_bgr_uint8, angle=45, color=white)
    # Corners will have border color due to rotation
    # Check a corner pixel (may not be exactly white due to interpolation)
    assert rotated[0, 0, 0] > 200  # Close to white


def test_rotate_border_color_bgra(frame_bgra_uint8):
    """Test rotation with custom border color for BGRA image."""
    white_opaque = (255, 255, 255, 255)
    rotated = rotate(frame_bgra_uint8, angle=45, color=white_opaque)
    assert rotated.shape[2] == 4
    assert rotated.dtype == frame_bgra_uint8.dtype
    # Corners will have border color
    assert rotated[0, 0, 3] > 200  # Alpha channel near 255


def test_rotate_greyscale(frame_grey_uint8):
    """Test rotation on greyscale image."""
    rotated = rotate(frame_grey_uint8, angle=90)
    assert rotated.shape == frame_grey_uint8.shape
    assert rotated.ndim == 2
    assert rotated.dtype == frame_grey_uint8.dtype
    # Top-left becomes bottom-left
    assert rotated[-1, 0] == frame_grey_uint8[0, 0]


def test_rotate_normalized_angles(frame_bgr_uint8):
    """Test that angles > 360 and < 0 are normalized correctly."""
    rotated_450 = rotate(frame_bgr_uint8, angle=450)  # Same as 90
    rotated_90 = rotate(frame_bgr_uint8, angle=90)
    assert np.array_equal(rotated_450, rotated_90)

    rotated_minus_90 = rotate(frame_bgr_uint8, angle=-90)  # Same as 270
    rotated_270 = rotate(frame_bgr_uint8, angle=270)
    assert np.array_equal(rotated_minus_90, rotated_270)
