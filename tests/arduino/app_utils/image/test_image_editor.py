# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import numpy as np
import cv2
from unittest.mock import patch
from arduino.app_utils.image.image_editor import (
    ImageEditor, 
    letterboxed, 
    resized, 
    adjusted, 
    greyscaled, 
    compressed_to_jpeg, 
    compressed_to_png
)
from arduino.app_utils.image.pipeable import PipeableFunction


class TestImageEditor:
    """Test cases for the ImageEditor class."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample RGB frame for testing."""
        # Create a 100x80 RGB frame with some pattern
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:, :40] = [255, 0, 0]  # Red left section
        frame[:, 40:] = [0, 255, 0]  # Green right section
        return frame
    
    @pytest.fixture
    def sample_grayscale_frame(self):
        """Create a sample grayscale frame for testing."""
        return np.random.randint(0, 256, (80, 100), dtype=np.uint8)
    
    def test_letterbox_make_square(self, sample_frame):
        """Test letterboxing to make frame square."""
        result = ImageEditor.letterbox(sample_frame)
        
        # Should make it square based on larger dimension (100)
        assert result.shape[:2] == (100, 100)
        assert result.shape[2] == 3  # Still RGB
    
    def test_letterbox_specific_size(self, sample_frame):
        """Test letterboxing to specific target size."""
        target_size = (200, 150)
        result = ImageEditor.letterbox(sample_frame, target_size=target_size)
        
        assert result.shape[:2] == (150, 200)  # Height, Width
        assert result.shape[2] == 3  # Still RGB
    
    def test_letterbox_custom_color(self, sample_frame):
        """Test letterboxing with custom padding color."""
        target_size = (200, 200)
        custom_color = (255, 255, 0)  # Yellow
        result = ImageEditor.letterbox(sample_frame, target_size=target_size, color=custom_color)
        
        assert result.shape[:2] == (200, 200)
        # Check that padding areas have the custom color
        # Top and bottom should have yellow padding
        assert np.array_equal(result[0, 0], custom_color)
    
    def test_resize_basic(self, sample_frame):
        """Test basic resizing functionality."""
        target_size = (50, 40)  # Smaller than original
        result = ImageEditor.resize(sample_frame, target_size=target_size)
        
        assert result.shape[:2] == (50, 40)
        assert result.shape[2] == 3  # Still RGB
    
    def test_resize_with_letterboxing(self, sample_frame):
        """Test resizing with maintain_ratio==True (uses letterboxing)."""
        target_size = (200, 200)
        result = ImageEditor.resize(sample_frame, target_size=target_size, maintain_ratio=True)
        
        assert result.shape[:2] == (200, 200)
        assert result.shape[2] == 3  # Still RGB
    
    def test_resize_interpolation_methods(self, sample_frame):
        """Test different interpolation methods."""
        target_size = (50, 40)
        
        # Test different interpolation methods
        for interpolation in [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]:
            result = ImageEditor.resize(sample_frame, target_size=target_size, interpolation=interpolation)
            assert result.shape[:2] == (50, 40)
    
    def test_adjust_brightness(self, sample_frame):
        """Test brightness adjustment."""
        # Increase brightness
        result = ImageEditor.adjust(sample_frame, brightness=50)
        assert result.shape == sample_frame.shape
        # Brightness should increase (but clipped at 255)
        assert np.all(result >= sample_frame)
        
        # Decrease brightness - should clamp at 0, so all values <= original
        result = ImageEditor.adjust(sample_frame, brightness=-50)
        assert result.shape == sample_frame.shape
        assert result.dtype == sample_frame.dtype
        assert np.all(result <= sample_frame)
        # Values should never go below 0
        assert np.all(result >= 0)
    
    def test_adjust_contrast(self, sample_frame):
        """Test contrast adjustment."""
        # Increase contrast
        result = ImageEditor.adjust(sample_frame, contrast=1.5)
        assert result.shape == sample_frame.shape
        
        # Decrease contrast
        result = ImageEditor.adjust(sample_frame, contrast=0.5)
        assert result.shape == sample_frame.shape
    
    def test_adjust_saturation(self, sample_frame):
        """Test saturation adjustment."""
        # Increase saturation
        result = ImageEditor.adjust(sample_frame, saturation=1.5)
        assert result.shape == sample_frame.shape
        
        # Decrease saturation (towards grayscale)
        result = ImageEditor.adjust(sample_frame, saturation=0.5)
        assert result.shape == sample_frame.shape
        
        # Zero saturation should be grayscale
        result = ImageEditor.adjust(sample_frame, saturation=0.0)
        # All channels should be equal for grayscale
        assert np.allclose(result[:, :, 0], result[:, :, 1], atol=1)
        assert np.allclose(result[:, :, 1], result[:, :, 2], atol=1)
    
    def test_adjust_combined(self, sample_frame):
        """Test combined brightness, contrast, and saturation adjustment."""
        result = ImageEditor.adjust(sample_frame, brightness=10, contrast=1.2, saturation=0.8)
        assert result.shape == sample_frame.shape
    
    def test_greyscale_conversion(self, sample_frame):
        """Test grayscale conversion."""
        result = ImageEditor.greyscale(sample_frame)
        
        assert len(result.shape) == 3 and result.shape[2] == 3
        assert result.shape[:2] == sample_frame.shape[:2]
    
    @patch('cv2.imencode')
    def test_compress_to_jpeg_success(self, mock_imencode, sample_frame):
        """Test successful JPEG compression."""
        mock_encoded = np.array([1, 2, 3, 4], dtype=np.uint8)
        mock_imencode.return_value = (True, mock_encoded)
        
        result = ImageEditor.compress_to_jpeg(sample_frame, quality=85)
        
        assert np.array_equal(result, mock_encoded)
        mock_imencode.assert_called_once()
        args, kwargs = mock_imencode.call_args
        assert args[0] == '.jpg'
        assert np.array_equal(args[1], sample_frame)
        assert args[2] == [cv2.IMWRITE_JPEG_QUALITY, 85]
    
    @patch('cv2.imencode')
    def test_compress_to_jpeg_failure(self, mock_imencode, sample_frame):
        """Test failed JPEG compression."""
        mock_imencode.return_value = (False, None)
        
        result = ImageEditor.compress_to_jpeg(sample_frame)
        
        assert result is None
    
    @patch('cv2.imencode')
    def test_compress_to_jpeg_exception(self, mock_imencode, sample_frame):
        """Test JPEG compression with exception."""
        mock_imencode.side_effect = Exception("Encoding error")
        
        result = ImageEditor.compress_to_jpeg(sample_frame)
        
        assert result is None
    
    @patch('cv2.imencode')
    def test_compress_to_png_success(self, mock_imencode, sample_frame):
        """Test successful PNG compression."""
        mock_encoded = np.array([1, 2, 3, 4], dtype=np.uint8)
        mock_imencode.return_value = (True, mock_encoded)
        
        result = ImageEditor.compress_to_png(sample_frame, compression_level=6)
        
        assert np.array_equal(result, mock_encoded)
        mock_imencode.assert_called_once()
        args, kwargs = mock_imencode.call_args
        assert args[0] == '.png'
        assert args[2] == [cv2.IMWRITE_PNG_COMPRESSION, 6]
    
    def test_compress_to_jpeg_dtype_preservation(self, sample_frame):
        """Test JPEG compression preserves input dtype."""
        # Create frame with different dtype
        frame_16bit = sample_frame.astype(np.uint16) * 256
        
        with patch('cv2.imencode') as mock_imencode:
            mock_imencode.return_value = (True, np.array([1, 2, 3]))
            result = ImageEditor.compress_to_jpeg(frame_16bit)
            
            args, kwargs = mock_imencode.call_args
            encoded_frame = args[1]
            assert encoded_frame.dtype == np.uint16
    
    def test_compress_to_png_dtype_preservation(self, sample_frame):
        """Test PNG compression preserves input dtype."""
        # Create frame with different dtype
        frame_16bit = sample_frame.astype(np.uint16) * 256
        
        with patch('cv2.imencode') as mock_imencode:
            mock_imencode.return_value = (True, np.array([1, 2, 3]))
            result = ImageEditor.compress_to_png(frame_16bit)
            
            args, kwargs = mock_imencode.call_args
            encoded_frame = args[1]
            assert encoded_frame.dtype == np.uint16


class TestPipeableFunctions:
    """Test cases for the pipeable wrapper functions."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample RGB frame for testing."""
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:, :40] = [255, 0, 0]  # Red left section
        frame[:, 40:] = [0, 255, 0]  # Green right section
        return frame
    
    def test_letterboxed_function_returns_pipeable(self):
        """Test that letterboxed function returns PipeableFunction."""
        result = letterboxed(target_size=(200, 200))
        assert isinstance(result, PipeableFunction)
    
    def test_letterboxed_pipe_operator(self, sample_frame):
        """Test letterboxed function with pipe operator."""
        result = letterboxed(target_size=(200, 200))(sample_frame)
        
        assert result.shape[:2] == (200, 200)
        assert result.shape[2] == 3
    
    def test_resized_function_returns_pipeable(self):
        """Test that resized function returns PipeableFunction."""
        result = resized(target_size=(50, 40))
        assert isinstance(result, PipeableFunction)
    
    def test_resized_pipe_operator(self, sample_frame):
        """Test resized function with pipe operator."""
        result = resized(target_size=(50, 40))(sample_frame)
        
        assert result.shape[:2] == (50, 40)
        assert result.shape[2] == 3
    
    def test_adjusted_function_returns_pipeable(self):
        """Test that adjusted function returns PipeableFunction."""
        result = adjusted(brightness=10, contrast=1.2)
        assert isinstance(result, PipeableFunction)
    
    def test_adjusted_pipe_operator(self, sample_frame):
        """Test adjusted function with pipe operator."""
        result = adjusted(brightness=10, contrast=1.2, saturation=0.8)(sample_frame)
        
        assert result.shape == sample_frame.shape
    
    def test_greyscaled_function_returns_pipeable(self):
        """Test that greyscaled function returns PipeableFunction."""
        result = greyscaled()
        assert isinstance(result, PipeableFunction)
    
    def test_greyscaled_pipe_operator(self, sample_frame):
        """Test greyscaled function with pipe operator."""
        result = greyscaled()(sample_frame)
        
        # Should have three channels
        assert len(result.shape) == 3 and result.shape[2] == 3
    
    def test_compressed_to_jpeg_function_returns_pipeable(self):
        """Test that compressed_to_jpeg function returns PipeableFunction."""
        result = compressed_to_jpeg(quality=85)
        assert isinstance(result, PipeableFunction)
    
    @patch('cv2.imencode')
    def test_compressed_to_jpeg_pipe_operator(self, mock_imencode, sample_frame):
        """Test compressed_to_jpeg function with pipe operator."""
        mock_encoded = np.array([1, 2, 3, 4], dtype=np.uint8)
        mock_imencode.return_value = (True, mock_encoded)
        
        pipe = compressed_to_jpeg(quality=85)
        result = pipe(sample_frame) 
        
        assert np.array_equal(result, mock_encoded)
    
    def test_compressed_to_png_function_returns_pipeable(self):
        """Test that compressed_to_png function returns PipeableFunction."""
        result = compressed_to_png(compression_level=6)
        assert isinstance(result, PipeableFunction)
    
    @patch('cv2.imencode')
    def test_compressed_to_png_pipe_operator(self, mock_imencode, sample_frame):
        """Test compressed_to_png function with pipe operator."""
        mock_encoded = np.array([1, 2, 3, 4], dtype=np.uint8)
        mock_imencode.return_value = (True, mock_encoded)
        
        pipe = compressed_to_png(compression_level=6)
        result = pipe(sample_frame)
        
        assert np.array_equal(result, mock_encoded)


class TestPipelineComposition:
    """Test cases for complex pipeline compositions."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample RGB frame for testing."""
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:, :40] = [255, 0, 0]  # Red left section
        frame[:, 40:] = [0, 255, 0]  # Green right section
        return frame
    
    def test_simple_pipeline(self, sample_frame):
        """Test simple pipeline composition."""
        # Create pipeline using function-to-function composition 
        pipe = letterboxed(target_size=(200, 200)) | resized(target_size=(100, 100))
        result = pipe(sample_frame)
        
        assert result.shape[:2] == (100, 100)
        assert result.shape[2] == 3
    
    def test_complex_pipeline(self, sample_frame):
        """Test complex pipeline with multiple operations."""
        # Create pipeline using function-to-function composition
        pipe = (letterboxed(target_size=(150, 150)) |
                   adjusted(brightness=10, contrast=1.1, saturation=0.9) |
                   resized(target_size=(75, 75)))
        result = pipe(sample_frame)
        
        assert result.shape[:2] == (75, 75)
        assert result.shape[2] == 3
    
    @patch('cv2.imencode')
    def test_pipeline_with_compression(self, mock_imencode, sample_frame):
        """Test pipeline ending with compression."""
        mock_encoded = np.array([1, 2, 3, 4], dtype=np.uint8)
        mock_imencode.return_value = (True, mock_encoded)
        
        # Create pipeline using function-to-function composition
        pipe = (letterboxed(target_size=(100, 100)) |
                adjusted(brightness=5) |
                compressed_to_jpeg(quality=90))
        result = pipe(sample_frame)
        
        assert np.array_equal(result, mock_encoded)
    
    def test_pipeline_with_greyscale(self, sample_frame):
        """Test pipeline with greyscale conversion."""
        # Create pipeline using function-to-function composition
        pipe = (letterboxed(target_size=(100, 100)) |
                greyscaled() |
                adjusted(brightness=10, contrast=1.2))
        result = pipe(sample_frame)
        
        assert len(result.shape) == 3 and result.shape[2] == 3
    
    def test_pipeline_error_propagation(self, sample_frame):
        """Test that errors in pipeline are properly propagated."""
        with patch.object(ImageEditor, 'letterbox', side_effect=ValueError("Test error")):
            pipe = letterboxed(target_size=(100, 100))
            with pytest.raises(ValueError, match="Test error"):
                pipe(sample_frame)
    
    def test_pipeline_with_no_args_functions(self, sample_frame):
        """Test pipeline with functions that take no additional arguments."""
        pipe = greyscaled()
        result = pipe(sample_frame)
        
        assert len(result.shape) == 3 and result.shape[2] == 3


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""
    
    def test_empty_frame(self):
        """Test handling of empty frames."""
        empty_frame = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        
        # Most operations should handle empty frames gracefully
        with pytest.raises((ValueError, cv2.error)):
            ImageEditor.letterbox(empty_frame)
    
    def test_single_pixel_frame(self):
        """Test handling of single pixel frames."""
        single_pixel = np.array([[[255, 0, 0]]], dtype=np.uint8)
        
        result = ImageEditor.letterbox(single_pixel, target_size=(10, 10))
        assert result.shape[:2] == (10, 10)
    
    def test_very_large_frame(self):
        """Test handling of large frames (memory considerations)."""
        # Create a moderately large frame to test without using too much memory
        large_frame = np.random.randint(0, 256, (500, 600, 3), dtype=np.uint8)
        
        result = ImageEditor.resize(large_frame, target_size=(100, 100))
        assert result.shape[:2] == (100, 100)
    
    def test_invalid_target_sizes(self):
        """Test handling of invalid target sizes."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Zero or negative dimensions should be handled
        with pytest.raises((ValueError, cv2.error)):
            ImageEditor.resize(frame, target_size=(0, 100))
        
        with pytest.raises((ValueError, cv2.error)):
            ImageEditor.resize(frame, target_size=(-10, 100))
    
    def test_extreme_adjustment_values(self, sample_frame=None):
        """Test extreme adjustment values."""
        if sample_frame is None:
            sample_frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        # Extreme brightness
        result = ImageEditor.adjust(sample_frame, brightness=1000)
        assert result.shape == sample_frame.shape
        assert np.all(result <= 255)  # Should be clipped
        
        result = ImageEditor.adjust(sample_frame, brightness=-1000)
        assert np.all(result >= 0)  # Should be clipped
        
        # Extreme contrast
        result = ImageEditor.adjust(sample_frame, contrast=100)
        assert result.shape == sample_frame.shape
        
        # Zero contrast
        result = ImageEditor.adjust(sample_frame, contrast=0)
        assert result.shape == sample_frame.shape