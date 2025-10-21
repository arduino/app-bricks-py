#!/usr/bin/env python3
"""
Test script to verify ImageEditor integration with Camera classes.
"""

import numpy as np
from arduino.app_peripherals.camera import ImageEditor
from arduino.app_peripherals.camera import image_editor as ie
from arduino.app_peripherals.camera.functional_utils import compose, curry, identity

def test_image_editor():
    """Test ImageEditor functionality."""
    # Create a test frame (100x150 RGB image)
    test_frame = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    
    print(f"Original frame shape: {test_frame.shape}")
    
    # Test letterboxing to make square
    letterboxed = ImageEditor.letterbox(test_frame)
    print(f"Letterboxed frame shape: {letterboxed.shape}")
    assert letterboxed.shape[0] == letterboxed.shape[1], "Letterboxed frame should be square"
    
    # Test letterboxing to specific size
    target_letterboxed = ImageEditor.letterbox(test_frame, target_size=(200, 200))
    print(f"Target letterboxed frame shape: {target_letterboxed.shape}")
    assert target_letterboxed.shape[:2] == (200, 200), "Should match target size"
    
    # Test PNG compression
    png_bytes = ImageEditor.compress_to_png(test_frame)
    print(f"PNG compressed size: {len(png_bytes) if png_bytes else 0} bytes")
    assert png_bytes is not None, "PNG compression should succeed"
    
    # Test JPEG compression
    jpeg_bytes = ImageEditor.compress_to_jpeg(test_frame)
    print(f"JPEG compressed size: {len(jpeg_bytes) if jpeg_bytes else 0} bytes")
    assert jpeg_bytes is not None, "JPEG compression should succeed"
    
    # Test PIL conversion
    pil_image = ImageEditor.numpy_to_pil(test_frame)
    print(f"PIL image size: {pil_image.size}, mode: {pil_image.mode}")
    assert pil_image.mode == 'RGB', "PIL image should be RGB"
    
    # Test numpy conversion back
    numpy_frame = ImageEditor.pil_to_numpy(pil_image)
    print(f"Converted back to numpy shape: {numpy_frame.shape}")
    assert numpy_frame.shape == test_frame.shape, "Round-trip conversion should preserve shape"
    
    # Test frame info
    info = ImageEditor.get_frame_info(test_frame)
    print(f"Frame info: {info}")
    assert info['width'] == 150 and info['height'] == 100, "Frame info should be correct"
    
    print("✅ All ImageEditor tests passed!")

def test_transformers():
    """Test transformer functionality."""
    print("\n=== Testing Transformers ===")
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    print(f"Original frame shape: {test_frame.shape}")
    
    # Test identity transformer
    identity_result = identity(test_frame)
    assert np.array_equal(identity_result, test_frame), "Identity should return unchanged frame"
    print("✅ Identity transformer works")
    
    # Test module-level API
    letterbox_transformer = ie.letterbox(target_size=(200, 200))
    letterboxed = letterbox_transformer(test_frame)
    print(f"Letterbox transformer result: {letterboxed.shape}")
    assert letterboxed.shape[:2] == (200, 200), "Transformer should produce correct size"
    print("✅ Letterbox transformer works")
    
    # Test resize transformer
    resize_transformer = ie.resize(target_size=(320, 240), maintain_aspect=False)
    resized = resize_transformer(test_frame)
    print(f"Resize transformer result: {resized.shape}")
    assert resized.shape[:2] == (240, 320), "Resize should produce correct dimensions"
    print("✅ Resize transformer works")
    
    # Test filter transformer
    filter_transformer = ie.filters(brightness=10, contrast=1.2, saturation=1.1)
    filtered = filter_transformer(test_frame)
    print(f"Filter transformer result: {filtered.shape}")
    assert filtered.shape == test_frame.shape, "Filter should preserve shape"
    print("✅ Filter transformer works")
    
    # Test pipeline composition
    pipeline_transformer = ie.pipeline(
        ie.letterbox(target_size=(200, 200)),
        ie.filters(brightness=5, contrast=1.1)
    )
    pipeline_result = pipeline_transformer(test_frame)
    print(f"Pipeline transformer result: {pipeline_result.shape}")
    assert pipeline_result.shape[:2] == (200, 200), "Pipeline should work correctly"
    print("✅ Pipeline transformer works")
    
    # Test standard processing
    standard_transformer = ie.standard_processing(target_size=(256, 256))
    standard_result = standard_transformer(test_frame)
    print(f"Standard processing result: {standard_result.shape}")
    assert standard_result.shape[:2] == (256, 256), "Standard processing should work"
    print("✅ Standard processing works")
    
    # Test webcam processing
    webcam_transformer = ie.webcam_processing()
    webcam_result = webcam_transformer(test_frame)
    print(f"Webcam processing result: {webcam_result.shape}")
    assert webcam_result.shape[:2] == (640, 640), "Webcam processing should work"
    print("✅ Webcam processing works")
    
    # Test mobile processing
    mobile_transformer = ie.mobile_processing()
    mobile_result = mobile_transformer(test_frame)
    print(f"Mobile processing result: {mobile_result.shape}")
    assert mobile_result.shape[:2] == (480, 480), "Mobile processing should work"
    print("✅ Mobile processing works")
    
    # Test with curry from functional_utils
    manual_letterbox = curry(ImageEditor.letterbox, target_size=(300, 300), color=(128, 128, 128))
    manual_result = manual_letterbox(test_frame)
    print(f"Manual curry result: {manual_result.shape}")
    assert manual_result.shape[:2] == (300, 300), "Manual curry should work"
    print("✅ Manual curry works")
    
    # Test compose from functional_utils
    composed_transformer = compose(
        ie.letterbox(target_size=(180, 180)),
        ie.filters(brightness=8)
    )
    composed_result = composed_transformer(test_frame)
    print(f"Compose result: {composed_result.shape}")
    assert composed_result.shape[:2] == (180, 180), "Compose should work"
    print("✅ Functional compose works")
    
    print("✅ All transformer tests passed!")

if __name__ == "__main__":
    test_image_editor()
    test_transformers()