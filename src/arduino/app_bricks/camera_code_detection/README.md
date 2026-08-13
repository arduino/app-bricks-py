# Camera Code Detection Brick

This Brick enables real-time barcode and QR code scanning from a camera video stream. 

## Overview

The Camera Code Detection Brick allows you to:

- Capture frames from a Camera (see Camera peripheral for supported cameras).
- Configure Camera settings (resolution and frame rate).
- Define the type of code to detect: barcodes and/or QR codes.
- Process detections with customizable callbacks.

## Features

- Supported Code Formats: 
  - **Linear**: EAN-13, EAN-8, UPC-A
  - **2D**: QR Code
- Single-code detection mode for focused scanning
- Multi-code detection for simultaneous barcode and QR code scanning
- Provides detection coordinates for precise code location

## Prerequisites

To use this Brick you can choose to plug a camera to your board or use a network-connected camera.

**Tip**: Use a USB-C® Hub with USB-A connectors to support commercial web cameras.

## Code example and usage

```python
from PIL.Image import Image
from arduino.app_utils import App
from arduino.app_bricks.camera_code_detection import CameraCodeDetection, Detection

def render_frame(frame: Image):
    ...

def handle_detected_code(frame: Image, detection: Detection):
    print(f"Detected {detection.type} with content: {detection.content}")

detector = CameraCodeDetection()
detector.on_frame(render_frame)
detector.on_detect(handle_detected_code)

App.run()
```

You can also select a specific camera to use:

```python
from PIL.Image import Image
from arduino.app_utils import App
from arduino.app_peripherals.camera import Camera
from arduino.app_bricks.camera_code_detection import CameraCodeDetection, Detection

def handle_detected_code(frame: Image, detection: Detection):
    ...

# Select the camera you want to use, its resolution and the max fps
camera = Camera(source="rtsp://...", resolution=(640, 360), fps=10)
detector = CameraCodeDetection(camera)
detector.on_detect(handle_detected_code)

App.run()
```

Notes:

- The constructor lets you restrict what is scanned: `CameraCodeDetection(camera=None, detect_qr=True, detect_barcode=True)`.
- Each `Detection` carries the decoded `content` (str), the `type` (`"QRCODE"` or `"BARCODE"`) and `coords`, a NumPy array with the four corner points of the code region.
- By default the `on_detect` callback is invoked once per detected code. To receive all codes of a frame in a single call, annotate the second parameter as `list[Detection]`: `def handle_detected_codes(frame: Image, detections: list[Detection])`.
- Use `on_error(callback)` to be notified of errors raised while scanning.