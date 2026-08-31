# Object Detection Brick

This Brick provides a Python interface for **detecting objects** within a given image.

## Overview

The Object Detection Brick allows you to:

- Detect objects in an image, either from a local file or directly from a camera feed.
- Locate detected objects in the image using bounding boxes.
- Get the detection confidence value of each object and its label.

## Features

- Performs real-time object detection on static images
- Outputs bounding boxes, class labels, and confidence scores for each detected object
- Supports multiple image formats, including JPEG, JPG, and PNG (default: JPG)
- Allows customization of detection confidence and non-maximum suppression (NMS) thresholds
- Easily integrates with PIL images or raw image byte streams

## Code example and usage

```python
from arduino.app_bricks.object_detection import ObjectDetection

object_detection = ObjectDetection()

# Image can be provided as bytes or PIL.Image
with open("path/to/your/image.jpg", "rb") as f:
    img = f.read()

out = object_detection.detect(img)
# You can also provide a confidence level
# out = object_detection.detect(img, confidence = 0.35)
if out and "detection" in out:
    for i, obj_det in enumerate(out["detection"]):
        # For every object detected, print its details
        detected_object = obj_det.get("class_name", None)
        confidence = obj_det.get("confidence", None)
        bounding_box = obj_det.get("bounding_box_xyxy", None)
        print(f"Detected '{detected_object}' with confidence {confidence}% at {bounding_box}")

# Draw the bounding boxes and save the resulting image
out_image = object_detection.draw_bounding_boxes(img, out)
if out_image is not None:
    out_image.save("result.png")
```

You can also detect objects directly from a file path:

```python
out = object_detection.detect_from_file("path/to/your/image.jpg")
```

Note: the `confidence` value returned in the results is a string formatted as a percentage on a 0-100 scale (e.g. `"87.50"`), while the `confidence` parameter accepted by the constructor and by `detect()` is a float threshold between 0.0 and 1.0.

