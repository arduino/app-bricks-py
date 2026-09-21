# Video Object Detection Brick

This Brick provides a Python interface for **detecting objects in real time from a USB camera video stream**.
It sends the camera frames to the Edge Impulse inference service running on the board, and produces detection events with predicted labels, bounding boxes, and confidence scores.

Beyond visualization, it allows you to **register callbacks** that react to detections, either for specific objects or for all detections, enabling event-driven logic in your applications.
It supports both **pre-trained models** provided by the framework and **custom models** trained with Edge Impulse.

## Overview

The Video Object Detection Brick allows you to:

- Continuously detect objects from a live camera or video stream.
- Get bounding boxes, labels, and confidence scores in real time, in the coordinates of the camera frame.
- Trigger custom Python functions when certain objects are detected.
- Handle all detections in a single callback if desired.
- Control confidence thresholds and debounce timing to avoid repeated triggers.
- Change the confidence threshold at runtime.

## Features

- Real-time detection stream with continuous object recognition.
- Outputs:
  - **Class label** (e.g., "person", "bicycle")
  - **Confidence score** for each detection
  - **Bounding boxes** for localized detections
- Two callback styles:
  - `on_detect("<label>", callback)` → React to a specific label.
  - `on_detect_all(callback)` → React to all detections at once.
- Configurable confidence threshold (default: `0.3`) and debounce time between repeated detections (default: `0s`, i.e. no debounce)
- Runtime threshold change with `override_threshold(value)`
- Clean lifecycle control with `start()` / `stop()` and integration with `App.run()`.
- Video stream with the bounding boxes on port `4912`, for browsers and embedded iframes.

## How it works

The models run in the **Edge Impulse inference service** (`arduino:edge_impulse`), one container shared by the bricks of the app, on the NPU where the board has one. The brick opens the model configured for it over the service socket (`/app/.cache/edge_impulse/ei.sock` in the app container), sends each camera frame as it is, and receives the boxes in the coordinates of that frame: the service resizes the frame to the model input in the model's own resize mode. Connections requesting the same model share it, and the model is released when the brick stops.

The model is the one selected for the brick in the app configuration (`EI_V_OBJ_DETECTION_MODEL`, set by the app CLI to the `.eim` file under its models directory). To open another model of that directory, pass its name to the constructor: the path relative to the models directory of the service without extension, for example `ootb/ei/yolo-x-nano` for a bundled model or `custom-ei/<id>/model` for a custom one.

## Video stream

The brick serves the camera video with the bounding boxes drawn on it on port `4912`: `http://<board>:4912/` is an MJPEG stream, which browsers show like an image at its natural size for an `<img>` tag or a player, and `http://<board>:4912/embed` is a bare page for embedding in an iframe. Frames are rendered only while someone is watching.
`stream_port=None` in the constructor disables the stream.

The video runs at the camera rate whatever the model takes: every frame is drawn with the boxes of the latest inference, so a slow model only makes the boxes lag behind moving objects until the next result replaces them, never the video stutter. Boxes older than twice the inference  period are dropped, so a model that stops answering leaves no stale boxes. Frames are rendered only while a viewer is connected.

## Prerequisites

To use this Brick you should have a USB camera connected to your board.

**Tip**: Use a USB-C® Hub with USB-A connectors to support commercial web cameras.

## Code example and usage

```python
from arduino.app_utils import App
from arduino.app_bricks.video_objectdetection import VideoObjectDetection

# Initialize detector with custom confidence and debounce settings
video_detector = VideoObjectDetection(confidence=0.4, debounce_sec=1.5)

# Callback when a "person" is detected
def on_person_detected():
    print("🚨 Person detected in the video stream!")

video_detector.on_detect("person", on_person_detected)

# Callback for all detections (takes one dict argument)
def on_all_detections(detections: dict):
    # Example: {"person": [{"confidence": 0.87, "bounding_box_xyxy": (10, 20, 110, 220)}]}
    print("All detections:", detections)

video_detector.on_detect_all(on_all_detections)

# Run the application (keeps the video detection loop active)
App.run()
```

Callback signatures:

- `on_detect(label, callback)`: the callback must be a plain function. With no parameters it is simply invoked; with one parameter it receives the detection details dict `{"confidence": float, "bounding_box_xyxy": (x1, y1, x2, y2)}`.
- `on_detect_all(callback)`: the callback receives one dict argument mapping each detected label to the list of its detections: `{label: [{"confidence": float, "bounding_box_xyxy": (x1, y1, x2, y2)}, ...], ...}`.
- With `VideoObjectDetection(camera_preview=True)`, a callback that declares a `frame` parameter (e.g. `def cb(detections, frame)`) also receives the camera frame the boxes were computed on, as raw JPEG bytes.

The constructor also accepts a `camera` parameter (`BaseCamera`) to use a specific camera instead of the default one, and a `model` parameter to open a model other than the configured one.
