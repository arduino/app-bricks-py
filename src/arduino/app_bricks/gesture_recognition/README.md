# Gesture Recognition Brick

This gesture recognition brick utilizes a pre-trained model to analyze video streams from a camera,
detecting hands and recognizing hand gestures, with the capability to trigger actions based on these
detections.

## Overview

The Gesture Recognition Brick allows you to:

- Detect hand gestures (e.g. `Victory`, `Open_Palm`, `Thumb_Up`) in real time from a camera stream.
- Trigger custom Python callbacks when a specific gesture is detected, optionally filtered by hand (`left`, `right` or `both`).
- React to hands entering or leaving the camera view.
- Access the raw camera frames via a frame callback.

## Prerequisites

To use this Brick you should have a USB camera connected to your board.

**Tip**: Use a USB-C® Hub with USB-A connectors to support commercial web cameras.

## Code example and usage

```python
from arduino.app_utils import App
from arduino.app_bricks.gesture_recognition import GestureRecognition

recognition = GestureRecognition()
recognition.on_gesture("Victory", lambda metadata: print("Victory!"))
recognition.on_gesture("Open_Palm", lambda metadata: print("Moving left!"), hand="left")
recognition.on_gesture("Open_Palm", lambda metadata: print("Moving right!"), hand="right")

App.run()
```

You can also react to hands appearing in or leaving the camera view:

```python
from arduino.app_utils import App
from arduino.app_bricks.gesture_recognition import GestureRecognition

recognition = GestureRecognition()
recognition.on_enter(lambda: print("Hi there!"))
recognition.on_exit(lambda: print("Goodbye!"))

App.run()
```

## Configuration

`GestureRecognition(camera=None, confidence=0.0)`:

- `camera` (`BaseCamera`, optional): the camera instance to use. If not provided, a default `Camera(fps=30)` is created.
- `confidence` (`float`): minimum confidence (0.0 to 1.0) required for a gesture detection to trigger callbacks.

## Methods

- **`on_gesture(gesture, callback, hand="both")`**: registers a callback for a specific gesture, optionally restricted to the `"left"` or `"right"` hand. The callback receives a metadata dict with the detection details: `hand`, `gesture`, `confidence`, `landmarks` (hand key points as `(x, y, z)` tuples) and `bounding_box_xyxy`. Pass `callback=None` to unregister.
- **`on_enter(callback)`**: registers a zero-argument callback invoked when at least one hand becomes visible.
- **`on_exit(callback)`**: registers a zero-argument callback invoked when no hands are visible anymore.
- **`on_frame(callback)`**: registers a callback that receives each camera frame as a NumPy array (e.g. to display or stream the video).

While a callback is still running, further events of the same kind are discarded instead of queueing up.
