# Video Object Tracking Brick

This Brick provides a Python interface for **tracking objects in real time from a USB camera video stream**.
It connects to a model runner over WebSocket, continuously analyzes incoming frames, and produces tracking events where each object carries a **stable identity** across frames.

Because every object keeps the same ID while it stays visible, the Brick can answer questions plain detection cannot: *how many distinct objects have I seen*, *how many crossed this line*, and *which way are they moving*.
It supports both **pre-trained models** provided by the framework and **custom models** trained with Edge Impulse.

## Overview

The Video Object Tracking Brick allows you to:

- Continuously track objects from a live camera or video stream, each with a persistent object ID.
- Count **unique** objects per label, instead of counting the same object again on every frame.
- Count objects crossing a virtual line (horizontal, vertical or diagonal).
- Follow the movement direction of each tracked object.
- Trigger custom Python functions when certain objects are tracked.
- Handle all tracked objects of a frame in a single callback if desired.
- Restrict counting and direction bookkeeping to a subset of labels.
- Override the tracker parameters dynamically at runtime (if supported by the model).

## Features

- Real-time tracking stream with persistent object identities.
- Outputs:
  - **Class label** (e.g., "person", "bicycle")
  - **Object ID**, stable for as long as the object is tracked
  - **Bounding boxes** for localized detections
- Two callback styles:
  - `on_detect("<label>", callback)` → React to a specific label.
  - `on_detect_all(callback)` → React to all tracked objects of a frame at once.
- Counters, readable at any time:
  - `get_unique_objects_count()` → distinct objects seen per label
  - `get_line_crossing_counts()` → line crossings per label
  - `get_objects_directions()` → movement history per object ID
- Virtual counting line via `set_horizontal_crossing_line(y)`, `set_vertical_crossing_line(x)` or `set_crossing_line_coordinates(x1, y1, x2, y2)`.
- Configurable confidence threshold (default: `0.4`) and debounce time between repeated callback invocations (default: `0s`, i.e. no debounce).
- Runtime tracker overrides: `override_confidence(value)`, `override_keep_grace(value)`, `override_max_observations(value)`, `override_iou_threshold(value)`, `override_euclidean_distance_threshold(value)`.
- Clean lifecycle control with `start()` / `stop()` and integration with `App.run()`.

## Prerequisites

To use this Brick you should have a USB camera connected to your board.

**Tip**: Use a USB-C® Hub with USB-A connectors to support commercial web cameras.

This Brick requires an **object tracking** model. The framework ships `yolox-object-tracking` as the default; custom Edge Impulse models are supported as long as the model runner emits `object_tracking` results.

## Code example and usage

```python
from arduino.app_utils import App
from arduino.app_bricks.video_object_tracking import VideoObjectTracking

# Track only people, with a 1.5s debounce between repeated callbacks
tracker = VideoObjectTracking(confidence=0.4, debounce_sec=1.5, labels_to_track=["person"])

# Count people crossing a horizontal line at y=240
tracker.set_horizontal_crossing_line(240)


# Callback when a "person" is tracked
def on_person_tracked(details: dict):
    # Example: {"object_id": 7.0, "bounding_box_xyxy": (10, 20, 110, 220)}
    print(f"🚶 Person {details['object_id']} at {details['bounding_box_xyxy']}")


tracker.on_detect("person", on_person_tracked)


# Callback for all tracked objects of a frame (takes one dict argument)
def on_all_tracked(objects: dict):
    # Example: {"person": {"object_id": 7.0, "bounding_box_xyxy": (10, 20, 110, 220)}}
    print("Tracked:", objects)
    print("Unique so far:", tracker.get_unique_objects_count())
    print("Line crossings:", tracker.get_line_crossing_counts())
    print("Directions:", tracker.get_objects_directions())


tracker.on_detect_all(on_all_tracked)

# Run the application (keeps the video tracking loop active)
App.run()
```

Callback signatures:

- `on_detect(label, callback)`: the callback must be a plain function. With no parameters it is simply invoked; with one parameter it receives the tracking details dict `{"object_id": float, "bounding_box_xyxy": (x1, y1, x2, y2)}`.
- `on_detect_all(callback)`: the callback receives one dict argument mapping each tracked label to its tracking details: `{label: {"object_id": float, "bounding_box_xyxy": (x1, y1, x2, y2)}, ...}`.

## Counting unique objects

`get_unique_objects_count()` returns a `{label: count}` dictionary of how many **distinct** objects have been seen since the last reset. An object is counted once, the first time its ID appears, so a person standing in front of the camera for a minute is counted once and not once per frame.

```python
tracker = VideoObjectTracking(labels_to_track=["car", "truck"])
# ...
print(tracker.get_unique_objects_count())  # {"car": 12, "truck": 3}
```

`reset_counters()` clears the unique-object counts, the line-crossing counts and the last-seen positions, so counting starts from scratch.

## Counting line crossings

Define a virtual line and the Brick counts, per label, every tracked object whose position moves from one side of it to the other:

```python
tracker.set_horizontal_crossing_line(240)  # spans x from 0 to 480 at y=240
tracker.set_vertical_crossing_line(320)  # spans y from 0 to 480 at x=320
tracker.set_crossing_line_coordinates(0, 100, 640, 380)  # arbitrary, diagonal line

print(tracker.get_line_crossing_counts())  # {"person": 5}
```

The two helpers span a fixed 480 px extent; use `set_crossing_line_coordinates()` to match a different frame size or to define a diagonal line.

**Note**: setting the line calls `reset_counters()`, so all counts collected so far are discarded. Set the line once, before or right after `App.run()` starts the Brick, rather than changing it while counting.

## Tracking movement direction

`get_objects_directions()` returns `{object_id: [direction, ...]}`, the sequence of direction changes observed for each tracked object. Consecutive repeats are collapsed, so a straight walk yields a single entry and the last element is the object's current direction.

Possible values are `up`, `down`, `left`, `right`, `up-left`, `up-right`, `down-left`, `down-right`. Horizontal directions are reported **mirrored** with respect to the frame: an object moving rightwards across the frame is reported as `left`. Vertical directions are not mirrored.

`min_movement_threshold` (default `10` px) is the minimum displacement needed for a movement to count as a direction change, which keeps bounding-box jitter from producing spurious directions.

## Tracker parameters

The constructor accepts the tracker knobs below, and each one has a matching `override_*` method to change it at runtime once the model runner is connected:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `confidence` | `0.4` | Minimum detection confidence, applied by the model runner. |
| `keep_grace` | `3` | Frames an object is kept alive after it disappears, before its ID is dropped. |
| `max_observations` | `3` | Maximum number of observations the tracker considers per object. |
| `iou_threshold` | `0.1` | Intersection-over-Union used to match tracks between frames. For bounding-box models such as YOLO. |
| `euclidean_distance_threshold` | `50` | Maximum distance in pixels used to match tracks between frames. For centroid models such as FOMO. |
| `debounce_sec` | `0.0` | Minimum seconds between repeated callback invocations for the same label. |
| `labels_to_track` | `None` | Labels included in the counters. `None` means all labels. |
| `min_movement_threshold` | `10` | Minimum displacement in pixels for a direction change. |

`iou_threshold` and `euclidean_distance_threshold` are alternatives: which one the model uses depends on its type, and each `override_*` method is a no-op on a model of the other type.

**Note**: `labels_to_track` filters the counters, the line crossings and the direction history only. Callbacks registered with `on_detect()` and `on_detect_all()` are invoked for every tracked label.
