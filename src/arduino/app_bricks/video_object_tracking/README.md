# Video Object Tracking Brick

This Brick provides a Python interface for **tracking objects in real time from a camera video stream**.
It recognizes the objects in front of the camera and follows each one while it stays in view, so every object keeps the same **ID** from frame to frame.

Because an object keeps its ID, the Brick can answer questions plain detection cannot: *how many different people walked by*, *how many crossed this line*, *who is inside this area*, *which way are they going*.
It supports both **pre-trained models** provided by the framework and **custom models** trained with Edge Impulse.

## Overview

The Video Object Tracking Brick allows you to:

- Follow the objects in a live camera stream, each with a persistent ID.
- Count **distinct** objects per label, instead of counting the same object again on every frame.
- Count the objects crossing a line you draw on the video, in each direction.
- Watch an area of the video and know who enters, who leaves and who is inside.
- Follow the direction each object moves in.
- Run your own Python functions when an object is tracked, crosses the line or enters the area.
- Restrict all of this to the labels you care about.
- Watch the result in a browser, with the boxes, IDs and your line or area drawn on the video.

## Features

- Real-time tracking with persistent object IDs.
- Outputs, for every tracked object:
  - **Class label** (e.g., "person", "bicycle")
  - **Object ID**, stable for as long as the object is in view
  - **Bounding box** in the coordinates of the camera frame
- Two callback styles:
  - `on_detect("<label>", callback)` → React to a specific label.
  - `on_detect_all(callback)` → React to all tracked objects of a frame at once.
- Event callbacks: `on_line_crossing(callback)`, `on_area_enter(callback)`, `on_area_exit(callback)`.
- Counters, readable at any time: `get_unique_objects_count()`, `get_line_crossing_counts()`, `get_objects_in_area()`, `get_area_counts()`, `get_objects_directions()`.
- Configurable confidence threshold (default: `0.4`) and debounce time between repeated callbacks (default: `0s`, i.e. no debounce).
- Clean lifecycle control with `start()` / `stop()` and integration with `App.run()`.
- Video stream with the tracked boxes and their IDs on port `4912`.

## Prerequisites

To use this Brick you should have a USB/CSI camera connected to your board.

## Code example and usage

```python
from arduino.app_utils import App
from arduino.app_bricks.video_object_tracking import VideoObjectTracking

# Track only people, with a 1.5s debounce between repeated callbacks
tracker = VideoObjectTracking(confidence=0.4, labels_to_track=["person"])


# Callback when a "person" is tracked
def on_person_tracked(details: dict):
    # Example: {"object_id": 7, "confidence": 0.87, "bounding_box_xyxy": (10, 20, 110, 220)}
    print(f"🚶 Person {details['object_id']} at {details['bounding_box_xyxy']}")


tracker.on_detect("person", on_person_tracked)


# Callback for all tracked objects of a frame (takes one dict argument)
def on_all_tracked(objects: dict):
    # Example: {"person": [{"object_id": 7, "confidence": 0.87, "bounding_box_xyxy": (10, 20, 110, 220)}]}
    print("Tracked:", objects)
    print("Distinct people so far:", tracker.get_unique_objects_count())


tracker.on_detect_all(on_all_tracked)

# Run the application (keeps the video tracking loop active)
App.run()
```

Callback signatures:

- `on_detect(label, callback)`: the callback must be a plain function. With no parameters it is simply invoked; with one parameter it receives the tracking details dict `{"object_id": int, "confidence": float, "bounding_box_xyxy": (x1, y1, x2, y2)}`, once per object of that label in the frame.
- `on_detect_all(callback)`: the callback receives one dict argument mapping each tracked label to the list of its objects in the frame: `{label: [{"object_id": int, "confidence": float, "bounding_box_xyxy": (x1, y1, x2, y2)}, ...], ...}`, so `len()` of a list is how many objects of that label are in view.

The constructor also accepts a `camera` parameter (`BaseCamera`) to use a specific camera instead of the default one, which is where the frame rate and resolution are set. A higher frame rate helps the tracker keep an identity on fast-moving objects.

While the app runs, open `http://<board>:4912/` in a browser to watch the video with the boxes, IDs and your line or area drawn on it. Pass `stream_port=None` to the constructor to turn the stream off.

## Counting distinct objects

`get_unique_objects_count()` returns a `{label: count}` dictionary of how many **distinct** objects have been seen since the last reset. An object is counted once, the first time its ID appears, so a person standing in front of the camera for a minute counts as one.

```python
tracker = VideoObjectTracking(labels_to_track=["car", "truck"])
# ...
print(tracker.get_unique_objects_count())  # {"car": 12, "truck": 3}
```

An object that disappears for a moment, because it was hidden or the model briefly mislabeled it, keeps its ID when it shows up again within about a second. After a longer absence it comes back as a new object. `keep_identities=False` turns this off and reports the IDs exactly as the model assigns them.

`reset_counters()` restarts every counter from zero: the distinct objects, the line crossings and the area entries and exits. The objects in view keep their IDs.

## Counting line crossings

Draw a line on the video and the Brick counts, per label and per direction, every object whose box centre moves from one side to the other. Without a line, nothing is counted.

```python
tracker.set_horizontal_crossing_line(240)  # horizontal line at y=240
tracker.set_vertical_crossing_line(320)  # vertical line at x=320
tracker.set_crossing_line_coordinates(0, 100, 640, 380)  # any line through two points

print(tracker.get_line_crossing_counts())  # {"person": {"left": 3, "right": 2, "all": 5}}
```

The line always runs across the whole frame, so an object is counted wherever it crosses it. The coordinates are pixels of the camera frame; pass `relative=True` to give them as fractions of the frame instead, from 0 to 1: `tracker.set_vertical_crossing_line(0.5, relative=True)` puts the line in the middle whatever the camera resolution.

To react to each crossing as it happens, register a callback with `on_line_crossing()`. It receives `{"label": str, "object_id": int, "direction": str}`, where the direction is the way the object went as seen on the screen: `up`, `down`, `left`, `right` or one of the four diagonals.

```python
def on_crossing(crossing: dict) -> None:
    print(f"{crossing['label']} #{crossing['object_id']} went {crossing['direction']}")


tracker.on_line_crossing(on_crossing)
```

The line is drawn on the video stream in white dashes. Pass `visible=False` to count without drawing it.

## Watching an area

Draw an area on the video and the Brick follows the objects whose box centre is inside it. The area is a rectangle, or any polygon from 3 to 100 points, in pixels or in fractions of the frame with `relative=True`.

```python
tracker.set_rectangular_area(200, 150, 440, 330)  # two opposite corners
tracker.set_area_coordinates([(100, 100), (300, 80), (320, 300), (120, 320)])  # any polygon

print(tracker.get_objects_in_area())  # {"apple": 1}
print(tracker.get_area_counts())  # {"apple": {"entered": 3, "exited": 2}}
```

An object **enters** when it is first seen inside, and **leaves** when it is seen outside or has been out of sight for about a second. Register a callback for each event; both receive `{"label": str, "object_id": int}`.

```python
def on_enter(event: dict) -> None:
    print(f"{event['label']} #{event['object_id']} entered the area")


def on_exit(event: dict) -> None:
    print(f"{event['label']} #{event['object_id']} left the area")


tracker.on_area_enter(on_enter)
tracker.on_area_exit(on_exit)
```

The outline of the area is drawn on the video stream in white dashes unless it is set with `visible=False`. `set_area_coordinates()` raises `ValueError` for a polygon whose border crosses itself.

## Tracking movement direction

`get_objects_directions()` returns `{object_id: [direction, ...]}`, the directions each object has moved in, in order, with repeats collapsed: a straight walk gives a single entry and the last one is the current direction. The names are the same as for the line crossings, as seen on the screen.

`min_movement_threshold` (default `10` px) is the smallest displacement that counts as a movement, so a box wobbling in place does not produce directions.

## Choosing what to track

- `labels_to_track=["person", "car"]` restricts everything the Brick reports, callbacks, counters and the boxes on the video, to those labels. `None`, the default, tracks every label of the model.
- `labels_to_ignore=["person"]` does the opposite and tracks everything but the listed labels.
- `confidence` is the minimum score for an object to be reported, `0.4` by default. `override_threshold(value)` changes it while the app runs.
- `debounce_sec` is the minimum time between two callbacks for the same label, useful to avoid a flood of calls while an object stays in view.

## Tracker parameters

The constructor also accepts the parameters of the tracker inside the model, each with a matching `override_*` method to change it at runtime. The defaults suit most cases.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `keep_grace` | `3` | Frames an object is kept after it disappears, before its ID is dropped. |
| `min_detections` | `3` | Detections an object needs before it is reported as tracked. |
| `iou_threshold` | `0.1` | How much two boxes must overlap between frames to be the same object. For bounding-box models such as YOLO. |
| `euclidean_distance_threshold` | `50` | How far an object may move between frames, in pixels, to be the same object. For centroid models such as FOMO. |
| `keep_identities` | `True` | Keep the ID of an object through a short loss or a change of label. |

`iou_threshold` and `euclidean_distance_threshold` are alternatives: the model uses the one matching its type and the other `override_*` method does nothing. An `override_*` method raises `RuntimeError` when the model has no such parameter. The parameters belong to the model, so they hold for every Brick of the app using the same model.

## Under the hood

The model runs in the **Edge Impulse inference server**, a container the Brick starts next to the app, on the NPU of the board. The Brick sends it the camera frames and gets back the tracked objects in the coordinates of the frame.

The model is the one selected for the Brick in the app configuration, or the default of the Brick for the board, `yolox-small-qnn-object-tracker`. A custom model trained with Edge Impulse works as long as it includes the **object tracking** block: the Brick checks this when the app starts and reports a model without it as an error.

The IDs start from zero on every run of the app.
