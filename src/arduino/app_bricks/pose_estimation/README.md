# Pose Estimation Brick

This pose estimation brick analyzes a camera video stream and detects the body poses of up to 10 people at a time, locating 17 keypoints per person (eyes, ears, nose, shoulders, elbows, wrists, hips, knees, ankles). The output is a video stream featuring the skeleton overlay, with the added capability to trigger actions based on the detected poses, people presence and people count.

## Overview

The Pose Estimation Brick allows you to:

- Detect body poses and keypoints of up to 10 people simultaneously in real-time from a camera stream.
- Trigger custom callbacks based on recognized poses (`left_arm_raised`, `right_arm_raised`, `sitting`, `standing`) or detect raw keypoint data for all visible people.
- React to people entering or leaving the camera view, and track changes in people count.
- Teach custom poses by providing training photos.
- Configure detection sensitivity, bounding box visualization, and keypoint confidence display.

## Prerequisites

To use this Brick you need to have a camera connected to your board.

**Tip**: Use a USB-C® Hub with USB-A connectors to support commercial web cameras.

## Features

- Detects up to 10 people simultaneously with 17 keypoint locations per person
- Provides callbacks for built-in poses and raw keypoint data
- Enables presence and people-counting automations
- Reports skeleton readability (whether classification can occur)
- Supports custom pose training with intuitive photo-based learning
- Real-time visualization with optional bounding boxes and confidence indicators
- Serves annotated video as MJPEG stream on port 5002
- Automatic person tracking for stable pose classification with hysteresis and enter/exit events

## Code example and usage

```python
from arduino.app_utils import App
from arduino.app_bricks.pose_estimation import PoseEstimation

pose_estimation = PoseEstimation()
pose_estimation.on_pose("standing", lambda pose: print(f"Standing: {pose.event}"))
pose_estimation.on_pose("left_arm_raised", lambda pose: print(f"Left arm raised: {pose.event}"))
pose_estimation.on_keypoints(lambda person: print(f"Person detected with {len(person.keypoints)} keypoints"))
pose_estimation.on_enter(lambda: print("Person detected!"))
pose_estimation.on_exit(lambda: print("No person detected"))

App.run()
```

## Configuration

`PoseEstimation(camera=None, confidence=0.25, count_debounce_sec=0.0, out_of_frame_tolerance=0.25, poses=None, custom_poses_dir="/app/poses", bbox_padding=0, draw_bboxes=False, draw_low_confidence_points=True)`:

- `camera` (`BaseCamera`, optional): the camera instance to use. If not provided, a default `Camera(fps=30)` is created.
- `confidence` (`float`): minimum confidence (0.0 to 1.0) for person detection. The value compares against the average of a person's 17 keypoint scores.
- `count_debounce_sec` (`float`): minimum seconds a person leaving, or the people count dropping, must hold before `on_exit`/`on_count_change` report it. Default is 0 (no debounce).
- `out_of_frame_tolerance` (`float`): how far past the frame edges a joint may be extrapolated before the skeleton counts as unreadable, as a fraction of the frame size. Default is 0.25; 0 demands every joint inside the picture.
- `poses` (`list`, optional): list of pose names or pose dicts to listen to. Defaults to the four built-in poses. See "Built-in Poses" below.
- `custom_poses_dir` (`str`): path to the folder containing custom pose training photos.
- `bbox_padding` (`float` or `tuple`): expands bounding boxes, CSS style; a single number applies to all sides, a 4-tuple is (top, right, bottom, left), each a fraction of the box height (top/bottom) or width (left/right) in [0.0, 1.0]. None by default.
- `draw_bboxes` (`bool`): whether to draw bounding boxes on the overlay.
- `draw_low_confidence_points` (`bool`): whether to show low-confidence keypoint marks on the overlay.

## Methods

- **`on_pose(name, callback)`**: registers a callback for a specific pose. The callback receives a `Pose` object with `event="enter"` or `"exit"`. Pose edges are stable with hysteresis and smoothing.
- **`on_keypoints(callback)`**: registers a callback that receives a `Person` object for each detected person every frame. Contains 17 named `Keypoint`s with pixel coordinates and confidence scores, plus the bounding box.
- **`on_enter(callback)`**: registers a zero-argument callback invoked when at least one person becomes visible.
- **`on_exit(callback)`**: registers a zero-argument callback invoked when no people are visible anymore.
- **`on_count_change(callback)`**: registers a callback that receives the new people count when it changes.
- **`on_readable_change(callback)`**: registers a callback invoked when the tracked person's skeleton readability changes (reports whether classification can occur).
- **`set_confidence(value)`**: changes the minimum person detection score at runtime.
- **`set_draw_bboxes(value)`**: enables or disables bounding box drawing on the overlay.
- **`set_draw_low_confidence_points(value)`**: shows or hides low-confidence keypoint marks on the overlay.
- **`set_bbox_padding(value)`**: changes bounding box padding at runtime.

## Properties

- **`readable`**: current readability state of the tracked person's skeleton (bool). Turns False when normalization anchors are guessed, a joint lands far outside the frame, or the torso collapses.
- **`people_count`**: current number of detected people (int).
- **`pose_names`**: list of active pose names the instance is listening to.

`BUILTIN_POSE_NAMES` is not an instance property but a module-level constant, importable with `from arduino.app_bricks.pose_estimation import BUILTIN_POSE_NAMES`: the tuple of all available built-in pose names.

## Technical Details

**Keypoints**: The 17 keypoints reported for every person, by name: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle.

**Detection score**: The `confidence` threshold compares against the average of a person's 17 keypoint scores, so it rises with skeleton completeness. A person is not detected unless at least one keypoint scores 0.25 or more.

**Classification**: The pose classifier is a k-NN model over a reference database shipped with the brick (`assets/pose_classifier.npz`, ~0.6 MB) containing labeled examples and per-pose thresholds. Thresholds are applied on an exponential moving average (0.31 s time constant by default, overridable per pose). When the tracked person disappears, active poses exit after a 0.7 s grace period.

**Runner**: The model runner performs internal person-tracking crops before inference. Reported coordinates are always in full-frame pixels. A periodic full-frame pass (every 10 frames) updates the tracking window, so people entering outside it are discovered within a few tenths of a second.

## Built-in Poses

The brick recognizes four poses by default:

- `left_arm_raised`: the person raises their left arm, as when asking to speak or waving hello. Left is the person's own left.
- `right_arm_raised`: the same with the right arm.
- `standing`: the person stands with the arms down.
- `sitting`: the person sits, on a chair or a bench, with the legs in view.

## Teaching your own poses

You can teach the brick to recognize custom poses by providing training photos. A custom pose is a folder of photos named like the pose (built-in pose names are not allowed) inside the `poses` folder at the root of your app. The running app sees that folder as `/app/poses`, the default `custom_poses_dir`:

```
poses/                           # in your app's root folder; /app/poses for the running app
  hands_on_hips/                 # the pose: IMG_0001.jpg, IMG_0002.jpg, ... (jpg or png, one person, whole body in the frame)
    report_<date>_<time>.txt     # written by the brick at every start that changes something
  other/                         # optional: photos of what is NOT any of your poses
  .cache/                        # what the brick already read; safe to delete
```

Photos: at least 40, taken on different occasions (distance, angle, room, clothes). Near-identical photos count as one, so 60 frames of a pose held still are one example: move between shots. A person partly out of the frame, or a photo with no person, is discarded and listed in the report. The `other/` photos are your own negatives: they set the firing threshold and never fire.

Declare the pose next to the built-in ones:

```python
pose_estimation = PoseEstimation(camera, poses=["standing", "hands_on_hips", {"name": "tennis_forehand", "type": "action", "duration": 0.7}])
pose_estimation.on_pose("hands_on_hips", on_hands_on_hips)
```

A custom pose is used only when declared in `poses`: with `poses=None` the brick listens to the four built-in poses and ignores the folders.

`type` is "state" (a held pose, the default) or "action" (a movement with a start and an end: one enter/exit pair per occurrence); `duration` is the typical length of one occurrence, in seconds, and can be specified only for actions; `thresholds` ({"enter", "exit"}) and `smoothing` (seconds) override the values the brick derives from the photos.

At `start()` the brick reads the photos through the model runner (a few fractions of a second per photo the first time, instant afterwards for the already computed ones thanks to `.cache/`), composes its classifier, measures whether each pose forms and writes a report in the log and in the pose folder. A pose that is not accepted stops `start()` with a `ValueError` carrying its verdict and next step; the app framework logs "Failed to start brick" and the brick stays stopped. The report, line by line:

- `photos: <found> found, <usable> usable, <discarded> discarded`, then one line per discarded photo with the reason (no person detected, person partly out of frame, skeleton incomplete, not an image).
- `groups: <n>`: how many distinct takes the photos amount to (photos closer than 1.0 to each other are one group); at least 5 are needed.
- `measure: <recall>% of your photos fire at threshold 0.55`: the pose is accepted above 70%, each photo judged with its near-identical photos left out. `own neighbours among the 9 nearest` is the same fact seen from the classifier.
- `learning curve` and `consistency`: how the measure grows with the photos; `mixed` means the photos spread over different variants of the pose.
- `verdict` and, when not accepted, `next step`: what to do, including how many photos in total are needed for 70% and 90%.
- `operating point`: the enter and exit thresholds derived from the photos (and from `other/` when present), and the smoothing.
- `confusion`: on what share of each active pose's photos every pose fires; above 15% the report warns that the two poses fire together, and it warns when a custom pose has more than 3 times the photos of another active pose.

What to expect: a compact held pose forms with a few dozen photos; a broad pose needs hundreds, and the report says how many. A variant of a built-in pose is the hardest to teach: it competes with hundreds of shipped examples. Recognizing people other than the ones in the photos takes photos of several people, or a lower `enter` and accept a few more false fires.

Actions: film a few repetitions from a fixed point, put the frames of the movement in the pose folder and the frames between one repetition and the next in `other/` (they are what keeps the pose from firing while you wait). A movement like a tennis stroke forms from the frames of a handful of repetitions.
