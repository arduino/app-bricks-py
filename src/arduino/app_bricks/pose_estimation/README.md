# Pose Estimation Brick

This pose estimation brick analyzes a camera video stream and detects the body poses of up to 10 people at a time, locating 17 keypoints per person (eyes, ears, nose, shoulders, elbows, wrists, hips, knees, ankles). The output is a video stream featuring the skeleton overlay, with the added capability to trigger actions based on the detected poses, people presence and people count.

Integration highlights:
- `on_keypoints` delivers one `Person` per detected person: their 17 named `Keypoint`s (a dict keyed by keypoint name, with pixel coordinates and confidence scores) plus the bounding box, for every processed frame with people in view, one callback invocation per person.
- `on_pose(name, callback)` triggers on the built-in poses `left_arm_raised`, `right_arm_raised`, `sitting` and `standing`. The classifier follows one person — the largest bounding box in view, normally the closest to the camera — and smooths per-frame classifications over time with hysteresis, so callbacks receive stable `Pose` edges: `event="enter"` when the tracked person assumes the pose, `"exit"` when they leave it (per-pose enter/exit thresholds shipped inside the classifier asset — 0.60/0.40 for the arms, 0.80/0.60 for standing, 0.55/0.35 for sitting — applied on an exponential moving average with a 0.31 s time constant, both overridable per pose through `poses`; when the person disappears, active poses exit after a 0.7 s grace period). Other people stay visible through `on_keypoints` but do not fire pose events.
- `on_enter` / `on_exit` / `on_count_change` enable presence and people-counting automations.
- `on_readable_change` reports whether the tracked person's skeleton can be classified: it turns False when the normalization anchors are all guessed, when a joint lands far outside the frame or when the torso collapses, and no pose event is emitted while it stays False.
- `readable` and `people_count` hold the current value of those two states, for clients that connect after the last change and would otherwise wait for the next one.
- `out_of_frame_tolerance` sets how far past the frame edges a joint may be extrapolated before the skeleton counts as unreadable, as a fraction of the frame size: 0.25 by default, 0 to demand a person entirely inside the picture.
- `poses` (constructor) declares the poses the instance listens to, built-in or your own: a list of names, or of dicts with `name` plus `type`, `duration`, `thresholds` and `smoothing`. The built-in poses left out stay in the classifier as negatives and never fire; `pose_names` lists the active ones.
- A custom pose is a folder of photos: see "Teaching your own poses" below.
- `BUILTIN_POSE_NAMES` lists the built-in pose names.
- `set_confidence` changes the minimum person detection score at runtime; the value is applied by the model runner itself, so the skeleton overlay only ever shows what the API reports.
- `set_draw_bboxes` (or `draw_bboxes=True` in the constructor) draws every detected person's bounding box on the overlay; off by default.
- `set_draw_low_confidence_points` (or `draw_low_confidence_points=False` in the constructor) shows or hides the low-confidence keypoint marks on the overlay; shown by default.
- `set_bbox_padding` (or `bbox_padding` in the constructor) expands every bounding box, CSS style: one number for all sides or a (top, right, bottom, left) tuple — top/bottom as a fraction of the box height, left/right of its width. It applies to both the reported `bounding_box_xyxy` and the drawn one; none by default.
- The skeleton overlay is drawn by the model runner, which serves the annotated video as an MJPEG stream on port 5002.

The 17 keypoints reported for every person, by name: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle.

Detection score: the `confidence` threshold (constructor and `set_confidence`) compares against the average of a person's 17 keypoint scores, so it rises with how complete the skeleton is as well as with how confident each keypoint is. Below that threshold, one limit stays: a person is not detected at all unless at least one of their keypoints scores 0.25 or more, the value the runner uses to start assembling a skeleton.

Classification note: the pose classifier is a k-NN over a reference database of labeled examples shipped with the brick (`assets/pose_classifier.npz`, ~0.6 MB) together with the exact dials and per-pose thresholds it was tuned with. The brick reads everything it needs (examples, dials, thresholds, calibration mask) from the file itself.

Runner note: the model runner performs an internal person-tracking crop before inference (people far from the camera would otherwise be too small in the model's letterboxed input and lose keypoint confidence). This is transparent to clients: reported coordinates are always in full-frame pixels. While the window is active, a periodic extra full-frame pass (every 10 frames) updates the tracking window only, so people entering the scene outside of it are discovered within a few tenths of a second without any quality dip in the reported results.

## Teaching your own poses

A custom pose is a folder of photos named like the pose (built-in pose names are not allowed) under `custom_poses_dir`, `/app/poses` by default:

```
/app/poses/
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

`type` is "state" (a held pose, the default) or "action" (a movement with a start and an end: one enter/exit pair per occurrence); `duration` is the typical length of one occurrence, in seconds, and can be specified only for actions; `thresholds` ({"enter", "exit"}) and `smoothing` (seconds) override the values the brick derives from the photos.

At `start()` the brick reads the photos through the model runner (about 0.2 s per photo the first time; the photos already read are not read again thanks to `.cache/`), composes its classifier, measures whether each pose forms and writes a report in the log and in the pose folder. A pose that is not accepted stops `start()` with a `ValueError` carrying its verdict and next step; the app framework logs "Failed to start brick" and the brick stays stopped. The report, line by line:

- `photos: <found> found, <usable> usable, <discarded> discarded`, then one line per discarded photo with the reason (no person detected, person partly out of frame, skeleton incomplete, not an image).
- `groups: <n>`: how many distinct takes the photos amount to (photos closer than 1.0 to each other are one group); at least 5 are needed.
- `measure: <recall>% of your photos fire at threshold 0.55`: the pose is accepted above 70%, each photo judged with its near-identical photos left out. `own neighbours among the 9 nearest` is the same fact seen from the classifier.
- `learning curve` and `consistency`: how the measure grows with the photos; `mixed` means the photos spread over different variants of the pose.
- `verdict` and, when not accepted, `next step`: what to do, including how many photos in total are needed for 70% and 90%.
- `operating point`: the enter and exit thresholds derived from the photos (and from `other/` when present), and the smoothing.
- `confusion`: on what share of each active pose's photos every pose fires; above 15% the report warns that the two poses fire together, and it warns when a custom pose has more than 3 times the photos of another active pose.

What to expect: a compact held pose forms with a few dozen photos; a broad pose needs hundreds, and the report says how many. A variant of a built-in pose (a hand at the ear against `right_arm_raised`) is the hardest to teach: it competes with hundreds of shipped examples. The photos teach the pose the way the people in them hold it: a different person lands farther from the examples and fires less often. If the app is for other people too, put several people in the photos, or lower `enter` in `thresholds` and accept a few more false fires.

Actions: film a few repetitions from a fixed point, put the frames of the movement in the pose folder and the frames between one repetition and the next in `other/` (they are what keeps the pose from firing while you wait). A movement like a tennis stroke forms from the frames of a handful of repetitions.
