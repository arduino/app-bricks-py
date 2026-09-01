# Pose Estimation Brick

This pose estimation brick analyzes a camera video stream and detects the body poses of up
to 10 people at a time, locating 17 keypoints per person (eyes, ears, nose, shoulders,
elbows, wrists, hips, knees, ankles). The output is a video stream featuring the skeleton
overlay, with the added capability to trigger actions based on the detected poses, people
presence and people count.

Integration highlights:
- `on_keypoints` delivers one `Person` per detected person: their 17 named `Keypoint`s
  (a dict keyed by keypoint name, with pixel coordinates and confidence scores) plus the
  bounding box, for every processed frame with people in view, one callback invocation
  per person.
- `on_pose(name, callback)` triggers on the built-in poses `left_arm_raised`,
  `right_arm_raised`, `sitting` and `standing`. The classifier follows one person —
  the largest bounding box in view, normally the closest to the camera — and smooths
  per-frame classifications over time with hysteresis, so callbacks receive stable
  `Pose` edges: `event="enter"` when the tracked person assumes the pose, `"exit"`
  when they leave it (per-pose enter/exit thresholds shipped inside the classifier
  asset — 0.60/0.40 for the arms, 0.80/0.60 for standing, 0.55/0.35 for sitting —
  applied on an exponential moving average with a 0.31 s time constant; when the
  person disappears, active poses exit after a 0.7 s grace period). Other people
  stay visible through `on_keypoints` but do not fire pose events.
- `on_enter` / `on_exit` / `on_count_change` enable presence and people-counting automations.
- `on_readable_change` reports whether the tracked person's skeleton can be classified: it
  turns False when the normalization anchors are all guessed, when a joint lands far outside
  the frame or when the torso collapses, and no pose event is emitted while it stays False.
- `readable` and `people_count` hold the current value of those two states, for clients that
  connect after the last change and would otherwise wait for the next one.
- `out_of_frame_tolerance` sets how far past the frame edges a joint may be extrapolated
  before the skeleton counts as unreadable, as a fraction of the frame size: 0.25 by
  default, 0 to demand a person entirely inside the picture.
- `POSE_NAMES` lists the built-in pose names `on_pose` accepts.
- `set_confidence` changes the minimum person detection score at runtime; the value is
  applied by the model runner itself, so the skeleton overlay only ever shows what the
  API reports.
- `set_draw_bboxes` (or `draw_bboxes=True` in the constructor) draws every detected person's
  bounding box on the overlay; off by default.
- `set_draw_low_confidence_points` (or `draw_low_confidence_points=False` in the constructor) shows or hides the
  low-confidence keypoint marks on the overlay; shown by default.
- `set_bbox_padding` (or `bbox_padding` in the constructor) expands every bounding box, CSS
  style: one number for all sides or a (top, right, bottom, left) tuple — top/bottom as a
  fraction of the box height, left/right of its width. It applies to both the reported
  `bounding_box_xyxy` and the drawn one; none by default.
- The skeleton overlay is drawn by the model runner, which serves the annotated video as an
  MJPEG stream on port 5002.

The 17 keypoints reported for every person, by name: nose, left_eye, right_eye,
left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle,
right_ankle.

Detection score: the `confidence` threshold (constructor and `set_confidence`) compares
against the average of a person's 17 keypoint scores, so it rises with how complete the
skeleton is as well as with how confident each keypoint is. Below that threshold, one limit
stays: a person is not detected at all unless at least one of their keypoints scores 0.25 or
more, the value the runner uses to start assembling a skeleton.

Classification note: the pose classifier is a k-NN over a reference database of labeled examples
shipped with the brick (`assets/pose_classifier.npz`, ~0.6 MB) together with the exact
dials and per-pose thresholds it was tuned with. The brick reads everything it needs
(examples, dials, thresholds, calibration mask) from the file itself.

Runner note: the model runner performs an internal person-tracking crop before inference
(people far from the camera would otherwise be too small in the model's letterboxed input
and lose keypoint confidence). This is transparent to clients: reported coordinates are
always in full-frame pixels. While the window is active, a periodic extra full-frame pass
(every 10 frames) updates the tracking window only, so people entering the scene outside
of it are discovered within a few tenths of a second without any quality dip in the
reported results.
