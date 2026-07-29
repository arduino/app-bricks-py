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
- `on_pose(name, callback)` is the planned trigger for named poses (e.g. "arms_up"):
  declared in this version but not implemented yet, it will be backed by built-in pose
  classification in a future release.
- `on_enter` / `on_exit` / `on_count_change` enable presence and people-counting automations.
- The skeleton overlay is drawn by the model runner, which serves the annotated video as an
  MJPEG stream on port 5002.

Runner note: the model runner performs an internal person-tracking crop before inference
(people far from the camera would otherwise be too small in the model's letterboxed input
and lose keypoint confidence). This is transparent to clients: reported coordinates are
always in full-frame pixels. While the window is active, a periodic extra full-frame pass
(every 10 frames) updates the tracking window only, so people entering the scene outside
of it are discovered within a few tenths of a second without any quality dip in the
reported results.
