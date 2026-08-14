# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from utils.constants import INPUT_HEIGHT, INPUT_WIDTH, MIN_KEYPOINT_SCORE, MIN_PERSON_SCORE
from utils.tf import load_qnn_delegate
from utils.image_processing import resize_pad
from utils.model_io_processing import decode_multiple_persons, dequantize, quantize
from utils.draw import draw_persons


# Load model
posenet = Interpreter(
    "models/posenet_mobilenet_w8a8.tflite",
    experimental_delegates=load_qnn_delegate(),
)
posenet.allocate_tensors()

posenet_input = posenet.get_input_details()
posenet_output = posenet.get_output_details()

# Runtime-tunable settings, updated by client config messages (wired in the
# base image's main.py).
_config = {"min_person_score": MIN_PERSON_SCORE, "draw_uncertain": True}


def apply_config(config: dict) -> None:
    """Apply a client configuration payload; unknown keys are ignored."""
    value = config.get("min_person_score")
    if value is not None:
        _config["min_person_score"] = max(0.0, min(1.0, float(value)))
        print(f"config: min_person_score set to {_config['min_person_score']}", flush=True)
    value = config.get("draw_uncertain")
    if value is not None:
        _config["draw_uncertain"] = bool(value)
        print(f"config: draw_uncertain set to {_config['draw_uncertain']}", flush=True)


# Person-tracking crop: instead of the full frame, the model gets a window cut
# around the people detected in the previous frame. The model input is fixed at
# 513x257 (letterboxed), so a smaller source image means people render larger
# in it, which raises keypoint confidence for people far from the camera.
# Detected coordinates are mapped back to full-frame pixels before being emitted.

MIN_CROP_H = 466  # 513/1.1: avoid >10% upscaling blur, which drops joint scores
MIN_CROP_W = 320  # narrower crops collapse elbow/wrist scores (measured on test images)
REFRESH_EVERY = 10  # full-frame tracker refresh cadence while cropping (~0.35 s at 29 fps)

_last_union_bbox: tuple[int, int, int, int] | None = None
_frame_count = 0


def _axis_window(center: float, size: float, limit: int) -> tuple[int, int]:
    """A window of `size` around `center`, shifted (not shrunk) to fit [0, limit]."""
    size = min(int(size), limit)
    start = int(round(center - size / 2))
    start = max(0, min(start, limit - size))
    return start, start + size


def _crop_rect(img_h: int, img_w: int, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Crop window around the tracked people, generous on top for raised arms.

    Returns (x1, y1, x2, y2), or None when the window degenerates or would be
    the whole frame anyway.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    mx1, mx2 = x1 - 0.30 * bw, x2 + 0.30 * bw
    my1, my2 = y1 - 0.45 * bh, y2 + 0.15 * bh  # top bias: raised wrists go well above the head
    cx1, cx2 = _axis_window((mx1 + mx2) / 2, max(mx2 - mx1, MIN_CROP_W), img_w)
    cy1, cy2 = _axis_window((my1 + my2) / 2, max(my2 - my1, MIN_CROP_H), img_h)
    if cx2 - cx1 < 48 or cy2 - cy1 < 48 or (cx2 - cx1 >= img_w and cy2 - cy1 >= img_h):
        return None
    return cx1, cy1, cx2, cy2


def _union_bbox(person_scores: np.ndarray, keypoint_scores: np.ndarray, coords_xy: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box around all detected people (confident keypoints only)."""
    boxes = []
    for person_score, kp_scores, kp_xy in zip(person_scores, keypoint_scores, coords_xy, strict=False):
        if person_score == 0.0:
            break
        confident = kp_scores >= MIN_KEYPOINT_SCORE
        pts = kp_xy[confident] if confident.any() else kp_xy
        boxes.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    if not boxes:
        return None
    return (
        int(min(b[0] for b in boxes)),
        int(min(b[1] for b in boxes)),
        int(max(b[2] for b in boxes)),
        int(max(b[3] for b in boxes)),
    )


def _merge_bbox(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    """Union rectangle of two optional bounding boxes."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _set_input(rgb_input: np.ndarray) -> None:
    """Quantize (if needed) and feed the preprocessed RGB input into the model.

    Args:
        rgb_input: Preprocessed RGB image of shape [1, H, W, 3], dtype uint8 in range [0, 255].
    """
    detail = posenet_input[0]
    if np.issubdtype(detail["dtype"], np.integer):
        # Quantized model: the input range is normalized [0, 1].
        normalized = rgb_input.astype(np.float32) / 255.0
        input_val = quantize(
            normalized,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    else:
        # Float model expects RGB in [0, 1].
        input_val = (rgb_input.astype(np.float32) / 255.0).astype(detail["dtype"])
    posenet.set_tensor(detail["index"], input_val)


def _get_output(detail: dict) -> np.ndarray:
    """Read one output tensor, dequantizing it if the model is quantized.

    The model emits channels-first tensors of shape [1, C, H, W]; the leading
    batch dimension is dropped to yield (C, H, W) as expected by the decoder.

    Args:
        detail: A single entry from posenet.get_output_details().

    Returns:
        np.ndarray: Output tensor with shape (C, H, W).
    """
    tensor = posenet.get_tensor(detail["index"])
    if np.issubdtype(detail["dtype"], np.integer):
        tensor = dequantize(
            tensor,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    # [1, C, H, W] -> (C, H, W)
    return tensor.squeeze(0)


def _run_model(frame: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run PoseNet on `frame` and return scores and full-frame (x, y) coordinates.

    Parameters
    ----------
    frame
        RGB image to infer on: the full frame or a crop of it.
    dx, dy
        Offset of `frame` inside the full frame, added back to the decoded
        coordinates so results are always in full-frame pixels.
    """
    input_val, scale, pad = resize_pad(frame, (INPUT_HEIGHT, INPUT_WIDTH))
    input_val = np.expand_dims(input_val, axis=0)

    _set_input(input_val)
    posenet.invoke()

    # Outputs follow the model's export order:
    #   heatmaps, offsets, displacement_fwd, displacement_bwd, max_vals
    heatmaps = _get_output(posenet_output[0])
    offsets = _get_output(posenet_output[1])
    displacement_fwd = _get_output(posenet_output[2])
    displacement_bwd = _get_output(posenet_output[3])
    max_vals = _get_output(posenet_output[4])

    person_scores, keypoint_scores, keypoint_coords = decode_multiple_persons(
        heatmaps,
        offsets,
        displacement_fwd,
        displacement_bwd,
        max_vals,
        min_person_score=_config["min_person_score"],
    )

    # Map (y, x) keypoint coordinates from network space back to full-frame pixels
    # (undo letterbox, then add the crop offset)
    pad_left, pad_top = pad
    keypoint_coords[..., 0] = (keypoint_coords[..., 0] - pad_top) / scale + dy
    keypoint_coords[..., 1] = (keypoint_coords[..., 1] - pad_left) / scale + dx

    return person_scores, keypoint_scores, keypoint_coords[..., ::-1]


def inference_callback(rgb_frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Process a single frame through the pose estimation pipeline.

    Args:
        rgb_frame: Input frame as RGB np.ndarray (H, W, 3).

    Returns:
        tuple[np.ndarray, dict]: contains (annotated_frame, metadata), where metadata contains:
            - 'persons': list of dicts, one per detected person, each containing:
                - 'score': float (pose confidence)
                - 'keypoints': list of 17 dicts with 'name', 'x', 'y', 'score',
                  where x, y are pixel coordinates in the frame
                - 'bounding_box_xyxy': list [x1, y1, x2, y2] in frame coordinates
    """
    global _last_union_bbox, _frame_count
    _frame_count += 1

    # Person-tracking crop: infer on a window around the previously seen people
    frame = rgb_frame
    dx = dy = 0
    crop_window = None
    if _last_union_bbox is not None:
        rect = _crop_rect(rgb_frame.shape[0], rgb_frame.shape[1], _last_union_bbox)
        if rect is not None:
            x1, y1, x2, y2 = rect
            frame = np.ascontiguousarray(rgb_frame[y1:y2, x1:x2])
            dx, dy = x1, y1
            crop_window = [x1, y1, x2, y2]

    person_scores, keypoint_scores, coords_xy = _run_model(frame, dx, dy)
    tracked = _union_bbox(person_scores, keypoint_scores, coords_xy)

    # Periodic full-frame pass, tracker-only: discovers people entering the
    # scene outside the crop window without touching the emitted results
    if crop_window is not None and _frame_count % REFRESH_EVERY == 0:
        full_scores, full_kp_scores, full_coords = _run_model(rgb_frame, 0, 0)
        tracked = _merge_bbox(tracked, _union_bbox(full_scores, full_kp_scores, full_coords))

    _last_union_bbox = tracked

    # Draw predictions on the full frame and get metadata; coordinates in (x, y) format
    metadata = draw_persons(rgb_frame, person_scores, keypoint_scores, coords_xy, _config["draw_uncertain"])
    metadata["crop_window"] = crop_window

    return rgb_frame, metadata
