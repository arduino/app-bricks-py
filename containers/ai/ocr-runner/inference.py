# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""EasyOCR (CRAFT detector + CRNN recognizer) on ONNX Runtime.

Pipeline mirrors `qai_hub_models/models/easyocr/app.py` from ai-hub-models v0.61.0,
reimplemented with numpy/OpenCV only - no torch, no easyocr package at runtime.

Both networks run on the QNN execution provider (Hexagon NPU) when it is available and
fall back to the ORT CPU provider otherwise; see `utils/onnx_ep.py`. The TFLite/LiteRT
version of this runner could not delegate the recognizer to the NPU at all, which is why
the runtime was switched.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from aihub.image_processing import denormalize_coordinates, resize_pad
from aihub.logging import logger

from utils.bbox_processing import box_4corners, box_xx_yy, diff, get_det_boxes, group_text_box
from utils.constants import (
    CHARACTERS,
    DETECTOR_ARGS,
    DETECTOR_MODEL_PATH,
    DETECTOR_OUTPUT_STRIDE,
    LANG_CHAR,
    RECOGNIZER_ARGS,
    RECOGNIZER_MODEL_PATH,
)
from utils.image_processing import adjust_contrast, four_point_transform
from utils.metadata import build_metadata
from utils.model_io_processing import ONNXModel
from utils.orientation import detections_to_original, parse_rotations, reading_score, rotate_image, select_orientation
from utils.post_processing import CTCLabelConverter

# Stage-by-stage timing and configuration prints, off by default; EASYOCR_DEBUG=1 enables
# them. One INFO summary line per frame is always logged (see inference_callback).
DEBUG_TIMING = os.environ.get("EASYOCR_DEBUG", "0") == "1"


def _debug(message: str) -> None:
    if DEBUG_TIMING:
        print(f"[ocr-debug] {message}", flush=True)


# Per-frame counters for the summary line, summed over the orientations read; reset by
# inference_callback, filled by recognizer_get_text.
_frame_stats = {"boxes": 0, "retries": 0}


def _load_models() -> tuple[ONNXModel, ONNXModel]:
    """
    Open both networks. The execution provider is picked per model, so the detector can
    sit on the NPU while the recognizer stays on the CPU (or vice versa) via
    EASYOCR_EP_DETECTOR / EASYOCR_EP_RECOGNIZER.

    On the QNN/HTP backend the first open of a model compiles its graph, which is
    single-threaded inside QNN and runs into minutes for the recognizer. The compiled
    result is cached as a context binary (see utils/onnx_ep.py), so that cost is paid once
    per cache - or never, when a pre-compiled binary ships next to the model. The two
    models are independent sessions, so EASYOCR_PARALLEL_INIT=1 compiles them on two
    threads; off by default because concurrent QNN backend initialisation is not something
    the EP documents as supported.
    """
    specs = (
        (os.environ.get("EASYOCR_DETECTOR_MODEL", DETECTOR_MODEL_PATH), os.environ.get("EASYOCR_EP_DETECTOR")),
        (os.environ.get("EASYOCR_RECOGNIZER_MODEL", RECOGNIZER_MODEL_PATH), os.environ.get("EASYOCR_EP_RECOGNIZER")),
    )

    t_start = time.perf_counter()
    if os.environ.get("EASYOCR_PARALLEL_INIT", "0") == "1":
        with ThreadPoolExecutor(max_workers=2) as pool:
            models = list(pool.map(lambda spec: ONNXModel(spec[0], backend=spec[1]), specs))
    else:
        models = [ONNXModel(path, backend=backend) for path, backend in specs]
    _debug(f"models loaded in {(time.perf_counter() - t_start) * 1000:.0f} ms")

    return models[0], models[1]


detector, recognizer = _load_models()
logger.info(
    f"ocr: detector on {detector.execution_provider}, recognizer on {recognizer.execution_provider}"
    + (
        " (measured)"
        if detector.placement or recognizer.placement
        else " (attached, not verified; run tools/compile_htp_context.py or set EASYOCR_QNN_VERIFY=1)"
    )
)

converter = CTCLabelConverter(CHARACTERS, LANG_CHAR)

# Client-configurable settings that are not part of the CTC converter. Read once per frame.
_settings: dict = {
    # Extra orientations (degrees counter-clockwise, multiples of 90) the whole image is
    # also read at; the most confident orientation wins. Empty = upright only. Config
    # key: "rotation".
    "rotations": [],
}


def apply_config(config: dict) -> None:
    """Apply a client configuration payload; unknown keys are ignored.

    Supported settings:
        allowlist (str): Restrict recognition to these characters (e.g. "0123456789").
            An empty string removes the restriction.
        rotation (list[int]): Also read the whole image turned counter-clockwise by
            these angles (90, 180, 270) and report the orientation that reads most
            confidently, for text that is not upright in the image (see
            utils/orientation.py). An empty list reads upright only. Each angle costs
            one more full pass, detection and recognition.
    """
    if "allowlist" in config:
        value = config.get("allowlist")
        allowlist = str(value) if value else None
        converter.set_allowlist(allowlist)
        _debug(f"config: allowlist set to {allowlist!r}")
    if "rotation" in config:
        try:
            _settings["rotations"] = parse_rotations(config.get("rotation"))
        except ValueError as exc:
            logger.warning(f"ocr: config rotation ignored ({exc}); keeping {_settings['rotations']}")
        else:
            _debug(f"config: rotations set to {_settings['rotations']}")


_recognizer_classes = int(recognizer.output_shapes[0][-1])
if _recognizer_classes != converter.num_classes:
    logger.warning(
        f"ocr: recognizer emits {_recognizer_classes} classes but the configured character "
        f"set has {converter.num_classes} (including the CTC blank). Decoded text will be wrong - "
        f"update CHARACTERS/LANG_CHAR in utils/constants.py to match the exported model."
    )


def detector_preprocess(rgb_frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize and pad an RGB image for the detector network.

    Parameters
    ----------
    rgb_frame
        [H, W, 3] uint8 RGB image.

    Returns
    -------
    detector_input : np.ndarray
        [1, H', W', 3] float32 in the range [0, 1].
    scale : float
        Scale factor applied to the original image.
    pad : (int, int)
        (pad_left, pad_top) added to the resized image.
    """
    frame_resized, scale, pad = resize_pad(rgb_frame.astype(np.float32) / 255.0, detector.image_shape)
    return np.expand_dims(frame_resized, axis=0), scale, pad


def detector_postprocess(
    result: np.ndarray,
    scale: float,
    pad: tuple[int, int],
) -> tuple[list[box_xx_yy], list[box_4corners]]:
    """
    Turn the detector score maps into text boxes in original-image coordinates.

    Parameters
    ----------
    result
        [H/2, W/2, 2] detector output: channel 0 is the region score, channel 1 the
        affinity (link) score.
    scale
        Scale factor returned by `detector_preprocess`.
    pad
        Padding returned by `detector_preprocess`.

    Returns
    -------
    horizontal_boxes : list[box_xx_yy]
        Axis-aligned boxes as (xmin, xmax, ymin, ymax), absolute pixels.
    free_boxes : list[box_4corners]
        Slanted boxes as ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), absolute pixels.
    """
    score_text = result[:, :, 0]
    score_link = result[:, :, 1]

    boxes = get_det_boxes(
        score_text,
        score_link,
        DETECTOR_ARGS["text_threshold"],
        DETECTOR_ARGS["link_threshold"],
        DETECTOR_ARGS["low_text"],
    )

    # Undo the resize/pad. Scale and padding are halved because the score maps are
    # emitted at half the network input resolution.
    detections: list[np.ndarray] = []
    for box in boxes:
        box = box.astype(np.float32)
        denormalize_coordinates(
            box,
            (1, 1),
            scale / DETECTOR_OUTPUT_STRIDE,
            (pad[0] // DETECTOR_OUTPUT_STRIDE, pad[1] // DETECTOR_OUTPUT_STRIDE),
        )
        detections.append(box.astype(np.int32).reshape(-1))

    # This reinterprets slanted boxes as horizontal ones when their slant is below slope_ths.
    horizontal_list_raw, free_list_raw = group_text_box(
        detections,
        slope_ths=DETECTOR_ARGS["slope_ths"],
        ycenter_ths=DETECTOR_ARGS["ycenter_ths"],
        height_ths=DETECTOR_ARGS["height_ths"],
        width_ths=DETECTOR_ARGS["width_ths"],
        add_margin=DETECTOR_ARGS["add_margin"],
    )

    horizontal_list: list[box_xx_yy] = [tuple(x) for x in horizontal_list_raw]
    free_list: list[box_4corners] = [tuple(tuple(y) for y in x) for x in free_list_raw]

    min_size = DETECTOR_ARGS["min_size"]
    if min_size:
        horizontal_list = [i for i in horizontal_list if max(i[1] - i[0], i[3] - i[2]) > min_size]
        free_list = [i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]

    return horizontal_list, free_list


def get_cutouts(
    img_grey: np.ndarray,
    horizontal_boxes: list[box_xx_yy],
    free_boxes: list[box_4corners],
) -> tuple[list[tuple[box_xx_yy | box_4corners, float]], list[np.ndarray]]:
    """
    Crop every detected text box out of the greyscale image.

    Parameters
    ----------
    img_grey
        [H, W] uint8 greyscale image.
    horizontal_boxes
        Axis-aligned boxes as (xmin, xmax, ymin, ymax).
    free_boxes
        Slanted boxes as 4 (x, y) corners.

    Returns
    -------
    boxes : list[tuple[box, float]]
        The box each cutout came from, plus its top y coordinate, sorted top to bottom.
    cutouts : list[np.ndarray]
        [h, w] uint8 greyscale crops, one per box, still to be sized for the recognizer
        with `prepare_recognizer_input`.
    """
    # If nothing was detected, read the whole image instead.
    if not horizontal_boxes and not free_boxes:
        y_max, x_max = img_grey.shape
        horizontal_boxes = [(0, x_max, 0, y_max)]

    cutouts: list[tuple[np.ndarray, box_xx_yy | box_4corners, float]] = []

    # Free boxes must be warped to a rectangle before cropping.
    for free_box in free_boxes:
        rect = np.array(free_box, dtype="float32")
        cutout = four_point_transform(img_grey, rect)
        if 0 in cutout.shape:
            continue
        cutouts.append((cutout, free_box, float(min(rect[:, 1]))))

    # Horizontal boxes can be cropped directly.
    for box in horizontal_boxes:
        x_min = max(0, int(box[0]))
        x_max = int(min(box[1], img_grey.shape[1]))
        y_min = max(0, int(box[2]))
        y_max = int(min(box[3], img_grey.shape[0]))

        if y_max - y_min <= 0 or x_max - x_min <= 0:
            continue

        cutouts.append((img_grey[y_min:y_max, x_min:x_max], box, float(y_min)))

    cutouts = sorted(cutouts, key=lambda item: item[2])

    boxes = [(box, y_min) for _, box, y_min in cutouts]
    return boxes, [cutout for cutout, _, _ in cutouts]


def prepare_recognizer_input(cutout: np.ndarray) -> np.ndarray:
    """
    Resize a greyscale cutout to the recognizer input shape.

    The text floats to the left of the canvas and the padding is filled with the
    cutout's top-left pixel, so the padded region blends into the page background.

    Parameters
    ----------
    cutout
        [H, W] uint8 greyscale crop.

    Returns
    -------
    frame : np.ndarray
        [1, 64, 800, 1] float32 in the range [0, 1].
    """
    img = cutout.astype(np.float32) / 255.0
    pad_value = float(img[0][0]) if img.size > 0 else 0.0
    frame_resized, _, _ = resize_pad(
        img,
        recognizer.image_shape,
        pad_value=pad_value,
        horizontal_float="left",
    )
    return frame_resized.reshape(1, *recognizer.image_shape, 1)


def recognizer_inference(cutout_frames: list[np.ndarray]) -> list[tuple[str, float]]:
    """
    Read the text out of a batch of prepared cutouts.

    Parameters
    ----------
    cutout_frames
        List of [1, 64, 800, 1] float32 recognizer inputs.

    Returns
    -------
    predictions : list[tuple[str, float]]
        One (text, confidence) pair per input frame.
    """
    if not cutout_frames:
        return []

    t_start = time.perf_counter()
    preds = np.concatenate([recognizer(frame)[0] for frame in cutout_frames], axis=0)
    t_invoke = time.perf_counter()
    preds_prob = converter.filter_probabilities(preds)
    predictions = converter.decode_greedy(preds_prob)
    t_decode = time.perf_counter()

    invoke_ms = (t_invoke - t_start) * 1000
    _debug(
        f"recognizer: {len(cutout_frames)} frames in {invoke_ms:.0f} ms "
        f"({invoke_ms / len(cutout_frames):.0f} ms/frame), decode {(t_decode - t_invoke) * 1000:.0f} ms"
    )
    return predictions


def recognizer_get_text(
    img_grey: np.ndarray,
    horizontal_boxes: list[box_xx_yy],
    free_boxes: list[box_4corners],
) -> tuple[list[tuple[box_xx_yy, str, float]], list[tuple[box_4corners, str, float]]]:
    """
    Run the recognizer over every detected box and clean up the predictions.

    Low-confidence cutouts are read a second time with boosted contrast, and the more
    confident of the two readings wins.

    Parameters
    ----------
    img_grey
        [H, W] uint8 greyscale image.
    horizontal_boxes
        Axis-aligned boxes as (xmin, xmax, ymin, ymax).
    free_boxes
        Slanted boxes as 4 (x, y) corners.

    Returns
    -------
    result_horizontal : list[tuple[box_xx_yy, str, float]]
        (box, text, confidence) per axis-aligned detection.
    result_free : list[tuple[box_4corners, str, float]]
        (box, text, confidence) per slanted detection.
    """
    t_start = time.perf_counter()
    boxes, cutouts = get_cutouts(img_grey, horizontal_boxes, free_boxes)
    cutout_frames = [prepare_recognizer_input(cutout) for cutout in cutouts]
    _debug(f"cutouts: {len(cutouts)} boxes prepared in {(time.perf_counter() - t_start) * 1000:.0f} ms")

    predictions = recognizer_inference(cutout_frames)
    _frame_stats["boxes"] += len(cutouts)

    # Re-read anything the recognizer was unsure about, with the contrast pushed up.
    contrast_ths = RECOGNIZER_ARGS["contrast_ths"]
    low_confidence_indices = [i for i, (_, confidence) in enumerate(predictions) if contrast_ths is not None and confidence < contrast_ths]
    if low_confidence_indices:
        _debug(f"low-confidence retry: re-reading {len(low_confidence_indices)}/{len(predictions)} cutouts with boosted contrast")
        contrast = 1 / contrast_ths if contrast_ths else 1
        high_contrast_predictions = recognizer_inference([adjust_contrast(cutout_frames[i], contrast) for i in low_confidence_indices])
    else:
        high_contrast_predictions = []
    _frame_stats["retries"] += len(low_confidence_indices)

    result_horizontal: list[tuple[box_xx_yy, str, float]] = []
    result_free: list[tuple[box_4corners, str, float]] = []

    for i, ((box, _), (text, confidence)) in enumerate(zip(boxes, predictions)):
        if i in low_confidence_indices:
            hc_text, hc_confidence = high_contrast_predictions[low_confidence_indices.index(i)]
            if hc_confidence > confidence:
                text, confidence = hc_text, hc_confidence

        if not text:
            continue

        # The recognizer can hallucinate these tokens when the cutout ends in
        # substantial empty space (padding is always added to the right).
        # TODO: verify other allucinations like '. Do specific tests for this.
        text = text.strip()
        if text and text[-1] in ("]", "|"):
            text = text[:-1].strip()
        if not text:
            continue

        if isinstance(box[0], tuple):
            result_free.append((box, text, confidence))
        else:
            result_horizontal.append((box, text, confidence))

    return result_horizontal, result_free


def read_frame(rgb_frame: np.ndarray) -> dict:
    """
    Run detection and recognition once, on the frame as given.

    Args:
        rgb_frame: Input frame as RGB np.ndarray (H, W, 3), uint8.

    Returns:
        dict: metadata as described in `inference_callback`, in the coordinates of
        `rgb_frame`.
    """
    t_start = time.perf_counter()
    grey_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

    # Run detector
    detector_input, scale, pad = detector_preprocess(rgb_frame)
    t_preprocess = time.perf_counter()
    detector_output = detector(detector_input)[0]
    t_invoke = time.perf_counter()

    # The exported graph may hand back the score maps channel-first.
    if detector_output.shape[-1] != 2 and detector_output.shape[1] == 2:
        detector_output = detector_output.transpose(0, 2, 3, 1)

    horizontal_boxes, free_boxes = detector_postprocess(detector_output[0], scale, pad)
    t_postprocess = time.perf_counter()
    _debug(
        f"detector: preprocess {(t_preprocess - t_start) * 1000:.0f} ms, "
        f"invoke {(t_invoke - t_preprocess) * 1000:.0f} ms, "
        f"postprocess {(t_postprocess - t_invoke) * 1000:.0f} ms "
        f"-> {len(horizontal_boxes)} horizontal + {len(free_boxes)} free boxes"
    )

    # Run recognizer
    result_horizontal, result_free = recognizer_get_text(grey_frame, horizontal_boxes, free_boxes)

    t_end = time.perf_counter()
    _debug(
        f"pass total {(t_end - t_start) * 1000:.0f} ms "
        f"(detector {(t_postprocess - t_start) * 1000:.0f} ms, recognizer {(t_end - t_postprocess) * 1000:.0f} ms)"
    )

    return build_metadata(result_horizontal, result_free)


def inference_callback(rgb_frame: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """
    Process a single frame through the EasyOCR pipeline.

    The frame is read upright and, for every configured rotation, turned by that angle;
    the orientation that reads most confidently is reported (see utils/orientation.py).

    Args:
        rgb_frame: Input frame as RGB np.ndarray (H, W, 3), uint8.

    Returns:
        tuple[np.ndarray | None, dict]: contains (None, metadata). The frame slot is
        always None - this pipeline produces text, not an annotated image, so the
        output sinks emit the metadata without a video feed. metadata contains:
            - 'text': str, all detected strings joined by newlines, in reading order
            - 'detections': list of dicts, each containing:
                - 'text': str
                - 'confidence': float
                - 'bounding_box_xyxy': list [x1, y1, x2, y2] in frame coordinates
                - 'polygon': list of 4 [x, y] vertices in frame coordinates, ordered
                  top-left, top-right, bottom-right, bottom-left of the text as read
                  (for rotated text the first vertex is not the frame's top-left one)
                - 'type': str ('horizontal' or 'free')
    """
    t_start = time.perf_counter()
    _frame_stats["boxes"] = _frame_stats["retries"] = 0

    angles = [0, *_settings["rotations"]]
    readings = {angle: read_frame(rotate_image(rgb_frame, angle)) for angle in angles}
    scores = {angle: reading_score((d["text"], d["confidence"]) for d in metadata["detections"]) for angle, metadata in readings.items()}
    angle = select_orientation(scores)
    metadata = readings[angle]
    detections_to_original(metadata["detections"], angle, rgb_frame.shape)
    if len(angles) > 1:
        _debug(f"orientations {', '.join(f'{a}: {s:.2f}' for a, s in scores.items())} -> {angle}")

    t_end = time.perf_counter()
    # One line per frame, always: enough to trace calls and performance from the container logs.
    logger.info(
        f"ocr: {rgb_frame.shape[1]}x{rgb_frame.shape[0]} frame, {len(metadata['detections'])} texts "
        f"read at {angle} deg ({len(angles)} orientation(s), {_frame_stats['boxes']} boxes, {_frame_stats['retries']} retries) "
        f"in {(t_end - t_start) * 1000:.0f} ms"
    )
    return None, metadata
