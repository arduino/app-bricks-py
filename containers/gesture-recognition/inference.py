# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter


from utils.constants import (
    INPUT_WIDTH,
    INPUT_HEIGHT,
    NMS_IOU_THRESHOLD,
    MIN_DETECTOR_BOX_SCORE,
    MIN_LANDMARK_SCORE,
    DETECTOR_SCORE_CLIPPING_THRESHOLD,
    GESTURE_LABELS,
    HAND_LANDMARK_CONNECTIONS,
)
from utils.tf import load_qnn_delegate
from utils.image_processing import (
    resize_pad,
    denormalize_coordinates,
    apply_affine_to_coordinates,
    apply_batched_affines_to_frame,
)
from utils.bbox_processing import box_xywh_to_xyxy, batched_nms, compute_box_affine_crop_resize_matrix
from utils.post_processing import decode_preds_from_anchors
from utils.model_io_processing import (
    compute_object_roi,
    split_into_singleton_arrays,
    preprocess_hand_x64,
    dequantize,
)
from utils.draw import draw_predictions


# Load models
hand_detector = Interpreter(
    "models/mediapipe_hand_gesture_w8a8_PalmDetector.tflite",
    experimental_delegates=load_qnn_delegate(),
)
landmark_detector = Interpreter(
    "models/mediapipe_hand_gesture_w8a8_HandLandmarkDetector.tflite",
    experimental_delegates=load_qnn_delegate(),
)
gesture_classifier = Interpreter("models/mediapipe_hand_gesture_float_CannedGestureClassifier.tflite")
anchor_detector = np.load("models/anchors_palm.npy").astype(np.float32)
anchor_detector = anchor_detector.reshape([*list(anchor_detector.shape)[:-1], -1, 2])

hand_detector.allocate_tensors()
landmark_detector.allocate_tensors()
gesture_classifier.allocate_tensors()

detector_input = hand_detector.get_input_details()
detector_output = hand_detector.get_output_details()

landmark_input = landmark_detector.get_input_details()
landmark_output = landmark_detector.get_output_details()

classifier_input = gesture_classifier.get_input_details()
classifier_output = gesture_classifier.get_output_details()


def inference_callback(rgb_frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Process a single frame through the gesture recognition pipeline.

    Args:
        rgb_frame: Input frame as RGB np.ndarray (H, W, 3).

    Returns:
        tuple[np.ndarray, dict]: contains (annotated_frame, metadata), where metadata contains:
            - 'hands': list of dicts, each containing:
                - 'hand': str ('right' or 'left')
                - 'gesture': str
                - 'confidence': float (gesture classification confidence)
                - 'bounding_box_xyxy': list [x1, y1, x2, y2] in frame coordinates
                - 'landmarks': list of shape [21, 3] with x, y, z coordinates
    """
    # Resize and pad for detector input
    input_val, scale, pad = resize_pad(rgb_frame, (INPUT_WIDTH, INPUT_HEIGHT))
    input_val = np.expand_dims(input_val, axis=0)

    # Run hand detector
    hand_detector.set_tensor(detector_input[0]["index"], input_val)
    hand_detector.invoke()

    # Get detector outputs
    box_coords = dequantize(
        hand_detector.get_tensor(detector_output[0]["index"]),
        zero_points=detector_output[0]["quantization_parameters"]["zero_points"],
        scales=detector_output[0]["quantization_parameters"]["scales"],
    )
    box_scores = dequantize(
        hand_detector.get_tensor(detector_output[1]["index"]),
        zero_points=detector_output[1]["quantization_parameters"]["zero_points"],
        scales=detector_output[1]["quantization_parameters"]["scales"],
    )

    # Process scores
    box_scores = np.clip(
        box_scores,
        -DETECTOR_SCORE_CLIPPING_THRESHOLD,
        DETECTOR_SCORE_CLIPPING_THRESHOLD,
    )
    box_scores = 1 / (1 + np.exp(-box_scores))  # sigmoid
    box_scores = np.squeeze(box_scores, axis=-1)

    # Process coordinates
    box_coords = box_coords.reshape((*box_coords.shape[:-1], -1, 2))
    decode_preds_from_anchors(box_coords, (256, 256), anchor_detector)
    box_coords = box_xywh_to_xyxy(box_coords)

    flattened_box_coords = box_coords.reshape([*list(box_coords.shape)[:-2], -1])

    # Run non maximum suppression
    batched_selected_coords, _ = batched_nms(
        NMS_IOU_THRESHOLD,
        MIN_DETECTOR_BOX_SCORE,
        flattened_box_coords,
        box_scores,
    )

    # Process selected boxes
    selected_boxes = []
    selected_keypoints = []
    for i in range(len(batched_selected_coords)):
        selected_coords = batched_selected_coords[i]
        if len(selected_coords) != 0:
            boxes_list = []
            kps_list = []
            for j in range(len(selected_coords)):
                selected_coords_ = selected_coords[j : j + 1].reshape([*list(selected_coords[j : j + 1].shape)[:-1], -1, 2])

                denormalize_coordinates(selected_coords_, (INPUT_WIDTH, INPUT_HEIGHT), scale, pad)

                boxes_list.append(selected_coords_[:, :2])
                kps_list.append(selected_coords_[:, 2:])

            if boxes_list:
                selected_boxes.append(np.concatenate(boxes_list, axis=0))
                selected_keypoints.append(np.concatenate(kps_list, axis=0))
            else:
                selected_boxes.append(np.empty(0, dtype=np.float32))
                selected_keypoints.append(np.empty(0, dtype=np.float32))
        else:
            selected_boxes.append(np.empty(0, dtype=np.float32))
            selected_keypoints.append(np.empty(0, dtype=np.float32))

    # Compute ROI for landmarks
    batched_roi_4corners = compute_object_roi(selected_boxes, selected_keypoints)
    batched_roi_4corners = split_into_singleton_arrays(batched_roi_4corners[0])

    batched_selected_landmarks: list[np.ndarray] = []
    batched_is_right_hand: list[list[bool]] = []
    batched_gesture_labels: list[list[str]] = []
    batched_gesture_confidences: list[list[float]] = []

    # Process each detected hand
    for _, roi_4corners in enumerate(batched_roi_4corners):
        if roi_4corners.size == 0:
            continue

        affines = compute_box_affine_crop_resize_matrix(roi_4corners[:, :3], (224, 224))
        # Create input images by applying the affine transforms
        keypoint_net_inputs = apply_batched_affines_to_frame(rgb_frame, affines, (224, 224)).astype(np.uint8, copy=False)

        landmark_detector.set_tensor(landmark_input[0]["index"], keypoint_net_inputs)

        # Compute landmarks
        landmark_detector.invoke()

        landmarks = dequantize(
            landmark_detector.get_tensor(landmark_output[0]["index"]),
            zero_points=landmark_output[0]["quantization_parameters"]["zero_points"],
            scales=landmark_output[0]["quantization_parameters"]["scales"],
        ).reshape(1, 21, 3)
        ld_scores = dequantize(
            landmark_detector.get_tensor(landmark_output[1]["index"]),
            zero_points=landmark_output[1]["quantization_parameters"]["zero_points"],
            scales=landmark_output[1]["quantization_parameters"]["scales"],
        )
        lr = dequantize(
            landmark_detector.get_tensor(landmark_output[2]["index"]),
            zero_points=landmark_output[2]["quantization_parameters"]["zero_points"],
            scales=landmark_output[2]["quantization_parameters"]["scales"],
        )

        all_landmarks = []
        all_lr = []
        gesture_label = []
        gesture_confidence = []
        for ld_batch_idx in range(landmarks.shape[0]):
            # Exclude landmarks that don't meet the appropriate score threshold
            if ld_scores[ld_batch_idx] >= MIN_LANDMARK_SCORE:
                # Apply the inverse of affine transform to landmark coordinates
                inverted_affine = cv2.invertAffineTransform(affines[ld_batch_idx]).astype(np.float32)
                landmarks[ld_batch_idx][:, :2] = apply_affine_to_coordinates(landmarks[ld_batch_idx][:, :2], inverted_affine)

                # Add the predicted landmarks to our list
                all_landmarks.append(landmarks[ld_batch_idx])
                all_lr.append(np.round(lr[ld_batch_idx]).item() == 1)

                hand = np.expand_dims(landmarks[ld_batch_idx], axis=0)
                lr_val = np.expand_dims(lr[ld_batch_idx], axis=0)

                x64_a = preprocess_hand_x64(hand, lr_val, mirror=False)
                x64_b = preprocess_hand_x64(hand, lr_val, mirror=True)

                gesture_classifier.set_tensor(classifier_input[0]["index"], x64_a)
                gesture_classifier.set_tensor(classifier_input[1]["index"], x64_b)

                gesture_classifier.invoke()

                score = gesture_classifier.get_tensor(classifier_output[0]["index"])
                gesture_id = np.argmax(score.flatten())
                gesture_label.append(GESTURE_LABELS[gesture_id])
                gesture_confidence.append(float(np.max(score.flatten())))

        # Add this batch of landmarks to the output list
        batched_selected_landmarks.append(np.stack(all_landmarks, axis=0) if all_landmarks else np.empty(0, dtype=np.float32))
        batched_is_right_hand.append(all_lr)
        batched_gesture_labels.append(gesture_label)
        batched_gesture_confidences.append(gesture_confidence)

    # Add empty entry for batches with no predicted bounding boxes
    batched_selected_landmarks.append(np.empty(0, dtype=np.float32))
    batched_is_right_hand.append([])
    batched_gesture_labels.append([])
    batched_gesture_confidences.append([])

    # Draw predictions on the frame and get metadata
    metadata = draw_predictions(
        [rgb_frame] * len(batched_selected_landmarks),
        batched_selected_landmarks,
        batched_is_right_hand,
        batched_gesture_labels,
        batched_gesture_confidences,
        landmark_connections=HAND_LANDMARK_CONNECTIONS,
    )

    return rgb_frame, metadata
