# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from PIL import ImageDraw, ImageFont
from PIL.Image import Image
import numpy as np
from .detection import Detection

font_size = 18
font = ImageFont.load_default(font_size)


# TODO: move these drawing functions to app_utils and merge them with the ones in object detection & friends
def draw_bounding_boxes(frame: Image, detections: list[Detection]) -> Image:
    for detection in detections:
        frame = draw_bounding_box(frame, detection)
        # code_content = detection.content
        # code_type = detection.type
        # code_coords = detection.coords

        # # Draw the bounding box and text on the frame
        # if code_coords is not None and code_coords.shape == (4, 2):
        #     draw.polygon(code_coords, outline=(0, 255, 0), width=3)

        #     # Calculate text position
        #     min_x = int(np.min(code_coords[:, 0]))
        #     min_y = int(np.min(code_coords[:, 1]))
        #     text_x = min_x
        #     text_y = min_y - (font_size + 5)

        #     if text_y < 5:  # If we're close to the top of the image
        #         # Move the text below the first point of the bounding box
        #         first_point_y = int(code_coords[0, 1])  # Assuming the first point is at index 0 of the second dimension
        #         text_y = first_point_y + 10  # Place text below the first point with some padding

        #     # Draw the text
        #     text_to_draw = f"[{code_type}] {code_content}"
        #     draw.text((text_x, text_y), text_to_draw, fill=(0, 255, 0), font=font)
        # else:
        #     print(f"Warning: Invalid or missing coordinates. Skipping detected code '{code_content}'.")
        #     continue

    return frame


def draw_bounding_box(frame: Image, detection: Detection) -> Image:
    """Draws a bounding box and label on an image for a detected QR code or barcode.

    This function overlays a green polygon around the detected code area and
    adds a text label above (or below) the bounding box with the code type and content.

    Args:
        frame (Image): The PIL Image object to draw on. This image will be modified in-place.
        detection (Detection): The detection result containing the code's content, type, and corner coordinates.

    Returns:
        Image: The annotated image with a bounding box and label drawn.
    """
    # Make a copy of the image to draw on
    draw = ImageDraw.Draw(frame)

    code_content = detection.content
    code_type = detection.type
    code_coords = detection.coords

    # Draw the bounding box and text on the frame
    if code_coords is not None and code_coords.shape == (4, 2):
        draw.polygon(code_coords, outline=(0, 255, 0), width=3)

        # Calculate text position
        min_x = int(np.min(code_coords[:, 0]))
        min_y = int(np.min(code_coords[:, 1]))
        text_x = min_x
        text_y = min_y - (font_size + 5)

        if text_y < 5:  # If we're close to the top of the image
            # Move the text below the first point of the bounding box
            first_point_y = int(code_coords[0, 1])  # Assuming the first point is at index 0 of the second dimension
            text_y = first_point_y + 10  # Place text below the first point with some padding

        # Draw the text
        text_to_draw = f"[{code_type}] {code_content}"
        draw.text((text_x, text_y), text_to_draw, fill=(0, 255, 0), font=font)
    else:
        print("Warning: Invalid or missing coordinates. Returning original image.")
        return frame

    return frame
