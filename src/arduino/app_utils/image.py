# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import io
from PIL import Image, ImageDraw, ImageFont
from arduino.app_utils import Logger

logger = Logger(__name__)


class Shape:
    RECTANGLE = "rectangle"
    CIRCLE = "circle"


# Define a mapping of confidence ranges to colors for bounding boxes
CONFIDENCE_MAP = {
    (0, 20): "#FF0976",  # Pink
    (21, 40): "#FF8131",  # Orange
    (41, 60): "#FFFC00",  # Yellow
    (61, 80): "#00DED7",  # Light blue
    (81, 100): "#1EFF00",  # Green
}

FONT_PATH = "/home/app/.fonts/OpenSans.ttf"


# Get the color for a given confidence value based on the defined ranges.
# If the confidence is outside the defined ranges, it defaults to green.
def get_box_color(confid):
    for (low, high), color in CONFIDENCE_MAP.items():
        if low <= confid <= high:
            return color
    return "#1EFF00"  # Default to Green if out of range


def _read(file_path: str) -> bytes:
    """Read an image from a file path and return a PIL Image object."""
    try:
        with open(file_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger(f"Error reading image: {e}")
        return None


def get_image_type(image_bytes: bytes | Image.Image) -> str | None:
    """Detect the type of image from bytes or a PIL Image object.

    Returns:
        str: The image type in lowercase (e.g., 'jpeg', 'png').
        None if the image type cannot be determined.
    """
    try:
        if isinstance(image_bytes, Image.Image):
            # If the input is already a PIL Image, we can directly get its format
            if image_bytes.format is not None:
                return image_bytes.format.lower()
        elif isinstance(image_bytes, bytes):
            image = Image.open(io.BytesIO(image_bytes))
            return image.format.lower()  # Returns 'jpeg', 'png', etc.
        return None
    except Exception as e:
        print(f"Error detecting image type: {e}")
        return None


def get_image_bytes(image: str | Image.Image | bytes) -> bytes | None:
    """Convert different type of image objects to bytes."""
    if image is None:
        return None
    try:
        if isinstance(image, Image.Image):
            byte_io = io.BytesIO()
            image.save(byte_io, "PNG")
            return byte_io.getvalue()
        elif isinstance(image, bytes):
            return image
        elif isinstance(image, str):
            return _read(image)
    except Exception as e:
        logger.error(f"Error converting image to bytes: {e}")
        return None


def draw_bounding_boxes(
    image: Image.Image | bytes,
    detection: dict,
    draw: ImageDraw.ImageDraw = None,
    shape: Shape = Shape.RECTANGLE,
) -> Image.Image | None:
    """Draw bounding boxes on an image using PIL.

    The thickness of the box and font size are scaled based on image size.

    Args:
        image (Image.Image|bytes): The image to draw on, can be a PIL Image or bytes.
        detection (dict): A dictionary containing detection results with keys 'class_name', 'bounding_box_xyxy', and
            'confidence'.
        draw (ImageDraw.ImageDraw, optional): An existing ImageDraw object to use. If None, a new one is created.
        shape (Shape, optional): Shape of the bounding box. Defaults to rectangle.
        itself. Defaults to False.
    """
    if isinstance(image, bytes):
        image_box = Image.open(io.BytesIO(image))
    else:
        image_box = image

    if draw is None:
        draw = ImageDraw.Draw(image_box)

    if not detection or "detection" not in detection:
        return None

    if shape not in (Shape.RECTANGLE, Shape.CIRCLE):
        logger.warning(f"Unsupported shape '{shape}'. Defaulting to rectangle.")
        shape = Shape.RECTANGLE

    detection = detection["detection"]

    # Scale font size and box thickness based on image size and number of detections
    ref_dim = max(image_box.size)
    n_detections = max(1, len(detection))
    # More aggressive scaling for many detections or small images
    font_size = max(8, int(ref_dim / (28 + n_detections * 3)))
    box_thickness = max(1, int(ref_dim / 250))
    label_vpad = max(2, int(font_size * 0.4))
    label_hpad = max(4, int(font_size * 0.8))

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception as e:
        logger.warning(f"Error loading custom font: {e}. Using default font.")
        font = ImageFont.load_default(14)

    for i, obj_det in enumerate(detection):
        if "class_name" not in obj_det or "bounding_box_xyxy" not in obj_det or "confidence" not in obj_det:
            continue

        class_name = obj_det["class_name"]
        box = obj_det["bounding_box_xyxy"]
        confid = float(obj_det["confidence"])

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        # Set box color based on confidence
        box_color = get_box_color(confid)

        # Prepare label text
        text = f"{class_name.capitalize()} {confid:.1f}%"

        text_box_size = font.getbbox(text)[2:]
        text_width, text_height = text_box_size[0], text_box_size[1]
        # Align label above the box
        label_gap = max(1, int(font_size * 0.15))  # space between box and label
        y1_text = y1 - text_height - label_vpad * 2 - label_gap
        if y1_text < 0:
            y1_text = y1 + label_gap  # fallback: label below the box if it goes out of bounds
        y2_text = y1_text + text_height + label_vpad * 2
        x2_text = x1 + text_width + label_hpad * 2

        # Draw bounding box
        if shape == Shape.CIRCLE:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            radius = 10
            bounding_box = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
            draw.ellipse(bounding_box, outline=box_color, width=2)
        else:
            draw.rectangle((x1, y1, x2, y2), outline=box_color, width=box_thickness)
        # Draw label background (dark gray, semi-transparent) on overlay
        label_bg_color = (0, 0, 0, 128)
        overlay = Image.new("RGBA", image_box.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle((x1, y1_text, x2_text, y2_text), fill=label_bg_color, outline=None)
        image_box = image_box.convert("RGBA")
        image_box = Image.alpha_composite(image_box, overlay)
        draw = ImageDraw.Draw(image_box)
        # Draw label text (same color as box, with padding)
        draw.text((x1 + label_hpad, y1_text + label_vpad), text, fill=box_color, font=font)

    return image_box


def draw_anomaly_markers(
    image: Image.Image | bytes,
    detection: dict,
    draw: ImageDraw.ImageDraw = None,
) -> Image.Image | None:
    """Draw bounding boxes on an image using PIL.

    The thickness of the box and font size are scaled based on image size.

    Args:
        image (Image.Image|bytes): The image to draw on, can be a PIL Image or bytes.
        detection (dict): A dictionary containing detection results with keys 'class_name', 'bounding_box_xyxy', and
            'score'.
        draw (ImageDraw.ImageDraw, optional): An existing ImageDraw object to use. If None, a new one is created.
        label_above_box (bool, optional): If True, labels are drawn above the bounding box. Defaults to False.
        colours (list, optional): List of colors to use for bounding boxes. Defaults to a predefined palette.
        text_color (str, optional): Color of the text labels. Defaults to "white".
    """
    if isinstance(image, bytes):
        image_box = Image.open(io.BytesIO(image))
    else:
        image_box = image

    if image_box.mode != "RGBA":
        image_box = image_box.convert("RGBA")

    if draw is None:
        draw = ImageDraw.Draw(image_box)

    max_anomaly_score = detection.get("anomaly_max_score", 0.0)

    if not detection or "detection" not in detection:
        return None
    detection = detection["detection"]

    # Scale font size and box thickness based on image size
    ref_dim = max(image_box.size)
    box_thickness = max(1, int(ref_dim / 400))

    for i, obj_det in enumerate(detection):
        if "class_name" not in obj_det or "bounding_box_xyxy" not in obj_det or "score" not in obj_det:
            continue

        box = obj_det["bounding_box_xyxy"]
        score = float(obj_det["score"])

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        normalized_score = score / max_anomaly_score if max_anomaly_score > 0 else 0
        alpha = int(255 * min(max(normalized_score, 0), 1))

        base_color_rgb = (255, 0, 0)
        outline_color = (0, 0, 0)
        fill_color_with_alpha = base_color_rgb + (alpha,)

        temp_layer = Image.new("RGBA", image_box.size, (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_layer)

        temp_draw.rectangle([x1, y1, x2, y2], fill=fill_color_with_alpha)
        temp_draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=box_thickness)
        image_box = Image.alpha_composite(image_box, temp_layer)

        draw = ImageDraw.Draw(image_box)

    return image_box
