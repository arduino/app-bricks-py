# Image Classification Brick

This Brick lets you perform image classification using a pre-trained neural network model.

## Overview

The Image Classification Brick allows you to:

- Analyze images and categorize their contents using a machine learning model.
- Use locally stored image files or camera feeds as input.
- Easy integration with your project using simple Python APIs.

## Features

- Classifies the content of a single image into one or more labels
- Returns class names and confidence scores for the predicted labels
- Supports input as bytes, file paths or PIL images
- Configurable model parameters (e.g., image type, confidence threshold)

## Code example and usage

```python
from arduino.app_bricks.image_classification import ImageClassification

image_classification = ImageClassification()

# Image frame can be as bytes or PIL image
with open("path/to/your/image.jpg", "rb") as f:
    frame = f.read()

out = image_classification.classify(frame)
# is it possible to customize image type and confidence level
# out = image_classification.classify(frame, image_type = "png", confidence = 0.35)
if out and "classification" in out:
    for i, obj_det in enumerate(out["classification"]):
        # For every label predicted, get its details
        detected_object = obj_det.get("class_name", None)
        confidence = obj_det.get("confidence", None)
        print(f"Detected '{detected_object}' with confidence: {confidence}%")
```

You can also classify an image directly from a file path:

```python
out = image_classification.classify_from_file("path/to/your/image.jpg")
```

Note: the `confidence` value returned in the results is a string formatted as a percentage on a 0-100 scale (e.g. `"97.35"`), while the `confidence` parameter accepted by the constructor and by `classify()` is a float threshold between 0.0 and 1.0.

## Image Classification Working Principle

Image classification models take an input image and assign one or more class labels to it, representing the most likely categories present in the image. These models analyze the image as a whole and do not localize objects within the frame. The result is a ranked list of predicted labels, each accompanied by a confidence score indicating the model's likelihood of each label being correct.