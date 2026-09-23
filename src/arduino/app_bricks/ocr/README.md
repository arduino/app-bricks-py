# OCR Brick

This OCR brick extracts the text visible in an image using the EasyOCR model
accelerated on the board's NPU. Given an image it returns the recognized text
in reading order, along with the position and confidence of every detected
text region.

The API is a single blocking call:

```python
from arduino.app_bricks.ocr import OCR

ocr = OCR()
result = ocr.extract_text("assests/photo.jpg")
print(result.text)
```

`extract_text` accepts a numpy array in BGR channel order (as returned by
`Camera.capture()`), the raw bytes of an encoded image file (e.g. JPEG or PNG),
or a path to an image file. It returns an `OcrResult`:

- `result.text` holds every recognized string joined by newlines, in reading
  order (top to bottom, left to right); it is empty when no text was found.
  `str(result)` yields the same text.
- `result.detections` lists one `TextDetection` per piece of text found, in
  reading order, each carrying the recognized `text`, the recognition
  `confidence`, the axis-aligned `bounding_box_xyxy` box and the `polygon` of
  the detected region — its 4 (x, y) vertices ordered top-left, top-right,
  bottom-right, bottom-left. Polygon and bounding box coincide for horizontal
  text; when the text is slanted, the polygon is the exact (rotated) region
  while the bounding box is the straight rectangle enclosing it.

Reading the text seen by a camera:

```python
from arduino.app_bricks.ocr import OCR
from arduino.app_peripherals.camera import Camera

ocr = OCR()
camera = Camera()
camera.start()

frame = camera.capture()
if frame is not None:
    result = ocr.extract_text(frame)
    for detection in result.detections:
        print(f"{detection.text} ({detection.confidence:.2f}) at {detection.bounding_box_xyxy}")
```

Tuning:

- `confidence` (constructor) drops detections whose recognition confidence is
  below the threshold and rebuilds `result.text` from the kept ones. Default is
  0.3; pass 0.0 to report everything the model finds.
- `allowlist` (constructor) restricts recognition to the given characters, e.g.
  `"0123456789"` to read only digits from a meter or a serial number. It is
  applied by the model runner while decoding — the excluded characters cannot be
  emitted at all — so it improves accuracy on constrained text rather than just
  filtering the output.

- `rotation` (constructor, overridable per call) also reads detected pieces of
  text rotated by the given angles (any of 90, 180, 270) and keeps the most
  confident reading, for photos where the text does not run left to right:
  vertical labels, an upside-down tag. Text is always read upright too; 90 and 270
  are only tried on regions taller than wide (that is what vertical text looks
  like), 180 on every region, and a rotated reading replaces the upright one only
  when it is clearly more confident. Each applicable angle costs one more recognizer pass
  per region, so leave it off when the orientation is known. Pass `[]` in a call
  to read upright only for that image. Phone photos usually need none of this:
  their EXIF orientation is applied when the image is decoded.
- `single_line` (constructor, overridable per call) joins every recognized piece
  of text with single spaces instead of newlines, so `result.text` is one line.
  Useful when the image holds one logical string split across regions, e.g. a
  plate or a serial number. Default is `False`.

```python
from arduino.app_bricks.ocr import OCR

ocr = OCR(confidence=0.5, allowlist="0123456789.")
reading = ocr.extract_text("assets/meter.jpg")
print(reading.text)

sideways = ocr.extract_text("assets/page.jpg", rotation=[90, 270])
plate = ocr.extract_text("assets/plate.jpg", single_line=True)  # one line, no newlines
```

Image size: the model looks at the whole image scaled to 800x608, so a piece of
text has to be reasonably large in the frame to be found, roughly at least 1.5% of
the image height (a whole A4 page photographed from afar is beyond it: crop or get
closer). Sending more pixels does not change that, so the brick downscales images
larger than 2048 px on their longest side before sending them. Positions in the
result always refer to the image you passed in. A dense image with many pieces of
text takes longer: each detected region is one recognizer pass, times the number
of orientations.
