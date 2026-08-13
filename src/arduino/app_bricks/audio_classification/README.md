# Audio Classification Brick

This Brick lets you perform audio classification using a pre-trained neural network model.

## Overview

The Audio Classification Brick allows you to:

- Analyze live audio from a microphone and detect specific sounds.
- Classify audio from existing .wav files.
- Register custom callbacks that trigger when a given class is detected.
- Easily integrate sound recognition into your project using simple Python APIs.

## Features

- Real-time audio classification from microphone input.
- Classifies sounds from .wav files of different bit depths (8, 16, 24, 32-bit).
- Configurable confidence threshold for detections.
- Callback support for specific class detections.
- Simple start/stop control for audio processing.

## Prerequisites

- USB-C hub (with USB A or 3.5 mm audio port)
- Analog 3.5 mm or USB microphone for real-time classification
- WAV audio files with supported bit depths: 8, 16, 24 or 32-bit

## Code example and usage

```python
from arduino.app_bricks.audio_classification import AudioClassification
from arduino.app_utils import App

classifier = AudioClassification()
classifier.on_detect("Glass_Breaking", lambda: print("Glass breaking sound detected!"))

App.run()
```

or using an existing audio file:

```python
from arduino.app_bricks.audio_classification import AudioClassification

classification = AudioClassification.classify_from_file("glass_breaking.wav")
print("Result:", classification)
```

The constructor accepts an optional microphone and a confidence threshold: `AudioClassification(mic=None, confidence=0.8)`. `classify_from_file(audio_path, confidence=0.8)` is a static method that returns a `{"class_name": ..., "confidence": ...}` dict, or `None` when nothing exceeds the threshold.

## Audio Classification Working Principle

Audio classification models take raw audio signals and extract numerical features representing the waveform. These features are processed by the model, which assigns one or more class labels to the input, representing the most likely sounds present.

When running in real time, the classifier continuously processes incoming audio data from the microphone, returning detected classes above a configurable confidence threshold. For offline usage, audio can be read from .wav files, decoded into features, and passed through the same classification pipeline.