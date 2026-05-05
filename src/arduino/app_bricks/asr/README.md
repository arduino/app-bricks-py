# Automatic Speech Recognition Brick

The `AutomaticSpeechRecognition` brick provides on-device automatic speech recognition (ASR) capabilities for audio streams and files. It offers a high-level interface for transcribing audio using a local model, with support for both real-time and batch processing.

## Overview

The ASR Brick allows you to:

- Stream audio from a microphone
- Transcribe WAV and PCM audio files
- Transcribe using a local model
- Use multiple languages

This Brick streams audio from a `Microphone` or `audio files` and gives you the transcribed text.

## Prerequisites

Before using the ASR brick, ensure you have the following components:

- USB microphone
OR
- WAV or PCM audio file

Tips:
- Use a USB-C® Hub with USB-A connectors to support commercial USB cameras with microphone. Note that the USB-C® Hub must have Power Delivery Support (PD).
- Microphones included in USB cameras/webcams are generally supported

## LocalASR Class Features

- All transcriptions are performed locally, ensuring data privacy and eliminating network dependencies.
- Supports the transcription of multiple spoken languages.
- Works with the Microphone peripheral as well as WAV and PCM audio files.
- Limits the number of simultaneous transcription sessions to avoid resource exhaustion.

## Code Example and Usage

This example transcribes audio captured from the microphone for 5 seconds.

```python
from arduino.app_bricks.asr import AutomaticSpeechRecognition
from arduino.app_peripherals.microphone import Microphone


mic = Microphone()
mic.start()

asr = AutomaticSpeechRecognition()
text = asr.transcribe_mic(mic, duration=5)
print(f"Transcription: {text}")

mic.stop()
```

This example transcribes audio from a file.

```python
from arduino.app_bricks.asr import AutomaticSpeechRecognition


asr = AutomaticSpeechRecognition()
with open("recording_01.wav", "rb") as wav_file:
    text = asr.transcribe_wav(wav_file.read())
    print(f"Transcription: {text}")
```

