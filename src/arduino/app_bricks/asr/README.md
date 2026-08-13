# Automatic Speech Recognition Brick

The ASR brick provides on-device automatic speech recognition (ASR) capabilities for audio streams and files. It offers a high-level interface for transcribing audio using a local model, with support for both real-time microphone capture (`AutomaticSpeechRecognition`) and in-memory audio (`WAVAutomaticSpeechRecognition`). With the possibility to use multiple languages.

## Features

- **Offline Operation:** All transcriptions are performed locally, ensuring data privacy and eliminating network dependencies.
- **Multi-Language Support:** Supports the transcription of multiple spoken languages. Language is auto-detected by default and can be overridden with the `language` constructor argument (e.g. `"en"`).
- **Flexible Audio Input:** `AutomaticSpeechRecognition` accepts a `BaseMicrophone` instance or `None` to use a default `Microphone()`. `WAVAutomaticSpeechRecognition` accepts a `bytes` WAV container or a raw `np.ndarray` of PCM samples (16 kHz mono).
- **Single-Session Semantics:** Each instance handles one transcription session at a time. For concurrent transcriptions on different microphones, create multiple `AutomaticSpeechRecognition` instances.

## Prerequisites

Before using the ASR brick, ensure you have the following components:

- USB microphone
OR
- WAV or PCM audio file

Tips:
- Use a USB-C® Hub with USB-A connectors to support commercial USB cameras with microphone. Note that the USB-C® Hub must have Power Delivery Support (PD).
- Microphones included in USB cameras/webcams are generally supported

## Code Example and Usage

This example transcribes audio captured from the microphone for 5 seconds at a time. The brick automatically uses the default microphone and handles its start and stop functions.

```python
from arduino.app_utils import App
from arduino.app_bricks.asr import AutomaticSpeechRecognition

asr = AutomaticSpeechRecognition()

print("Please start speaking for transcription...")


def transcribe():
    text = asr.transcribe(duration=5)
    print(f"Transcription: {text}")


App.run(user_loop=transcribe)
```

This example transcribes audio from a WAV file.

```python
from arduino.app_utils import App
from arduino.app_bricks.asr import WAVAutomaticSpeechRecognition

with open("recording_01.wav", "rb") as wav_file:
    audio_bytes = wav_file.read()
    asr = WAVAutomaticSpeechRecognition(audio_bytes)
    App.start_brick(asr)
    text = asr.transcribe()
    print(f"Transcription: {text}")

App.run()
```

## Methods

Both classes share the same transcription API (durations/timeouts apply to the microphone class; the WAV class always consumes the whole buffer):

- `transcribe(duration=60) -> str`: transcribes audio and returns the final text (`WAVAutomaticSpeechRecognition.transcribe()` takes no arguments).
- `transcribe_stream(duration=0) -> TranscriptionStream[ASREvent]`: yields intermediate transcription events; use it in a `with` block.
- `transcribe_sentence(timeout=0) -> str`: transcribes until the first complete sentence is detected.
- `transcribe_sentence_stream(timeout=0) -> TranscriptionStream[ASREvent]`: streams events until the first complete sentence.
- `transcribe_until_cancelled() -> TranscriptionStream[ASREvent]`: streams events until `cancel()` is called.
- `cancel()`: cancels the active transcription session, if any.
- `is_transcribing() -> bool`: returns whether a session is currently active.

## Errors

- `ASRBusyError`: raised if you call `transcribe()` / `transcribe_stream()` while the instance already has an active session. Fix by awaiting the current session or using a separate instance.
- `ASRServiceBusyError`: raised when the inference server rejects session creation because it is currently serving another client. The caller decides whether to retry.
- `ASRUnavailableError`: raised when the inference service is unreachable (container down, network error) or the WebSocket connection drops mid-session. The caller decides whether to retry.
- `ASRError`: base class for all of the above.

## Source Ownership

- When `mic` is `None`, `AutomaticSpeechRecognition` constructs a default `Microphone()` and manages its lifecycle through `asr.start()` / `asr.stop()` (called automatically by `App.run()`).
- When `mic` is a `BaseMicrophone` you pass in, **you** own its lifecycle — call `mic.start()` before transcribing and `mic.stop()` when done.
- In-memory sources (`bytes`, `np.ndarray` passed to `WAVAutomaticSpeechRecognition`) have no device lifecycle.
