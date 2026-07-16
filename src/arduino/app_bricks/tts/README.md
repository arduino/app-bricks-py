# Text To Speech (TTS) Brick

The `TextToSpeech` brick provides a completely offline text-to-speech (TTS) solution for Arduino Apps. It's designed to convert text input into spoken audio using locally available TTS engines, ensuring privacy and low-latency performance without reliance on cloud services.

## Features

- **Offline Operation:** All text-to-speech functions are performed locally, ensuring data privacy and eliminating network dependencies.
- **Multiple Language Support:** Select a language by configuring the corresponding TTS model (e.g. `melo-tts-en`, `melo-tts-es`, `melo-tts-zh`, `piper-tts-de`, `piper-tts-en`, `piper-tts-it`) in `brick_config.yaml` or override per-app in `app.yaml`.
- **Audio Output Formats:** Directly output synthesized speech to a Speaker instance or to WAV, PCM, or PCM audio.
- **Long Text Support:** `speak()` splits long input into sentence-aware chunks of up to 1024 characters before synthesis.
- **Streaming Playback:** `speak()` plays PCM chunks as they arrive from the local TTS service instead of waiting for the full rendered response.
- **Cancellable Playback:** Use `cancel()` to stop the current spoken sequence and notify the local TTS service without stopping the TTS brick or speaker.
- **Single-Session Semantics:** Each instance handles one speech session at a time. For concurrent speech, create multiple `TextToSpeech` instances.

## Prerequisites

Before using the TTS Speak Through Speaker example, shown in the next section, ensure you have the following:

- USB-C® Hub with external power supply (5V, 3A)
- USB audio device (USB speaker or USB-C → 3.5mm adapter)
- Arduino VENTUNO Q running in Network Mode or SBC Mode (USB-C port needed for the hub)

## Code Example and Usage

This example shows how to convert text into spoken audio, which will be played on a speaker.

```python
from arduino.app_bricks.tts import TextToSpeech
from arduino.app_utils import App
import time


tts = TextToSpeech()


def runner():
    tts.speak("Hello world, Arduino!")
    time.sleep(5)  # Wait for the speech to finish before ending the app


App.run(user_loop=runner)
```

### Save as WAV

This example shows how to convert text into spoken audio and save it as a WAV file.

```python
from arduino.app_bricks.tts import TextToSpeech

tts = TextToSpeech()

wav = tts.synthesize_wav("Hello, Arduino world!")
with open("synthesized_speech.wav", "wb") as f:
    f.write(wav)
```

## Errors

- `TTSBusyError`: raised if you call `speak()`, `synthesize_pcm()`, `synthesize_pcm_stream()`, or `synthesize_wav()` while the instance already has an active session. Fix by awaiting the current session or using a separate instance.
- `TTSError`: base class for all of the above.
