# Microphone peripheral

The `Microphone` peripheral allows you to capture audio from audio devices.

## Usage

This will instantiate a Microphone that streams audio chunks read from a physically connected microphone.

```python
from arduino.app_peripherals.microphone import Microphone

mic = Microphone(device=0)
mic.start()

for chunk in mic.stream():  # Returns a numpy array iterator
    # ...

mic.stop()
```

You can also expose a WebSocket address to be used by clients to remotely stream PCM content:

```python
from arduino.app_peripherals.microphone import Microphone

mic = Microphone(device="ws://0.0.0.0:8080")
mic.start()

for chunk in mic.stream():  # Returns a numpy array iterator
    # ...

mic.stop()
```

# Note: clients of the WebSocket version are expected to respect the sample rate, channels, format, and chunk size specified during initialization.

## Detecting pauses

`PauseDetector` tells speech pauses from speech in the chunks you read, adapting to the
room noise, so no fixed threshold has to be tuned per microphone. Create one detector per
consumer of the audio and feed it every chunk; `paused` stays `True` for as long as the
pause lasts. `chunk_level(chunk)` returns the RMS level of a single chunk on a 0..1 scale,
whatever its sample format.

```python
from arduino.app_peripherals.microphone import Microphone, PauseDetector

mic = Microphone()
mic.start()
detector = PauseDetector(pause_s=0.25)

for chunk in mic.stream():
    detector.update(chunk, mic.sample_rate, mic.channels, mic.format_is_packed)
    if detector.paused:
        print(f"Pause of {detector.quiet_s:.2f} s")

mic.stop()
```

## Parameters

- `device`: (optional) microphone selector (default: 0). An integer index selects the n-th plugged microphone, giving priority to USB microphones and then jack microphones if supported by the platform. You can also pass an explicit ALSA device name/path, a `Microphone.USB_MIC_x`/`Microphone.JACK_MIC_x` shorthand, or a WebSocket address to expose to clients.
- `rate`: (optional) sampling frequency (default: 16000 Hz)
- `channels`: (optional) number channels (default: 1)
- `format`: (optional) Aaudio format (default: 'S16_LE')
- `periodsize`: (optional) buffer chunk dymension (default: 1024)
