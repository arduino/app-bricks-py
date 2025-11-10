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

## Parameters

- `device`: (optional) ALSA device index or name or websocket address to expose to clients (default: 0)
- `rate`: (optional) sampling frequency (default: 16000 Hz)
- `channels`: (optional) number channels (default: 1)
- `format`: (optional) Aaudio format (default: 'S16_LE')
- `periodsize`: (optional) buffer chunk dymension (default: 1024)
