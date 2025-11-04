# Microphone peripheral

The `Microphone` peripheral allows you to capture audio from audio devices.

## Usage

```python
from arduino.app_peripherals.microphone import Microphone

mic = Microphone(device=0, channels=1)
mic.start()

for chunk in mic.stream():  # Returns a numpy array iterator
    # ...

mic.stop()
```

## Parameters

- `device`: (optional) ALSA device index or name (default: 0)
- `rate`: (optional) sampling frequency (default: 16000 Hz)
- `channels`: (optional) channels (default: 1)
- `format`: (optional) ALSA audio format (default: 'S16_LE')
- `periodsize`: (optional) buffer chunk dymension (default: 1024)
