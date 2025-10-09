# Microphone peripheral

This brick allow you to capture audio from Alsa Linux audio subsystem.

## Usage

```python
from arduino.app_peripherals.microphone import Microphone

mic = Microphone(device='USB_MIC_1', channels=1)
mic.start()
audio_chunk_iterator = mic.stream()  # Returns a numpy array iterator
for chunk in audio_chunk_iterator:
    # ...
mic.stop()
```

## Parameters

- `device`: (optional) ALSA device name (default: 'USB_MIC_1'. It can be the real ALSA device nome or USB_MIC_1, USB_MIC_2, ..)
- `rate`: (optional) sampling frequency (default: 16000 Hz)
- `channels`: (optional) channels (default: 1)
- `format`: (optional) ALSA audio format (default: 'S16_LE')
- `periodsize`: (optional) buffer chunk dymension (default: 1024)
