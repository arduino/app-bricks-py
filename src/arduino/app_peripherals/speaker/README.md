# Speaker peripheral

This brick allow you to play audio using Alsa Linux audio subsystem

## Usage

```python
from arduino.app_peripherals.speaker import Speaker

speak = Speaker(device='USB_SPEAKER_1')
speak.start()
# data is a byte array
speak.play(data)
speak.stop()
```

## Parameters

- `device`: (optional) ALSA device name (default: 'USB_MIC_1'. It can be the real ALSA device nome or USB_MIC_1, USB_MIC_2, ..)
- `rate`: (optional) sampling frequency (default: 16000 Hz)
- `channels`: (optional) channels (default: 1)
- `format`: (optional) ALSA audio format (default: 'S16_LE')
