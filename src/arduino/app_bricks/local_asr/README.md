# Local ASR Brick Overview

The `LocalASR` brick provides on-device automatic speech recognition (ASR) capabilities for audio streams and files. It offers a high-level interface for transcribing audio using a local model, with support for both real-time and batch processing.

## LocalASR Class Features

- **Offline Operation:** All transcriptions are performed locally, ensuring data privacy and eliminating network dependencies.
- **English Language Support:** Supports the transcription of spoken english.
- **Audio Input Formats**: Designed to work with the Microphone peripheral, WAV and PCM audio.
- **Concurrency Control**: Limits the number of simultaneous transcription sessions to avoid resource exhaustion.
