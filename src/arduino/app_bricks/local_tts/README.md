# LocalTTS Module Overview

The `local_tts` module provides a completely offline text-to-speech (TTS) solution for Arduino Apps. At its core is the `LocalTTS` class, designed to convert text input into spoken audio using locally available TTS engines, ensuring privacy and low-latency performance without reliance on cloud services.

## Key Features

- **Offline Operation:** All speech synthesis is performed locally, ensuring data privacy and eliminating network dependencies.
- **Multiple Language Support:** Easily switch between different languages (en, es, zh).
- **Audio Output Formats:** Directly output synthesized speech to a Speaker instance or to WAV or PCM audio.
