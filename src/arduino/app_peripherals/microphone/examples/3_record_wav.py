# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Record a WAV audio to a file"
# EXAMPLE_REQUIRES = "Requires a connected microphone"
from pathlib import Path
from arduino.app_peripherals.microphone import Microphone


mic = Microphone()
mic.start()
wav_audio = mic.record_wav(5)  # Record 5 seconds of audio
out_file = Path("/recording.wav")
out_file.write_bytes(wav_audio.tobytes())
mic.stop()

# Otherwise, you can use contexts
with Microphone() as mic:
    wav_audio = mic.record_wav(5)
    out_file = Path("/recording.wav")
    out_file.write_bytes(wav_audio.tobytes())
