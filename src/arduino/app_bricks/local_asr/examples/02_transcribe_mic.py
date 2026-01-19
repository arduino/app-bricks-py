# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Transcribe audio from microphone"
# EXAMPLE_REQUIRES = "Requires a microphone device"
from arduino.app_bricks.local_asr import LocalASR

asr = LocalASR()
text = asr.transcribe_mic(duration=5)
print(f"Transcription: {text}")
