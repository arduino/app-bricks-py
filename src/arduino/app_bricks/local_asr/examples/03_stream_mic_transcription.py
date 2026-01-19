# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Transcribe audio from microphone and stream the results"
# EXAMPLE_REQUIRES = "Requires a microphone device"
from arduino.app_bricks.local_asr import LocalASR

asr = LocalASR()
with asr.transcribe_mic_stream(duration=5) as stream:
    for chunk in stream:
        match chunk.type:
            case "partial_text":
                print(f"Partial: {chunk.data}")
            case "full_text":
                print(f"Final: {chunk.data}")
                break
