# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_bricks.local_asr import LocalASR

asr = LocalASR()
with asr.transcribe_stream(duration=5) as stream:
    for chunk in stream:
        match chunk.type:
            case "partial_text":
                print(f"Partial: {chunk.data}")
            case "full_text":
                print(f"Final: {chunk.data}")
                break
