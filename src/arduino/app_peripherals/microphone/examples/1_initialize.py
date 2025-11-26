# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Initialize microphone"
# EXAMPLE_REQUIRES = "Requires a connected microphone"
from arduino.app_peripherals.microphone import Microphone, ALSAMicrophone, WebSocketMicrophone
from arduino.app_peripherals.microphone.config import RATE_48K, CHANNELS_STEREO, FORMAT_S24_LE, CHUNK_LOW_LATENCY


default = Microphone()  # Uses default microphone

custom = Microphone(
    device=0,  # The first ALSA device will be selected
    sample_rate=RATE_48K,
    channels=CHANNELS_STEREO,
    format=FORMAT_S24_LE,
    chunk_size=CHUNK_LOW_LATENCY,
)
# Note: Microphone's constructor arguments (except those in its signature)
# must be provided in keyword format to forward them correctly to the
# specific microphone implementations.

# The following two are equivalent
mic = Microphone(0)  # Infers microphone type
alsa = ALSAMicrophone()  # Explicitly request ALSA microphone

# The following two are equivalent
mic = Microphone("ws://0.0.0.0:8080")  # Infers microphone type
wsm = WebSocketMicrophone()  # Explicitly request WebSocket microphone
