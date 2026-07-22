# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Initialize microphone"
# EXAMPLE_REQUIRES = "Requires a connected microphone"
import numpy as np

from arduino.app_peripherals.microphone import Microphone, ALSAMicrophone, WebSocketMicrophone


default = Microphone()  # Uses default microphone

custom = Microphone(
    device=Microphone.USB_MIC_1,  # The first USB microphone will be selected
    sample_rate=Microphone.RATE_48K,
    channels=Microphone.CHANNELS_STEREO,
    format=np.int32,
    buffer_size=Microphone.BUFFER_SIZE_REALTIME,
)
# Note: Microphone's constructor arguments other than those in its signature
# must be provided in keyword format to forward them correctly to the
# specific microphone implementations.

# The following two are equivalent
mic = Microphone(0)  # Infers microphone type
alsa = ALSAMicrophone()  # Explicitly request ALSA microphone

# The following two are equivalent
mic = Microphone("ws://0.0.0.0:8080")  # Infers microphone type
wsm = WebSocketMicrophone()  # Explicitly request WebSocket microphone
