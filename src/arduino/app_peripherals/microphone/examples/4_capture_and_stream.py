# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Capture audio chunks from the microphone stream"
# EXAMPLE_REQUIRES = "Requires a connected microphone"
import time

import numpy as np
from arduino.app_peripherals.microphone import Microphone

mic = Microphone()
mic.start()

# Capture audio for 5 seconds
start_time = time.time()
while time.time() - start_time < 5:
    audio: np.ndarray = mic.capture()
    # You can process the audio here if needed, e.g save it

# Indefinitely produce audio chunks, call stop() or break to end
for chunk in mic.stream():
    print(f"Received audio chunk of size {len(chunk)}")
    mic.stop()
