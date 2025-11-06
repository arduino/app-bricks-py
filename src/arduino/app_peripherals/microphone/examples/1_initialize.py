# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Initialize microphone input"
# EXAMPLE_REQUIRES = "Requires a connected microphone"
from arduino.app_peripherals.microphone import Microphone
from arduino.app_peripherals.microphone.alsa_microphone import ALSAMicrophone
from arduino.app_peripherals.microphone.config import RATE_48K, STEREO, FORMAT_S24_LE, LOW_LATENCY_CHUNK


default = Microphone()  # Uses default microphone

# The following two are equivalent
mic = Microphone("ws://0.0.0.0:8080")  # Infers microphone type
alsa = ALSAMicrophone(RATE_48K, STEREO, FORMAT_S24_LE, LOW_LATENCY_CHUNK)  # Explicitly request ALSA microphone

# Note: Microphone's constructor arguments (except those in its signature)
# must be provided in keyword format to forward them correctly to the
# specific microphone implementations.
