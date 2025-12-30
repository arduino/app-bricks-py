# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from unittest.mock import patch, MagicMock
from arduino.app_peripherals.speaker import Speaker, ALSASpeaker

from arduino.app_peripherals.speaker.errors import SpeakerConfigError


# Mock data for ALSA
MOCK_USB_S_CARDS = ["UH34", "OtherCard", "OtherCard2", "OtherCard3"]
MOCK_USB_S_CARD_INDEXES = [i for i in range(len(MOCK_USB_S_CARDS))]
MOCK_USB_S_CARD_DESCS = [
    ("UH34", "Audio Device"),
    ("OtherCard", "Other USB Device"),
    ("OtherCard2", "Other USB Device 2"),
    ("OtherCard3", "Other USB Device 3"),
]
MOCK_USB_S_PCM_DEVICES = [
    "plughw:CARD=UH34,DEV=0",
    "plughw:CARD=OtherCard,DEV=0",
    "plughw:CARD=OtherCard2,DEV=0",
    "plughw:CARD=OtherCard3,DEV=0",
]


@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.cards", return_value=MOCK_USB_S_CARDS)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.card_indexes", return_value=MOCK_USB_S_CARD_INDEXES)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.card_name", side_effect=lambda idx: MOCK_USB_S_CARD_DESCS[idx])
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.pcms", return_value=MOCK_USB_S_PCM_DEVICES)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM")
@patch("arduino.app_peripherals.speaker.alsa_speaker.Path.exists")
@patch(
    "arduino.app_peripherals.speaker.alsa_speaker.Path.resolve",
    return_value="/sys/devices/platform/soc@0/4ef8800.usb/4e00000.usb/xhci-hcd.2.auto/usb1/1-1/1-1.3/1-1.3:1.0/sound/card0/pcmC0D0c",
)
def test_list_usb_devices(mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
    """Test listing USB devices using alsaaudio mocks."""
    mock_exists.side_effect = [False, True, True, True]
    usb_devices = ALSASpeaker.list_usb_devices()
    assert usb_devices == [
        "plughw:CARD=OtherCard,DEV=0",
        "plughw:CARD=OtherCard2,DEV=0",
        "plughw:CARD=OtherCard3,DEV=0",
    ], "Should return only USB plughw devices"


@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.cards", return_value=MOCK_USB_S_CARDS)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.card_indexes", return_value=MOCK_USB_S_CARD_INDEXES)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.card_name", side_effect=lambda idx: MOCK_USB_S_CARD_DESCS[idx])
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.pcms", return_value=MOCK_USB_S_PCM_DEVICES)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM")
@patch("arduino.app_peripherals.speaker.alsa_speaker.Path.exists")
@patch(
    "arduino.app_peripherals.speaker.alsa_speaker.Path.resolve",
    return_value="/sys/devices/platform/soc@0/4ef8800.usb/4e00000.usb/xhci-hcd.2.auto/usb1/1-1/1-1.3/1-1.3:1.0/sound/card0/pcmC0D0c",
)
def test_speaker_init_usb_1(mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
    """Test initializing Speaker with one USB device."""
    mock_exists.side_effect = [False, True, True, True]
    pcm_instance = MagicMock()
    mock_pcm.return_value = pcm_instance
    spkr = Speaker(device=Speaker.USB_SPEAKER_1)
    assert spkr.device_stable_ref == "plughw:CARD=OtherCard,DEV=0"


@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.cards", return_value=MOCK_USB_S_CARDS)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.card_indexes", return_value=MOCK_USB_S_CARD_INDEXES)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.card_name", side_effect=lambda idx: MOCK_USB_S_CARD_DESCS[idx])
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.pcms", return_value=MOCK_USB_S_PCM_DEVICES)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM")
@patch("arduino.app_peripherals.speaker.alsa_speaker.Path.exists")
@patch(
    "arduino.app_peripherals.speaker.alsa_speaker.Path.resolve",
    return_value="/sys/devices/platform/soc@0/4ef8800.usb/4e00000.usb/xhci-hcd.2.auto/usb1/1-1/1-1.3/1-1.3:1.0/sound/card0/pcmC0D0c",
)
def test_speaker_init_usb_3(mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
    """Test initializing Speaker without using available macros."""
    mock_exists.side_effect = [False, True, True, True]
    pcm_instance = MagicMock()
    mock_pcm.return_value = pcm_instance
    spkr = Speaker(device="usb:3")
    assert spkr.device_stable_ref == "plughw:CARD=OtherCard3,DEV=0"


@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.cards", return_value=MOCK_USB_S_CARDS)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.card_indexes", return_value=MOCK_USB_S_CARD_INDEXES)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.card_name", side_effect=lambda idx: MOCK_USB_S_CARD_DESCS[idx])
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.pcms", return_value=MOCK_USB_S_PCM_DEVICES)
def test_speaker_init_usb_errors(mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
    """Test initializing Speaker without using available macros."""
    with pytest.raises(SpeakerConfigError):
        Speaker(device="usb:something")

    with pytest.raises(SpeakerConfigError):
        Speaker(device="usb:5")
