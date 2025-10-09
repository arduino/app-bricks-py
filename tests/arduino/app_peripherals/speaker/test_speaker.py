# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from unittest.mock import patch, MagicMock
from arduino.app_peripherals.speaker import Speaker, SpeakerException
from unittest.mock import MagicMock as MagicMockType

# Mock data for ALSA
MOCK_USB_S_CARDS = ["UH34", "OtherCard", "2OtherCard", "3OtherCard"]
MOCK_USB_S_CARD_INDEXES = [0, 1, 2, 3]
MOCK_USB_S_CARD_DESCS = [
    ("UH34", "Audio Device"),
    ("OtherCard", "Other USB Device"),
    ("2OtherCard", "Other USB Device 2"),
    ("3OtherCard", "Other USB Device 3"),
]
MOCK_USB_S_PCM_DEVICES = [
    "plughw:CARD=UH34,DEV=0",
    "plughw:CARD=OtherCard,DEV=0",
    "plughw:CARD=OtherCard2,DEV=0",
    "plughw:CARD=OtherCard3,DEV=0",
]


@patch("alsaaudio.cards", return_value=MOCK_USB_S_CARDS)
@patch("alsaaudio.card_indexes", return_value=MOCK_USB_S_CARD_INDEXES)
@patch("alsaaudio.card_name", side_effect=lambda idx: MOCK_USB_S_CARD_DESCS[idx])
@patch("alsaaudio.pcms", return_value=MOCK_USB_S_PCM_DEVICES)
@patch("alsaaudio.PCM")
def test_list_usb_devices(
    mock_pcm: MagicMockType,
    mock_pcms: MagicMockType,
    mock_card_name: MagicMockType,
    mock_card_indexes: MagicMockType,
    mock_cards: MagicMockType,
) -> None:
    """Test listing USB devices using alsaaudio mocks."""
    usb_devices = Speaker.list_usb_devices()
    assert usb_devices == [
        "plughw:CARD=OtherCard,DEV=0",
        "plughw:CARD=OtherCard2,DEV=0",
        "plughw:CARD=OtherCard3,DEV=0",
    ], "Should return only USB plughw devices"


@patch("alsaaudio.cards", return_value=MOCK_USB_S_CARDS)
@patch("alsaaudio.card_indexes", return_value=MOCK_USB_S_CARD_INDEXES)
@patch("alsaaudio.card_name", side_effect=lambda idx: MOCK_USB_S_CARD_DESCS[idx])
@patch("alsaaudio.pcms", return_value=MOCK_USB_S_PCM_DEVICES)
@patch("alsaaudio.PCM")
def test_microphone_init_usb_1(
    mock_pcm: MagicMockType,
    mock_pcms: MagicMockType,
    mock_card_name: MagicMockType,
    mock_card_indexes: MagicMockType,
    mock_cards: MagicMockType,
) -> None:
    """Test initializing Speaker with one USB device."""
    # PCM mock instance
    pcm_instance = MagicMock()
    mock_pcm.return_value = pcm_instance
    mic = Speaker(device=Speaker.USB_SPEAKER_1)
    assert mic.device == "plughw:CARD=OtherCard,DEV=0"


@patch("alsaaudio.cards", return_value=MOCK_USB_S_CARDS)
@patch("alsaaudio.card_indexes", return_value=MOCK_USB_S_CARD_INDEXES)
@patch("alsaaudio.card_name", side_effect=lambda idx: MOCK_USB_S_CARD_DESCS[idx])
@patch("alsaaudio.pcms", return_value=MOCK_USB_S_PCM_DEVICES)
@patch("alsaaudio.PCM")
def test_microphone_init_usb_5_error(
    mock_pcm: MagicMockType,
    mock_pcms: MagicMockType,
    mock_card_name: MagicMockType,
    mock_card_indexes: MagicMockType,
    mock_cards: MagicMockType,
) -> None:
    """Test initializing Speaker with USB_SPEAKER_5 when only one USB device is available."""
    # Only one USB device, so USB_SPEAKER_2 should raise
    with pytest.raises(SpeakerException):
        Speaker(device="USB_SPEAKER_5")


@patch("alsaaudio.cards", return_value=MOCK_USB_S_CARDS)
@patch("alsaaudio.card_indexes", return_value=MOCK_USB_S_CARD_INDEXES)
@patch("alsaaudio.card_name", side_effect=lambda idx: MOCK_USB_S_CARD_DESCS[idx])
@patch("alsaaudio.pcms", return_value=MOCK_USB_S_PCM_DEVICES)
@patch("alsaaudio.PCM")
def test_microphone_init_usb_3(
    mock_pcm: MagicMockType,
    mock_pcms: MagicMockType,
    mock_card_name: MagicMockType,
    mock_card_indexes: MagicMockType,
    mock_cards: MagicMockType,
) -> None:
    """Test initializing Speaker with USB_SPEAKER_3 when only one USB device is available."""
    pcm_instance = MagicMock()
    mock_pcm.return_value = pcm_instance
    mic = Speaker(device="USB_SPEAKER_3")
    assert mic.device == "plughw:CARD=OtherCard3,DEV=0"
