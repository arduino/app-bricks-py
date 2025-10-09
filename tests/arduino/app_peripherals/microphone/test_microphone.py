# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from unittest.mock import patch, MagicMock
from arduino.app_peripherals.microphone import Microphone, MicrophoneException
import numpy as np
from unittest.mock import MagicMock as MagicMockType
from typing import Any, Callable

# Mock data for ALSA
MOCK_USB_CARDS = ["UH34", "OtherCard"]
MOCK_USB_CARD_INDEXES = [0, 1]
MOCK_USB_CARD_DESCS = [("UH34", "USB Audio Device"), ("OtherCard", "Other Device")]
MOCK_USB_PCM_DEVICES = [
    "plughw:CARD=UH34,DEV=0",
    "plughw:CARD=OtherCard,DEV=0",
]


@patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
@patch("alsaaudio.card_indexes", return_value=MOCK_USB_CARD_INDEXES)
@patch("alsaaudio.card_name", side_effect=lambda idx: MOCK_USB_CARD_DESCS[idx])
@patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
@patch("alsaaudio.PCM")
def test_list_usb_devices(
    mock_pcm: MagicMockType,
    mock_pcms: MagicMockType,
    mock_card_name: MagicMockType,
    mock_card_indexes: MagicMockType,
    mock_cards: MagicMockType,
) -> None:
    """Test listing USB devices using alsaaudio mocks."""
    usb_devices = Microphone.list_usb_devices()
    assert usb_devices == ["plughw:CARD=UH34,DEV=0"], "Should return only USB plughw devices"


@patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
@patch("alsaaudio.card_indexes", return_value=MOCK_USB_CARD_INDEXES)
@patch("alsaaudio.card_name", side_effect=lambda idx: MOCK_USB_CARD_DESCS[idx])
@patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
@patch("alsaaudio.PCM")
def test_microphone_init_usb_1(
    mock_pcm: MagicMockType,
    mock_pcms: MagicMockType,
    mock_card_name: MagicMockType,
    mock_card_indexes: MagicMockType,
    mock_cards: MagicMockType,
) -> None:
    """Test initializing Microphone with one USB device."""
    # PCM mock instance
    pcm_instance = MagicMock()
    mock_pcm.return_value = pcm_instance
    mic = Microphone(device=Microphone.USB_MIC_1)
    assert mic.device == "plughw:CARD=UH34,DEV=0"


@patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
@patch("alsaaudio.card_indexes", return_value=MOCK_USB_CARD_INDEXES)
@patch("alsaaudio.card_name", side_effect=lambda idx: MOCK_USB_CARD_DESCS[idx])
@patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
@patch("alsaaudio.PCM")
def test_microphone_init_usb_2_error(
    mock_pcm: MagicMockType,
    mock_pcms: MagicMockType,
    mock_card_name: MagicMockType,
    mock_card_indexes: MagicMockType,
    mock_cards: MagicMockType,
) -> None:
    """Test initializing Microphone with USB_MIC_2 when only one USB device is available."""
    # Only one USB device, so USB_MIC_2 should raise
    with pytest.raises(MicrophoneException):
        Microphone(device=Microphone.USB_MIC_2)


@patch("alsaaudio.cards", return_value=[])
@patch("alsaaudio.card_indexes", return_value=[])
@patch("alsaaudio.pcms", return_value=[])
@patch("alsaaudio.PCM")
def test_microphone_no_usb_found(
    mock_pcm: MagicMockType,
    mock_pcms: MagicMockType,
    mock_card_indexes: MagicMockType,
    mock_cards: MagicMockType,
) -> None:
    """Test initializing Microphone when no USB devices are found."""
    with pytest.raises(MicrophoneException):
        Microphone(device=Microphone.USB_MIC_1)


def _mock_pcm_read_factory(dtype: Any, n: int = 8) -> Callable[[], tuple[int, bytes]]:
    # Return n samples of the correct dtype as bytes
    arr = np.arange(n, dtype=dtype)
    return lambda: (n, arr.tobytes())


@pytest.mark.parametrize("fmt, expected_dtype", [(fmt, dtype) for fmt, (_, dtype) in Microphone.FORMAT_MAP.items() if dtype is not None])
@patch("alsaaudio.cards", return_value=["USB"])
@patch("alsaaudio.card_indexes", return_value=[0])
@patch("alsaaudio.card_name", return_value=("USB", "USB Audio Device"))
@patch("alsaaudio.pcms", return_value=["plughw:CARD=USB,DEV=0"])
@patch("alsaaudio.PCM")
def test_microphone_stream_supported_formats(
    mock_pcm: MagicMockType,
    mock_pcms: MagicMockType,
    mock_card_name: MagicMockType,
    mock_card_indexes: MagicMockType,
    mock_cards: MagicMockType,
    fmt: str,
    expected_dtype: Any,
) -> None:
    """Test Microphone stream with supported formats."""
    pcm_instance = MagicMock()
    mock_pcm.return_value = pcm_instance
    pcm_instance.read.side_effect = _mock_pcm_read_factory(expected_dtype)
    mic = Microphone(device=Microphone.USB_MIC_1, format=fmt)
    mic.start()
    stream = mic.stream()
    arr = next(stream)
    assert arr.dtype == np.dtype(expected_dtype)
    assert arr.shape[0] == 8
    mic.stop()


@pytest.mark.parametrize("fmt", [fmt for fmt, (_, dtype) in Microphone.FORMAT_MAP.items() if dtype is None])
@patch("alsaaudio.cards", return_value=["USB"])
@patch("alsaaudio.card_indexes", return_value=[0])
@patch("alsaaudio.card_name", return_value=("USB", "USB Audio Device"))
@patch("alsaaudio.pcms", return_value=["plughw:CARD=USB,DEV=0"])
@patch("alsaaudio.PCM")
def test_microphone_stream_unsupported_formats(
    mock_pcm: MagicMockType,
    mock_pcms: MagicMockType,
    mock_card_name: MagicMockType,
    mock_card_indexes: MagicMockType,
    mock_cards: MagicMockType,
    fmt: str,
) -> None:
    """Test Microphone with unsupported formats: should raise NotImplementedError at instantiation."""
    with pytest.raises(NotImplementedError):
        Microphone(device=Microphone.USB_MIC_1, format=fmt)


# Test context manager usage
@patch("alsaaudio.cards", return_value=["USB"])
@patch("alsaaudio.card_indexes", return_value=[0])
@patch("alsaaudio.card_name", return_value=("USB", "USB Audio Device"))
@patch("alsaaudio.pcms", return_value=["plughw:CARD=USB,DEV=0"])
@patch("alsaaudio.PCM")
def test_microphone_context_manager(
    mock_pcm: MagicMockType,
    mock_pcms: MagicMockType,
    mock_card_name: MagicMockType,
    mock_card_indexes: MagicMockType,
    mock_cards: MagicMockType,
) -> None:
    """Test Microphone context manager start/stop and event handling."""
    pcm_instance = MagicMock()
    mock_pcm.return_value = pcm_instance
    with Microphone(device=Microphone.USB_MIC_1) as mic:
        assert mic.is_recording.is_set()
        stream = mic.stream()
        # Simula una lettura
        pcm_instance.read.return_value = (8, b"\x00" * 16)
        next(stream)
    # Dopo il context manager, l'evento deve essere cleared
    assert not mic.is_recording.is_set()


@patch("alsaaudio.cards", return_value=["USB"])
@patch("alsaaudio.card_indexes", return_value=[0])
@patch("alsaaudio.card_name", return_value=("USB", "USB Audio Device"))
@patch("alsaaudio.pcms", return_value=["plughw:CARD=USB,DEV=0"])
@patch("alsaaudio.PCM")
def test_microphone_stop_without_start(
    mock_pcm: MagicMockType,
    mock_pcms: MagicMockType,
    mock_card_name: MagicMockType,
    mock_card_indexes: MagicMockType,
    mock_cards: MagicMockType,
) -> None:
    """Test that calling stop() without start() does not raise."""
    mic = Microphone(device=Microphone.USB_MIC_1)
    mic.stop()  # Should not raise
