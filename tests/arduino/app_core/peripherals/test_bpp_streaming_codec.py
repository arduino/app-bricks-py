# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import struct
from arduino.app_internal.core.peripherals import BPPCodec, BPPStreamCodec

SECRET = "stream_secret"
PAYLOAD = b"some_data"


@pytest.fixture
def stream_wrapper():
    codec = BPPCodec(secret=SECRET, enable_encryption=True)
    return BPPStreamCodec(codec)


def test_stream_simple_encode_decode(stream_wrapper):
    """Test standard flow: encode -> decode"""
    # Encode (Magic + Len + Checksum + BPP)
    wire_data = stream_wrapper.encode(PAYLOAD)

    assert wire_data[0] == 0xAA
    length = struct.unpack(">I", wire_data[1:5])[0]
    # Header is 6 bytes
    assert len(wire_data) == 6 + length

    result = next(stream_wrapper.decode(wire_data))
    assert result == PAYLOAD


def test_stream_fragmentation(stream_wrapper):
    """Test receiving data byte-by-byte (BLE MTU fragmentation simulation)"""
    wire_data = stream_wrapper.encode(PAYLOAD)
    results = []

    # Feed one byte at a time using decode()
    for byte in wire_data:
        chunk = bytes([byte])
        for msg in stream_wrapper.decode(chunk):
            if msg:
                results.append(msg)

    assert len(results) == 1
    assert results[0] == PAYLOAD


def test_stream_concatenation(stream_wrapper):
    """Test receiving multiple packets in a single chunk (UART buffering)"""
    packet1 = stream_wrapper.encode(b"packet_one")
    packet2 = stream_wrapper.encode(b"packet_two")

    big_chunk = packet1 + packet2

    idx = 0
    for msg in stream_wrapper.decode(big_chunk):
        assert msg is not None
        if idx == 0:
            assert msg == b"packet_one"
        elif idx == 1:
            assert msg == b"packet_two"
        else:
            pytest.fail("Unexpected extra message")
        idx += 1


def test_stream_garbage_resync(stream_wrapper):
    """Test recovery from garbage data before a valid packet"""
    valid_packet = stream_wrapper.encode(PAYLOAD)

    # Inject garbage bytes before valid packet
    garbage = b"\x00\x11\x22" + valid_packet

    n_msgs = 0
    for msg in stream_wrapper.decode(garbage):
        # The garbage data should have been skipped
        assert msg is not None
        assert msg == PAYLOAD
        n_msgs += 1

    assert n_msgs == 1


def test_stream_garbage_packet_resync(stream_wrapper):
    """Test recovery when garbage contains an invalid packet"""
    valid_packet = stream_wrapper.encode(PAYLOAD)

    # Inject garbage packet before valid packet
    garbage_packet = b"\xaa\x00\x00" + b"\xff" * 5
    stream = garbage_packet + valid_packet

    n_msgs = 0
    for msg in stream_wrapper.decode(stream):
        # The garbage packet should have been skipped
        assert msg is not None
        assert msg == PAYLOAD
        n_msgs += 1

    assert n_msgs == 1


def test_stream_invalid_checksum_resync(stream_wrapper):
    """Test recovery when garbage contains the magic byte but invalid checksum"""
    valid_packet = stream_wrapper.encode(PAYLOAD)

    # Inject invalid packet before valid packet
    # 0xAA ^ 0 ^ 0 ^ 0 ^ 5 = 0xAF, but we put 0xFF as checksum to force mismatch.
    invalid_packet = b"\xaa\x00\x00\x00\x05\xff" + b"\xff" * 5
    stream = invalid_packet + valid_packet

    n_msgs = 0
    for msg in stream_wrapper.decode(stream):
        # The invalid packet should have been skipped
        assert msg is not None
        assert msg == PAYLOAD
        n_msgs += 1

    assert n_msgs == 1
