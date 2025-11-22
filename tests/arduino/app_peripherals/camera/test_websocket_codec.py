# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import base64
import json
from arduino.app_peripherals.camera.websocket_codec import BinaryCodec, JsonCodec

SECRET = "topsecret"
DATA = b"hello world"
DATA_STR = "hello world".encode()
UNICODE_STR = "héllo 🌍".encode()


def test_binary_codec_no_security():
    codec = BinaryCodec()

    # Test with binary input
    encoded = codec.encode(DATA)
    assert encoded == DATA
    decoded = codec.decode(encoded)
    assert decoded == DATA

    # Test with string input
    encoded_str = codec.encode(DATA_STR)
    assert encoded_str == DATA_STR
    decoded_str = codec.decode(encoded_str)
    assert decoded_str is not None
    assert decoded_str == DATA_STR

    # Test with unicode string input
    encoded_unicode = codec.encode(UNICODE_STR)
    decoded_unicode = codec.decode(encoded_unicode)
    assert decoded_unicode is not None
    assert decoded_unicode == UNICODE_STR


def test_binary_codec_authenticated():
    codec = BinaryCodec(secret=SECRET)

    # Test with binary input
    encoded = codec.encode(DATA)
    sig_len = encoded[0]
    assert sig_len > 0
    signature = encoded[1 : 1 + sig_len]
    assert len(signature) == sig_len
    timestamp = int.from_bytes(encoded[1 + sig_len : 1 + sig_len + 8], "big")
    assert timestamp > 0
    data = encoded[1 + sig_len + 8 :]
    assert data == DATA
    decoded = codec.decode(encoded)
    assert decoded == DATA

    # Test with string input
    encoded_str = codec.encode(DATA_STR)
    data_str = encoded_str[1 + encoded_str[0] + 8 :]
    assert data_str is not None
    assert data_str == DATA_STR
    decoded_str = codec.decode(encoded_str)
    assert decoded_str is not None
    assert decoded_str == DATA_STR

    # Test with unicode string input
    encoded_unicode = codec.encode(UNICODE_STR)
    data_unicode = encoded_unicode[1 + encoded_unicode[0] + 8 :]
    assert data_unicode == UNICODE_STR
    decoded_unicode = codec.decode(encoded_unicode)
    assert decoded_unicode is not None
    assert decoded_unicode == UNICODE_STR


def test_binary_codec_encrypted():
    codec = BinaryCodec(secret=SECRET, enable_encryption=True)

    # Test with binary input
    encoded = codec.encode(DATA)
    encrypted_len = int.from_bytes(encoded[0:4], "big")
    encrypted_data = encoded[12 : 12 + encrypted_len]
    assert len(encrypted_data) == encrypted_len
    assert encrypted_data != DATA
    timestamp = int.from_bytes(encoded[4:12], "big")
    assert timestamp > 0
    decoded = codec.decode(encoded)
    assert decoded == DATA

    # Test with string input
    encoded_str = codec.encode(DATA_STR)
    decoded_str = codec.decode(encoded_str)
    assert decoded_str is not None
    assert decoded_str == DATA_STR

    # Test with unicode string input
    encoded_unicode = codec.encode(UNICODE_STR)
    decoded_unicode = codec.decode(encoded_unicode)
    assert decoded_unicode is not None
    assert decoded_unicode == UNICODE_STR


def test_json_codec_no_security():
    codec = JsonCodec()

    # Test with binary input
    encoded = codec.encode(DATA)
    payload = json.loads(encoded)
    assert "data" in payload
    assert "timestamp" not in payload
    assert "signature" not in payload
    assert base64.b64decode(payload["data"]) == DATA
    decoded = codec.decode(encoded)
    assert decoded == DATA

    # Test with string input
    encoded_str = codec.encode(DATA_STR)
    payload_str = json.loads(encoded_str)
    assert base64.b64decode(payload_str["data"]) == DATA_STR
    decoded_str = codec.decode(encoded_str)
    assert decoded_str is not None
    assert decoded_str == DATA_STR

    # Test with unicode string input
    encoded_unicode = codec.encode(UNICODE_STR)
    payload_unicode = json.loads(encoded_unicode)
    assert base64.b64decode(payload_unicode["data"]) == UNICODE_STR
    decoded_unicode = codec.decode(encoded_unicode)
    assert decoded_unicode is not None
    assert decoded_unicode == UNICODE_STR


def test_json_codec_authenticated():
    codec = JsonCodec(secret=SECRET)

    # Test with binary input
    encoded = codec.encode(DATA)
    payload = json.loads(encoded)
    assert "data" in payload
    data = base64.b64decode(payload["data"])
    assert data == DATA
    assert "signature" in payload
    signature = base64.b64decode(payload["signature"])
    assert len(signature) > 0
    assert "timestamp" in payload
    timestamp = payload["timestamp"]
    assert type(timestamp) is int
    assert timestamp > 0
    decoded = codec.decode(encoded)
    assert decoded == DATA

    # Test with string input
    encoded_str = codec.encode(DATA_STR)
    payload_str = json.loads(encoded_str)
    assert base64.b64decode(payload_str["data"]) == DATA_STR
    decoded_str = codec.decode(encoded_str)
    assert decoded_str is not None
    assert decoded_str == DATA_STR

    # Test with unicode string
    encoded_unicode = codec.encode(UNICODE_STR)
    payload_unicode = json.loads(encoded_unicode)
    assert base64.b64decode(payload_unicode["data"]) == UNICODE_STR
    decoded_unicode = codec.decode(encoded_unicode)
    assert decoded_unicode is not None
    assert decoded_unicode == UNICODE_STR


def test_json_codec_encrypted():
    codec = JsonCodec(secret=SECRET, enable_encryption=True)

    # Test with binary input
    encoded = codec.encode(DATA)
    payload = json.loads(encoded)
    assert "data" in payload
    encrypted_data = base64.b64decode(payload["data"])
    assert encrypted_data != DATA
    assert "timestamp" in payload
    timestamp = payload["timestamp"]
    assert type(timestamp) is int
    assert timestamp > 0
    assert "signature" not in payload
    decoded = codec.decode(encoded)
    assert decoded == DATA

    # Test with string input
    encoded_str = codec.encode(DATA_STR)
    decoded_str = codec.decode(encoded_str)
    assert decoded_str is not None
    assert decoded_str == DATA_STR

    # Test with unicode string input
    encoded_unicode = codec.encode(UNICODE_STR)
    decoded_unicode = codec.decode(encoded_unicode)
    assert decoded_unicode is not None
    assert decoded_unicode == UNICODE_STR
