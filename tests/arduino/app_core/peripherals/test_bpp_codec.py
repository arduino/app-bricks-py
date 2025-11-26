# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import time
from arduino.app_internal.core.peripherals.bpp_codec import BPPCodec

SECRET = "super_secret_key_12345"
PAYLOAD = b'{"command": "turn_on", "value": 1}'


# FIXTURES
@pytest.fixture
def codec_none():
    """Codec configured for No Security"""
    return BPPCodec()


@pytest.fixture
def codec_sign():
    """Codec configured for HMAC Signing (Mode 1)"""
    return BPPCodec(secret=SECRET, enable_encryption=False)


@pytest.fixture
def codec_enc():
    """Codec configured for Encryption (Mode 2)"""
    return BPPCodec(secret=SECRET, enable_encryption=True)


# HAPPY PATH TESTS
def test_mode_none_success(codec_none):
    """Test standard encoding/decoding for Mode 0"""
    encoded = codec_none.encode(PAYLOAD)
    decoded = codec_none.decode(encoded)

    assert decoded == PAYLOAD
    assert encoded[0] == 0x00  # V0
    assert encoded[1] == 0x00  # Mode 0


def test_mode_sign_success(codec_sign):
    """Test standard encoding/decoding for Mode 1"""
    encoded = codec_sign.encode(PAYLOAD)
    decoded = codec_sign.decode(encoded)

    assert decoded == PAYLOAD
    assert encoded[1] == 0x01  # Mode 1
    assert len(encoded) == 14 + len(PAYLOAD) + 32  # Header(14) + Payload + Signature(32)


def test_mode_enc_success(codec_enc):
    """Test standard encoding/decoding for Mode 2"""
    encoded = codec_enc.encode(PAYLOAD)
    decoded = codec_enc.decode(encoded)

    assert decoded == PAYLOAD
    assert encoded[1] == 0x02  # Mode 2
    assert PAYLOAD not in encoded[14:]  # Skip header


# SECURITY ENFORCEMENT (DOWNGRADE ATTACKS)
def test_downgrade_prevention_sign_rejects_none(codec_sign, codec_none):
    """
    Scenario: Server expects Signature. Attacker sends Mode 0 (None).
    Result: Should return None.
    """
    attack_packet = codec_none.encode(PAYLOAD)
    decoded = codec_sign.decode(attack_packet)

    assert decoded is None  # Rejected


def test_downgrade_prevention_enc_rejects_sign(codec_enc, codec_sign):
    """
    Scenario: Server expects Encryption. Attacker sends Mode 1 (Signed).
    Result: Should return None.
    """
    attack_packet = codec_sign.encode(PAYLOAD)
    decoded = codec_enc.decode(attack_packet)

    assert decoded is None  # Rejected


def test_downgrade_prevention_enc_rejects_none(codec_enc, codec_none):
    """
    Scenario: Server expects Encryption. Attacker sends Mode 0 (None).
    Result: Should return None (rejected).
    """
    # Attacker creates a valid Mode 0 packet
    attack_packet = codec_none.encode(PAYLOAD)
    # Victim tries to decode
    decoded = codec_enc.decode(attack_packet)

    assert decoded is None  # Rejected


# REPLAY PROTECTION & EXPIRATION
def test_replay_attack_prevention(codec_enc):
    """
    Scenario: Valid packet sent twice.
    Result: First succeeds, second fails.
    """
    packet = codec_enc.encode(PAYLOAD)

    # First decode: succeeds
    assert codec_enc.decode(packet) == PAYLOAD
    # Second decode (replay): fails
    assert codec_enc.decode(packet) is None


def test_packet_expiration(codec_enc, monkeypatch):
    """
    Scenario: Packet is 11 seconds old.
    Result: Should fail time window check.
    """
    packet = codec_enc.encode(PAYLOAD)

    # Fast forward time by 11 seconds
    original_ns = time.time_ns
    monkeypatch.setattr(time, "time_ns", lambda: original_ns() + (11_000_000 * 1000))

    assert codec_enc.decode(packet) is None  # Rejected due to expiration


# INTEGRITY (TAMPERING)
def test_tampered_signature(codec_sign):
    """
    Scenario: Mode 1 packet, change one byte of the signature.
    Result: HMAC verification should fail.
    """
    packet = bytearray(codec_sign.encode(PAYLOAD))
    packet[-1] ^= 0xFF  # Flip signature's last byte to corrupt it

    decoded = codec_sign.decode(bytes(packet))
    assert decoded is None  # Rejected


def test_tampered_ciphertext(codec_enc):
    """
    Scenario: Mode 2 packet, change one byte of the ciphertext.
    Result: Poly1305 verification should fail.
    """
    packet = bytearray(codec_enc.encode(PAYLOAD))
    packet[14] ^= 0xFF  # Flip ciphertext's first byte to corrupt it

    decoded = codec_enc.decode(bytes(packet))
    assert decoded is None  # Rejected


def test_tampered_header_aad(codec_enc):
    """
    Scenario: Mode 2 packet. Attacker tries to modify the Timestamp in the header
    to bypass replay protection, but keeps the original Tag.
    Result: Poly1305 verification should fail because Header is AAD.
    """
    packet = bytearray(codec_enc.encode(PAYLOAD))
    # Note: This is a bit manual, assuming big-endian layout
    packet[9] += 1  # Increment LSB of Timestamp

    decoded = codec_enc.decode(bytes(packet))
    assert decoded is None  # Rejected


# TEXT WRAPPER TESTS
def test_text_wrapper_workflow(codec_enc):
    """Verify string encoding works"""
    token = codec_enc.encode_text(PAYLOAD)
    decoded = codec_enc.decode_text(token)

    assert decoded == PAYLOAD


def test_text_wrapper_tampering(codec_enc):
    """Verify tampering with the base64 string fails"""
    b64_text = codec_enc.encode_text(PAYLOAD)
    # Tamper with the base64 string (change last char)
    tampered_token = b64_text[:-1] + ("A" if b64_text[-1] != "A" else "B")

    decoded = codec_enc.decode_text(tampered_token)
    assert decoded is None  # Rejected
