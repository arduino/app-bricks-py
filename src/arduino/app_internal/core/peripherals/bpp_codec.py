# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import base64
import time
import hashlib
import hmac
import secrets
import struct
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.exceptions import InvalidTag

from arduino.app_utils.logger import Logger

logger = Logger("BPPCodec")

BPP_VERSION = 0x00
MODE_NONE = 0x00
MODE_SIGN = 0x01
MODE_ENC = 0x02

# Big-endian header: [Version:1] [Mode:1] [Timestamp:8] [Random:4]
HEADER_FORMAT = ">BBQL"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 14 bytes
WINDOW_US = 10_000_000  # 10s in µs


class ReplayProtection:
    """
    Manages the sliding window replay protection and the temporary cache storing
    the IVs already seen within the validity window.
    """

    def __init__(self, window_us: int = WINDOW_US):
        self.window_us = window_us
        self.cache: dict[bytes, int] = {}  # IV -> Expiration timestamp

    def check_and_update(self, iv: bytes, timestamp_us: int) -> bool:
        """
        Determines if the message is valid by assessing replay attack conditions:
        timestamp out of validity window and IV reuse.
        """
        now = time.time_ns() // 1_000

        # Check time window
        if abs(now - timestamp_us) > self.window_us:
            logger.warning(f"Message outside validity window. Drift: {(now - timestamp_us) / 1000}ms")
            return False

        # Check IV reuse
        if iv in self.cache:
            logger.warning("IV reuse detected")
            return False

        # Prune old entries if cache grows too large
        if len(self.cache) > 1000:
            self._prune(now)

        self.cache[iv] = now + self.window_us

        return True

    def _prune(self, now: int):
        # Remove expired entries
        expired_keys = [k for k, v in self.cache.items() if now > v]
        for k in expired_keys:
            del self.cache[k]


class BPPCodec:
    """
    Binary Peripheral Protocol (BPP) Codec.
    Implements a secure container format for peripherals and allows to encode and
    decode payloads.
    This codec is intended to be used with message-based protocols, i.e. with builtin
    message boundaries (e.g., WebSocket). If used with stream-based protocols (e.g.,
    TCP, BLE, UART), it must be wrapped in BPPStreamCodec.

    The protocol supports three security modes:
    - Mode 0: No Security;
    - Mode 1: HMAC-SHA256 Signing, useful for authentication and data integrity;
    - Mode 2: ChaCha20-Poly1305 Encryption and Signing, providing confidentiality,
        authentication and data integrity.

    The binary format is as follows:

    [Version (1)] [Mode (1)] [Timestamp (8)] [Random (4)] [Payload (Var)] [AuthTag/Sig (16/32)]

    - Version: Protocol version (currently 0x01).
    - Mode: Security mode (0x00: None, 0x01: HMAC-SHA256, 0x02: ChaCha20-Poly1305).
    - Timestamp: Microsecond-precision timestamp (Unix epoch).
    - Random: 32-bit random value for uniqueness.
    - Payload: Actual data being transmitted.
    - AuthTag/Sig: HMAC signature (32 bytes for Mode 1) or AuthTag (16 bytes for Mode 2).

    Text-safe encoding/decoding via Base64URL are also provided.
    """

    def __init__(self, secret: str = "", enable_encryption: bool = False):
        """
        Initialize codec.

        Args:
            secret: Pre-shared secret. Default: empty (no security).
            enable_encryption: If True, uses ChaCha20-Poly1305. If False, uses HMAC-SHA256 if
                secret is provided. Default: False.
        """
        self.secret = secret.encode() if secret else b""
        self.enable_encryption = enable_encryption and bool(secret)
        self.cc_cipher = None

        if self.enable_encryption:
            # Derive 32-byte key for ChaCha20
            key = hashlib.sha256(self.secret).digest()
            self.cc_cipher = ChaCha20Poly1305(key)

        self.replay_protection = ReplayProtection()

    def encode(self, data: bytes) -> bytes:
        """
        Packs data into a BPP message and returns its bytes.

        Args:
            data: The payload to encode.
        Returns:
            The complete BPP message (bytes).
        """
        # Assemble the header
        mode = MODE_ENC if self.enable_encryption else (MODE_SIGN if self.secret else MODE_NONE)
        timestamp_us = time.time_ns() // 1_000
        random_val = secrets.randbits(32)
        header = struct.pack(HEADER_FORMAT, BPP_VERSION, mode, timestamp_us, random_val)

        if mode == MODE_ENC and self.cc_cipher:
            # Encrypt with ChaCha20-Poly1305, use header as AAD
            # Note: cryptography lib appends the 16-byte Poly1305 AuthTag automatically
            iv = header[2:]  # Last 12 bytes of header (Timestamp + Random)
            encrypted_payload = self.cc_cipher.encrypt(iv, data, header)
            return header + encrypted_payload

        elif mode == MODE_SIGN and self.secret:
            # HMAC Signature
            msg_to_sign = header + data
            signature = hmac.new(self.secret, msg_to_sign, hashlib.sha256).digest()
            return header + data + signature

        else:
            # No Security
            return header + data

    def decode(self, message: bytes) -> bytes | None:
        """
        Unpacks a BPP message and returns its payload.

        Args:
            message: The complete BPP message to decode.
        Returns:
            The decoded payload (bytes) if valid, else None.
        """
        if len(message) < HEADER_SIZE:
            logger.warning("Message too short for header")
            return None

        try:
            ver, mode, timestamp_us, random_val = struct.unpack(HEADER_FORMAT, message[:HEADER_SIZE])
        except struct.error:
            logger.warning("Header parsing failed")
            return None

        if ver != BPP_VERSION:
            logger.warning(f"Unsupported version {ver}")
            return None

        # Check expected minimum size
        footer_size = 16 if mode == MODE_ENC else (32 if mode == MODE_SIGN else 0)
        min_size = HEADER_SIZE + footer_size
        if len(message) < min_size:
            logger.warning("Message too short (truncated)")
            return None

        # Check for downgrade attacks
        if self.enable_encryption:
            if mode != MODE_ENC:
                logger.warning(f"Security mode mismatch: expected Mode 2 (encrypt), but received Mode {mode}.")
                return None
        elif self.secret:
            if mode != MODE_SIGN:
                logger.warning(f"Security mode mismatch: expected Mode 1 (sign), but received Mode {mode}.")
                return None
        else:
            if mode != MODE_NONE:
                logger.warning(f"Security mode mismatch: expected Mode 0 (none), but received Mode {mode}.")
                return None

        # Check for replay attacks
        replay_id = message[2:HEADER_SIZE]  # Timestamp (8) + Random (4)
        if not self.replay_protection.check_and_update(replay_id, timestamp_us):
            return None

        header_bytes = message[:HEADER_SIZE]

        # Decrypt/verify
        try:
            if mode == MODE_ENC:
                iv = replay_id
                ciphertext_with_tag = message[HEADER_SIZE:]
                return self.cc_cipher.decrypt(iv, ciphertext_with_tag, header_bytes)

            elif mode == MODE_SIGN:
                if len(message) < HEADER_SIZE + 32:
                    return None

                payload = message[HEADER_SIZE:-32]
                received_sig = message[-32:]

                msg_to_verify = header_bytes + payload
                expected_sig = hmac.new(self.secret, msg_to_verify, hashlib.sha256).digest()

                if not hmac.compare_digest(received_sig, expected_sig):
                    logger.warning("HMAC verification failed")
                    return None

                return payload

            elif mode == MODE_NONE:
                return message[HEADER_SIZE:]

        except InvalidTag:
            logger.warning(f"Decryption failed: encryption key or data integrity issue")
            return None
        except Exception as e:
            logger.error(f"Unknown error while decoding: {e} ({type(e)})")
            return None

    def encode_text(self, data: bytes) -> str:
        """
        Encodes a text-safe BPP packet to a Base64URL string.
        """
        binary_packet = self.encode(data)
        return base64.b64encode(binary_packet).decode("ascii")

    def decode_text(self, b64_str: str) -> bytes | None:
        """
        Decodes a text-safe BPP packet from a Base64URL string.
        """
        try:
            binary_packet = base64.b64decode(b64_str)
            return self.decode(binary_packet)

        except Exception as e:
            logger.warning(f"Text decode failed: {e}")
            return None
