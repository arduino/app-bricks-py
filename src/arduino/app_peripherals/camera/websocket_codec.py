# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import json
import base64
import time
import hashlib
from abc import ABC, abstractmethod
import hmac
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from arduino.app_utils.logger import Logger

logger = Logger("Codecs")


class MessageCodec(ABC):
    """
    Abstract base class for message encoding/decoding.

    Handles any type of data with authentication or encryption, letting the
    caller decide how to interpret the decoded byte.
    """

    def __init__(self, secret: str = "", enable_encryption: bool = False):
        """
        Initialize codec with optional authentication/encryption.

        Args:
            secret: Pre-shared secret for authentication/encryption (empty = no security)
            enable_encryption: Use encryption (ChaCha20-Poly1305) instead of just auth (HMAC-SHA256)
        """
        self.secret = secret.encode() if secret else b""
        self.enable_encryption = enable_encryption and bool(secret)

        if secret:
            if enable_encryption:
                # ChaCha20-Poly1305 for authenticated encryption (fastest AEAD)
                # Derive 256-bit key from secret
                key = hashlib.sha256(self.secret).digest()
                self.cc_cipher = ChaCha20Poly1305(key)
            else:
                # HMAC-SHA256 for authentication only
                self.cc_cipher = None
        else:
            self.cc_cipher = None

        self._last_rx_timestamp = 0  # For replay protection on received messages
        self._last_tx_timestamp = 0  # For generating unique timestamps on sent messages

    def _get_next_timestamp(self) -> int:
        """Get next microsecond timestamp for outgoing messages."""
        timestamp_us = int(time.time() * 1_000_000)
        # Ensure strictly increasing
        if timestamp_us <= self._last_tx_timestamp:
            timestamp_us = self._last_tx_timestamp + 1
        self._last_tx_timestamp = timestamp_us
        return timestamp_us

    def _verify_timestamp(self, timestamp_us: int) -> bool:
        """Verify timestamp for replay protection on incoming messages."""
        current_time_us = int(time.time() * 1_000_000)
        time_diff_us = current_time_us - timestamp_us

        # Reject if timestamp is in future or too old (30 seconds)
        if time_diff_us < 0 or time_diff_us > 30_000_000:
            logger.warning(f"Timestamp outside valid window: {time_diff_us / 1_000_000:.2f}s")
            return False

        # Prevent replay - timestamp must be strictly increasing
        if timestamp_us <= self._last_rx_timestamp:
            logger.warning(f"Timestamp replay detected: {timestamp_us} <= {self._last_rx_timestamp}")
            return False

        self._last_rx_timestamp = timestamp_us
        return True

    # === ENCRYPTION/AUTHENTICATION HELPERS ===

    def _encrypt_data(self, data: bytes, timestamp_us: int) -> bytes:
        """Encrypt data with ChaCha20-Poly1305 (includes authentication)."""
        if not self.cc_cipher:
            return data

        # Use timestamp as part of the nonce for uniqueness (96 bits total)
        # First 64 bits: timestamp, last 32 bits: zeros
        nonce = timestamp_us.to_bytes(8, "big") + b"\x00" * 4

        # Encrypt and authenticate in one step
        return self.cc_cipher.encrypt(nonce, data, str(timestamp_us).encode())

    def _decrypt_data(self, ciphertext: bytes, timestamp_us: int) -> bytes | None:
        """Decrypt data with ChaCha20-Poly1305 (includes authentication)."""
        if not self.cc_cipher:
            return ciphertext

        if not self._verify_timestamp(timestamp_us):
            return None

        # Reconstruct nonce
        nonce = timestamp_us.to_bytes(8, "big") + b"\x00" * 4

        try:
            # Decrypt and verify in one step
            return self.cc_cipher.decrypt(nonce, ciphertext, str(timestamp_us).encode())
        except Exception as e:
            logger.warning(f"Decryption/authentication failed: {e}")
            return None

    def _generate_signature(self, data: bytes, timestamp_us: int) -> bytes:
        """Generate HMAC-SHA256 for authentication only."""
        if not self.secret:
            return b""

        message = data + str(timestamp_us).encode()
        return hmac.new(self.secret, message, hashlib.sha256).digest()

    def _verify_signature(self, data: bytes, mac: bytes, timestamp_us: int) -> bool:
        """Verify HMAC-SHA256 for authentication only."""
        if not self.secret:
            return True

        if not self._verify_timestamp(timestamp_us):
            return False

        message = data + str(timestamp_us).encode()
        expected_mac = hmac.new(self.secret, message, hashlib.sha256).digest()
        return hmac.compare_digest(mac, expected_mac)

    # === ABSTRACT METHODS FOR IMPLEMENTATION ===

    @abstractmethod
    def decode(self, message: bytes) -> bytes | None:
        """
        Decode and authenticate/decrypt incoming message.
        Returns raw bytes if valid, None if authentication fails.
        """
        pass

    @abstractmethod
    def encode(self, data: bytes) -> bytes:
        """
        Encode and authenticate/encrypt outgoing data.
        Returns formatted message based on codec type (binary/json).
        """
        pass


class BinaryCodec(MessageCodec):
    """
    Binary protocol with authentication or authentication+encryption.

    No security format: [data]
    Authenticated format: [sig_len:1][signature][timestamp:8][data]
    Encrypted format: [encrypted_len:4][timestamp:8][encrypted_data]
    """

    def decode(self, message: bytes) -> bytes | None:
        """Decode binary message with security."""
        try:
            if self.enable_encryption:
                if len(message) < 12:
                    return None

                encrypted_len = int.from_bytes(message[0:4], "big")
                timestamp_us = int.from_bytes(message[4:12], "big")
                encrypted_data = message[12 : 12 + encrypted_len]

                return self._decrypt_data(encrypted_data, timestamp_us)

            elif self.secret:
                if len(message) < 1:
                    return None

                sig_len = message[0]
                if len(message) - 1 - 8 < sig_len:
                    return None

                signature = message[1 : 1 + sig_len]
                timestamp_us = int.from_bytes(message[1 + sig_len : 1 + sig_len + 8], "big")
                data = message[1 + sig_len + 8 :]

                if not self._verify_signature(data, signature, timestamp_us):
                    logger.warning("Binary message authentication failed")
                    return None

                return data

            else:
                return message

        except Exception as e:
            logger.warning(f"Error decoding binary message: {e}")
            return None

    def encode(self, data: bytes) -> bytes:
        """Encode data to binary with security."""
        if self.enable_encryption:
            timestamp_us = self._get_next_timestamp()
            encrypted = self._encrypt_data(data, timestamp_us)
            return len(encrypted).to_bytes(4, "big") + timestamp_us.to_bytes(8, "big") + encrypted

        elif self.secret:
            timestamp_us = self._get_next_timestamp()
            signature = self._generate_signature(data, timestamp_us)
            return len(signature).to_bytes(1) + signature + timestamp_us.to_bytes(8, "big") + data

        else:
            return data


class JsonCodec(MessageCodec):
    """
    JSON protocol with authentication or authentication+encryption.

    No security: {"data": "base64_data"}
    Authenticated: {"data": "base64_data", "timestamp": ..., "signature": "..."}
    Encrypted: {"data": "encrypted_base64_data", "timestamp": ...}
    """

    def decode(self, message: bytes) -> bytes | None:
        """Decode JSON message with security, returns raw bytes."""
        try:
            payload = json.loads(message)

            if "data" not in payload:
                logger.warning("JSON message missing data field")
                return None
            if (self.enable_encryption or self.secret) and "timestamp" not in payload:
                logger.warning("JSON message missing timestamp field")
                return None

            if self.enable_encryption:
                timestamp_us = payload.get("timestamp", 0)
                encrypted_b64 = payload["data"]
                encrypted_data = base64.b64decode(encrypted_b64)
                return self._decrypt_data(encrypted_data, timestamp_us)

            elif self.secret:
                timestamp_us = payload.get("timestamp", 0)
                signature_b64 = payload.get("signature")
                signature = base64.b64decode(signature_b64)
                data_b64 = payload.get("data")
                data = base64.b64decode(data_b64)

                if not self._verify_signature(data, signature, timestamp_us):
                    logger.warning("JSON message authentication failed")
                    return None

                return data

            else:
                # No security - extract raw data
                data_b64 = payload.get("data")
                return base64.b64decode(data_b64)

        except Exception as e:
            logger.warning(f"Error decoding JSON message: {e}")
            return None

    def encode(self, data: bytes) -> bytes:
        """Encode data to JSON with security."""
        if self.enable_encryption:
            timestamp_us = self._get_next_timestamp()
            encrypted_data = self._encrypt_data(data, timestamp_us)
            encrypted_data_b64 = base64.b64encode(encrypted_data).decode("latin-1")
            message = json.dumps({"data": encrypted_data_b64, "timestamp": timestamp_us})
            return message.encode("latin-1")

        elif self.secret:
            timestamp_us = self._get_next_timestamp()
            signature = self._generate_signature(data, timestamp_us)
            signature_b64 = base64.b64encode(signature).decode("latin-1")
            data_b64 = base64.b64encode(data).decode("latin-1")
            message = json.dumps({"data": data_b64, "timestamp": timestamp_us, "signature": signature_b64})
            return message.encode("latin-1")

        else:
            data_b64 = base64.b64encode(data).decode("latin-1")
            message = json.dumps({"data": data_b64})
            return message.encode("latin-1")
