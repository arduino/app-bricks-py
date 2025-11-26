# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import struct
from typing import Iterator

from .bpp_codec import BPPCodec

MAGIC = 0xAA
HEADER_FORMAT = ">BIB"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 6 bytes


class BPPStreamCodec:
    """
    Wraps a BPPCodec to provide support for stream-based protocols (e.g. TCP,
    BLE, UART).

    The binary format is as follows:

    [Magic(1)] [Length(4)] [HeaderCRC(1)] [BPP Packet]

    - Magic Byte: 0xAA, marks the start of a BPP frame.
    - Length: 4-byte big-endian unsigned int indicating the length of the BPP packet.
    - HeaderCRC: Simple checksum over the Length and Magic byte for header integrity.
    - BPP Packet: The actual BPP-encoded packet as per BPPCodec.
    """

    def __init__(self, codec: BPPCodec):
        self.codec = codec
        self._buffer = bytearray()

    def encode(self, data: bytes) -> bytes:
        bpp_packet = self.codec.encode(data)
        length = len(bpp_packet)
        checksum = self._calc_header_checksum(length)
        header = struct.pack(HEADER_FORMAT, MAGIC, length, checksum)
        return header + bpp_packet

    def decode(self, chunk: bytes = b"") -> Iterator[bytes]:
        """
        Ingests a stream chunk and yields all fully decoded BPP payloads found.

        Yields:
            Decoded payloads (bytes)
        """
        if chunk:
            self._buffer.extend(chunk)

        while True:
            if not self._buffer:
                break

            # Look for the Magic byte
            if self._buffer[0] != MAGIC:
                try:
                    idx = self._buffer.index(MAGIC)
                    del self._buffer[:idx]
                except ValueError:
                    self._buffer.clear()
                    break

            # Do we have a full header?
            if len(self._buffer) < HEADER_SIZE:
                break  # Need more data, wait for next chunk

            magic, length, checksum = struct.unpack(HEADER_FORMAT, self._buffer[:HEADER_SIZE])
            if self._calc_header_checksum(length) != checksum:
                del self._buffer[0]
                continue

            total_frame_size = HEADER_SIZE + length

            # Do we have the full frame?
            if len(self._buffer) < total_frame_size:
                break  # Need more data, wait for next chunk

            bpp_raw = self._buffer[HEADER_SIZE:total_frame_size]
            del self._buffer[:total_frame_size]

            # Decode BPP payload and yield if valid
            payload = self.codec.decode(bytes(bpp_raw))
            if payload is not None:
                yield payload

    def _calc_header_checksum(self, length: int) -> int:
        """ "Calculates simple checksum for header integrity verification."""
        b0 = (length >> 24) & 0xFF
        b1 = (length >> 16) & 0xFF
        b2 = (length >> 8) & 0xFF
        b3 = length & 0xFF
        return (MAGIC ^ b0 ^ b1 ^ b2 ^ b3) & 0xFF
