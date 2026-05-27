# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Initialize a remote sensor"
from arduino.app_peripherals.remote_sensor import RemoteSensor


# Default configuration: plaintext WebSocket server on port 8080, no authentication
default = RemoteSensor()

# Custom port and timeout
custom = RemoteSensor(port=9000, timeout=5)

# Authenticated mode (HMAC-SHA256): clients must share the same secret
authenticated = RemoteSensor(secret="my-shared-secret")

# Encrypted mode (ChaCha20-Poly1305): wraps payloads with authenticated encryption
encrypted = RemoteSensor(secret="my-shared-secret", encrypt=True)

# Transport-level security via TLS (clients connect with wss://)
secure = RemoteSensor(use_tls=True)
