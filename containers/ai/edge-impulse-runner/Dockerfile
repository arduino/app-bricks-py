# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ghcr.io/astral-sh/uv:0.10.3@sha256:7a88d4c4e6f44200575000638453a5a381db0ae31ad5c3a51b14f8687c9d93a3 AS uv

# The inference server and its locked packages, shared with edge-impulse-npu-runner
FROM ${REGISTRY}app-bricks/edge-impulse-server:${BASE_IMAGE_VERSION} AS edge-impulse-server

FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

# The server: the layer of edge-impulse-server itself, the very blob edge-impulse-npu-runner carries
COPY --link --from=edge-impulse-server /app/ /app/

# Runtime settings expected from the compose file:
# - the .eim files mounted read-only at /models
# - a host directory at /ipc for the socket
# - a read_only rootfs with a tmpfs /tmp for the .eim sockets
# - shm_size for the feature buffers
# - cap_drop ALL
# uv and the server's lock are only mounted while installing, they are not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=from=edge-impulse-server,source=/deps,target=/deps \
    set -ex; \
    apt-get update -y; \
    # libatomic1 is linked by the .eim binaries and is the only library the base does not carry
    apt-get install -y --no-install-recommends --no-install-suggests libatomic1; \
    rm -rf /var/lib/apt/lists/*; \
    uv export --frozen --project /deps | uv pip install --system --no-cache-dir --require-hashes -r -; \
    # The numpy test suites are never run by the server
    find /usr/local/lib/python3.13/site-packages -type d -name tests -prune -exec rm -rf {} +; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall -q /usr/local/lib /app

USER arduino

WORKDIR /app

ENV USER=arduino
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV EI_ACCEL=cpu

# Base arguments, the compose file appends the limits of the board in "command"
ENTRYPOINT ["python3", "/app/inference_server.py", "--socket=/ipc/ei.sock", "--models-dir=/models"]
