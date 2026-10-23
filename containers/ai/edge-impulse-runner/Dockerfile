# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ghcr.io/astral-sh/uv:0.10.3@sha256:7a88d4c4e6f44200575000638453a5a381db0ae31ad5c3a51b14f8687c9d93a3 AS uv

FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

# The server, its locked packages and the out-of-the-box models come from the "server" context,
# containers/ai/edge-impulse-server, shared with edge-impulse-npu-runner and linked in docker-bake.hcl.
# Standalone build: docker build --build-context server=../edge-impulse-server .
COPY --from=server ./src /app
COPY --from=server --chown=arduino:arduino --chmod=755 ./models/ei-ootb-models/arm64/*.eim /models/ootb/ei/
COPY --from=server ./pyproject.toml ./uv.lock /tmp/deps/

# Runtime settings expected from the compose file:
# - the .eim files mounted read-only at /models
# - a host directory at /ipc for the socket
# - a read_only rootfs with a tmpfs /tmp for the .eim sockets
# - shm_size for the feature buffers
# - cap_drop ALL
# uv is only mounted while installing, it is not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    set -ex; \
    apt-get update -y; \
    # libatomic1 is linked by the .eim binaries and is the only library the base does not carry
    apt-get install -y --no-install-recommends --no-install-suggests libatomic1; \
    rm -rf /var/lib/apt/lists/*; \
    uv export --frozen --project /tmp/deps | uv pip install --system --no-cache-dir --require-hashes -r -; \
    rm -rf /tmp/deps; \
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
