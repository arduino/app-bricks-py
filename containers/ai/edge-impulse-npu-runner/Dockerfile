# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ghcr.io/astral-sh/uv:0.10.3@sha256:7a88d4c4e6f44200575000638453a5a381db0ae31ad5c3a51b14f8687c9d93a3 AS uv

FROM ${REGISTRY}app-bricks/qairt-common-base:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

# The server, its locked packages and the out-of-the-box models come from the "server" context,
# containers/ai/edge-impulse-server, shared with edge-impulse-runner and linked in docker-bake.hcl.
# Standalone build: docker build --build-context server=../edge-impulse-server .
COPY --from=server ./src /app
COPY --from=server --chown=arduino:arduino --chmod=755 ./models/ei-ootb-models/arm64/*.eim /models/ootb/ei/
COPY --from=server ./pyproject.toml ./uv.lock /tmp/deps/
# The QNN builds of the out-of-the-box models, next to the CPU ones
COPY --chown=arduino:arduino --chmod=755 ./models/ei-ootb-models/arm64/*.eim /models/ootb/ei/

# Runtime settings expected from the compose file:
# - the QNN .eim files mounted read-only at /models
# - a host directory at /ipc for the socket
# - the fastrpc and DMA heap devices
# - the device model and the host DSP payload mounted for /qairt-entrypoint.sh
# - a tmpfs /tmp for the .eim sockets
# - shm_size for the feature buffers
# - cap_drop ALL
# uv is only mounted while installing, it is not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    set -ex; \
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
# The server checks the NPU devices at startup and fails without them
ENV EI_ACCEL=qnn

# The entrypoint wrapper merges the host DSP payload into /usr/share/qcom, then runs the server.
# Base arguments, the compose file appends the limits of the board in "command"
ENTRYPOINT ["/qairt-entrypoint.sh", "python3", "/app/inference_server.py", "--socket=/ipc/ei.sock", "--models-dir=/models"]
