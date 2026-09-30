# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ghcr.io/astral-sh/uv:0.12.21@sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711 AS uv

FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

COPY ./pyproject.toml ./uv.lock /tmp/deps/

COPY --chown=arduino:arduino --chmod=755 ./src /app

# From the bake context pointing at the repository's models/ directory
COPY --from=models --chown=arduino:arduino --chmod=644 models-list.yaml /app/models-list.yaml

# uv is only mounted while installing, it is not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    set -ex; \
    uv export --frozen --project /tmp/deps | uv pip install --system --no-cache-dir --require-hashes -r -; \
    rm -rf /tmp/deps; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib

USER arduino

WORKDIR /app

ENV HOME=/app
ENV USER=arduino
