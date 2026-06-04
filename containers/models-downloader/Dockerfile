# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION}

COPY --chown=arduino:arduino --chmod=755 ./requirements.txt /app/requirements.txt

COPY --chown=arduino:arduino --chmod=755 ./src /app

COPY --chown=arduino:arduino --chmod=644 ./models-list.yaml /app/models-list.yaml

RUN set -ex; \
    pip install --no-cache-dir -r /app/requirements.txt && rm /app/requirements.txt; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib

USER arduino

WORKDIR /app

ENV HOME=/app
ENV USER=arduino
