# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION}

COPY ./requirements.txt /app/requirements.txt

COPY --chown=arduino:arduino ./src /app

RUN set -ex; \
    chown -R arduino:arduino /app; \
    pip install --no-cache-dir -r /app/requirements.txt && rm /app/requirements.txt; \
    # ensure permissions for /run.sh
    chmod 755 /*.sh; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib

USER arduino

WORKDIR /app

ENV HOME=/app
ENV USER=arduino
