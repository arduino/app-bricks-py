# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG DEBIAN_FRONTEND=noninteractive

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ghcr.io/astral-sh/uv:0.10.3@sha256:7a88d4c4e6f44200575000638453a5a381db0ae31ad5c3a51b14f8687c9d93a3 AS uv

FROM ${REGISTRY}app-bricks/qairt-common-base:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

COPY pyproject.toml uv.lock /tmp/deps/

ENV PYTHONUNBUFFERED=1

# uv is only mounted while installing, it is not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    set -ex; \
    # Make the venv the default for child images
    python -m venv --system-site-packages /opt/venv; \
    # Install the locked Python dependencies into the venv
    uv export --frozen --project /tmp/deps | uv pip install --python /opt/venv/bin/python --no-cache-dir --require-hashes -r -; \
    /opt/venv/bin/python -m compileall /opt/venv/lib; \
    rm -rf /tmp/deps

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

ENV PYTHONPATH="/opt"
COPY aihub/ /opt/aihub/

# Prepare the app directory for child images
RUN mkdir -p /app
WORKDIR /app
COPY main.py inference.py requirements.txt ./

ENTRYPOINT [ "/qairt-entrypoint.sh", "python", "main.py", "--input", "websocket", "--output", "mjpeg", "websocket" ]