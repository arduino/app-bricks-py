# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG DEBIAN_FRONTEND=noninteractive

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/qairt-common-base:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

COPY requirements-base.txt .

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN set -ex; \
    # Make the venv the default for child images
    python -m venv --system-site-packages /opt/venv; \
    # Install Python dependencies into the venv
    /opt/venv/bin/pip install --no-cache-dir -r requirements-base.txt; \
    /opt/venv/bin/python -m compileall /opt/venv/lib; \
    rm requirements-base.txt

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

ENV PYTHONPATH="/opt"
COPY aihub/ /opt/aihub/

# Prepare the app directory for child images
RUN mkdir -p /app
WORKDIR /app
COPY main.py inference.py requirements.txt ./

ENTRYPOINT [ "python", "main.py", "--input", "websocket", "--output", "mjpeg", "websocket" ]