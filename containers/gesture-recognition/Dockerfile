# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/aihub-models-runner:${BASE_IMAGE_VERSION}

COPY . .

RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m compileall /opt/venv \
    && rm requirements.txt