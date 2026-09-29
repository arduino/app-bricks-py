# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG DEBIAN_FRONTEND=noninteractive

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

# The `aihub` framework and the app skeleton are shared with aihub-onnx-models-runner
# and live in their own source-only image; see containers/base/aihub-framework.
FROM ${REGISTRY}app-bricks/aihub-framework:${BASE_IMAGE_VERSION} AS framework

FROM ${REGISTRY}app-bricks/qairt-common-base:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

COPY --from=framework /opt/aihub-dist/requirements-base.txt .
COPY requirements.txt .

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN set -ex; \
    # Make the venv the default for child images
    python -m venv --system-site-packages /opt/venv; \
    # The framework's own dependencies, then what every LiteRT runner on this base needs
    # (OpenCV and the LiteRT interpreter): installing them here rather than in each runner
    # keeps one copy of the wheels instead of one per runner.
    /opt/venv/bin/pip install --no-cache-dir -r requirements-base.txt; \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt; \
    /opt/venv/bin/python -m compileall /opt/venv/lib; \
    rm requirements-base.txt requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

ENV PYTHONPATH="/opt"
COPY --from=framework /opt/aihub-dist/aihub/ /opt/aihub/

# Prepare the app directory for child images
RUN mkdir -p /app
WORKDIR /app
COPY --from=framework /opt/aihub-dist/app/ ./

ENTRYPOINT [ "/qairt-entrypoint.sh", "python", "main.py", "--input", "websocket", "--output", "mjpeg", "websocket" ]
