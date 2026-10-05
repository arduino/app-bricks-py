# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG DEBIAN_FRONTEND=noninteractive

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ghcr.io/astral-sh/uv:0.12.21@sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711 AS uv

FROM ${REGISTRY}app-bricks/qairt-common-base:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

COPY pyproject.toml uv.lock /tmp/deps/

ENV PYTHONUNBUFFERED=1

# uv is only mounted while installing, it is not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    set -ex; \
    # Make the venv the default for child images
    python -m venv --system-site-packages /opt/venv; \
    # The framework's own dependencies plus what every LiteRT runner on this base needs
    # (OpenCV and the LiteRT interpreter): installing them here rather than in each runner
    # keeps one copy of the wheels instead of one per runner.
    uv export --frozen --project /tmp/deps | uv pip install --python /opt/venv/bin/python --no-cache-dir --require-hashes -r -; \
    /opt/venv/bin/python -m compileall /opt/venv/lib; \
    rm -rf /tmp/deps

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

# The `aihub` framework and the app skeleton are shared with aihub-onnx-models-runner:
# "framework" is the bake context pointing at containers/base/aihub-framework.
ENV PYTHONPATH="/opt"
COPY --from=framework aihub/ /opt/aihub/

# Prepare the app directory for child images
RUN mkdir -p /app
WORKDIR /app
COPY --from=framework app/ ./

ENTRYPOINT [ "/qairt-entrypoint.sh", "python", "main.py", "--input", "websocket", "--output", "mjpeg", "websocket" ]
