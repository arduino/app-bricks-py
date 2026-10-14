# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG DEBIAN_FRONTEND=noninteractive

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv

# The `aihub` framework and the app skeleton, shared with aihub-onnx-models-runner.
FROM ${REGISTRY}app-bricks/aihub-framework:${BASE_IMAGE_VERSION} AS aihub-framework

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

# The `aihub` framework, importable as `aihub`, and the app skeleton: the layers of aihub-framework
# themselves, not a copy of their files. --link builds each one apart from what is below it, so
# it comes out as the very blob of the layer image, shared with aihub-onnx-models-runner
# and never rebuilt when the base changes. The /app it carries is root's, like the runners.
ENV PYTHONPATH="/opt"
COPY --link --from=aihub-framework /opt/aihub/ /opt/aihub/

# Prepare the app directory for child images
RUN mkdir -p /app
WORKDIR /app
COPY --link --from=aihub-framework /app/ ./

ENTRYPOINT [ "/qairt-entrypoint.sh", "python", "main.py", "--input", "websocket", "--output", "mjpeg", "websocket" ]
