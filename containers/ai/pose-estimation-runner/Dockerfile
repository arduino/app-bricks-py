# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ghcr.io/astral-sh/uv:0.10.3@sha256:7a88d4c4e6f44200575000638453a5a381db0ae31ad5c3a51b14f8687c9d93a3 AS uv

FROM ${REGISTRY}app-bricks/aihub-models-runner:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

COPY . .

# uv is only mounted while installing, it is not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    set -ex; \
    uv export --frozen | uv pip install --python /opt/venv/bin/python --no-cache-dir --require-hashes -r -; \
    rm pyproject.toml uv.lock; \
    python -m compileall /opt/venv
