# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv

FROM ${REGISTRY}app-bricks/qairt-common-base:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_URL=https://github.com/arduino/app-bricks-py/releases/download/llamacpp%2F20261028/llamacpp-hexagon-20261028.tar.gz
ARG LLAMA_CPP_DIGEST=sha256:9ccf05899491098afbab59a44a99ddfd8c8478ca0068b3170d86c3cb095c3461

ADD --checksum=${LLAMA_CPP_DIGEST} ${LLAMA_CPP_URL} /tmp/llamacpp.tar.gz

COPY ./scripts/* /
COPY ./pyproject.toml ./uv.lock /tmp/deps/

# uv is only mounted while installing, it is not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    set -ex; \
    uv export --frozen --project /tmp/deps | uv pip install --system --no-cache-dir --require-hashes -r -; \
    rm -rf /tmp/deps; \
    tar -xzf /tmp/llamacpp.tar.gz -C /opt; \
    rm /tmp/llamacpp.tar.gz; \
    chown -R arduino:arduino /opt/pkg-snapdragon; \
    chmod +x /opt/pkg-snapdragon/bin/*; \
    mkdir -p /models; \
    chown arduino:arduino /models; \
    chmod +x /*.sh

EXPOSE 9999

USER arduino

ENV HOME=/home/arduino
ENV USER=arduino

ENV LD_LIBRARY_PATH=/opt/pkg-snapdragon/lib
ENV ADSP_LIBRARY_PATH=/opt/pkg-snapdragon/lib

ENTRYPOINT ["/qairt-entrypoint.sh", "/run-model-router.sh"]
