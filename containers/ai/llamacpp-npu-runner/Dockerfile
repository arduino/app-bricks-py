# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ghcr.io/astral-sh/uv:0.10.3@sha256:7a88d4c4e6f44200575000638453a5a381db0ae31ad5c3a51b14f8687c9d93a3 AS uv

FROM ${REGISTRY}app-bricks/qairt-common-base:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_URL
ARG LLAMA_CPP_DIGEST

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
