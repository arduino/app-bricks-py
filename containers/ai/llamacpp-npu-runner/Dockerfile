# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
ARG LLAMA_CPP_URL=https://github.com/arduino/app-bricks-py/releases/download/llamacpp%2F20261113-2/llamacpp-hexagon-20261113-2.tar.gz
ARG LLAMA_CPP_DIGEST=sha256:8cab02c679fe0063c91630df573051f6496456bf8e0edb76beecf98e5d58a703

FROM ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv

FROM scratch AS llamacpp
ARG LLAMA_CPP_URL
ARG LLAMA_CPP_DIGEST
ADD --checksum=${LLAMA_CPP_DIGEST} ${LLAMA_CPP_URL} /llamacpp.tar.gz

FROM ${REGISTRY}app-bricks/qairt-common-base:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

COPY ./scripts/* /
COPY ./pyproject.toml ./uv.lock /tmp/deps/

# uv and the llama.cpp tarball are only mounted while installing, they are not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=from=llamacpp,source=/llamacpp.tar.gz,target=/mnt/llamacpp.tar.gz \
    set -ex; \
    uv export --frozen --project /tmp/deps | uv pip install --system --no-cache-dir --require-hashes -r -; \
    rm -rf /tmp/deps; \
    tar -xzf /mnt/llamacpp.tar.gz -C /opt; \
    # keep the HTP skels of the supported NPUs (v73, v75), drop the other Hexagon versions
    find /opt/pkg-snapdragon/lib -name 'libggml-htp-v*.so' ! -name 'libggml-htp-v73.so' ! -name 'libggml-htp-v75.so' -delete; \
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
