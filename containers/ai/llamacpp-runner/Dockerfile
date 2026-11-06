# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
ARG LLAMA_CPP_URL=https://github.com/rjtokenring/app-bricks-py/releases/download/llamacpp%2F20261106/llamacpp-cpu-20261106.tar.gz
ARG LLAMA_CPP_DIGEST=sha256:622ed53798e89da9d014380ab6dcecc862387c05b982325cb97bebab66fbe6fc

FROM ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv

FROM scratch AS llamacpp
ARG LLAMA_CPP_URL
ARG LLAMA_CPP_DIGEST
ADD --checksum=${LLAMA_CPP_DIGEST} ${LLAMA_CPP_URL} /llamacpp.tar.gz

FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

COPY ./scripts/* /
COPY ./pyproject.toml ./uv.lock /tmp/deps/

# uv and the llama.cpp tarball are only mounted while installing, they are not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=from=llamacpp,source=/llamacpp.tar.gz,target=/mnt/llamacpp.tar.gz \
    set -ex; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --no-install-suggests libgomp1; \
    apt-get dist-clean; \
    uv export --frozen --project /tmp/deps | uv pip install --system --no-cache-dir --require-hashes -r -; \
    tar -xzf /mnt/llamacpp.tar.gz -C /opt; \
    chown -R arduino:arduino /opt/pkg-cpu; \
    chmod +x /opt/pkg-cpu/bin/*; \
    mkdir -p /models; \
    chown arduino:arduino /models; \
    chmod +x /*.sh; \
    # cleanup unnecessary files to reduce image size
    rm -rf /tmp/* /var/tmp/*; \
    rm -rf /var/lib/apt/lists/*; \
    rm -fr /var/cache/apt/archives/*.deb \
        /var/cache/apt/archives/partial/*.deb \
        /usr/share/man/* \
        /usr/share/locale/* \
        /var/cache/debconf/* \
        /var/lib/dpkg/info/* \
        /var/cache/apt/*.bin; \
    find /usr/share/doc -type f ! -name 'copyright' -delete; \
    find /usr/share/doc -type d -empty -delete; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib; \
    ldconfig

EXPOSE 9999

USER arduino

ENV HOME=/home/arduino
ENV USER=arduino

ENV LD_LIBRARY_PATH=/opt/pkg-cpu/lib

ENTRYPOINT ["/run-model-router.sh"]
