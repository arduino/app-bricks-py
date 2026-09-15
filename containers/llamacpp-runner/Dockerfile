# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ghcr.io/astral-sh/uv:0.10.3@sha256:7a88d4c4e6f44200575000638453a5a381db0ae31ad5c3a51b14f8687c9d93a3 AS uv

FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_URL=https://github.com/arduino/app-bricks-py/releases/download/llamacpp%2F20260914/llamacpp-cpu-20260914.tar.gz
ARG LLAMA_CPP_DIGEST=sha256:e6bd5706ce4826842d6a4ad770a6bf825c6bf158fa9aab00561476e6995b5fc6

ADD --checksum=${LLAMA_CPP_DIGEST} ${LLAMA_CPP_URL} /tmp/llamacpp.tar.gz

COPY ./scripts/* /
COPY ./pyproject.toml ./uv.lock /tmp/deps/

# uv is only mounted while installing, it is not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    set -ex; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --no-install-suggests libgomp1; \
    apt-get dist-clean; \
    uv export --frozen --project /tmp/deps | uv pip install --system --no-cache-dir --require-hashes -r -; \
    tar -xzf /tmp/llamacpp.tar.gz -C /opt; \
    rm /tmp/llamacpp.tar.gz; \
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
