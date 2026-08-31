# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/qairt-common-base:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_URL
ARG LLAMA_CPP_DIGEST

ADD --checksum=${LLAMA_CPP_DIGEST} ${LLAMA_CPP_URL} /tmp/llamacpp.tar.gz

COPY ./scripts/* /
COPY ./models-list.yaml /models-list.yaml

RUN set -ex; \
    # pyyaml: configure-llamacpp.py parses /models-list.yaml
    pip install --no-cache-dir pyyaml; \
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
