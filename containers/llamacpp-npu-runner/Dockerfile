# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/aihub-models-runner:${BASE_IMAGE_VERSION}

ARG LLAMA_CPP_GITHUB_REPO=arduino/app-bricks-py
ARG LLAMA_CPP_RELEASE_VERSION=20260504
ARG LLAMA_CPP_SHA256=0000000000000000000000000000000000000000000000000000000000000000

ADD --checksum=sha256:${LLAMA_CPP_SHA256} \
    https://github.com/${LLAMA_CPP_GITHUB_REPO}/releases/download/llamacpp/${LLAMA_CPP_RELEASE_VERSION}/llamacpp-${LLAMA_CPP_RELEASE_VERSION}.tar.gz \
    /tmp/llamacpp.tar.gz

COPY ./scripts/* /

RUN set -ex; \
    tar -xzf /tmp/llamacpp.tar.gz -C /opt; \
    rm /tmp/llamacpp.tar.gz; \
    chown -R ubuntu:ubuntu /opt/pkg-snapdragon; \
    chmod +x /opt/pkg-snapdragon/bin/*; \
    mkdir -p /models; \
    chown ubuntu:ubuntu /models; \
    chmod +x /*.sh

EXPOSE 9999

USER ubuntu

ENV HOME=/home/ubuntu
ENV USER=ubuntu

ENV LD_LIBRARY_PATH=/opt/pkg-snapdragon/lib
ENV ADSP_LIBRARY_PATH=/opt/pkg-snapdragon/lib

ENTRYPOINT ["/run-model-router.sh"]
