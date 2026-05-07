# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/aihub-models-runner:${BASE_IMAGE_VERSION}

ENV LLAMA_CPP_VERSION=20260504

COPY ./llamacpp-hexagon-${LLAMA_CPP_VERSION}.tgz /tmp/llamacpp-hexagon.tgz
COPY ./scripts/* /

RUN set -ex; \
    tar -xzf /tmp/llamacpp-hexagon.tgz -C /opt; \
    rm /tmp/llamacpp-hexagon.tgz; \
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
