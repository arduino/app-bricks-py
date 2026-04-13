# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/base:${BASE_IMAGE_VERSION} AS production

ENV LLAMA_CPP_VERSION=b8777

COPY ./llamacpp-hexagon-${LLAMA_CPP_VERSION}.tgz /tmp/llamacpp-hexagon.tgz
COPY ./scripts/*.sh /

RUN set -ex; \
    tar -xzf /tmp/llamacpp-hexagon.tgz -C /opt; \
    rm /tmp/llamacpp-hexagon.tgz; \
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

ENTRYPOINT ["/run.sh"]
