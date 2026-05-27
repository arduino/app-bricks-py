# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

FROM python:3.13-slim

ARG LLAMA_CPP_URL
ARG LLAMA_CPP_DIGEST

ADD --checksum=${LLAMA_CPP_DIGEST} ${LLAMA_CPP_URL} /tmp/llamacpp.tar.gz

COPY ./scripts/* /

RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 ubuntu \
    && useradd -u 1000 -g ubuntu -ms /bin/bash ubuntu \
    && tar -xzf /tmp/llamacpp.tar.gz -C /opt \
    && rm /tmp/llamacpp.tar.gz \
    && chown -R ubuntu:ubuntu /opt/pkg-cpu \
    && chmod +x /opt/pkg-cpu/bin/* \
    && mkdir -p /models \
    && chown ubuntu:ubuntu /models \
    && chmod +x /*.sh

EXPOSE 9999

USER ubuntu

ENV HOME=/home/ubuntu
ENV USER=ubuntu

ENV LD_LIBRARY_PATH=/opt/pkg-cpu/lib

ENTRYPOINT ["/run-model-router.sh"]
