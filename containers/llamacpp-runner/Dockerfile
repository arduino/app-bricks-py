# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION}

ARG LLAMA_CPP_URL
ARG LLAMA_CPP_DIGEST

ADD --checksum=${LLAMA_CPP_DIGEST} ${LLAMA_CPP_URL} /tmp/llamacpp.tar.gz

COPY ./scripts/* /

RUN set -x; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --no-install-suggests libgomp1; \
    apt-get dist-clean; \
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
    python -m compileall /usr/local/lib    

EXPOSE 9999

USER arduino

ENV HOME=/home/arduino
ENV USER=arduino

ENV LD_LIBRARY_PATH=/opt/pkg-cpu/lib

ENTRYPOINT ["/run-model-router.sh"]
