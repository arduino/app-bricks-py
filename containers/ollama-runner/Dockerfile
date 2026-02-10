# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG OLLAMA_VERSION=0.15.4

FROM ollama/ollama:${OLLAMA_VERSION} AS ollama-builder


FROM ${REGISTRY}app-bricks/base:latest AS production

COPY --from=ollama-builder /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama-builder /usr/lib/ollama/libggml* /usr/lib/ollama/

COPY ./scripts/*.sh /

RUN set -ex; \
    chmod +x /usr/bin/ollama; \
    chmod +x /*.sh

# Ollama settings
ENV OLLAMA_HOST=0.0.0.0
ENV OLLAMA_PORT=11434
ENV OLLAMA_MODELS=/models

EXPOSE 11434

USER arduino

ENV HOME=/home/arduino
ENV USER=arduino

ENTRYPOINT ["/run.sh"]
