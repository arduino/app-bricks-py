# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM debian:trixie AS llamacpp-builder

ENV LLAMA_CPP_VERSION=b8407

# Install Dependencies
RUN export DEBIAN_FRONTEND=noninteractive; \
    apt update; \
    mkdir -p ~/dev/llm; \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    ninja-build \
    ca-certificates \
    libgomp1 \
    libssl-dev:arm64 \
    && update-ca-certificates

COPY ./tools-download.patch /tmp/tools-download.patch

RUN cd ~/dev/llm; \
    git clone https://github.com/ggml-org/llama.cpp; \
    cd llama.cpp; \
    git checkout ${LLAMA_CPP_VERSION}; \
    git apply /tmp/tools-download.patch; \
    mkdir -p build; \
    cd build; \
    cmake .. -G Ninja \
		-DCMAKE_SYSTEM_NAME=Linux \
		-DCMAKE_SYSTEM_PROCESSOR=aarch64 \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_BUILD_TYPE=Release \
		-DLLAMA_OPENSSL=ON \
		-DGGML_NATIVE=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		-DGGML_BACKEND_DL=ON \
		-DGGML_USE_CPU_REPACK=ON \
		-DGGML_CPU_ALL_VARIANTS=ON; \
    ninja -j`nproc`

RUN cd ~/dev/llm/llama.cpp/build/; \
    mv bin bin-full; \
    mkdir -p bin; \
    cp bin-full/llama-cli bin/; \
    cp bin-full/llama-server bin/; \
    cp bin-full/llama-pull bin/; \
    cp bin-full/lib* bin/

FROM ${REGISTRY}app-bricks/base:${BASE_IMAGE_VERSION} AS production

COPY --from=llamacpp-builder /root/dev/llm/llama.cpp/build/bin /usr/local/bin/
COPY ./scripts/*.sh /

RUN set -ex; \
    mkdir -p /models; \
    chown arduino:arduino /models; \
    chmod +x /usr/local/bin/*; \
    chmod +x /*.sh

EXPOSE 9999

USER arduino

ENV HOME=/home/arduino
ENV USER=arduino

ENTRYPOINT ["/run.sh"]
