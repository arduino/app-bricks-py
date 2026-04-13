# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM --platform=linux/amd64 ghcr.io/snapdragon-toolchain/arm64-linux:v0.1

ENV LLAMA_CPP_VERSION=b8778

COPY ./tools-download.patch /tmp/tools-download.patch

RUN mkdir /workspace; \
    cd /workspace; \
    git clone https://github.com/ggml-org/llama.cpp; \
    cd llama.cpp; \
    git checkout ${LLAMA_CPP_VERSION}; \
    git apply /tmp/tools-download.patch; \
    cp docs/backend/snapdragon/CMakeUserPresets.json . ; \
    cmake --preset arm64-linux-snapdragon-release -B build-snapdragon; \
    cmake --build build-snapdragon -j $(nproc); \
    cmake --install build-snapdragon --prefix pkg-snapdragon; \
    rm -fr pkg-snapdragon/bin/test*; \
    tar -czvf llama-cpp-snapdragon.tar.gz pkg-snapdragon

# TODO: save output to extenal volume
