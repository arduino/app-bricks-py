# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

# Official build steps: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/linux.md
FROM --platform=linux/amd64 ghcr.io/snapdragon-toolchain/arm64-linux:v0.1

ENV LLAMA_CPP_VERSION=b9016

COPY ./tools-download.patch /tmp/tools-download.patch
COPY ./CMakeUserPresets.json /tmp/CMakeUserPresets.json

RUN mkdir /workspace; \
    cd /workspace; \
    git clone https://github.com/ggml-org/llama.cpp; \
    cd llama.cpp; \
    git checkout ${LLAMA_CPP_VERSION}; \
    git apply /tmp/tools-download.patch; \
    cp /tmp/CMakeUserPresets.json docs/backend/snapdragon/CMakeUserPresets.json; \
    cmake --preset arm64-linux-snapdragon-release -B build-snapdragon; \
    cmake --build build-snapdragon -j $(nproc); \
    cmake --install build-snapdragon --prefix pkg-snapdragon; \
    rm -fr pkg-snapdragon/bin/test*; \
    tar -czvf llama-cpp-snapdragon.tar.gz pkg-snapdragon; \
    mv llama-cpp-snapdragon.tar.gz /output/llamacpp-hexagon.tgz

# TODO: save output to extenal volume
