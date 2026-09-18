# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/aihub-litert-models-runner:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

# Code and models only: OpenCV and ai-edge-litert, which this runner used to install
# itself, now come from the base image, where every LiteRT runner shares one copy.
COPY . .
