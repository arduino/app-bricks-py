# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/aihub-models-runner:${BASE_IMAGE_VERSION}

RUN mkdir -p /home/app

WORKDIR /home/app

COPY . .