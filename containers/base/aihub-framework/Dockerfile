# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Source-only image: it carries the `aihub` runner framework and the app skeleton
# so that every runner base image gets them from one place.
#
# Nothing runs from this image. The runner bases have different runtimes
# (aihub-litert-models-runner is LiteRT/QAIRT on qairt-common-base,
# aihub-onnx-models-runner is ONNX Runtime/QNN on python-slim) and CI builds each
# container with its own directory as the build context, so a plain `COPY ../..`
# between them is not possible: they pick the files up with `COPY --from` on this
# image instead.
#
# Layout under /opt/aihub-dist, mirroring this directory:
#   aihub/                  the importable package -> /opt/aihub (PYTHONPATH=/opt)
#   app/                    the app skeleton -> /app; a runner overwrites
#                           inference.py (and requirements.txt) with its own
#   requirements-base.txt   the framework's own runtime dependencies

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}

COPY aihub/ /opt/aihub-dist/aihub/
COPY app/ /opt/aihub-dist/app/
COPY requirements-base.txt /opt/aihub-dist/requirements-base.txt
