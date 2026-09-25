# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

# ONNX Runtime with the QNN execution provider, the FastRPC client libraries and the aihub
# framework, on python-slim. Not aihub-litert-models-runner: that one carries the QAIRT
# SDK and the LiteRT delegate, and this runner uses neither - the QAIRT it talks to the
# NPU with is the one bundled in the onnxruntime-qnn wheel the base image installs.
FROM ${REGISTRY}app-bricks/aihub-onnx-models-runner:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

# Code, ONNX models and the HTP context binaries compiled on the board with
# tools/compile_htp_context.py, next to them. Nothing to install: onnxruntime,
# onnxruntime-qnn, numpy and OpenCV all come from the base image, which pins and
# hash-locks them once for every ONNX runner.
COPY . .
