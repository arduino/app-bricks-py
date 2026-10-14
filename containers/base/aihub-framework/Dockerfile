# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# The `aihub` runner framework and the app skeleton, as a layer: an image holding only those
# files, at the paths where they land in the images using them. It is never run. An image
# imports it through a stage of its own and adds its layers:
#
#   ARG REGISTRY
#   ARG BASE_IMAGE_VERSION=latest
#   FROM ${REGISTRY}app-bricks/aihub-framework:${BASE_IMAGE_VERSION} AS aihub-framework
#   FROM <its own base>
#   ...
#   ENV PYTHONPATH="/opt"
#   COPY --link --from=aihub-framework /opt/aihub/ /opt/aihub/
#   WORKDIR /app
#   COPY --link --from=aihub-framework /app/ ./
#
# and its docker-bake.hcl target links it with parent_context("aihub-framework"), next to its
# parent. With --link each COPY reproduces one layer below, byte for byte and whatever the base:
# the image carries the same blobs as this one. Without it, the files are copied into new layers.
# scratch has no Python: the framework's dependencies are installed by each image adding it,
# from its own pyproject.toml.

FROM scratch

COPY aihub/ /opt/aihub/
COPY app/ /app/
