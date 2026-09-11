# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

# Copy requirements file
COPY --chown=root:root ./requirements.txt /app/requirements.txt

COPY --chown=root:root ./src /app

# Install WiFi scanning tools and Python dependencies
# Note: this container requires the following runtime settings to allow iw scanning:
#   network_mode: host
#   cap_add: [NET_ADMIN, NET_RAW]
RUN set -ex; \
    apt-get update -y && apt-get install -y --no-install-recommends --no-install-suggests iw wireless-tools; \
    python -m pip install --no-cache-dir -r /app/requirements.txt && rm /app/requirements.txt; \
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib; \
    chmod +x /app/start-scan-server.sh

# Run as root: iw scan requires NET_ADMIN capability which is granted at runtime
USER root
ENV USER=root

WORKDIR /app
ENV HOME=/app

ENTRYPOINT ["/app/start-scan-server.sh"]
